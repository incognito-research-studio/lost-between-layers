"""E1 v2 scoring: offline and deterministic, from stored replies.

Per model, case and repeat, the stored replies go through the case pipeline again (the
snapshot each reply was given is recomputed and its hash compared), and the decision is
taken under four arms:
  point        the nominal gate on the interpreter's point readings
  stated       the box-robust gate on the intervals the model states
  calibrated   the box-robust gate on the stated intervals widened by the model's
               split-conformal margin q, fitted on the calibration cases only
  recheck      the system-of-record recheck (A-ORACLE holds by construction): a reference
Definitions (plant X is the planned entity; R(C) is X's capacity in weeks 40-49):
  joint coverage      every week's true X capacity lies in the reported interval (an
                      unchanged week reports its nominal point), i.e. S* in the reported
                      set: numeric, scope and entity errors all count as misses
  scope_exact         the weeks of X that changed are the weeks the notice affects
  entity_ok           the plants whose capacity changed are the plants the notice affects
  quiet_ok            for a case expected to change nothing at X, X is unchanged
  released            the arm executed an action
  violation           the executed action is infeasible for the true capacity (a
                      rejection executes nothing and is no violation): Pr[V] per case
  violation | release violations among released cases: Pr[V | R]
  excess cost         shortfall of the executed plan above the true optimum (units)
Calibration: for each calibration case where the interpreter changed some affected week
of X, the score is the largest distance by which a true value in those weeks falls
outside the stated interval; q is the ceil((n + 1)(1 - alpha))-th smallest score. Weeks the
interpreter left unchanged (scope misses, abstentions) cannot be widened and stay misses.
Rates are reported on the test cases, with 95% Wilson intervals.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from dsoi import properties as P
from dsoi.canon import state_hash
from dsoi.constraints import solve_lp
from dsoi.experiments.e1v2_notices import HORIZON, NOMINAL, Case
from dsoi.experiments.e1v2_run import parse
from dsoi.experiments.e1v2_world import (
    T_DECIDE,
    messages,
    pipeline,
    scenario,
    sha,
    snapshot_text,
    state_of,
)
from dsoi.layers.l1_interpreter import InterpretResult
from dsoi.layers.l5_gate import BoxRobustGate, NominalGate, RecheckGate
from dsoi.sim.intervals import wilson
from dsoi.types import Abstain, Delta, Quantity

ARMS = ("point", "stated", "calibrated", "recheck")


def _gate(arm, scn):
    if arm == "point":
        return NominalGate()
    if arm == "recheck":
        return RecheckGate(scn.true_reader(), fallback="resolve")
    return BoxRobustGate(mode="resolve")


def widen(result: InterpretResult, q: float) -> InterpretResult:
    """The calibrated reading: every capacity interval widened by q, within [0, nominal]."""
    if not isinstance(result.output, Delta) or q == 0:
        return result
    ops = []
    for op in result.output.ops:
        if op.p == "capacity" and isinstance(op.o, Quantity):
            lo = op.o.value if op.o.lo is None else op.o.lo
            hi = op.o.value if op.o.hi is None else op.o.hi
            op = replace(op, o=Quantity(op.o.value, max(0.0, lo - q), min(NOMINAL, hi + q),
                                        op.o.unit))
        ops.append(op)
    return replace(result, output=Delta(tuple(ops)), nu_i=f"{result.nu_i}+cal{q:g}")


class Replay:
    """The L1 for scoring: returns the stored reply for each message, after checking that
    the snapshot it was given equals the snapshot the pipeline now shows."""

    def __init__(self, replies: dict, nu_i: str, q: float = 0.0):
        self.replies, self.nu_i, self.q = replies, nu_i, q
        self.p, self.times, self.snapshot_ok = None, {}, True

    def interpret(self, message, schema):
        reply = self.replies.get(message.message_id)
        if reply is None:
            return InterpretResult(Abstain("no stored reply"), nu_i=self.nu_i)
        if sha(snapshot_text(self.p, self.times[message.message_id])) != reply["snapshot_sha"]:
            self.snapshot_ok = False
        return widen(parse(reply, self.nu_i, message, schema), self.q)


def _run_arm(case: Case, arm: str, replies: dict, nu_i: str, q: float = 0.0):
    scn = scenario(case)
    replay = Replay(replies, nu_i, q if arm == "calibrated" else 0.0)
    _, p = pipeline(case, _gate(arm, scn), replay, nu_i, historical=True)
    replay.p, replay.times = p, {m.message_id: t for m, t in messages(case)}
    outcomes = [p.ingest_message(m, t) for m, t in messages(case)]
    try:
        record = p.decide(T_DECIDE, scn.theta()).record
    except Exception:
        # A cycle that ends before execution (e.g. L4's MissingState after a delta that
        # retracted the state it reads) is logged by L6 as "error" and executes nothing.
        # Deviation from the frozen scorer, which let the exception through (DEVIATIONS.md).
        record = p.audit.records[-1]
        if record.outcome != "error":
            raise
    return scn, p, record, outcomes, replay.snapshot_ok


def _bounds(x: dict) -> tuple[dict, dict]:
    lo = {w: v[0] if v[1] is None else v[1] for w, v in x.items()}
    hi = {w: v[0] if v[2] is None else v[2] for w, v in x.items()}
    return lo, hi


def _covered(x: dict, truth: dict) -> bool:
    """A week without a value (deleted by a delta: DEVIATIONS.md) does not contain the truth."""
    lo, hi = _bounds(x)
    return all(w in lo and lo[w] - 1e-6 <= truth[w] <= hi[w] + 1e-6 for w in HORIZON)


def arm_row(case: Case, arm: str, replies: dict, nu_i: str, q: float = 0.0) -> dict:
    """Release, violation and excess cost of one arm; the calibrated arm adds its coverage."""
    scn, p, record, _, _ = _run_arm(case, arm, replies, nu_i, q)
    inst = p.formulation.build(scn.true_state(), scn.theta())
    best = inst.objective(solve_lp(inst.c, inst.system).x)
    executed = record.x_exec
    row = {f"released_{arm}": executed is not None,
           f"error_{arm}": record.outcome == "error",
           f"violation_{arm}": not P.k_exec_true(executed, scn.true_state(), scn.theta(),
                                                 p.formulation),
           f"excess_{arm}": (None if executed is None
                             else inst.objective(np.array(executed)) - best)}
    if arm == "calibrated":
        row["covered_calibrated"] = _covered(state_of(p, T_DECIDE).get("plant:X", {}),
                                             case.true_capacity("plant:X"))
    return row


def score_case(case: Case, replies: dict, model: str, nu_i: str, repeat: int) -> dict:
    """Everything except the calibrated arm, which needs q from the calibration rows."""
    row = {"model": model, "case_id": case.id, "category": case.category, "split": case.split,
           "expect": case.expect, "repeat": repeat}
    truth_x = case.true_capacity("plant:X")
    affected = {w for w in HORIZON if truth_x[w] != NOMINAL}
    _, p, _, outcomes, snap_ok = _run_arm(case, "stated", replies, nu_i)
    state = state_of(p, T_DECIDE)
    x = state.get("plant:X", {})
    # A week left without a value counts as changed; it cannot be covered by any margin,
    # so its calibration score is the margin that spans the domain (DEVIATIONS.md).
    missing = {w for w in HORIZON if w not in x}
    changed = ({e for e, ws in state.items() for v in ws.values() if v != (NOMINAL, None, None)}
               | {e for e, ws in state.items() if any(w not in ws for w in HORIZON)})
    changed_x = {w for w, v in x.items() if v != (NOMINAL, None, None)} | missing
    lo, hi = _bounds(x)
    hits = changed_x & affected
    present = hits - missing
    row.update({
        "outcomes": ",".join(outcomes), "abstained": "abstained" in outcomes,
        "snapshot_ok": snap_ok,
        "entity_ok": (changed == set(case.truth)) if case.truth else None,
        "scope_exact": changed_x == affected, "quiet_ok": not changed_x,
        "covered_stated": _covered(x, truth_x),
        "cal_score": (None if not hits else NOMINAL if hits & missing else
                      max(max(lo[w] - truth_x[w], truth_x[w] - hi[w], 0.0) for w in hits)),
        "width": float(np.mean([hi[w] - lo[w] for w in present])) if present else None,
        "state_hash": state_hash(p.state.read(p.decision_class, T_DECIDE)[0]),
    })
    for arm in ("point", "stated", "recheck"):
        row.update(arm_row(case, arm, replies, nu_i))
    return row


def calibrate(rows: list[dict], alpha: float) -> dict[str, float]:
    """q per model from its calibration rows (split-conformal, joint over the changed weeks)."""
    out = {}
    for model in sorted({r["model"] for r in rows}):
        scores = sorted(r["cal_score"] for r in rows
                        if r["model"] == model and r["split"] == "calibration"
                        and r["cal_score"] is not None)
        n = len(scores)
        k = math.ceil((n + 1) * (1 - alpha))
        out[model] = float(scores[k - 1]) if 0 < k <= n else NOMINAL
    return out


def _rate(rows, key):
    vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    k, n = sum(vals), len(vals)
    lo, hi = wilson(k, n)
    return {"rate": k / n, "k": k, "n": n, "ci95": [round(lo, 4), round(hi, 4)]}


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict], q: dict[str, float], alpha: float) -> dict:
    out = {"alpha": alpha, "margin_q": q, "models": {}, "by_category": {}}
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model and r["split"] == "test"]
        change = [r for r in rs if r["expect"] == "change"]
        quiet = [r for r in rs if r["expect"] != "change"]
        groups = defaultdict(list)
        for r in rs:
            groups[r["case_id"]].append(r)
        m = {
            "test_rows": len(rs), "change_rows": len(change), "quiet_rows": len(quiet),
            "snapshot_mismatches": sum(not r["snapshot_ok"] for r in rs),
            "abstained": _rate(change, "abstained"),
            "quiet_ok": _rate(quiet, "quiet_ok"),
            "entity_ok": _rate(rs, "entity_ok"),
            "scope_exact": _rate([r for r in change if r["category"] != "other_entity"],
                                 "scope_exact"),
            "coverage_stated": _rate(change, "covered_stated"),
            "coverage_calibrated": _rate(change, "covered_calibrated"),
            "mean_stated_width": _mean(change, "width"),
            "state_varies_across_repeats": float(np.mean(
                [len({r["state_hash"] for r in g}) > 1 for g in groups.values()]))
            if groups else None,
        }
        for arm in ARMS:
            released = [r for r in change if r.get(f"released_{arm}")]
            m[arm] = {
                "error": _rate(change, f"error_{arm}"),
                "release": _rate(change, f"released_{arm}"),
                "violation": _rate(change, f"violation_{arm}"),
                "violation_given_release": _rate(released, f"violation_{arm}"),
                "mean_excess_when_released": _mean(released, f"excess_{arm}"),
            }
        out["models"][model] = m
        out["by_category"][model] = {
            cat: {"rows": len(cr), "coverage_stated": _rate(cr, "covered_stated"),
                  "coverage_calibrated": _rate(cr, "covered_calibrated"),
                  "scope_exact": _rate(cr, "scope_exact"), "entity_ok": _rate(cr, "entity_ok"),
                  "quiet_ok": _rate(cr, "quiet_ok"), "abstained": _rate(cr, "abstained")}
            for cat in sorted({r["category"] for r in rs})
            for cr in [[r for r in rs if r["category"] == cat]]
        }
    return out


def score(cases: list[Case], replies: dict[tuple, dict], alpha: float, out_dir: Path,
          manifest: str) -> dict:
    """Two passes: the stated arms for every row (calibration scores), then q per model from
    the calibration rows, then the calibrated arm. Writes rows and the summary."""
    by_key: dict[tuple, dict] = defaultdict(dict)
    nu: dict[str, str] = {}
    for (model, case_id, repeat, message_id), rep in replies.items():
        by_key[(model, case_id, repeat)][message_id] = rep
        nu[model] = rep["nu_i"]
    by_id = {c.id: c for c in cases}
    keys = sorted(by_key)
    rows = [score_case(by_id[c], by_key[(m, c, r)], m, nu[m], r) for m, c, r in keys]
    q = calibrate(rows, alpha)
    for row, (m, c, r) in zip(rows, keys, strict=True):
        row.update(arm_row(by_id[c], "calibrated", by_key[(m, c, r)], nu[m], q[m]))
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(out_dir / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = {"manifest": manifest, **summarize(rows, q, alpha)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary

