"""E1 scoring: offline, deterministic, from stored replies.

Per reply: the L1 contract (parse, typing, spans), the committed interpreted
state, and interpreter metrics against the notice's ground truth:
  scope_exact     the weeks the interpreter changed equal the truly affected weeks
  covered         every affected week's true capacity lies in the reported interval (A2)
  unsafe_miss     some affected week's true capacity lies below the reported lower bound
  no_change_ok    for out-of-scope and vague notices: no capacity was changed
Per change notice, the interpreted state then goes through the nominal, box,
state-vertex and recheck gates, and K-exec is checked against the truth.
Across repeats: whether the interpreted state and the box-gate release vary.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dsoi import properties as P
from dsoi.canon import state_hash
from dsoi.context import VersionVector
from dsoi.experiments.e1_harness import message_for
from dsoi.experiments.e1_notices import HORIZON, NOMINAL, Notice
from dsoi.layers.l1_claude import interpret_reply
from dsoi.layers.l5_gate import BoxRobustGate, NominalGate, RecheckGate, StateVertexGate
from dsoi.scenarios import REPO_ROOT, VERSIONS_FILE, Scenario, build_pipeline

GATES = {
    "nominal": lambda s: NominalGate(),
    "box_robust": lambda s: BoxRobustGate(mode="resolve"),
    "state_vertex": lambda s: StateVertexGate(),
    "recheck": lambda s: RecheckGate(s.true_reader(), fallback="resolve"),
}


def world_scenario(world: dict, notice: Notice | None = None) -> Scenario:
    spec = {
        "id": "E1", "model": "capacity_plan",
        "world": {"plant": world["plant"], "horizon": list(HORIZON),
                  "nominal_capacity": world["nominal_capacity"],
                  "demand_per_week": world["demand_per_week"],
                  "erp_source_id": "erp:capacity-master",
                  "true_capacity": {} if notice is None else notice.truth()},
        "times": {"erp_load": 1, "message": 2, "decide": 3},
    }
    return Scenario(REPO_ROOT / "scenarios", spec)


class _Replay:
    """An L1 that returns one stored reply (scoring re-runs nothing against the API)."""

    def __init__(self, result):
        self.result = result

    def interpret(self, message, schema):
        return self.result


def _pipeline(scn, gate, result, served):
    versions = VersionVector.load(VERSIONS_FILE).bump(interpreter=served)
    return build_pipeline(scn, gate, _Replay(result), versions)


def score_record(rec: dict, notice: Notice, world: dict) -> dict:
    scn = world_scenario(world, notice)
    served = rec.get("served_model") or rec["model"]
    row = {"custom_id": rec["custom_id"], "model": rec["model"], "notice_id": notice.id,
           "category": notice.category, "expect": notice.expect, "repeat": rec["repeat"],
           "cost_usd": rec.get("cost_usd", 0.0)}
    if rec.get("error"):
        return {**row, "outcome": f"api_error:{rec['error']}"}
    msg = message_for(notice)
    result = interpret_reply(served, rec.get("stop_reason"), rec.get("text"), msg,
                             scn.schema(), rec.get("refusal_category"))
    p = _pipeline(scn, NominalGate(), result, served)
    row["outcome"] = p.ingest_message(msg, scn.time("message"))
    row["reason"] = p.audit.ingests[-1].reason
    state, _ = p.state.read(scn.decision_class(), scn.time("decide"))
    row["state_hash"] = state_hash(state)
    vals = {k[2]: (v, lo, hi) for k, v, lo, hi, _u in state}
    changed = {int(w) for w, (v, lo, hi) in vals.items()
               if (v, lo, hi) != (NOMINAL, None, None)}
    row["changed_weeks"] = ",".join(map(str, sorted(changed)))
    if notice.expect != "change":
        row["no_change_ok"] = not changed
        return row
    truth = notice.truth()
    affected = notice.weeks
    row["scope_exact"] = changed == set(affected)
    lows = {w: (vals[float(w)][1] if vals[float(w)][1] is not None else vals[float(w)][0])
            for w in affected}
    highs = {w: (vals[float(w)][2] if vals[float(w)][2] is not None else vals[float(w)][0])
             for w in affected}
    row["covered"] = all(lows[w] - 1e-6 <= truth[w] <= highs[w] + 1e-6 for w in affected)
    row["unsafe_miss"] = any(truth[w] < lows[w] - 1e-6 for w in affected)
    row["point_abs_error"] = float(np.mean([abs(vals[float(w)][0] - truth[w])
                                            for w in affected]))
    row["interval_width"] = float(np.mean([highs[w] - lows[w] for w in affected]))
    for name, make in GATES.items():
        g = _pipeline(scn, make(scn), result, served)
        g.ingest_message(msg, scn.time("message"))
        r = g.decide(scn.time("decide"), scn.theta())
        row[f"kexec_{name}"] = P.k_exec_true(r.record.x_exec, scn.true_state(), scn.theta(),
                                             g.formulation)
        if name == "box_robust":
            row["box_release"] = json.dumps(r.record.x_released)
    row["box_attribution_ok"] = row["kexec_box_robust"] or row["unsafe_miss"]
    return row


def _rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict]) -> dict:
    out: dict = {"models": {}, "by_category": {}}
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        change = [r for r in rs if r["expect"] == "change" and "scope_exact" in r]
        quiet = [r for r in rs if r["expect"] in ("no_change", "abstain")]
        groups = defaultdict(list)
        for r in rs:
            groups[r["notice_id"]].append(r)
        multi = [g for g in groups.values() if len(g) > 1]
        out["models"][model] = {
            "replies": len(rs),
            "api_errors": sum(r["outcome"].startswith("api_error") for r in rs),
            "abstained": sum(r["outcome"] == "abstained" for r in rs),
            "no_change_correct": _rate(quiet, "no_change_ok"),
            "scope_exact": _rate(change, "scope_exact"),
            "coverage_A2": _rate(change, "covered"),
            "unsafe_miss": _rate(change, "unsafe_miss"),
            "point_abs_error": _rate(change, "point_abs_error"),
            "interval_width": _rate(change, "interval_width"),
            **{f"violations_{g}": (1 - _rate(change, f"kexec_{g}")) if change else None
               for g in GATES},
            "box_attribution_exceptions": sum(not r["box_attribution_ok"] for r in change),
            "state_varies_across_repeats": (float(np.mean(
                [len({r.get("state_hash") for r in g}) > 1 for g in multi])) if multi else None),
            "box_release_varies_across_repeats": (float(np.mean(
                [len({r.get("box_release") for r in g if "box_release" in r}) > 1
                 for g in multi if g[0]["expect"] == "change"])) if multi else None),
            "cost_usd": float(sum(r.get("cost_usd") or 0 for r in rs)),
        }
        out["by_category"][model] = {
            cat: {"n": len(cr),
                  "scope_exact": _rate(cr, "scope_exact"),
                  "coverage_A2": _rate(cr, "covered"),
                  "no_change_correct": _rate(cr, "no_change_ok")}
            for cat in sorted({r["category"] for r in rs})
            for cr in [[r for r in rs if r["category"] == cat]]
        }
    return out


def score(records: list[dict], notices: list[Notice], world: dict, out_dir: Path,
          phase: str | None = None) -> dict:
    by_id = {n.id: n for n in notices}
    rows = [score_record(r, by_id[r["notice_id"]], world) for r in records
            if phase is None or r["phase"] == phase]
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out_dir / "scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
