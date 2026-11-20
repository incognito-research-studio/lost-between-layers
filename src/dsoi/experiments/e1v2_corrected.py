"""Corrected, explicitly post hoc replay of E1 v2; no new model calls.

The historical scorer remains available for reproducing the frozen results. Here
single-source notices outrank the baseline master, and interval widening is a
versioned gate transformation after interpretation, not a change to nu_I. Stored
reply snapshots must still match. The fitted margin retains the historical selection
rule and is a descriptive heuristic, not a validated conformal guarantee.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from dsoi import properties as P
from dsoi.constraints import solve_lp
from dsoi.experiments import e1v2_score as old
from dsoi.experiments.e1v2_notices import HORIZON, NAMES, NOMINAL, Case
from dsoi.experiments.e1v2_run import parse
from dsoi.experiments.e1v2_world import T_DECIDE, messages, pipeline, scenario, schema, state_of
from dsoi.layers.l5_gate import BoxRobustGate
from dsoi.types import Delta, OpKind, Quantity

ARMS = ("point", "stated", "calibrated", "recheck", "stated_hold")


def widen_state(state: tuple, margin: float) -> tuple:
    """Widen only reported intervals; do not silently invent scope or deleted facts."""
    return tuple((key, v, max(0.0, lo - margin), min(NOMINAL, hi + margin), unit)
                 if key[1] == "capacity" and lo is not None and hi is not None
                 else (key, v, lo, hi, unit) for key, v, lo, hi, unit in state)


class MarginGate(BoxRobustGate):
    def __init__(self, margin: float):
        super().__init__(mode="resolve")
        self.margin = margin

    @property
    def label(self) -> str:
        return f"{super().label}/reported-margin={self.margin:g}"

    def apply(self, x, inst, ctx, formulation):
        ctx = replace(ctx, state=widen_state(ctx.state, self.margin))
        widened = formulation.build(ctx.state, ctx.theta_dict())
        result = super().apply(x, widened, ctx, formulation)
        return replace(result, detail={**result.detail, "reported_margin": self.margin})


def run_arm(case: Case, arm: str, replies: dict, nu_i: str, margin: float = 0.0):
    scn = scenario(case)
    replay = old.Replay(replies, nu_i)
    gate = MarginGate(margin) if arm == "calibrated" else old._gate(arm, scn)
    _, p = pipeline(case, gate, replay, nu_i)
    replay.p, replay.times = p, {m.message_id: t for m, t in messages(case)}
    outcomes = [p.ingest_message(m, t) for m, t in messages(case)]
    if not replay.snapshot_ok:
        raise ValueError(f"{case.id}: corrected replay differs from the model's input snapshot")
    try:
        record = p.decide(T_DECIDE, scn.theta()).record
    except Exception:
        record = p.audit.records[-1]
        if record.outcome != "error":
            raise
    return scn, p, record, outcomes


def emitted_coverage(case: Case, replies: dict, nu_i: str) -> dict:
    """Diagnostic before L3: apply emitted operations in order, ignoring source ranks.

    An assertion's proposed value replaces the previous proposed value here. This is
    an emitted-value diagnostic, not the store's assert semantics or an independent
    semantic annotation. Retractions remove the proposed value; abstentions do nothing.
    """
    proposed = {s: {w: (NOMINAL, None, None) for w in HORIZON} for s in NAMES}
    declined = rejected = 0
    for message, _ in messages(case):
        reply = replies[message.message_id]
        result = parse(reply, nu_i, message, schema())
        if not isinstance(result.output, Delta):
            # A malformed or contract-rejected reply is distinct from an explicit refusal.
            try:
                explicit = json.loads(reply.get("text") or "null").get("kind") == "abstain"
            except (ValueError, AttributeError):
                explicit = False
            declined += int(explicit)
            rejected += int(not explicit)
            continue
        for op in result.output.ops:
            for w in HORIZON:
                if op.vt[0] <= w < op.vt[1]:
                    if op.kind is OpKind.RETRACT:
                        proposed[op.s].pop(w, None)
                    elif isinstance(op.o, Quantity):
                        proposed[op.s][w] = (op.o.value, op.o.lo, op.o.hi)
    return {"emitted_covered": old._covered(proposed["plant:X"],
                                           case.true_capacity("plant:X")),
            "model_abstentions": declined, "contract_rejections": rejected}


def arm_values(case: Case, arm: str, replies: dict, nu_i: str, margin: float = 0.0):
    scn, p, record, outcomes = run_arm(case, arm, replies, nu_i, margin)
    inst = p.formulation.build(scn.true_state(), scn.theta())
    optimum = inst.objective(solve_lp(inst.c, inst.system).x)
    executed = record.x_exec
    values = {f"released_{arm}": executed is not None,
              f"error_{arm}": record.outcome == "error",
              f"violation_{arm}": not P.k_exec_true(executed, scn.true_state(), scn.theta(),
                                                    p.formulation),
              f"excess_{arm}": None if executed is None else
                  inst.objective(np.array(executed)) - optimum}
    return values, p, record, outcomes


def score_case(case: Case, replies: dict, model: str, nu_i: str, repeat: int) -> dict:
    row = {"model": model, "case_id": case.id, "category": case.category,
           "split": case.split, "expect": case.expect, "repeat": repeat,
           **emitted_coverage(case, replies, nu_i)}
    values, p, _, outcomes = arm_values(case, "stated", replies, nu_i)
    row.update(values)
    state = state_of(p, T_DECIDE)
    x = state.get("plant:X", {})
    truth = case.true_capacity("plant:X")
    affected = {w for w in HORIZON if truth[w] != NOMINAL}
    missing = {w for w in HORIZON if w not in x}
    changed_x = {w for w, v in x.items() if v != (NOMINAL, None, None)} | missing
    changed = {e for e, ws in state.items()
               if any(v != (NOMINAL, None, None) for v in ws.values())
               or any(w not in ws for w in HORIZON)}
    hits = changed_x & affected
    lo, hi = old._bounds(x)
    row.update({"outcomes": ",".join(outcomes), "abstained": "abstained" in outcomes,
                "snapshot_ok": True, "scope_exact": changed_x == affected,
                "entity_ok": changed == set(case.truth) if case.truth else None,
                "quiet_ok": not changed_x, "covered_stated": old._covered(x, truth),
                "cal_score": None if not hits else NOMINAL if hits & missing else
                    max(max(lo[w] - truth[w], truth[w] - hi[w], 0.0) for w in hits)})
    for arm in ("point", "recheck"):
        row.update(arm_values(case, arm, replies, nu_i)[0])
    held = row["abstained"]
    for key in ("released", "error", "violation"):
        row[f"{key}_stated_hold"] = row[f"{key}_stated"] and not held
    row["excess_stated_hold"] = None if held else row["excess_stated"]
    return row


def score(cases: list[Case], replies: dict, alpha: float, out: Path) -> tuple[list[dict], dict]:
    grouped, versions = defaultdict(dict), {}
    for (model, cid, repeat, mid), reply in replies.items():
        grouped[(model, cid, repeat)][mid] = reply
        versions[model] = reply["nu_i"]
    by_id = {c.id: c for c in cases}
    keys = sorted(grouped)
    rows = [score_case(by_id[c], grouped[(m, c, r)], m, versions[m], r) for m, c, r in keys]
    margins = old.calibrate(rows, alpha)
    for row, key in zip(rows, keys, strict=True):
        model, cid, _ = key
        case = by_id[cid]
        values, p, _, _ = arm_values(case, "calibrated", grouped[key], versions[model],
                                     margins[model])
        row.update(values)
        canon, _ = p.state.read(p.decision_class, T_DECIDE)
        widened = {k[2]: (v, lo, hi) for k, v, lo, hi, _ in widen_state(canon, margins[model])
                   if k[0] == "plant:X"}
        row["covered_calibrated"] = old._covered(widened, case.true_capacity("plant:X"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    return rows, margins
