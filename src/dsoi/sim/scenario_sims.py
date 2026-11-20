"""Monte Carlo for the scenarios with a stochastic element beyond S1: S3, S4 and S8.

S3  A suspension notice can be missed by the interpreter, and the vendor master
    can be stale. The system-of-record recheck fails only when both happen, so its
    violation rate is P(miss) * P(stale | miss). The simulation checks the
    pipeline against that product and shows what correlation between the two
    failures does to it.

S4  Two sources report the same capacity cut. The planner's estimate is
    approximate (point and interval); the supplier portal gives an exact figure
    that is sometimes outdated. Each L3 conflict policy and gate is run on the
    same draws, with random arrival order.

S8  A pallet-configuration notice gives units per pallet as an approximate cut.
    The same draws go through the nominal, budgeted, box, state-vertex and recheck
    rungs; the gap between box and state-vertex receipts is the cost of treating
    linked coefficients as independent.

All parameters are illustrative, not calibrated.
"""

from __future__ import annotations

import numpy as np

from dsoi import properties as P
from dsoi.constraints import solve_lp
from dsoi.layers.l5_gate import (
    BoxRobustGate,
    BudgetedGate,
    NominalGate,
    RecheckGate,
    StateVertexGate,
)
from dsoi.scenarios import Deferred, Scenario, run_events

# -- S3 --------------------------------------------------------------------------


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run_s3(config: dict) -> tuple[list[dict], dict]:
    base = Scenario.load("S3")
    rng = np.random.default_rng(config["seed"])
    # Common random numbers: every cell reuses the same uniforms, so cells differ
    # only in their probabilities (and the realized miss set is shared when p_miss is).
    u_miss, u_stale = rng.random(config["runs"]), rng.random(config["runs"])
    rows = []
    for cell in config["cells"]:
        p_miss, p_stale_miss, p_stale_ok = (cell["p_miss"], cell["p_stale_given_miss"],
                                            cell["p_stale_given_detected"])
        for run in range(config["runs"]):
            miss = bool(u_miss[run] < p_miss)
            stale = bool(u_stale[run] < (p_stale_miss if miss else p_stale_ok))
            scn = base.with_world(master_stale=stale)
            fixtures = {"msg-suspend": "fixtures/missed.json"} if miss else None
            truth, theta = scn.true_state(), scn.theta()
            for name, gate in (("nominal", NominalGate()),
                               ("recheck", RecheckGate(scn.true_reader(), fallback="resolve"))):
                p, (r,) = run_events(scn, gate, fixtures=fixtures)
                x_exec = r.record.x_exec
                inst = p.formulation.build(truth, theta)
                j_star = inst.objective(solve_lp(inst.c, inst.system).x)
                rows.append({**cell, "run": run, "gate": name, "miss": miss, "stale": stale,
                             "k_exec": P.k_exec_true(x_exec, truth, theta, p.formulation),
                             "released": x_exec is not None,
                             "excess_cost": None if x_exec is None
                             else inst.objective(np.array(x_exec)) - j_star})
    summary = []
    for cell in config["cells"]:
        for name in ("nominal", "recheck"):
            rs = [r for r in rows if r["gate"] == name and all(r[k] == v for k, v in cell.items())]
            missed = [r for r in rs if r["miss"]]
            summary.append({
                **cell, "gate": name,
                "realized_miss_rate": len(missed) / len(rs),
                "k_exec_violation_rate": 1 - np.mean([r["k_exec"] for r in rs]),
                "closed_form": cell["p_miss"] * (cell["p_stale_given_miss"]
                                                 if name == "recheck" else 1.0),
                # should match p_stale_given_miss for the recheck, 1 for nominal
                "violation_rate_given_miss":
                    1 - np.mean([r["k_exec"] for r in missed]) if missed else None,
                # usefulness beside safety: how often a plan is released, and its cost
                # above the true optimum when it complies
                "release_rate": float(np.mean([r["released"] for r in rs])),
                "mean_excess_cost_when_compliant": _mean_or_none(
                    [r["excess_cost"] for r in rs if r["released"] and r["k_exec"]]),
            })
    return rows, {"config": config, "cells": summary}


# -- S4 --------------------------------------------------------------------------

S4_CONFIGS = {  # label -> (resolution policy, gate factory)
    "latest + nominal": ("latest", NominalGate),
    "priority(portal) + nominal": ("priority", NominalGate),
    "priority(portal) + box": ("priority", BoxRobustGate),
    "span + box": ("span", BoxRobustGate),
    "block": ("block", NominalGate),
}
PLANNER_SPAN = "capacity down about 30% for weeks 42-44"
PORTAL_SPAN = "capacity for weeks 42-44: 600 units per week"


def _claim(version: str, value: float, lo: float | None, hi: float | None, span: str) -> dict:
    return {"interpreter_version": version, "output": {"kind": "delta", "ops": [{
        "op": "supersede", "subject": "plant:X", "predicate": "capacity", "value": value,
        "lo": lo, "hi": hi, "unit": "units/week", "valid_from": 42, "valid_to": 45,
        "span_text": span}]}}


def run_s4(config: dict) -> tuple[list[dict], dict]:
    base = Scenario.load("S4")
    version = base.spec["interpreter_version"]
    rng = np.random.default_rng(config["seed"])
    nominal = float(base.world["nominal_capacity"])
    rows = []
    for cell in config["cells"]:
        for run in range(config["runs"]):
            r_true = rng.uniform(*config["true_reduction"])
            stated = round(r_true * 20) / 20
            r_hat = stated + cell["planner_sd"] * rng.standard_normal()
            hw = cell["planner_half_width"]
            portal_wrong = bool(rng.random() < cell["portal_error_rate"])
            r_portal = rng.uniform(*config["portal_stale_reduction"]) if portal_wrong else r_true
            portal_first = bool(rng.random() < 0.5)
            clip = lambda v: float(np.clip(v, 0, nominal))  # noqa: E731
            payloads = {
                "msg-planner": _claim(version, clip(nominal * (1 - r_hat)),
                                      clip(nominal * (1 - r_hat - hw)),
                                      clip(nominal * (1 - r_hat + hw)), PLANNER_SPAN),
                "msg-portal": _claim(version, round(nominal * (1 - r_portal), 6), None, None,
                                     PORTAL_SPAN),
            }
            times = {"msg-planner": 3, "msg-portal": 2} if portal_first else None
            cap = nominal * (1 - r_true)
            scn = base.with_world(true_capacity={w: cap for w in (42, 43, 44)})
            truth, theta = scn.true_state(), scn.theta()
            for label, (policy, gate) in S4_CONFIGS.items():
                p, (r,) = run_events(scn, gate(), payloads=payloads, times=times,
                                     resolution=policy)
                row = {**cell, "run": run, "config": label, "portal_wrong": portal_wrong,
                       "portal_first": portal_first}
                if isinstance(r, Deferred):
                    rows.append({**row, "deferred": True, "k_exec": True, "excess": None})
                    continue
                inst = p.formulation.build(truth, theta)
                j_star = inst.objective(solve_lp(inst.c, inst.system).x)
                rows.append({**row, "deferred": False,
                             "k_exec": P.k_exec_true(r.record.x_exec, truth, theta,
                                                     p.formulation),
                             "excess": inst.objective(np.array(r.record.x_exec)) - j_star})
    summary = []
    for cell in config["cells"]:
        for label in S4_CONFIGS:
            rs = [r for r in rows if r["config"] == label
                  and all(r[k] == v for k, v in cell.items())]
            decided = [r for r in rs if not r["deferred"]]
            compliant = [r["excess"] for r in decided if r["k_exec"]]
            summary.append({
                **cell, "config": label,
                "deferral_rate": 1 - len(decided) / len(rs),
                "k_exec_violation_rate": (1 - np.mean([r["k_exec"] for r in decided])
                                          if decided else None),
                "mean_excess_shortfall_when_compliant":
                    float(np.mean(compliant)) if compliant else None,
            })
    return rows, {"config": config, "cells": summary}


# -- S8 --------------------------------------------------------------------------

S8_GATES = {
    "nominal": lambda scn: NominalGate(),
    "budgeted:1": lambda scn: BudgetedGate(1),
    "box_robust": lambda scn: BoxRobustGate(mode="resolve"),
    "state_vertex": lambda scn: StateVertexGate(),
    "recheck": lambda scn: RecheckGate(scn.true_reader(), fallback="resolve"),
}


def run_s8(config: dict) -> tuple[list[dict], dict]:
    base = Scenario.load("S8")
    version = base.spec["interpreter_version"]
    nominal = float(base.world["units_per_pallet"])
    rng = np.random.default_rng(config["seed"])
    n = config["runs"]
    r_true = rng.uniform(*config["true_reduction"], n)   # common random numbers
    z = rng.standard_normal(n)
    rows = []
    for cell in config["cells"]:
        for run in range(n):
            stated = round(r_true[run] * 20) / 20
            pct = round(100 * stated)
            r_hat = stated + cell["point_sd"] * z[run]
            hw = cell["half_width"]
            text = (f"From this week, the new pallet configuration holds about {pct}% "
                    f"fewer units per pallet.")
            payload = {"interpreter_version": version, "output": {"kind": "delta", "ops": [{
                "op": "supersede", "subject": "dc:D1", "predicate": "units_per_pallet",
                "value": nominal * (1 - r_hat), "lo": nominal * (1 - r_hat - hw),
                "hi": nominal * (1 - r_hat + hw), "unit": "units/pallet",
                "valid_from": 42, "valid_to": 43,
                "span_text": f"holds about {pct}% fewer units per pallet"}]}}
            c_true = nominal * (1 - r_true[run])
            covered = nominal * (1 - r_hat - hw) <= c_true <= nominal * (1 - r_hat + hw)
            scn = base.with_world(truth={"units_per_pallet": c_true})
            truth, theta = scn.true_state(), scn.theta()
            inst = scn.model.formulation.build(truth, theta)
            j_star = inst.objective(solve_lp(inst.c, inst.system).x)
            for name, gate in S8_GATES.items():
                p, (r,) = run_events(scn, gate(scn), payloads={"msg-pallets": payload},
                                     texts={"msg-pallets": text})
                x = r.record.x_exec
                k_exec = P.k_exec_true(x, truth, theta, p.formulation)
                robust = name in ("budgeted:1", "box_robust", "state_vertex")
                rows.append({**cell, "run": run, "gate": name, "covered": bool(covered),
                             "rejected": x is None, "k_exec": k_exec,
                             # robust rungs may violate only when the interval misses the truth
                             "attribution_ok": not (robust and covered and not k_exec),
                             "receipts": None if x is None else x[0],
                             "excess": None if x is None else inst.objective(np.array(x)) - j_star})
    summary = []
    for cell in config["cells"]:
        for name in S8_GATES:
            rs = [r for r in rows if r["gate"] == name and all(r[k] == v for k, v in cell.items())]
            ok = [r for r in rs if r["k_exec"] and not r["rejected"]]
            summary.append({
                **cell, "gate": name,
                "k_exec_violation_rate": 1 - np.mean([r["k_exec"] for r in rs]),
                "rejection_rate": np.mean([r["rejected"] for r in rs]),
                "coverage_rate": np.mean([r["covered"] for r in rs]),
                "attribution_violations": int(sum(not r["attribution_ok"] for r in rs)),
                "mean_receipts_when_compliant":
                    float(np.mean([r["receipts"] for r in ok])) if ok else None,
                "mean_excess_when_compliant":
                    float(np.mean([r["excess"] for r in ok])) if ok else None,
            })
    return rows, {"config": config, "cells": summary}
