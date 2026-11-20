"""Monte Carlo over S1-like capacity notices, comparing gate rungs on the same draws.

Per run: draw the true reduction r*, the supplier states it rounded to 5%, the
synthetic interpreter reads the notice, and each gate rung decides on the same
interpreted state (common random numbers). Every run also checks two
attribution claims; a nonzero count would contradict the implementation of
Prop. 3(c) or Cor. 2(a):
  - nominal gate, K-prop holds but K-exec fails  =>  interpretation differs on R(C)
  - box-robust gate, K-exec fails                =>  a true value lies outside its interval
"""

from __future__ import annotations

import numpy as np

from dsoi import properties as P
from dsoi.constraints import solve_lp
from dsoi.context import VersionVector
from dsoi.layers.l5_gate import (
    BoxRobustGate,
    BudgetedGate,
    ChanceGate,
    NominalGate,
    RecheckGate,
    StateVertexGate,
)
from dsoi.scenarios import VERSIONS_FILE, Scenario, build_pipeline
from dsoi.sim.noise import VERSION, CapacityNotice, NoiseModel, SyntheticInterpreter

GATES = {
    "nominal": lambda scn: NominalGate(mode="project"),
    "box_robust": lambda scn: BoxRobustGate(mode="project"),
    "recheck": lambda scn: RecheckGate(scn.true_reader(), fallback="project"),
}


def make_gate(spec: str, scn: Scenario):
    """Gate from a config string: nominal | box_robust | state_vertex | recheck |
    chance:<epsilon> | budgeted:<gamma>."""
    if spec.startswith("chance:"):
        return ChanceGate(float(spec.split(":", 1)[1]))
    if spec.startswith("budgeted:"):
        return BudgetedGate(int(spec.split(":", 1)[1]))
    if spec == "state_vertex":
        return StateVertexGate()
    return GATES[spec](scn)


def run_s1(config: dict) -> tuple[list[dict], dict]:
    base = Scenario.load("S1")
    noise = NoiseModel.from_dict(config.get("interpreter", {}))
    lo, hi = config["true_reduction"]["low"], config["true_reduction"]["high"]
    first, last = config.get("weeks", [42, 44])
    nominal = float(base.world["nominal_capacity"])
    versions = VersionVector.load(VERSIONS_FILE).bump(interpreter=VERSION)
    rng = np.random.default_rng(config["seed"])
    rows: list[dict] = []
    for run in range(config["runs"]):
        r_true = float(rng.uniform(lo, hi))
        seed = int(rng.integers(2**32))
        stated = round(r_true * 20) / 20  # "about 30%": supplier rounds to 5%
        cap_true = nominal * (1 - r_true)
        scn = base.with_world(true_capacity={w: cap_true for w in range(first, last + 1)})
        notice = CapacityNotice(scn.subject, stated, first, last, nominal)
        true_state, theta = scn.true_state(), scn.theta()
        p_true = build_pipeline(scn, NominalGate()).formulation.build(true_state, theta)
        best = solve_lp(p_true.c, p_true.system)
        j_star = p_true.objective(best.x)
        for name in config["gates"]:
            interp = SyntheticInterpreter(notice, noise, seed)
            p = build_pipeline(scn, make_gate(name, scn), interp, versions)
            msg = scn.message(text=notice.text())
            ingest = p.ingest_message(msg, scn.time("message"))
            res = p.decide(scn.time("decide"), theta)
            rec = res.record
            k_prop = P.k_prop(rec, p.formulation)
            k_exec = P.k_exec_true(rec.x_exec, true_state, theta, p.formulation)
            differs = P.interpretation_differs(rec.state, true_state)
            uncovered = P.coverage_failures(rec.state, true_state)
            unsafe = P.unsafe_coverage_failures(rec.state, true_state)
            attribution_ok = True
            if name == "nominal" and k_prop and not k_exec:
                attribution_ok = bool(differs)
            if name == "box_robust" and not k_exec:
                attribution_ok = bool(unsafe)
            j = None if rec.x_exec is None else p_true.objective(np.array(rec.x_exec))
            rows.append({
                "run": run, "gate": name, "r_true": r_true, "stated": stated,
                "ingest": ingest, "gate_outcome": rec.gate_outcome,
                "k_prop": k_prop, "k_exec": k_exec,
                "covered": not uncovered, "unsafe_miss": bool(unsafe),
                "interp_exact": not differs,
                "shortfall": j, "true_opt_shortfall": j_star,
                "excess_shortfall": None if j is None else j - j_star,
                "attribution_ok": attribution_ok,
            })
    return rows, summarize(rows, config)


def summarize(rows: list[dict], config: dict) -> dict:
    out = {"config": config, "gates": {}}
    for name in config["gates"]:
        rs = [r for r in rows if r["gate"] == name]
        compliant = [r for r in rs if r["k_exec"] and r["excess_shortfall"] is not None]
        out["gates"][name] = {
            "runs": len(rs),
            "k_exec_violation_rate": 1 - sum(r["k_exec"] for r in rs) / len(rs),
            "coverage_rate": sum(r["covered"] for r in rs) / len(rs),
            "unsafe_miscoverage_rate": sum(r["unsafe_miss"] for r in rs) / len(rs),
            "mean_excess_shortfall": float(np.mean([r["excess_shortfall"] for r in rs
                                                    if r["excess_shortfall"] is not None])),
            "mean_excess_shortfall_when_compliant":
                float(np.mean([r["excess_shortfall"] for r in compliant])) if compliant else None,
            "rejected": sum(r["gate_outcome"] == "rejected" for r in rs),
            "attribution_violations": sum(not r["attribution_ok"] for r in rs),
        }
    return out
