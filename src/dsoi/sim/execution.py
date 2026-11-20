"""Execution-side gaps on S1: faulty execution and time-of-check/time-of-use.

Two failures that A1-A3 of the GSN pattern do not cover:
  fault   execution deviates from the released action (x^e != x), with
          probability p_exec_fault, by up to +-executor_max_dev per position
  toctou  capacity drops again after the recheck's re-read and before execution,
          with probability p_toctou, by a fraction drawn from toctou_drop

Each run classifies a K-exec violation by cause and records whether the audit
log alone reveals it: execution infidelity is visible (the log holds both x and
x^e); a TOCTOU drop is not, because the log shows a compliant release, a
faithful execution and a passed re-read. Illustrative parameters.
"""

from __future__ import annotations

import numpy as np

from dsoi import properties as P
from dsoi.constraints import solve_lp
from dsoi.context import VersionVector
from dsoi.layers.l5_gate import BoxRobustGate, RecheckGate
from dsoi.pipeline import NoisyExecutor
from dsoi.scenarios import VERSIONS_FILE, Scenario, build_pipeline
from dsoi.sim.noise import VERSION, CapacityNotice, NoiseModel, SyntheticInterpreter

GATES = ("box_robust", "recheck")


def run_exec(config: dict) -> tuple[list[dict], dict]:
    base = Scenario.load("S1")
    noise = NoiseModel.from_dict(config.get("interpreter", {}))
    nominal = float(base.world["nominal_capacity"])
    versions = VersionVector.load(VERSIONS_FILE).bump(interpreter=VERSION)
    rng = np.random.default_rng(config["seed"])
    n = config["runs"]
    # Common random numbers across cells and gates.
    r_true = rng.uniform(config["true_reduction"]["low"], config["true_reduction"]["high"], n)
    seeds = rng.integers(2**32, size=(n, 2))
    u_toctou = rng.random(n)
    drop = rng.uniform(*config["toctou_drop"], n)
    rows = []
    for cell in config["cells"]:
        for run in range(n):
            stated = round(r_true[run] * 20) / 20
            cap_check = nominal * (1 - r_true[run])
            toctou = bool(u_toctou[run] < cell["p_toctou"])
            cap_exec = cap_check * (1 - drop[run]) if toctou else cap_check
            weeks = range(42, 45)
            scn_check = base.with_world(true_capacity={w: cap_check for w in weeks})
            scn_exec = base.with_world(true_capacity={w: cap_exec for w in weeks})
            notice = CapacityNotice(scn_check.subject, stated, 42, 44, nominal)
            for name in GATES:
                gate = (BoxRobustGate() if name == "box_robust"
                        else RecheckGate(scn_check.true_reader(), fallback="resolve"))
                executor = NoisyExecutor(cell["p_exec_fault"], config["executor_max_dev"],
                                         np.random.default_rng(int(seeds[run, 1])))
                p = build_pipeline(scn_check, gate,
                                   SyntheticInterpreter(notice, noise, int(seeds[run, 0])),
                                   versions, executor=executor)
                p.ingest_message(scn_check.message(text=notice.text()), scn_check.time("message"))
                rec = p.decide(scn_check.time("decide"), scn_check.theta()).record
                theta = scn_exec.theta()
                k_exec = P.k_exec_true(rec.x_exec, scn_exec.true_state(), theta, p.formulation)
                faithful = P.execution_fidelity(rec)
                released_ok = P.k_exec_true(rec.x_released, scn_exec.true_state(), theta,
                                            p.formulation)
                interp = bool(P.unsafe_coverage_failures(rec.state, scn_check.true_state()))
                true_inst = p.formulation.build(scn_exec.true_state(), theta)
                j_star = true_inst.objective(solve_lp(true_inst.c, true_inst.system).x)
                causes = []
                if not k_exec:
                    if not faithful:
                        causes.append("execution")
                    if toctou and not released_ok:
                        causes.append("toctou")
                    if name == "box_robust" and interp:
                        causes.append("interpretation")
                rows.append({**cell, "run": run, "gate": name, "k_exec": k_exec,
                             "faithful": faithful, "toctou": toctou,
                             "unsafe_miss": interp, "causes": "+".join(causes),
                             "log_visible": (not k_exec) and not faithful,
                             "unexplained": (not k_exec) and not causes,
                             "released": rec.x_released is not None,
                             "excess_shortfall": None if rec.x_exec is None else
                             true_inst.objective(np.array(rec.x_exec)) - j_star})
    return rows, {"config": config, "cells": summarize(rows, config)}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _share(rows: list[dict], pred) -> float | None:
    return sum(pred(r) for r in rows) / len(rows) if rows else None


def summarize(rows: list[dict], config: dict) -> list[dict]:
    out = []
    for cell in config["cells"]:
        for name in GATES:
            rs = [r for r in rows if r["gate"] == name and all(r[k] == v for k, v in cell.items())]
            bad = [r for r in rs if not r["k_exec"]]
            out.append({
                **cell, "gate": name,
                "k_exec_violation_rate": len(bad) / len(rs),
                "share_visible_in_log": _share(bad, lambda r: r["log_visible"]),
                "share_toctou_only": _share(bad, lambda r: r["causes"] == "toctou"),
                "share_interpretation": _share(bad, lambda r: "interpretation" in r["causes"]),
                "unexplained": sum(r["unexplained"] for r in rs),
                # usefulness beside safety: release rate, and shortfall above the true
                # optimum at execution time for compliant executions
                "release_rate": _share(rs, lambda r: r["released"]),
                "mean_excess_shortfall_when_compliant": _mean(
                    [r["excess_shortfall"] for r in rs
                     if r["k_exec"] and r["excess_shortfall"] is not None]),
            })
    return out
