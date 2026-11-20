"""Post hoc arm of the E1 v2 study: the calibrated gate as the protocol describes it.

    uv run python tools/e1v2_posthoc.py        # after `dsoi-e1v2 score`

The pre-registered calibrated arm (src/dsoi/experiments/e1v2_score.py) pins the pipeline
to the model's interpreter version, but `widen` labels each widened reading with
"+cal<q>", so whenever q > 0 L1's version check rejects every widened reply and the arm
decides on the stale nominal state (data/e1v2/DEVIATIONS.md, item 2). That arm is
reported as frozen. This script runs the arm the protocol describes: the same replies,
the same margin q from results/e1v2/summary.json, the same box-robust gate, with the
pipeline pinned to the widened label.

It adds a second post hoc arm, "stated, hold on abstention": the box-robust gate on the
stated intervals, with L2 holding the decision whenever a message of the case ended in an
abstention, as if the abstained message were queued for a person and the decisions of
its decision class waited for it (Section 9.5 names this remedy). The arm needs no new
run: it is derived from the ingest outcomes the stated arm logged (rows.csv), and a held
case releases nothing. Holding also stops decisions after quiet notices that the model
rightly abstained on; that cost is reported as the held share of quiet notices.

Writes results/e1v2/posthoc.json and posthoc_rows.csv; everything here is post hoc.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dsoi import properties as P
from dsoi.constraints import solve_lp
from dsoi.experiments import e1v2_score as sc
from dsoi.experiments.e1v2_notices import load
from dsoi.experiments.e1v2_run import load_replies
from dsoi.experiments.e1v2_world import T_DECIDE, messages, pipeline, scenario, state_of
from dsoi.layers.l5_gate import BoxRobustGate

ROOT = Path(__file__).resolve().parents[1]
HOLD_ON = {"abstained"}                  # ingest outcomes that put a message in the queue


def run_arm(case, replies: dict, nu_i: str, q: float):
    """The calibrated arm with L1 pinned to the label of the widened reading."""
    scn = scenario(case)
    replay = sc.Replay(replies, nu_i, q)
    _, p = pipeline(case, BoxRobustGate(mode="resolve"), replay,
                    nu_i if q == 0 else f"{nu_i}+cal{q:g}", historical=True)
    replay.p, replay.times = p, {m.message_id: t for m, t in messages(case)}
    for m, t in messages(case):
        p.ingest_message(m, t)
    try:
        record = p.decide(T_DECIDE, scn.theta()).record
    except Exception:
        record = p.audit.records[-1]
        if record.outcome != "error":
            raise
    return scn, p, record


def row(case, replies: dict, model: str, nu_i: str, repeat: int, q: float) -> dict:
    scn, p, record = run_arm(case, replies, nu_i, q)
    inst = p.formulation.build(scn.true_state(), scn.theta())
    best = inst.objective(solve_lp(inst.c, inst.system).x)
    executed = record.x_exec
    violation = not P.k_exec_true(executed, scn.true_state(), scn.theta(), p.formulation)
    return {"model": model, "case_id": case.id, "category": case.category,
            "split": case.split, "expect": case.expect, "repeat": repeat,
            "covered": sc._covered(state_of(p, T_DECIDE).get("plant:X", {}),
                                   case.true_capacity("plant:X")),
            "error": record.outcome == "error", "released": executed is not None,
            "violation": violation,
            "excess": None if executed is None else inst.objective(np.array(executed)) - best}


def hold_rows(scored: list[dict]) -> list[dict]:
    """The stated arm with L2 holding the decision after any abstention (from rows.csv)."""
    out = []
    for r in scored:
        held = bool(set(r["outcomes"].split(",")) & HOLD_ON)
        released = r["released_stated"] == "True" and not held
        out.append({"model": r["model"], "case_id": r["case_id"], "category": r["category"],
                    "split": r["split"], "expect": r["expect"], "repeat": int(r["repeat"]),
                    "held": held, "error": r["error_stated"] == "True" and not held,
                    "released": released,
                    "violation": r["violation_stated"] == "True" and not held,
                    "excess": float(r["excess_stated"]) if released else None})
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    for model in sorted({r["model"] for r in rows}):
        test = [r for r in rows if r["model"] == model and r["split"] == "test"]
        change = [r for r in test if r["expect"] == "change"]
        released = [r for r in change if r["released"]]
        feasible = [r["excess"] for r in released if not r["violation"]]
        m = {"error": sc._rate(change, "error"), "release": sc._rate(change, "released"),
             "violation": sc._rate(change, "violation"),
             "violation_given_release": sc._rate(released, "violation"),
             "mean_excess_when_released": sc._mean(released, "excess"),
             "mean_excess_when_released_and_feasible":
                 float(np.mean(feasible)) if feasible else None,
             "released_and_feasible": len(feasible)}
        if "covered" in change[0]:
            m["coverage"] = sc._rate(change, "covered")
        if "held" in change[0]:
            m["held"] = sc._rate(change, "held")
            m["held_quiet"] = sc._rate([r for r in test if r["expect"] != "change"], "held")
        out[model] = m
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "e1v2")
    args = ap.parse_args(argv)
    summary = json.loads((args.results / "summary.json").read_text())
    cases = {c.id: c for c in load(ROOT / "data" / "e1v2" / "cases.jsonl")}
    by_key: dict[tuple, dict] = defaultdict(dict)
    nu: dict[str, str] = {}
    for (model, case_id, repeat, message_id), rep in load_replies(
            args.results / "replies.jsonl").items():
        by_key[(model, case_id, repeat)][message_id] = rep
        nu[model] = rep["nu_i"]
    rows = [row(cases[c], by_key[(m, c, r)], m, nu[m], r, summary["margin_q"][m])
            for m, c, r in sorted(by_key)]
    with open(args.results / "posthoc_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.results / "rows.csv", newline="") as fh:
        held = hold_rows(list(csv.DictReader(fh)))
    out = {"post_hoc": True, "manifest": summary["manifest"], "margin_q": summary["margin_q"],
           "arms": {"calibrated_pinned": summarize(rows), "stated_hold": summarize(held)},
           "note": " ".join(__doc__.split("\n\n")[1:3]).replace("\n", " ")}
    (args.results / "posthoc.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for arm, models in out["arms"].items():
        print(arm, json.dumps({m: {k: v["rate"] if isinstance(v, dict) else v
                                   for k, v in s.items()} for m, s in models.items()},
                              indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
