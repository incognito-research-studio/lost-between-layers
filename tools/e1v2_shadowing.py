"""Post hoc: how much of E1 v2's result comes from asserts shadowed by the capacity master.

    uv run python tools/e1v2_shadowing.py

Found on 3 Nov 2026 (data/e1v2/DEVIATIONS.md, item 6). A single-source notice reaches L3 from
`email:e1`, which has no entry in the study's source priorities, so it ranks 0, as does the
authenticated capacity master. An interpreted `assert` adds a fact beside the master's without
closing it (Definition 3), and L3's `resolve` then prefers the authenticated master: the reading
is committed but never read. Only `supersede`, or `retract` before `assert`, replaces the
master's value, and the prompt does not say so. Conflict cases are unaffected, since their
sources rank 1 and 2.

This script re-scores the stored replies with one change, made in memory: `email:e1` ranks 1,
above the master. Nothing else changes (replies, snapshots, gates, scorer), and no stored file
is rewritten. Output: results/e1v2_shadowing.json. Everything here is post hoc.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dsoi.experiments import e1v2_score as sc
from dsoi.experiments import e1v2_world as world
from dsoi.experiments.e1v2_notices import EMAIL, load
from dsoi.experiments.e1v2_run import load_replies

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1v2_clustered as ec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ec.MODELS


def ranked_pipeline(case, gate, interpreter, nu_i, *, historical=False):
    p0, p = world.pipeline(case, gate, interpreter, nu_i, historical=historical)
    p.state.source_priority = {**p.state.source_priority, EMAIL: 1}
    return p0, p


def main() -> int:
    results = ROOT / "results" / "e1v2"
    cases = {c.id: c for c in load(ROOT / "data" / "e1v2" / "cases.jsonl")}
    replies: dict[tuple, dict] = defaultdict(dict)
    nu: dict[str, str] = {}
    for (model, case_id, repeat, mid), rep in load_replies(results / "replies.jsonl").items():
        replies[(model, case_id, repeat)][mid] = rep
        nu[model] = rep["nu_i"]
    frozen = {(r["model"], r["case_id"], int(r["repeat"])): r
              for r in csv.DictReader(open(results / "rows.csv", newline=""))
              if r["split"] == "test"}

    sc.pipeline = ranked_pipeline          # the only change: email:e1 outranks the master
    out: dict = {"post_hoc": True, "note": " ".join(__doc__.split("\n\n")[1:3]).replace("\n", " "),
                 "models": {}}
    for model in MODELS:
        allkeys = sorted(k for k in frozen if k[0] == model)
        new = {k: sc.score_case(cases[k[1]], replies[k], model, nu[model], k[2]) for k in allkeys}
        quiet = [k for k in allkeys if frozen[k]["expect"] != "change"]
        keys = [k for k in allkeys if frozen[k]["expect"] == "change"]
        n = len(keys)
        changed = Counter()
        for k in keys:
            for col in ("covered_stated", "violation_stated", "violation_point", "scope_exact"):
                if str(new[k][col]) != frozen[k][col]:
                    changed[col] += 1
        out["models"][model] = {
            "n": n,
            "abstained": sum(new[k]["abstained"] for k in keys),
            "coverage_stated": [sum(frozen[k]["covered_stated"] == "True" for k in keys),
                                sum(new[k]["covered_stated"] for k in keys)],
            "scope_exact": [sum(frozen[k]["scope_exact"] == "True" for k in keys),
                            sum(new[k]["scope_exact"] for k in keys)],
            "violation_point": [sum(frozen[k]["violation_point"] == "True" for k in keys),
                                sum(new[k]["violation_point"] for k in keys)],
            "violation_stated": [sum(frozen[k]["violation_stated"] == "True" for k in keys),
                                 sum(new[k]["violation_stated"] for k in keys)],
            "error_stated": [sum(frozen[k]["error_stated"] == "True" for k in keys),
                             sum(new[k]["error_stated"] for k in keys)],
            "rows_changed": dict(changed),
            "quiet_ok": [sum(frozen[k]["quiet_ok"] == "True" for k in quiet),
                         sum(new[k]["quiet_ok"] for k in quiet), len(quiet)],
            "snapshot_mismatches": sum(not new[k]["snapshot_ok"] for k in allkeys),
            "answered": sum(not new[k]["abstained"] for k in keys),
            "answered_covered": sum(new[k]["covered_stated"] and not new[k]["abstained"]
                                    for k in keys),
        }
    (ROOT / "results" / "e1v2_shadowing.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["models"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
