"""What the corrected replay's abstentions say, and the counts the workshop paper adds for
review round R1 (post hoc, 17 Nov 2026; no model calls).

    python tools/e1v2_abstention_reasons.py

Reads data/e1v2/cases.jsonl, results/e1v2/replies.jsonl and results/e1v2_corrected/rows.csv,
and reruns tools/e1v2_abstention_policies.score_case for the test change notices. Writes
results/e1v2_abstention_reasons.json with:

- split: per model, change-notice case-repeats of the test part with an explicit model
  abstention (the raw reply's kind is "abstain") and with only an L1 rejection;
- reasons: per model, case-repeats with a model abstention whose stated reason matches each
  keyword pattern below. The patterns overlap and were not hand-coded; every change notice
  names plant X and weeks inside the horizon, so "out of scope" is a misreading here;
- truth_cells: plant-week truth values in the benchmark, by expectation and plant;
- standing_vs_keep: case-repeats where holding with the standing plan and keeping the old
  state differ in violation, with the case's category and its abstention/rejection counts;
- checks: checks per configuration in tools/ft_harness.CHECKS.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

MODELS = ("qwen2.5:7b-instruct", "llama3.1:8b", "phi4:latest")
PATTERNS = {
    "out_of_scope": r"out of scope|outside the scope|not in scope",
    "ambiguous_scope": r"ambiguous about scope|scope is ambiguous|scope of the message is ambiguous",
    "plant_name_or_subject_list": r"northfield|riverside|subject identifier|allowed subjects"
                                  r"|not listed as a subject",
}


def main() -> int:
    rows = [r for r in csv.DictReader(open(ROOT / "results/e1v2_corrected/rows.csv"))
            if r["split"] == "test" and r["expect"] == "change"]
    key = {(r["model"], r["case_id"], r["repeat"]): r for r in rows}
    split = {m: {"model_abstention": sum(int(r["model_abstentions"]) > 0 for r in rows
                                         if r["model"] == m),
                 "l1_rejection_only": sum(int(r["model_abstentions"]) == 0
                                          and int(r["contract_rejections"]) > 0
                                          for r in rows if r["model"] == m)}
             for m in MODELS}

    hits: dict = defaultdict(lambda: defaultdict(set))
    for line in open(ROOT / "results/e1v2/replies.jsonl"):
        x = json.loads(line)
        k = (x["model"], x["case_id"], str(x["repeat"]))
        if k not in key or int(key[k]["model_abstentions"]) == 0:
            continue
        try:
            j = json.loads(x.get("text") or "null")
        except ValueError:
            continue
        if not (isinstance(j, dict) and j.get("kind") == "abstain"):
            continue
        reason = (j.get("reason") or "").lower()
        for name, pat in PATTERNS.items():
            if re.search(pat, reason):
                hits[x["model"]][name].add(k)
    reasons = {m: {name: len(hits[m][name]) for name in PATTERNS} for m in MODELS}

    cells: Counter = Counter()
    for line in open(ROOT / "data/e1v2/cases.jsonl"):
        c = json.loads(line)
        for plant, weeks in (c.get("truth") or {}).items():
            for v in weeks.values():
                cells[(c["expect"], plant, v < 1000)] += 1
    truth_cells = {"total": sum(cells.values()),
                   "below_nominal": sum(n for (e, p, b), n in cells.items() if b),
                   "by_expect_plant": {f"{e}/{p}": n for (e, p, b), n in sorted(cells.items())}}

    import e1v2_abstention_policies as T
    by_id = {c.id: c for c in T.load(ROOT / "data/e1v2/cases.jsonl")}
    grouped, nu = defaultdict(dict), {}
    for (m, cid, rep, mid), reply in T.load_replies(ROOT / "results/e1v2/replies.jsonl").items():
        grouped[(m, cid, rep)][mid] = reply
        nu[m] = reply["nu_i"]
    diff = {}
    for model in MODELS:
        out = []
        for (m, cid, rep) in sorted(grouped):
            if m != model or by_id[cid].split != "test" or by_id[cid].expect != "change":
                continue
            row = T.score_case(by_id[cid], grouped[(m, cid, rep)], nu[m], rep)
            if row["retain"]["violation"] != row["hold_standing"]["violation"]:
                out.append({"case_id": cid, "repeat": rep, "category": by_id[cid].category,
                            "abstentions": row["abstentions"], "rejections": row["rejections"],
                            "keep_violates": row["retain"]["violation"],
                            "standing_violates": row["hold_standing"]["violation"]})
        diff[model] = out

    import ft_harness as H
    checks = Counter(c.config for c in H.CHECKS)

    res = {"post_hoc": True, "fresh_model_calls": 0, "split": split, "patterns": PATTERNS,
           "reasons": reasons, "truth_cells": truth_cells, "standing_vs_keep": diff,
           "checks": dict(checks)}
    (ROOT / "results/e1v2_abstention_reasons.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "standing_vs_keep"}, indent=1))
    print({m: len(v) for m, v in diff.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
