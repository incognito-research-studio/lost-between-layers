"""Human reading test for the E1 v2 study (WP2, R1-P1-03): can a person read the notices as posed?

    uv run python tools/e1v2_human_reading.py packet          # writes the blind packet
    uv run python tools/e1v2_human_reading.py score ANSWERS    # scores a filled answer sheet

`packet` draws 30 test cases, two per category plus one more conflict and one more
correction (seed 20261104), and writes, for a reader who did not write the generator:
the instructions, the notices with the snapshot each first message is given, and an empty
answer sheet in the L1 delta schema. The truth and the case ids stay in a separate key.

`score` turns the filled sheet into stored replies (one per message, as the models' are),
runs them through the same pipeline and scorer (`e1v2_run.run`, `e1v2_score.score`), and
reports coverage, scope and entity per category, with every disagreement listed against the
generator's truth (data/e1v2/TRUTH_RULES.md). Offline; no model is called.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from dsoi.experiments.e1v2_notices import CATEGORIES, load
from dsoi.experiments.e1v2_run import load_replies, run
from dsoi.experiments.e1v2_score import score
from dsoi.experiments.e1v2_world import messages, pipeline, snapshot_text
from dsoi.layers.l5_gate import NominalGate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e1v2_human"
SEED = 20261104
EXTRA = ("conflict", "correction")
FIELDS = [
    "item",
    "message",
    "kind",
    "plant",
    "first_week",
    "last_week",
    "value",
    "lo",
    "hi",
    "span_text",
    "reason",
]
CODE = {"X": "plant:X", "Y": "plant:Y"}

INSTRUCTIONS = """# Reading test: capacity notices

You will read {n} short notices about two plants. For each, record what the planning system
should store about plant capacity, or that it should store nothing. Work alone, and don't
look at the study's code, data or results until you have finished.

**The world.** Each notice comes with the system's current state, as the "Current state"
block shows it. The planning weeks are 40-49; capacity is in units per week.

**For each message, fill one or more rows of `answers.csv`:**
- `item` and `message` as in `notices.md` (message 1, or 2 where an item has two).
- `kind`: `delta` if the message changes capacity, `abstain` if the system should not store
  anything from it (and say why in `reason`).
- For a `delta`, one row per plant and week range: `plant` (X or Y), `first_week` and
  `last_week` (inclusive), `value` (your best estimate, units per week), `lo` and `hi`
  (a range you are confident contains the true capacity; equal to `value` if the notice is
  exact), and `span_text` (the words of the message that support the row, copied exactly).
- Convert percentages and per-day figures to units per week yourself.
- Approvals or releases in a message are not capacity; don't record them.
- Where an item has two messages, answer message 1 first, then message 2 as if your answer
  to message 1 had already been stored.

There is no time limit. Note anything you found ambiguous in `reason`; those notes are
part of the result.
"""


def packet() -> None:
    cases = [c for c in load(ROOT / "data" / "e1v2" / "cases.jsonl") if c.split == "test"]
    rng = random.Random(SEED)
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c.category].append(c)
    chosen = []
    for cat in CATEGORIES:
        chosen += rng.sample(by_cat[cat], 3 if cat in EXTRA else 2)
    rng.shuffle(chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Notices", ""]
    for i, c in enumerate(chosen, 1):
        _, p = pipeline(c, NominalGate(), None, "human")
        (m0, t0), *rest = messages(c)
        lines += [f"## Item {i}", "", "Current state:", "", "```", snapshot_text(p, t0), "```", ""]
        for j, (m, _) in enumerate(messages(c), 1):
            lines += [f"**Message {j}** (source `{m.source_id}`):", "", f"> {m.text}", ""]
    (OUT / "notices.md").write_text("\n".join(lines))
    (OUT / "INSTRUCTIONS.md").write_text(INSTRUCTIONS.format(n=len(chosen)))
    with open(OUT / "answers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, c in enumerate(chosen, 1):
            for j in range(1, len(c.messages) + 1):
                w.writerow({"item": i, "message": j})
    with open(OUT / "key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", "case_id", "category", "expect", "truth"])
        w.writeheader()
        for i, c in enumerate(chosen, 1):
            w.writerow(
                {
                    "item": i,
                    "case_id": c.id,
                    "category": c.category,
                    "expect": c.expect,
                    "truth": json.dumps(c.truth, sort_keys=True),
                }
            )
    print(f"wrote {len(chosen)} items to {OUT}")


class SheetBackend:
    """An `e1v2_run` backend that answers each message from the filled sheet."""

    name = "human"

    def __init__(self, answers: dict):
        self.answers = answers

    def nu_i(self, model: str) -> str:
        return f"human:{model}"

    def send(self, message, schema, snapshot, model, seed, case) -> dict:
        rows = self.answers.get(message.message_id, [])
        if not rows or all(r["kind"].strip().lower() != "delta" for r in rows):
            reason = " / ".join(r["reason"] for r in rows if r.get("reason")) or "no answer"
            payload = {"kind": "abstain", "reason": reason, "ops": []}
        else:
            ops = []
            for r in rows:
                if r["kind"].strip().lower() != "delta":
                    continue
                v = float(r["value"])
                lo = float(r["lo"]) if r["lo"].strip() else v
                hi = float(r["hi"]) if r["hi"].strip() else v
                ops.append(
                    {
                        "op": "supersede",
                        "subject": CODE[r["plant"].strip().upper()],
                        "predicate": "capacity",
                        "value": v,
                        "lo": lo,
                        "hi": hi,
                        "unit": "units/week",
                        "valid_from": int(r["first_week"]),
                        "valid_to": int(r["last_week"]) + 1,
                        "span_text": r["span_text"],
                    }
                )
            payload = {"kind": "delta", "reason": "human reader", "ops": ops}
        return {"text": json.dumps(payload), "stop_reason": "stop", "served_model": model}


def score_sheet(path: Path) -> None:
    key = {r["item"]: r for r in csv.DictReader(open(OUT / "key.csv"))}
    cases = {c.id: c for c in load(ROOT / "data" / "e1v2" / "cases.jsonl")}
    answers = defaultdict(list)
    for r in csv.DictReader(open(path)):
        cid = key[r["item"]]["case_id"]
        answers[f"{cid}-m{int(r['message']) - 1}"].append(r)
    chosen = [cases[key[i]["case_id"]] for i in sorted(key, key=int)]
    out = OUT / "scored"
    store = out / "replies.jsonl"
    if store.exists():
        store.unlink()
    run(chosen, SheetBackend(answers), ["reader"], 1, store, "human-reading")
    summary = score(chosen, load_replies(store), 0.1, out, "human-reading")
    rows = list(csv.DictReader(open(out / "rows.csv")))
    item_of = {key[i]["case_id"]: i for i in key}
    report = {
        "by_category": summary["by_category"]["reader"],
        "disagreements": [
            {
                "item": item_of[r["case_id"]],
                "case_id": r["case_id"],
                "category": r["category"],
                "covered": r["covered_stated"],
                "scope_exact": r["scope_exact"],
                "entity_ok": r["entity_ok"],
                "quiet_ok": r["quiet_ok"],
                "abstained": r["abstained"],
                "truth": cases[r["case_id"]].truth,
                "answer": answers.get(f"{r['case_id']}-m0", [])
                + answers.get(f"{r['case_id']}-m1", []),
            }
            for r in rows
            if (r["expect"] == "change" and r["covered_stated"] != "True")
            or (r["expect"] != "change" and r["quiet_ok"] != "True")
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    for cat, m in report["by_category"].items():

        def cell(k, values=m):
            return "-" if not values[k] else f"{values[k]['k']}/{values[k]['n']}"

        print(
            f"  {cat:15s} covered {cell('coverage_stated'):5s} scope {cell('scope_exact'):5s} "
            f"entity {cell('entity_ok'):5s} quiet {cell('quiet_ok')}"
        )
    print(f"disagreements: {len(report['disagreements'])} (report.json)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["packet"]:
        packet()
    elif sys.argv[1:2] == ["score"]:
        score_sheet(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
