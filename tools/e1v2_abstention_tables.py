"""Tables of the E1 v2 abstention reasons (R1-P1-03), after `tools/e1v2_abstentions.py`.

    uv run python tools/e1v2_abstention_tables.py

Merges the rule codes (replies_coded.csv) with the provisional hand codes (hand_codes.csv),
counts abstentions by model and code at two units, the reply and the case-repeat (the
scorer's unit), and writes a blind, stratified sample of 40 hand-coded replies for an
independent coder, with the answer key kept in a separate file. Offline; no model is called.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from dsoi.experiments.e1v2_notices import load
from dsoi.experiments.e1v2_run import load_replies, parse
from dsoi.experiments.e1v2_world import messages, pipeline, snapshot_text
from dsoi.layers.l5_gate import NominalGate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e1v2_diagnostics"
MODELS = ("qwen2.5:7b-instruct", "llama3.1:8b", "phi4:latest")
CODES = "HPEWSCQVAXOI"
LABELS = {
    "H": "weeks taken as outside the horizon",
    "P": "plant name taken as unknown or unusable",
    "E": "plant taken as unclear, or the other plant as missing",
    "W": "weeks taken as missing or unclear",
    "S": "current record taken as blocking a change",
    "C": "size taken as not convertible to units per week",
    "Q": "approximate or bounded size taken as unusable",
    "V": "valid time taken as impossible to set",
    "A": "approval text taken as blocking",
    "X": "conflicting sources taken as blocking",
    "O": "message taken as out of scope",
    "I": "message taken as vague or lacking information",
    "contract": "L1 contract rejection",
}
KEY = ("model", "case_id", "repeat", "message_id")
SEED = 20261103


def key(r):
    return tuple(str(r[k]) for k in KEY)


def main() -> None:
    replies = list(csv.DictReader(open(OUT / "replies_coded.csv")))
    hand = {key(r): r for r in csv.DictReader(open(OUT / "hand_codes.csv"))}
    for r in replies:
        r["code"], r["coded_by"], r["verdict"] = "", "", ""
        if r["source"] == "contract":
            r["code"], r["coded_by"], r["verdict"] = "contract", "contract", "rejected"
        elif r["source"] == "model":
            if r["primary"]:
                r["code"], r["coded_by"], r["verdict"] = r["primary"], "rule", r["primary_verdict"]
            else:
                h = hand[key(r)]
                r["code"], r["coded_by"], r["verdict"] = h["hand_code"], "hand", h["hand_verdict"]
    assert all(r["code"] for r in replies if r["abstained"] == "True"), "uncoded abstention"

    test_change = [r for r in replies if r["split"] == "test" and r["expect"] == "change"]
    out: dict = {"unit_reply": {}, "unit_case_repeat": {}}
    # Reply level: every abstained reply in the test change cases.
    for m in MODELS:
        ab = [r for r in test_change if r["model"] == m and r["abstained"] == "True"]
        out["unit_reply"][m] = {
            "replies": sum(r["model"] == m for r in test_change),
            "abstained": len(ab),
            "codes": dict(Counter(r["code"] for r in ab)),
            "coded_by": dict(Counter(r["coded_by"] for r in ab)),
            "verdict": dict(Counter(r["verdict"] for r in ab)),
        }
    # Case-repeat level: one row per (model, case, repeat); the row's code is that of its
    # first abstained message, a model abstention taking precedence over a contract one
    # (the rule that gives the paper's 424 + 8).
    rows = defaultdict(list)
    for r in test_change:
        rows[(r["model"], r["case_id"], r["repeat"])].append(r)
    for m in MODELS:
        codes, n, mixed = Counter(), 0, 0
        for (mm, _, _), rs in rows.items():
            if mm != m:
                continue
            n += 1
            ab = sorted((r for r in rs if r["abstained"] == "True"), key=lambda r: r["message_id"])
            if not ab:
                continue
            model_ab = [r for r in ab if r["source"] == "model"]
            mixed += bool(model_ab) and len({r["source"] for r in ab}) > 1
            codes[(model_ab or ab)[0]["code"]] += 1
        out["unit_case_repeat"][m] = {
            "case_repeats": n,
            "abstained": sum(codes.values()),
            "codes": dict(codes),
            "mixed_model_and_contract": mixed,
        }
    pooled = Counter()
    for m in MODELS:
        pooled.update(out["unit_case_repeat"][m]["codes"])
    out["unit_case_repeat"]["pooled"] = {"abstained": sum(pooled.values()), "codes": dict(pooled)}

    # Examples: up to three verbatim reasons per code, from the test change replies.
    rng = random.Random(SEED)
    ex = {}
    for c in list(CODES) + ["contract"]:
        pool = [r for r in test_change if r["code"] == c]
        ex[c] = [
            {
                "model": r["model"],
                "case_id": r["case_id"],
                "repeat": r["repeat"],
                "message": r["message"],
                "reason": r["reason"],
                "coded_by": r["coded_by"],
            }
            for r in rng.sample(pool, min(3, len(pool)))
        ]
    out["examples"] = ex
    out["labels"] = LABELS
    (OUT / "abstention_tables.json").write_text(json.dumps(out, indent=1) + "\n")

    # Blind sample for an independent coder: 40 hand-coded replies, stratified by model and
    # by change / quiet notice, proportional with at least one per non-empty stratum.
    handed = [r for r in replies if r["coded_by"] == "hand"]
    strata = defaultdict(list)
    for r in handed:
        strata[(r["model"], r["expect"] == "change")].append(r)
    total, want = len(handed), 40
    alloc = {s: max(1, round(want * len(v) / total)) for s, v in strata.items()}
    while sum(alloc.values()) > want:
        s = max(alloc, key=lambda s: alloc[s] - want * len(strata[s]) / total)
        alloc[s] -= 1
    while sum(alloc.values()) < want:
        s = max(alloc, key=lambda s: want * len(strata[s]) / total - alloc[s])
        alloc[s] += 1
    sample = [r for s, v in sorted(strata.items()) for r in rng.sample(v, alloc[s])]
    rng.shuffle(sample)
    snaps = snapshots({key(r) for r in sample})
    with open(OUT / "second_coder_sample.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "item",
                "message",
                "snapshot",
                "reason",
                "code",
                "misread (yes/no)",
                "notes",
            ],
        )
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow(
                {
                    "item": i,
                    "message": r["message"],
                    "snapshot": snaps[key(r)],
                    "reason": r["reason"],
                    "code": "",
                    "misread (yes/no)": "",
                    "notes": "",
                }
            )
    with open(OUT / "second_coder_key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", *KEY, "category", "expect", "code", "verdict"])
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow(
                {
                    "item": i,
                    **{k: r[k] for k in KEY},
                    "category": r["category"],
                    "expect": r["expect"],
                    "code": r["code"],
                    "verdict": r["verdict"],
                }
            )
    print(json.dumps({k: out[k] for k in ("unit_reply", "unit_case_repeat")}, indent=1))
    print(
        "sample strata",
        {f"{m[:5]}/{'change' if c else 'quiet'}": n for (m, c), n in sorted(alloc.items())},
    )


def snapshots(keys: set) -> dict:
    """The snapshot each sampled reply was given, rebuilt from the stored replies."""
    cases = {c.id: c for c in load(ROOT / "data" / "e1v2" / "cases.jsonl")}
    store = load_replies(ROOT / "results" / "e1v2" / "replies.jsonl")
    out = {}
    for model, cid, rep, mid in keys:
        case = cases[cid]
        rs = {m: r for (mm, c, rr, m), r in store.items() if (mm, c, str(rr)) == (model, cid, rep)}
        nu_i = next(iter(rs.values()))["nu_i"]

        class L1:
            def __init__(self, key, replies, version, times):
                self.key, self.replies, self.version, self.times = key, replies, version, times
                self.p = None

            def interpret(self, message, schema):
                if message.message_id == self.key[-1]:
                    out[self.key] = snapshot_text(self.p, self.times[message.message_id])
                return parse(self.replies[message.message_id], self.version, message, schema)

        times = {m.message_id: t for m, t in messages(case)}
        reader = L1((model, cid, rep, mid), rs, nu_i, times)
        _, p = pipeline(case, NominalGate(), reader, nu_i)
        reader.p = p
        for m, t in messages(case):
            p.ingest_message(m, t)
    return out


if __name__ == "__main__":
    main()
