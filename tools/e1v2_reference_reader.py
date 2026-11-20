"""Reference reader for the E1 v2 study (WP2, R1-P1-03): is the task solvable as posed?

    uv run python tools/e1v2_reference_reader.py dev     # calibration cases only
    uv run python tools/e1v2_reference_reader.py final   # every case, scored once

A deterministic interpreter, written from the case generator's phrasing templates
(`e1v2_notices.make`) and the snapshot format (`e1v2_notices.snapshot`), that reads only
what a model reads: the message text and the snapshot. It never sees the case's truth.
Its replies go through the same L1 contract, pipeline and scorer as the models' replies
(`e1v2_run.run` with this backend, then `e1v2_score.score`), so a correct reading by it
shows that the contract, the snapshot and the scorer admit a correct reading.

What it does not show: that the phrasings are unambiguous to a reader who does not know
the templates. For hedged, "up to" and "at least" notices it states the generator's own
truth intervals, so its coverage there holds by construction.

Development used the calibration part only (`dev`); `final` records the SHA-256 of this
file as the run's manifest and is run once. Outside src/dsoi, so the freeze is unaffected.
Writes results/e1v2_reference/.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from dsoi.experiments.e1v2_notices import load
from dsoi.experiments.e1v2_run import load_replies, run
from dsoi.experiments.e1v2_score import score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e1v2_reference"
MODEL = "reference-reader"
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5}
HEDGE = r"(?:about|roughly|approximately|around)"


def plant_map(snapshot: str) -> dict[str, str]:
    """Names to subject ids, from the snapshot's "- plant:X is Plant X, also called ..." lines."""
    out = {}
    for sid, code, name in re.findall(r"- (plant:\w+) is ([^,]+), also called ([^.]+)\.", snapshot):
        for n in (code, name, re.sub(r"^the ", "", name)):
            out[n.lower()] = sid
    return out


def recorded(snapshot: str) -> tuple[float, dict[str, dict[int, float]]]:
    """Nominal capacity and recorded weekly points; unlisted weeks are nominal."""
    nominal = float(re.search(r"Nominal capacity: ([\d.]+) units per week", snapshot).group(1))
    rec: dict[str, dict[int, float]] = defaultdict(dict)
    for sid, body in re.findall(
        r"- (plant:\w+) currently recorded: (.*?); other weeks nominal\.", snapshot
    ):
        for w, v in re.findall(r"week (\d+): ([\d.]+)", body):
            rec[sid][int(w)] = float(v)
    return nominal, rec


def weeks(text: str) -> tuple[int, int] | None:
    """The week range a clause states, in the generator's formats."""
    m = re.search(r"the (\w+) weeks starting week (\d+)", text)
    if m and m.group(1) in NUMBER_WORDS:
        a = int(m.group(2))
        return a, a + NUMBER_WORDS[m.group(1)] - 1
    m = re.search(r"weeks (\d+)\s*(?:-|to|through)\s*(\d+)", text) or re.search(
        r"W(\d+)-W(\d+)", text
    )
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\bweek (\d+)\b", text) or re.search(r"\bW(\d+)\b", text)
    if m:
        return int(m.group(1)), int(m.group(1))
    return None


def subject(text: str, names: dict[str, str]) -> str | None:
    hits = [(text.lower().find(n), sid) for n, sid in names.items() if n in text.lower()]
    return min(hits)[1] if hits else None


def cut_bounds(clause: str) -> tuple[float, float, float] | None:
    """Percentage cut as (point, low, high), by the phrasing's truth rule."""
    if m := re.search(HEDGE + r" (\d+(?:\.\d+)?)%", clause):
        p = float(m.group(1))
        return p, p - 2.5, p + 2.5
    if m := re.search(r"up to (\d+(?:\.\d+)?)%", clause):
        p = float(m.group(1))
        return p, max(p - 10, 0.0), p
    if m := re.search(r"at least (\d+(?:\.\d+)?)%", clause):
        p = float(m.group(1))
        return p, p, p + 10
    if m := re.search(r"(\d+(?:\.\d+)?)%", clause):
        p = float(m.group(1))
        return p, p, p
    return None


def op(sid, a, b, value, lo, hi, span):
    return {
        "op": "supersede",
        "subject": sid,
        "predicate": "capacity",
        "value": value,
        "lo": lo,
        "hi": hi,
        "unit": "units/week",
        "valid_from": a,
        "valid_to": b + 1,
        "span_text": span,
    }


def read(text: str, snapshot: str) -> dict:
    """One message under the L1 output schema: a delta, or an abstention with a reason."""
    names = plant_map(snapshot)
    nominal, rec = recorded(snapshot)

    def base(sid, w):
        return rec.get(sid, {}).get(w, nominal)

    # The approval sentence of an injected notice asserts nothing; only the capacity part is read.
    body = (
        text.split("Also, ", 1)[1] if text.startswith("Approved by") and "Also, " in text else text
    )
    ops = []
    if m := re.search(
        r"This is week (\d+)\. Starting next week, (.*?) will run (\d+(?:\.\d+)?)% "
        r"below normal capacity for (\d+) weeks",
        body,
    ):
        now, n, p = int(m.group(1)), int(m.group(4)), float(m.group(3))
        sid = subject(m.group(2), names)
        a, b = now + 1, now + n
        if sid:
            v = round(base(sid, a) * (1 - p / 100), 2)
            ops.append(op(sid, a, b, v, v, v, m.group(0)))
        return delta(ops) if ops else abstain("plant not resolved")
    if m := re.search(
        r"confirmed capacity at (.*?) for (.*?) is (\d+(?:\.\d+)?) units per week", body
    ):
        sid, wk = subject(m.group(1), names), weeks(m.group(2))
        if sid and wk:
            v = float(m.group(3))
            ops.append(op(sid, wk[0], wk[1], v, v, v, m.group(0)))
        return delta(ops) if ops else abstain("plant or weeks not resolved")
    if "Correction to the earlier notice" in body and (
        m := re.search(r"applies to (.*?), not ", body)
    ):
        wk, sid, cb = weeks(m.group(1)), subject(body, names), cut_bounds(body)
        if wk and sid and cb:
            ops += pct_ops(sid, wk, cb, base, body)
        return delta(ops) if ops else abstain("correction not resolved")
    for clause in [c.strip() for c in body.split(";")]:
        sid, wk = subject(clause, names), weeks(clause)
        if not (sid and wk):
            continue
        if m := re.search(r"at most (\d+(?:\.\d+)?) units per week", clause):
            v = float(m.group(1))
            ops.append(op(sid, wk[0], wk[1], v, v, v, clause))
        elif m := re.search(r"drops by (\d+(?:\.\d+)?) units per working day", clause):
            days = re.search(r"\((\w+) working days per week\)", clause)
            k = NUMBER_WORDS.get(days.group(1), None) if days else None
            if k is None:
                continue
            for w in range(wk[0], wk[1] + 1):
                v = base(sid, w) - k * float(m.group(1))
                ops.append(op(sid, w, w, v, v, v, clause))
        elif re.search(r"capacity|down|loses|below normal", clause) and (cb := cut_bounds(clause)):
            ops += pct_ops(sid, wk, cb, base, clause)
    if ops:
        return delta(ops)
    if not re.search(r"\d", body):
        return abstain("no size or weeks stated")
    return abstain("no capacity change stated")


def pct_ops(sid, wk, cb, base, span):
    p, lo_cut, hi_cut = cb
    out = []
    for w in range(wk[0], wk[1] + 1):
        c = base(sid, w)
        out.append(
            op(
                sid,
                w,
                w,
                round(c * (1 - p / 100), 2),
                round(c * (1 - hi_cut / 100), 2),
                round(c * (1 - lo_cut / 100), 2),
                span,
            )
        )
    return out


def delta(ops):
    return {"kind": "delta", "reason": "reference reader", "ops": ops}


def abstain(reason):
    return {"kind": "abstain", "reason": reason, "ops": []}


class ReaderBackend:
    """An `e1v2_run` backend that answers with `read`; it ignores the case it is handed."""

    name = "reference"

    def nu_i(self, model: str) -> str:
        return f"reference:{model}@{READER_SHA[:12]}"

    def send(self, message, schema, snapshot, model, seed, case) -> dict:
        return {
            "text": json.dumps(read(message.text, snapshot)),
            "stop_reason": "stop",
            "served_model": model,
        }


READER_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    cases = load(ROOT / "data" / "e1v2" / "cases.jsonl")
    if mode == "dev":
        cases = [c for c in cases if c.split == "calibration"]
    out = OUT / mode
    store = out / "replies.jsonl"
    if store.exists():
        store.unlink()  # deterministic: a rerun replaces the replies
    run(cases, ReaderBackend(), [MODEL], 1, store, READER_SHA)
    replies = load_replies(store)
    if mode == "dev":
        # The scorer reports the test part; for development, score every case as test.
        from dataclasses import replace

        cases = [replace(c, split="test") for c in cases]
    summary = score(cases, replies, 0.1, out, READER_SHA)
    (out / "reader_sha256.txt").write_text(READER_SHA + "\n")
    cats = summary["by_category"][MODEL]
    print(f"{mode}: reader {READER_SHA[:12]}")
    for cat, m in cats.items():

        def cell(k, values=m):
            return "-" if not values[k] else f"{values[k]['k']}/{values[k]['n']}"

        print(
            f"  {cat:15s} rows {m['rows']:3d}  covered {cell('coverage_stated'):7s} scope "
            f"{cell('scope_exact'):7s} entity {cell('entity_ok'):7s} quiet {cell('quiet_ok'):7s} "
            f"abstained {cell('abstained')}"
        )


if __name__ == "__main__":
    main()
