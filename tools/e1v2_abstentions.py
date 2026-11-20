"""Post hoc diagnostic of the E1 v2 abstentions (R1-P1-03): what the models abstained on, and why.

    uv run python tools/e1v2_abstentions.py        # after `dsoi-e1v2 score`

Offline and deterministic: no model is called. Each stored reply is replayed through the
case pipeline (as `e1v2_score` does), the snapshot it was given is rebuilt and its hash
checked against the stored one, and the reply is classified under the L1 contract. Model
abstentions are then coded by the scheme in results/e1v2_diagnostics/CODING_SCHEME.md:
facts about weeks and plants are computed, claims are detected by the fixed patterns
below, and the rest is left for hand coding (hand_codes.csv, provisional).

Outside src/dsoi, so the freeze is unaffected. Writes results/e1v2_diagnostics/.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from dsoi.experiments.e1v2_notices import HORIZON, NAMES, load
from dsoi.experiments.e1v2_run import load_replies, parse
from dsoi.experiments.e1v2_world import messages, pipeline, sha, snapshot_text
from dsoi.layers.l5_gate import NominalGate
from dsoi.types import Abstain

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e1v2_diagnostics"

# Claim patterns (CODING_SCHEME.md, original scheme and amendment 1), matched
# case-insensitively against the reason. A pattern detects a claim; whether the claim is a
# misreading is decided by a computed fact (see `code`).
OUTSIDE = (
    r"(outside|not within|not in|beyond|exceed\w*|out of|fall\w* out|not part of|not covered|"
    r"not included|after the|past the|later than|not (?:yet )?(?:recorded|present|available|"
    r"listed)|no (?:capacity )?(?:data|record\w*|information) for|does not (?:cover|include|"
    r"extend)|doesn't (?:cover|include|extend)|are not (?:part|within))"
    r"[^.;]{0,80}?(horizon|planning (?:week|period|range|window|horizon)|recording period|"
    r"current (?:state|planning|data|snapshot|weeks)|recorded weeks|weeks? 40|40-49|40 to 49|"
    r"range|snapshot|state)")
UNKNOWN_PLANT = (
    r"(unknown|not (?:listed|recognized|recognised|mapped|defined|found|known|an allowed|"
    r"among|in the allowed|one of|a valid|valid|included|present|specified in|mentioned in)|"
    r"no (?:mapping|subject|identifier)|does not (?:match|map|correspond)|doesn't (?:match|map|"
    r"correspond)|unrecogni[sz]ed|unmapped|cannot (?:be )?(?:map|identif|match))"
    r"[^.;]{0,60}?(plant|subject|identifier|entity|northfield|riverside|name)"
    r"|(plant|subject|identifier|entity|northfield|riverside)[^.;]{0,60}?(unknown|not (?:listed|"
    r"recognized|recognised|mapped|defined|found|known|an allowed|among|one of|valid|included|"
    r"present)|unrecogni[sz]ed|unmapped|does not (?:match|map|correspond|exist)|doesn't (?:match|"
    r"map|correspond|exist))"
    r"|by (?:its )?name\b|only (?:by )?(?:its )?name|not (?:by|using|use|specified by) (?:the )?"
    r"(?:code|identifier|subject identifier)|instead of (?:the )?(?:code|identifier)|"
    r"(?:does not|doesn't) use the (?:subject )?identifier")
NEG = r"(?:does not|doesn't|did not|do not|not|no|without)"
CLAIMS = {k: re.compile(v, re.I) for k, v in {
    "H": OUTSIDE,
    "P": UNKNOWN_PLANT,
    "E": (r"either plant|which plant|plant X or (?:plant )?Y|\bX or Y\b|both plants|"
          + NEG + r" (?:explicitly)?(?:mention|specify|say|refer to|indicate|address|include)\w*"
          r"[^.;]{0,40}(?:plant[ :]?Y|riverside|other plant)|unclear about (?:the )?"
          r"(?:subject|plant|entity)|(?:subject|plant|entity) (?:is )?(?:unclear|ambiguous|"
          r"not clear)"),
    "W": (NEG + r" (?:explicitly )?(?:specify|specified|state|give|provide|mention|indicate|"
          r"include)\w*[^.;]{0,40}(?:week|time|period|date|when|duration|start)|which weeks|"
          r"unclear (?:which|what) weeks|weeks? (?:are|is) (?:not|un)(?:clear|specified)|"
          r"ambiguous about (?:the )?(?:weeks|time|timing|period)"),
    "S": (r"already (?:record|set|at nominal|nominal|reflect)|current(?:ly)? (?:state )?"
          r"(?:record\w*|shows|indicates|lists|has)[^.;]{0,40}nominal|recorded at nominal|"
          r"no (?:indication|evidence)[^.;]{0,40}chang|matches the current|"
          r"current state[^.;]{0,60}(?:already|nominal)"),
    "C": (r"(?:percent\w*|%|per (?:working )?day|daily)[^.;]{0,80}" + NEG +
          r"[^.;]{0,40}(?:unit|convert|weekly|per week|absolute)|" + NEG +
          r" (?:specify|provide|give|state)\w*[^.;]{0,30}(?:units|absolute|weekly figure|"
          r"new capacity value)|only (?:a|the) percentage|without convert|not (?:directly )?"
          r"convert"),
    "Q": (r"approximat|\b(?:about|around|roughly) \d|up to \d|at least \d|'up to|'at least|"
          r"not precise|imprecise|uncertain|estimat|exact (?:value|amount|reduction|figure|"
          r"capacity|number|percentage)|not (?:a )?(?:specific|definite|exact)|range of possible"
          r"|hedg"),
    "V": r"valid_to|valid_from|valid time|half-open|time range|\[valid",
    "A": r"approv|release|QA manager|\blot\b",
    "X": r"conflict|contradict|planner|supplier|confirm",
    "O": (r"out of scope|not (?:related|relevant) to|unrelated|does not (?:concern|relate)|"
          r"not about (?:capacity|production)|no (?:information|mention)[^.;]{0,30}capacity"),
    "I": (r"ambiguous|vague|unclear|not clear|insufficient|not enough|more (?:information|"
          r"details)|lack|missing|does not specify|doesn't specify|not specified"),
}.items()}
ORDER = "HPEWSCQVAXOI"
TARGET = {"H": "snapshot", "P": "snapshot", "E": "message", "W": "message", "S": "snapshot",
          "C": "snapshot", "Q": "prompt", "V": "snapshot", "A": "prompt", "X": "message",
          "O": "message", "I": "message"}

HORIZON_RANGE = re.compile(r"\b40\s*(?:-|–|to|through)\s*49\b")
WEEK_WORD = re.compile(r"\b(?:weeks?|W)\b\s*|\bW(?=\d)", re.I)
PLANT_REFS = {
    "plant:X": re.compile(r"plant[ :]?X\b|northfield", re.I),
    "plant:Y": re.compile(r"plant[ :]?Y\b|riverside", re.I),
}
OTHER_PLANT = re.compile(r"\bplant[ :]([A-WZ])\b", re.I)   # a plant code that is not X or Y


def weeks_in(text: str) -> tuple[set[int], bool]:
    """Week numbers a text names after "week(s)" or "W", without the horizon range 40-49
    itself; and whether the text names the horizon range."""
    horizon = bool(HORIZON_RANGE.search(text))
    t = HORIZON_RANGE.sub(" ", text)
    weeks: set[int] = set()
    for m in WEEK_WORD.finditer(t):
        seg = re.split(r"[.;:]", t[m.end():m.end() + 30])[0]
        for a, b in re.findall(r"(\d{1,2})\s*(?:-|–|to|through|until)\s*(?:weeks?\s*|W)?(\d{1,2})",
                               seg):
            if int(a) <= int(b) and int(b) - int(a) < 20:
                weeks |= set(range(int(a), int(b) + 1))
        weeks |= {int(n) for n in re.findall(r"(?<![\d.$])(\d{1,2})(?![\d%])", seg)}
    return {w for w in weeks if 1 <= w <= 60}, horizon


def plants_in(text: str) -> set[str]:
    out = {e for e, rx in PLANT_REFS.items() if rx.search(text)}
    out |= {f"plant:{m.group(1).upper()}" for m in OTHER_PLANT.finditer(text)}
    return out


class Capture:
    """The L1 for this diagnostic: replays the stored reply and keeps the snapshot it was given."""

    def __init__(self, replies: dict, nu_i: str, case):
        self.replies, self.nu_i, self.case = replies, nu_i, case
        self.p, self.times = None, {m.message_id: t for m, t in messages(case)}
        self.seen: dict[str, dict] = {}

    def interpret(self, message, schema):
        reply = self.replies[message.message_id]
        snap = snapshot_text(self.p, self.times[message.message_id])
        result = parse(reply, self.nu_i, message, schema)
        self.seen[message.message_id] = {"snapshot": snap, "snapshot_ok":
                                         sha(snap) == reply["snapshot_sha"], "result": result}
        return result


def source_of(reply: dict, result) -> tuple[str, str]:
    """Who produced the abstention: the model, the L1 contract, or the transport."""
    text = reply.get("text")
    stop = reply.get("stop_reason")
    if text is None or stop in ("max_tokens", "refusal") or str(stop).startswith("error"):
        return "transport", str(stop)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "transport", "invalid JSON"
    if isinstance(payload, dict) and payload.get("kind") == "abstain":
        return "model", ""
    reason = result.output.reason if isinstance(result.output, Abstain) else ""
    sub = ("span not found" if "span not found" in reason else
           "empty delta" if "empty delta" in reason else
           "interval" if "interval" in reason or "point" in reason else
           "valid time" if "valid time" in reason else
           "schema" if "out-of-schema" in reason or "malformed" in reason else reason[:60])
    return "contract", sub


def notice_weeks(case) -> set[int]:
    return {w for ws in case.truth.values() for w in ws} if case.truth else set()


def code(reason: str, case) -> dict:
    """Facts computed from the reason, the case and the snapshot; claims detected by the
    patterns; and for each claim whether it misreads the case (CODING_SCHEME.md)."""
    weeks, names_horizon = weeks_in(reason)
    plants = plants_in(reason)
    stated, msg_plants = set(), set()
    for _, text in case.messages:
        stated |= weeks_in(text)[0]
        msg_plants |= plants_in(text)
    change = case.expect == "change"
    claims = [k for k in ORDER if CLAIMS[k].search(reason)]
    facts = {
        "weeks_named": " ".join(map(str, sorted(weeks))), "names_horizon": names_horizon,
        "weeks_in_horizon": all(HORIZON[0] <= w <= HORIZON[-1] for w in weeks) if weeks else None,
        "weeks_match_notice": (weeks == stated) if weeks and stated else None,
        "plant_named": " ".join(sorted(plants)),
        "plant_in_map": all(p in NAMES for p in plants) if plants else None,
        "message_weeks_stated": bool(stated) or case.category == "scope_relative",
        "message_plants_resolve": bool(msg_plants) and all(p in NAMES for p in msg_plants),
    }
    verdict = {}
    for k in claims:
        if k == "H":
            v = (None if not weeks else "misread" if facts["weeks_in_horizon"] else "correct")
        elif k == "P":
            v = "misread" if facts["plant_in_map"] is not False else "correct"
        elif k == "E":
            v = "misread" if facts["message_plants_resolve"] else "correct"
        elif k == "W":
            v = "misread" if facts["message_weeks_stated"] else "correct"
        elif k in "OI":
            v = "misread" if change else "correct"
        else:
            v = "misread" if change else "quiet"
        verdict[k] = v
    # H without a week number cannot be decided by the fact; it stays a detected claim only.
    decided = [k for k in claims if verdict[k] is not None]
    primary = decided[0] if decided else ""
    return {**facts, "claims": "".join(claims),
            "verdicts": " ".join(f"{k}:{verdict[k]}" for k in claims),
            "primary": primary, "primary_verdict": verdict.get(primary, "") if primary else "",
            "primary_target": TARGET.get(primary, "")}


def main() -> None:
    cases = {c.id: c for c in load(ROOT / "data" / "e1v2" / "cases.jsonl")}
    store = load_replies(ROOT / "results" / "e1v2" / "replies.jsonl")
    by_key: dict[tuple, dict] = defaultdict(dict)
    for (model, cid, rep, mid), r in store.items():
        by_key[(model, cid, int(rep))][mid] = r

    replies_out, rows_out = [], []
    for (model, cid, rep), rs in sorted(by_key.items()):
        case = cases[cid]
        nu_i = next(iter(rs.values()))["nu_i"]
        cap = Capture(rs, nu_i, case)
        _, p = pipeline(case, NominalGate(), cap, nu_i)
        cap.p = p
        outcomes = [p.ingest_message(m, t) for m, t in messages(case)]
        msg_rows = []
        for (m, _), outcome in zip(messages(case), outcomes, strict=True):
            r, seen = rs[m.message_id], cap.seen[m.message_id]
            abstained = outcome == "abstained"
            row = {"model": model, "case_id": cid, "category": case.category,
                   "split": case.split, "expect": case.expect, "repeat": rep,
                   "message_id": m.message_id, "source_id": m.source_id,
                   "manifest": r["manifest"][:16], "snapshot_ok": seen["snapshot_ok"],
                   "outcome": outcome, "abstained": abstained, "source": "", "subtype": "",
                   "reason": "", "message": m.text}
            if abstained:
                src, sub = source_of(r, seen["result"])
                if src == "contract" or src == "transport":
                    row["reason"] = seen["result"].output.reason
                elif not isinstance(seen["result"].output, Abstain):
                    src = "pipeline"
                else:
                    row["reason"] = seen["result"].output.reason
                row.update(source=src, subtype=sub)
                if src == "model":
                    row.update(code(row["reason"], case))
            msg_rows.append(row)
        replies_out += msg_rows
        ab = [r for r in msg_rows if r["abstained"]]
        rows_out.append({"model": model, "case_id": cid, "category": case.category,
                         "split": case.split, "expect": case.expect, "repeat": rep,
                         "messages": len(msg_rows), "abstained": bool(ab),
                         "abstained_messages": len(ab),
                         "sources": "+".join(sorted({r["source"] for r in ab})),
                         "abstained_message_ids": " ".join(r["message_id"][-2:] for r in ab)})

    OUT.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in replies_out for k in r})
    with open(OUT / "replies_coded.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(replies_out)
    with open(OUT / "case_repeats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    # Cross-check against the scorer's rows: same abstained flag per case-repeat.
    scored = {}
    with open(ROOT / "results" / "e1v2" / "rows.csv") as fh:
        for r in csv.DictReader(fh):
            scored[(r["model"], r["case_id"], int(r["repeat"]))] = r["abstained"] == "True"
    mism = [k for k in scored if scored[k] != next(
        r["abstained"] for r in rows_out if (r["model"], r["case_id"], r["repeat"]) == k)]
    summary = {
        "replies": len(replies_out), "case_repeats": len(rows_out),
        "snapshot_mismatches": sum(not r["snapshot_ok"] for r in replies_out),
        "abstained_flag_mismatches_vs_rows_csv": len(mism),
        "contract_reason_examples": Counter(
            r["reason"][:90] for r in replies_out if r["source"] == "contract").most_common(),
    }
    (OUT / "extract_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
