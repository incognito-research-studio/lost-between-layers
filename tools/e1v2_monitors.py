"""Frozen log monitors for the E1 v2 pipeline, for the fault trial (tools/fault_trial.py).

    uv run python tools/e1v2_monitors.py            # counts on the unmodified pipeline

The two predicates of tools/e1v2_log_checks.py
were written after the defects they find were known. This module fixes a wider set of
monitors before any fault is planted, so that a fault trial run by someone else can test
whether checks the architecture already defines detect integration faults nobody here
wrote them for. Every monitor reads the audit log, the store or a replay of logged inputs;
none reads the generator's truth. Fixed on 4 Nov 2026, before any fault was written; any
later change to MONITORS must be recorded as a deviation.

  G1               every fact a decision read has an authenticated source or a span of a
                   logged message (R1)
  R2               every message has a logged outcome, and re-applying the logged deltas
                   reproduces the store (R2(ii), R2(iii))
  C1               replaying the logged inputs releases the logged action (R3)
  K-prop           the released action satisfies the gate's logged set (R4)
  audit            hash chain, execution acknowledgements and encodings are complete (R1)
  G2               the logged support alone reproduces the decision (Prop. 4(b)); can fail
                   for a correct decision whose support is rank-deficient, so only a change
                   against the unmodified pipeline counts
  exec_fidelity    what execution reported equals what the gate released
  shadowed         a logged conflict between an interpreted fact and another source on a
                   key whose fact, as read, came from the capacity master
  version_refusal  an ingest refused for an interpreter-version mismatch
  committed_unread a message L1 committed contributes no fact to the decision record
                   (its span appears in no fact read); quiet notices about plant Y do this
                   legitimately, so only a change against the unmodified pipeline counts

The run uses the corrected replay of the stored replies (test part, three models, widened
box gate with the stored margins). It reports, per monitor, the number of case-repeats on
which the monitor fires, and outcome counts (abstentions, releases, violations) that the
trial uses only to tell whether a fault changed anything.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dsoi import properties as P
from dsoi.experiments import e1v2_score as old
from dsoi.experiments.e1v2_corrected import MarginGate
from dsoi.experiments.e1v2_notices import load
from dsoi.experiments.e1v2_run import load_replies
from dsoi.experiments.e1v2_world import MASTER, T_DECIDE, messages, pipeline, scenario

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen2.5:7b-instruct", "llama3.1:8b", "phi4:latest")
MONITORS = ("G1", "R2", "C1", "K-prop", "audit", "G2", "exec_fidelity", "shadowed",
            "version_refusal", "committed_unread")


def _safe(check) -> bool:
    """A check that raises counts as failing: the log could not support it."""
    try:
        return bool(check())
    except Exception:
        return False


def fired(p, ids: list[str]) -> dict[str, bool]:
    """Which monitors fire on one case-repeat's pipeline (True = a problem is flagged)."""
    log = p.audit
    decided = log.decisions()
    inv = {}
    try:
        inv = P.inv_report(p, ids)
    except Exception:
        inv = dict.fromkeys(("G1", "R2", "C1", "K-prop", "audit"), False)
    rec = log.records[-1] if log.records else None
    rho = (rec.rho if rec else None) or {}
    facts, conflicts = rho.get("facts", {}), rho.get("conflicts", {})
    spans = {f["span"][0] for f in facts.values() if f.get("span")}
    committed = {r.message_id for r in log.ingests if r.outcome == "committed"}
    return {
        **{k: not v for k, v in inv.items()},
        "G2": not all(_safe(lambda r=r: P.g2(r, p.formulation)) for r in decided),
        "exec_fidelity": not all(_safe(lambda r=r: P.execution_fidelity(r)) for r in decided),
        "shadowed": any(any(s.startswith("email:") for s in srcs)
                        and facts.get(key, {}).get("source_id") == MASTER
                        for key, srcs in conflicts.items()),
        "version_refusal": any(r.reason.startswith("nu_I mismatch") for r in log.ingests),
        "committed_unread": bool(committed - spans),
    }


def run_one(case, replies: dict, nu_i: str, q: float) -> tuple[dict, dict]:
    scn = scenario(case)
    replay = old.Replay(replies, nu_i)
    _, p = pipeline(case, MarginGate(q), replay, nu_i)
    replay.p, replay.times = p, {m.message_id: t for m, t in messages(case)}
    ids = []
    for m, t in messages(case):
        ids.append(m.message_id)
        p.ingest_message(m, t)
    try:
        record = p.decide(T_DECIDE, scn.theta()).record
    except Exception:
        record = p.audit.records[-1] if p.audit.records else None
        if record is None or record.outcome != "error":
            raise
    executed = record.x_exec
    outcome = {"abstained": any(r.outcome == "abstained" for r in p.audit.ingests),
               "released": executed is not None,
               "violation": not P.k_exec_true(executed, scn.true_state(), scn.theta(),
                                              p.formulation)}
    return fired(p, ids), outcome


def evaluate(cases_path: Path, replies_path: Path, margins: dict) -> dict:
    by_id = {c.id: c for c in load(cases_path)}
    grouped, nu = defaultdict(dict), {}
    for (model, cid, repeat, mid), reply in load_replies(replies_path).items():
        grouped[(model, cid, repeat)][mid] = reply
        nu[model] = reply["nu_i"]
    out: dict = {}
    for model in MODELS:
        counts = dict.fromkeys(MONITORS, 0)
        outcomes = {"abstained": 0, "released": 0, "violation": 0, "crashed": 0}
        keys = sorted(k for k in grouped if k[0] == model and by_id[k[1]].split == "test")
        for key in keys:
            try:
                flags, outcome = run_one(by_id[key[1]], grouped[key], nu[model],
                                         margins[model])
            except Exception:
                outcomes["crashed"] += 1
                continue
            for k, v in flags.items():
                counts[k] += v
            for k, v in outcome.items():
                outcomes[k] += v
        out[model] = {"case_repeats": len(keys), "monitors": counts, "outcomes": outcomes}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=ROOT / "data/e1v2/cases.jsonl")
    ap.add_argument("--replies", type=Path, default=ROOT / "results/e1v2/replies.jsonl")
    ap.add_argument("--margins", type=Path, default=ROOT / "results/e1v2_components.json")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    margins = json.loads(args.margins.read_text())["margins"]["corrected"]
    res = evaluate(args.cases, args.replies, margins)
    text = json.dumps(res, indent=1, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
