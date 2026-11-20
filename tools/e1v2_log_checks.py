"""Would the architecture's own audit evidence have exposed the two E1 v2 harness defects?

    uv run python tools/e1v2_log_checks.py

Review of 4 Nov 2026 (fourth round, point 5). A reanalysis of the stored replies; no model is
called. Each case-repeat runs through the four cells of tools/e1v2_components.py, and two
predicates are evaluated on the resulting audit log only, without the generator's truth:

  shadowed   a decision record lists a conflict on a key between an interpreted (e-mail)
             fact and another source (rho["conflicts"], R1), and the fact the decision
             actually read for that key (rho["facts"]) comes from the capacity master.
             An interpreted value was committed and then never used.
  version    an ingest record reports an interpreter-version mismatch: L1 refused a reading
             whose nu_I differs from the pinned one (R3). The record's outcome is
             "abstained", so a count of outcomes alone conflates it with model abstention;
             the reason field separates them.

Both predicates read fields the architecture already requires. Neither was evaluated before
review found the defects. Output: results/e1v2_log_checks.json (test part, per model and cell).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from e1v2_clustered import MODELS
from e1v2_components import CELLS

from dsoi.experiments import e1v2_score as old
from dsoi.experiments.e1v2_corrected import MarginGate
from dsoi.experiments.e1v2_notices import load
from dsoi.experiments.e1v2_run import load_replies
from dsoi.experiments.e1v2_world import MASTER, T_DECIDE, messages, pipeline, scenario
from dsoi.layers.l5_gate import BoxRobustGate

ROOT = Path(__file__).resolve().parents[1]
INTERPRETED = "email:"


def audit_of(case, replies: dict, nu_i: str, priority: str, widening: str, q: float):
    """Run one case-repeat through one cell; return its ingest records and decision record."""
    scn = scenario(case)
    relabel = widening == "relabel"
    replay = old.Replay(replies, nu_i, q if relabel else 0.0)
    gate = MarginGate(q) if (q and not relabel) else BoxRobustGate(mode="resolve")
    _, p = pipeline(case, gate, replay, nu_i, historical=priority == "historical")
    replay.p, replay.times = p, {m.message_id: t for m, t in messages(case)}
    for m, t in messages(case):
        p.ingest_message(m, t)
    try:
        record = p.decide(T_DECIDE, scn.theta()).record
    except Exception:
        record = p.audit.records[-1]
        if record.outcome != "error":
            raise
    return p.audit.ingests, record


def shadowed(record) -> bool:
    rho = record.rho or {}
    facts, conflicts = rho.get("facts", {}), rho.get("conflicts", {})
    return any(any(s.startswith(INTERPRETED) for s in sources)
               and facts.get(key, {}).get("source_id") == MASTER
               for key, sources in conflicts.items())


def version_rejected(ingests) -> bool:
    return any(rec.reason.startswith("nu_I mismatch") for rec in ingests)


def model_abstained(ingests) -> bool:
    return any(rec.outcome == "abstained" and not rec.reason.startswith("nu_I mismatch")
               for rec in ingests)


def analyse(cases: list, replies: dict, margins: dict) -> dict:
    grouped, nu = defaultdict(dict), {}
    for (model, cid, repeat, mid), reply in replies.items():
        grouped[(model, cid, repeat)][mid] = reply
        nu[model] = reply["nu_i"]
    by_id = {c.id: c for c in cases}
    out: dict = {"post_hoc": True, "fresh_model_calls": 0, "split": "test",
                 "note": " ".join(__doc__.split("\n\n")[1:3]).replace("\n", " "), "models": {}}
    for model in MODELS:
        keys = sorted(k for k in grouped if k[0] == model and by_id[k[1]].split == "test")
        cells = {}
        for priority, widening in CELLS:
            q = margins[priority][model]
            counts = {"case_repeats": len(keys), "shadowed": 0, "version_rejected": 0,
                      "logged_abstained": 0, "model_abstained": 0}
            for key in keys:
                ingests, record = audit_of(by_id[key[1]], grouped[key], nu[model],
                                           priority, widening, q)
                counts["shadowed"] += shadowed(record)
                counts["version_rejected"] += version_rejected(ingests)
                counts["logged_abstained"] += any(r.outcome == "abstained" for r in ingests)
                counts["model_abstained"] += model_abstained(ingests)
            cells[f"{priority}/{widening}"] = counts
        out["models"][model] = cells
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/e1v2_log_checks.json")
    ap.add_argument("--components", type=Path, default=ROOT / "results/e1v2_components.json")
    ap.add_argument("--cases", type=Path, default=ROOT / "data/e1v2/cases.jsonl")
    ap.add_argument("--replies", type=Path, default=ROOT / "results/e1v2/replies.jsonl")
    args = ap.parse_args(argv)
    margins = json.loads(args.components.read_text())["margins"]
    res = analyse(load(args.cases), load_replies(args.replies), margins)
    res["margins"] = margins
    res["inputs_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in (("cases", args.cases), ("replies", args.replies))}
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res["models"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
