"""Deviation 1 of the fault trial (data/fault_trial/DEVIATIONS.md): an extended effect trace.

    PYTHONPATH=<patched src> python tools/ft_trace_ext.py --out runs.json [--negatives DIR]

Run by tools/ft_reanalysis.py in place of tools/ft_harness.py. It imports the frozen harness
unchanged and replaces only its effect trace: the frozen trace digests the store, the
records, the ingests and the approvals, and so cannot see a fault whose only effect is on
what the system returns when it replays a decision or reloads its saved log. The extended
trace adds, for every decided record, the action `Pipeline.replay` returns from the logged
state and from the store, and the content of the log after `AuditLog.to_jsonl` and
`AuditLog.from_jsonl`. These are outputs of the system's own functions, not of any check;
the checks, configurations, classification and scoring are those of the frozen tools.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_harness  # noqa: E402  (the frozen harness, unchanged)

frozen_trace = ft_harness.trace


def _outcome(fn):
    try:
        v = fn()
        return None if v is None else [round(float(x), 6) for x in v]
    except Exception as exc:
        return f"raised {type(exc).__name__}"


def extended_trace(run) -> str:
    p = run.p
    replays = []
    for r in p.audit.decisions():
        replays.append([r.id, _outcome(lambda r=r: p.replay(r.id)),
                        _outcome(lambda r=r: p.replay(r.id, from_store=True))])
    try:
        from dsoi.layers.l6_audit import AuditLog
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            head = p.audit.to_jsonl(path)
            loaded = AuditLog.from_jsonl(path, expected_head=head)
        reloaded = json.dumps({
            "messages": loaded.messages,
            "ingests": [asdict(x) for x in loaded.ingests],
            "approvals": loaded.approvals,
            "records": [r.record_hash for r in loaded.records],
            "deltas": [[t, repr(dl)] for t, dl in loaded.deltas],
            "encodings": sorted(loaded.encodings)}, sort_keys=True, default=str)
    except Exception as exc:
        reloaded = f"raised {type(exc).__name__}"
    body = {"frozen": frozen_trace(run), "replays": replays, "reloaded": reloaded}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


ft_harness.trace = extended_trace

if __name__ == "__main__":
    raise SystemExit(ft_harness.main())
