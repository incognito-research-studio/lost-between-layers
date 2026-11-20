"""Deviation 1 of the fault trial: rerun every patch with the extended effect trace.

    uv run python tools/ft_reanalysis.py --faults data/fault_trial/set-1/patches \
        --negatives data/fault_trial/set-1/negatives --manifest-sha SHA \
        --out results/fault_trial_v2_reanalysis.json

Decided after the frozen results were unblinded (data/fault_trial/DEVIATIONS.md). It runs
tools/ft_trial.py's own functions (copying src/, applying each patch, the determinism
check, `compare`, `negatives_report`) with one substitution: the harness subprocess is
tools/ft_trace_ext.py, the frozen harness with the extended effect trace. Nothing else
changes; the frozen tools are not edited. The frozen result, results/fault_trial_v2.json,
remains the primary result.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_trial  # noqa: E402  (the frozen trial runner, unchanged)

ft_trial.HARNESS = ft_trial.ROOT / "tools/ft_trace_ext.py"

if __name__ == "__main__":
    code = ft_trial.main()
    out = None
    for i, a in enumerate(sys.argv):
        if a == "--out":
            out = Path(sys.argv[i + 1])
    if out is not None:
        import json
        res = json.loads(out.read_text())
        res["deviation"] = {
            "id": 1, "note": "extended effect trace (replay outputs and the reloaded log); "
            "decided after unblinding; see data/fault_trial/DEVIATIONS.md",
            "harness_used": "tools/ft_trace_ext.py",
            "harness_sha256": hashlib.sha256(ft_trial.HARNESS.read_bytes()).hexdigest()}
        out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    raise SystemExit(code)
