"""Round 3, deviation 2: run tools/ft_round3.py with tools/ft_repaired_dev2.py as the harness.

    uv run python tools/ft_r3_deviation2.py --exceptions OUT.json <ft_round3.py arguments>

Post hoc (data/fault_trial/round3/DEVIATIONS.md); the frozen results stay primary. Besides
ft_round3's result file, writes every exception a check raised (patch, gating or not, run,
check, message) and a count by patch and check to the --exceptions file."""
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_round3  # noqa: E402  (frozen for round 3)

ft_round3.HARNESS = ft_round3.ROOT / "tools/ft_repaired_dev2.py"
_in_copy = ft_round3.in_copy


def in_copy(patch, extra):
    os.environ["FT_PATCH"] = patch.name if patch is not None else "clean"
    try:
        return _in_copy(patch, extra)
    finally:
        os.environ.pop("FT_PATCH", None)


ft_round3.in_copy = in_copy

if __name__ == "__main__":
    argv = sys.argv[1:]
    i = argv.index("--exceptions")
    target = Path(argv[i + 1])
    del argv[i:i + 2]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        log = Path(fh.name)
    os.environ["FT_EXC_LOG"] = str(log)
    rc = ft_round3.main(argv)
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    log.unlink()
    by = Counter((r["patch"], r["check"], r["gating"]) for r in rows)
    target.write_text(json.dumps({
        "post_hoc": True,
        "by_patch_check": [{"patch": p, "check": c, "gating": g, "exceptions": n}
                           for (p, c, g), n in sorted(by.items(), key=str)],
        "exceptions": rows}, indent=1, sort_keys=True) + "\n")
    raise SystemExit(rc)
