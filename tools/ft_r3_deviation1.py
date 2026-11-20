"""Round 3, deviation 1: run tools/ft_round3.py with tools/ft_repaired_dev1.py as the harness.

    uv run python tools/ft_r3_deviation1.py <ft_round3.py arguments>

Post hoc (data/fault_trial/round3/DEVIATIONS.md); the frozen results stay primary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_round3  # noqa: E402  (frozen for round 3)

ft_round3.HARNESS = ft_round3.ROOT / "tools/ft_repaired_dev1.py"

if __name__ == "__main__":
    raise SystemExit(ft_round3.main())
