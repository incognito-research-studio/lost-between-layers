"""Round 3, deviation 1 (data/fault_trial/round3/DEVIATIONS.md): the repaired harness with the
state-vertex gate's rung name added to `robust_release`'s list, nothing else changed. Post hoc."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_repaired  # noqa: E402  (frozen for round 3)

ft_repaired.ROBUST_RUNGS = ("box_robust", "vertex", "state_vertex")

if __name__ == "__main__":
    raise SystemExit(ft_repaired.main())
