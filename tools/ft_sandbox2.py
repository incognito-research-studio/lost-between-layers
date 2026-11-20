"""Build the round-2 fault-generation sandbox (data/fault_trial/round2/PROTOCOL.md).

    uv run python tools/ft_sandbox2.py DIR

The round-1 sandbox (tools/ft_sandbox.py, frozen, reused unchanged) plus what lets the
generator run cases and see what its faults do:
  src/dsoi/layers/l5_monitors.py   an interface-only stub (round2/l5_monitors_stub.py); the
                                   real trace-monitor rules stay withheld
  run_case.py                      a behaviour-only runner for one case directory
  .venv/                           Python with the artifact's pinned numpy, scipy, pyyaml
  FAULT_BRIEF.md                   the round-2 brief (replaces round 1's)
  episodes/                        for the generator's input episodes
Still withheld: src/dsoi/properties.py, src/dsoi/trace.py, the real l5_monitors.py,
everything under tools/, tests/ and results/, the paper's checkability table and Section
10.3, the review files, both protocols and the round-1 fault set.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_sandbox  # noqa: E402  (frozen in round 1, reused unchanged)

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "data/fault_trial/round2"
PINS = ("numpy==2.5.4", "scipy==1.18.1", "pyyaml==6.0.3")


def build(dest: Path) -> None:
    ft_sandbox.build(dest)
    shutil.copy2(R2 / "l5_monitors_stub.py", dest / "src/dsoi/layers/l5_monitors.py")
    shutil.copy2(R2 / "run_case.py", dest / "run_case.py")
    shutil.copy2(R2 / "FAULT_BRIEF.md", dest / "FAULT_BRIEF.md")
    (dest / "episodes").mkdir()
    with open(dest / ".gitignore", "a") as fh:
        fh.write("episodes/\n.venv/\n__pycache__/\n")
    subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(dest / ".venv")], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(dest / ".venv/bin/python"),
                    *PINS], check=True)
    for cmd in (["git", "add", "-A"],
                ["git", "-c", "user.name=sandbox", "-c", "user.email=sandbox@localhost",
                 "commit", "-qm", "round 2: runnable sandbox"]):
        subprocess.run(cmd, cwd=dest, check=True)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
    print("sandbox ready")
