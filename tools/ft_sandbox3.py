"""Build the round-3 fault-generation sandbox (data/fault_trial/round3/PROTOCOL.md).

    uv run python tools/ft_sandbox3.py DIR

The round-2 sandbox (tools/ft_sandbox2.py, frozen, reused unchanged), with the round-3 brief
in place of round 2's and an empty negatives/ folder. Nothing else differs: the same source
without the withheld files, the same interface-only monitor stub, the same behaviour-only
runner and pinned environment. No fault set of rounds 1 or 2, no results and no tools are
copied (tools/ft_sandbox.py copies none of them).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_sandbox2  # noqa: E402  (frozen in round 2, reused unchanged)

import ft_sandbox  # noqa: E402  (frozen in round 1)

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "data/fault_trial/round3"
# The frozen round-1 builder copies the layer contracts from mdpi-paper/tables/contracts.tex at
# the project top. On 12 Nov 2026 the manuscripts moved under manuscripts/; the MDPI copy is
# frozen and byte-identical to the file rounds 1 and 2 used (this hash, from commit 0ec46ba).
OLD_CONTRACTS = ROOT.parent / "mdpi-paper/tables/contracts.tex"
MOVED_CONTRACTS = ROOT.parent / "manuscripts/mdpi-paper/tables/contracts.tex"
CONTRACTS_SHA256 = "56189d2fae59bdb878aba68635f71feb6ca9f14c0eb35080b8a12a405c6b87ba"


def _copy2(src, dst, *a, **k):
    if Path(src) == OLD_CONTRACTS and not OLD_CONTRACTS.exists():
        if hashlib.sha256(MOVED_CONTRACTS.read_bytes()).hexdigest() != CONTRACTS_SHA256:
            raise SystemExit("the moved contracts table differs from the one rounds 1-2 used")
        src = MOVED_CONTRACTS
    return _real_copy2(src, dst, *a, **k)


_real_copy2 = shutil.copy2


def build(dest: Path) -> None:
    ft_sandbox.shutil.copy2 = _copy2      # only the contracts path is redirected
    try:
        ft_sandbox2.build(dest)
    finally:
        ft_sandbox.shutil.copy2 = _real_copy2
    shutil.copy2(R3 / "FAULT_BRIEF.md", dest / "FAULT_BRIEF.md")
    (dest / "negatives").mkdir(exist_ok=True)
    (dest / "patches").mkdir(exist_ok=True)
    for cmd in (["git", "add", "-A"],
                ["git", "-c", "user.name=sandbox", "-c", "user.email=sandbox@localhost",
                 "commit", "-qm", "round 3: brief"]):
        subprocess.run(cmd, cwd=dest, check=True)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
    print("sandbox ready")
