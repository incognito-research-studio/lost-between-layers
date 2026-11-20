"""Build the fault-generation sandbox (data/fault_trial/PROTOCOL.md, roles).

    uv run python tools/ft_sandbox.py DIR

Copies into DIR, as a fresh git repository with one commit:
  src/dsoi/...      the pipeline source, without the files that hold checks
                    (WITHHELD) and without code that only scores or simulates
  scenarios/        every scenario with its fixtures (the input formats)
  configs/versions.yaml
  CONTRACTS.tex     the layer contracts (Table 2 of the paper)
  FAULT_BRIEF.md, NEGATIVE_FORMAT.md
Nothing under tools/, tests/ or results/ is copied, nor the paper's checkability table or
Section 10.3. The sandbox is for reading and writing patches; it is not meant to run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WITHHELD = {"properties.py", "layers/l5_monitors.py", "trace.py"}
SKIP_DIRS = {"sim", "__pycache__"}
EXPERIMENTS_KEPT = {"e1_notices.py", "e1v2_notices.py", "e1v2_world.py", "e1v2_corrected.py",
                    "__init__.py"}


def build(dest: Path) -> list[str]:
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty")
    src = ROOT / "src/dsoi"
    copied = []
    for f in sorted(src.rglob("*.py")):
        rel = f.relative_to(src).as_posix()
        parts = rel.split("/")
        if rel in WITHHELD or SKIP_DIRS & set(parts):
            continue
        if parts[0] == "experiments" and parts[-1] not in EXPERIMENTS_KEPT:
            continue
        if rel == "witnesses.py":
            continue
        out = dest / "src/dsoi" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        copied.append(f"src/dsoi/{rel}")
    shutil.copytree(ROOT / "scenarios", dest / "scenarios",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (dest / "configs").mkdir()
    shutil.copy2(ROOT / "configs/versions.yaml", dest / "configs/versions.yaml")
    shutil.copy2(ROOT.parent / "mdpi-paper/tables/contracts.tex", dest / "CONTRACTS.tex")
    for name in ("FAULT_BRIEF.md", "NEGATIVE_FORMAT.md"):
        shutil.copy2(ROOT / "data/fault_trial" / name, dest / name)
    (dest / "negatives").mkdir()
    (dest / "patches").mkdir()
    (dest / ".gitignore").write_text("patches/\nnegatives/\nmanifest.json\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.name=sandbox", "-c", "user.email=sandbox@localhost",
                 "commit", "-qm", "sandbox"]):
        subprocess.run(cmd, cwd=dest, check=True)
    return copied


if __name__ == "__main__":
    files = build(Path(sys.argv[1]))
    print(f"{len(files)} source files copied; withheld: {sorted(WITHHELD)}")
