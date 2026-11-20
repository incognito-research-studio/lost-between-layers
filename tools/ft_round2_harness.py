"""Round-2 harness driver (data/fault_trial/round2/PROTOCOL.md).

    PYTHONPATH=<patched src> python tools/ft_round2_harness.py --out runs.json \
        [--episodes DIR] [--negatives DIR] [--no-fixed]

Imports the frozen round-1 harness (tools/ft_harness.py) unchanged, with the extended effect
trace of tools/ft_trace_ext.py (adopted for round 2 before any round-2 fault existed), and
adds one world: the generator's episodes, scenario directories run through the frozen
`episode` function under ids `episodes/<name>`. The checks, their configurations, tags and
evaluation are the frozen ones. `--no-fixed` runs only the given directories (used to run
the negative cases on the clean pipeline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_harness  # noqa: E402
import ft_trace_ext  # noqa: E402,F401  (installs the extended trace on ft_harness)
import yaml  # noqa: E402


def dir_runs(root: Path, prefix: str):
    from dsoi.scenarios import Scenario
    for d in sorted(p for p in root.iterdir() if (p / "scenario.yaml").exists()):
        spec = yaml.safe_load((d / "scenario.yaml").read_text())
        trial = spec.pop("trial", {})
        run_id = f"{prefix}/{d.name}"
        yield run_id, (lambda d=d, spec=spec, trial=trial, run_id=run_id:
                       ft_harness.episode(Scenario(d, spec), trial, run_id))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=Path)
    ap.add_argument("--negatives", type=Path)
    ap.add_argument("--no-fixed", action="store_true")
    args = ap.parse_args(argv)
    gens = [] if args.no_fixed else [ft_harness.e1v2_runs(), ft_harness.scenario_runs()]
    if args.episodes:
        gens.append(dir_runs(args.episodes, "episodes"))
    if args.negatives:
        gens.append(dir_runs(args.negatives, "negatives"))
    runs = {}
    for gen in gens:
        for run_id, go in gen:
            runs[run_id] = ft_harness.evaluate(go)
    meta = {c.name: {"config": c.config, "layers": list(c.layers), "timing": c.timing,
                     "relative": c.relative} for c in ft_harness.CHECKS}
    args.out.write_text(json.dumps({"checks": meta, "runs": runs}, indent=1,
                                   sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
