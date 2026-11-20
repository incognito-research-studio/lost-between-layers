"""dsoi-sweep: run the S1 Monte Carlo over a grid of interpreter noise parameters.

Every cell reuses the base seed, so cells see the same true reductions and the
same noise draws, scaled by the cell's parameters (common random numbers across
cells as well as across gate rungs).
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import time
from pathlib import Path

import yaml

from dsoi.sim.runner import run_s1


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def run_sweep(config: dict) -> list[dict]:
    base = config["base"]
    cells = []
    for i, combo in enumerate(expand_grid(config["grid"])):
        cfg = copy.deepcopy(base)
        cfg["interpreter"] = {**base.get("interpreter", {}), **combo}
        _, summary = run_s1(cfg)
        for gate, s in summary["gates"].items():
            cells.append({"cell": i, **combo, "gate": gate, **s})
    return cells


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsoi-sweep", description=__doc__)
    ap.add_argument("config", type=Path, help="YAML config, e.g. configs/sweep_s1.yaml")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    out = args.out or Path("results") / config.get("name", args.config.stem)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cells = run_sweep(config)
    with open(out / "cells.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    n_cells = len({c["cell"] for c in cells})
    print(f"{n_cells} cells x {len(config['base']['gates'])} gates in "
          f"{time.perf_counter() - started:.0f}s -> {out}/cells.csv")
    attrib = sum(c["attribution_violations"] for c in cells)
    print(f"attribution exceptions across all cells: {attrib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
