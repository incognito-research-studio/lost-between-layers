"""dsoi-sim: run a simulation config and write runs.csv and summary.json.

The config's `scenario` picks the study: S1 (gate ladder under interpreter noise),
S3 (missed suspension x stale vendor master), S4 (contradictory sources x L3 policy),
S8 (coefficient uncertainty: box vs state-vertex),
EXEC (faulty execution and time-of-check/time-of-use on S1).
A `grid:` block in an S3/S4 config is expanded into `cells`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from dsoi.sim.execution import run_exec
from dsoi.sim.runner import run_s1
from dsoi.sim.scenario_sims import run_s3, run_s4, run_s8
from dsoi.sim.sweep import expand_grid

SCENARIOS = {"S1": run_s1, "S3": run_s3, "S4": run_s4, "S8": run_s8, "EXEC": run_exec}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsoi-sim", description=__doc__)
    ap.add_argument("config", type=Path, help="YAML config, e.g. configs/sim_s1.yaml")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    args = ap.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    if "grid" in config:
        config["cells"] = expand_grid(config.pop("grid"))
    out = args.out or Path("results") / config.get("name", args.config.stem)
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = SCENARIOS[config["scenario"]](config)

    with open(out / "runs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    if "gates" in summary:  # S1
        print(f"{'gate':<12}{'K-exec viol.':>14}{'coverage':>10}{'excess shortfall':>18}"
              f"{'rejected':>10}{'attrib. viol.':>15}")
        for name, s in summary["gates"].items():
            excess = s["mean_excess_shortfall_when_compliant"]
            print(f"{name:<12}{s['k_exec_violation_rate']:>14.3f}{s['coverage_rate']:>10.3f}"
                  f"{'-' if excess is None else f'{excess:.1f}':>18}{s['rejected']:>10}"
                  f"{s['attribution_violations']:>15}")
    else:
        cells = summary["cells"]
        keys = list(cells[0])
        print("  ".join(f"{k:>14}" for k in keys))
        for c in cells:
            print("  ".join(f"{_fmt(c[k]):>14}" for k in keys))
    print(f"wrote {out}/runs.csv and {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
