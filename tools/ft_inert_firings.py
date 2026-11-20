"""Deviation 1, descriptive part (data/fault_trial/DEVIATIONS.md).

    uv run python tools/ft_inert_firings.py      # from the package root

For patches inert under the extended effect trace, list the checks that newly fire against
the clean run. Descriptive only: such firings are not counted as detections, because the
classification rules say an inert patch is excluded before detection is looked at.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_trial  # noqa: E402  (the frozen trial runner, unchanged)

ft_trial.HARNESS = ft_trial.ROOT / "tools/ft_trace_ext.py"
ROOT = ft_trial.ROOT


def main() -> int:
    reanalysis = json.loads((ROOT / "results/fault_trial_v2_reanalysis.json").read_text())
    inert = sorted(n for n, f in reanalysis["faults"].items() if f["status"] == "inert")
    base = ft_trial.in_copy(None)["result"]
    out = {}
    for name in inert:
        res = ft_trial.in_copy(ROOT / "data/fault_trial/set-1/patches" / name)["result"]
        new = {k: sorted(set(res["runs"][k]["fired"]) - set(v["fired"]))
               for k, v in base["runs"].items()}
        new = {k: v for k, v in new.items() if v}
        out[name] = {"runs_with_new_firings": len(new),
                     "checks": sorted({c for v in new.values() for c in v}),
                     "example_runs": sorted(new)[:3]}
        print(name, out[name], flush=True)
    (ROOT / "results/fault_trial_v2_inert_firings.json").write_text(json.dumps(
        {"descriptive_only": True, "note": "checks that newly fire on patches inert under the "
         "extended effect trace (deviation 1); not counted as detections", "patches": out},
        indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
