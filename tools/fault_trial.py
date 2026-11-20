"""Blind fault trial: do the frozen monitors detect integration faults planted by someone else?

    uv run python tools/fault_trial.py --faults path/to/faults --out results/fault_trial.json

Protocol in data/e1v2/FAULT_TRIAL.md. A
person who did not write the monitors writes each fault as a patch against src/dsoi
(paths relative to replication-package/, applied with `patch -p1`). For each patch this
tool copies src/ to a temporary directory, applies the patch there, and runs the frozen
monitors of tools/e1v2_monitors.py against the patched package in a fresh interpreter.
configs/ and scenarios/ are linked into the copy, so a patch must touch src/ only. The
working tree is never modified. The tool reads only the patches, never a key or a
description, so the classification below is fixed before anyone sees which fault is which.

Per fault, against the unmodified pipeline:
  rejected   the patch does not apply
  crashed    the patched package fails to import or the monitor run fails
  inert      no outcome count and no monitor count changes: the fault did nothing here
  detected   some monitor count changes (for any model); the monitors that changed are listed
  missed     some outcome count (abstentions, releases, violations, crashes) changes and no
             monitor count does: a fault that changed decisions without leaving a trace
The detection rate is detected / (detected + missed). Outcome counts use the benchmark
truth only to say whether a fault was effective, never to detect it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONITORS = ROOT / "tools/e1v2_monitors.py"


def monitor_run(src: Path, workdir: Path) -> tuple[dict | None, str]:
    """Run the monitors with `src` first on the path; check that it is the package imported."""
    out = workdir / "monitors.json"
    env = {**os.environ, "PYTHONPATH": str(src)}
    probe = subprocess.run(
        [sys.executable, "-c", "import dsoi, sys; print(dsoi.__file__)"],
        env=env, capture_output=True, text=True, cwd=workdir)
    if probe.returncode != 0:
        return None, probe.stderr[-2000:]
    if not probe.stdout.strip().startswith(str(src)):
        raise RuntimeError(f"monitors would not import the patched package: {probe.stdout}")
    run = subprocess.run(
        [sys.executable, str(MONITORS), "--out", str(out),
         "--cases", str(ROOT / "data/e1v2/cases.jsonl"),
         "--replies", str(ROOT / "results/e1v2/replies.jsonl"),
         "--margins", str(ROOT / "results/e1v2_components.json")],
        env=env, capture_output=True, text=True, cwd=workdir)
    if run.returncode != 0:
        return None, run.stderr[-2000:]
    return json.loads(out.read_text()), ""


def changed(base: dict, res: dict, part: str) -> dict:
    return {m: {k: [base[m][part][k], res[m][part][k]] for k in base[m][part]
                if base[m][part][k] != res[m][part][k]}
            for m in base if any(base[m][part][k] != res[m][part][k] for k in base[m][part])}


def trial(patch: Path | None) -> dict:
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        shutil.copytree(ROOT / "src", work / "src")
        for name in ("configs", "scenarios"):   # read relative to the package (scenarios.py)
            (work / name).symlink_to(ROOT / name, target_is_directory=True)
        if patch is not None:
            applied = subprocess.run(["patch", "-p1", "-s", "--no-backup-if-mismatch",
                                      "-d", str(work), "-i", str(patch.resolve())],
                                     capture_output=True, text=True)
            if applied.returncode != 0:
                return {"status": "rejected", "detail": applied.stdout + applied.stderr}
        res, err = monitor_run(work / "src", work)
    if res is None:
        return {"status": "crashed", "detail": err}
    return {"status": "ran", "result": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faults", type=Path, required=True, help="directory of *.patch files")
    ap.add_argument("--out", type=Path, default=ROOT / "results/fault_trial.json")
    args = ap.parse_args(argv)
    patches = sorted(args.faults.glob("*.patch"))
    base = trial(None)
    if base["status"] != "ran":
        raise SystemExit(f"unmodified pipeline failed: {base.get('detail')}")
    base = base["result"]
    if any(m["outcomes"]["crashed"] for m in base.values()):
        raise SystemExit("unmodified pipeline crashed in the trial copy; fix the harness")
    faults = {}
    for patch in patches:
        t = trial(patch)
        entry = {"sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                 "status": t["status"]}
        if t["status"] == "ran":
            mon = changed(base, t["result"], "monitors")
            out = changed(base, t["result"], "outcomes")
            entry["monitors_changed"], entry["outcomes_changed"] = mon, out
            entry["status"] = "detected" if mon else "missed" if out else "inert"
            entry["detected_by"] = sorted({k for m in mon.values() for k in m})
        else:
            entry["detail"] = t["detail"]
        faults[patch.name] = entry
        print(patch.name, entry["status"], entry.get("detected_by", ""))
    counts = {s: sum(f["status"] == s for f in faults.values())
              for s in ("detected", "missed", "inert", "crashed", "rejected")}
    effective = counts["detected"] + counts["missed"]
    res = {"post_hoc": False, "fresh_model_calls": 0,
           "monitors_sha256": hashlib.sha256(MONITORS.read_bytes()).hexdigest(),
           "baseline": base, "faults": faults, "counts": counts,
           "detection_rate": None if not effective else counts["detected"] / effective}
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"counts": counts, "detection_rate": res["detection_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
