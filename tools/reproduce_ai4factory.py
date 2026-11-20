"""Reproduce the AI4Factory workshop paper's results from stored data (no model calls, no network).

    python tools/reproduce_ai4factory.py [--quick] [--out DIR]

Run from the package root (or anywhere; paths are resolved from this file). Each step reruns one
analysis on the stored replies and data and compares its output with the stored result file: JSON
field by field, other files byte for byte. The four tools that write in place
(e1v2_shadowing.py, e1v2_abstention_reasons.py, laya_conformance.py analyse, ft_r3_trace.py) must leave their result
file's SHA-256 unchanged.
Exits 1 on any
mismatch or failed step. --quick skips the gated round-3 rerun (about 4 minutes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # package root
PAPER = ROOT.parent / "paper"
R = ROOT / "results"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(new: Path, old: Path) -> bool:
    if new.suffix == ".json":
        return json.loads(new.read_text()) == json.loads(old.read_text())
    return new.read_bytes() == old.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="where reruns write (default: a temp dir)")
    a = ap.parse_args()
    out = a.out or Path(tempfile.mkdtemp(prefix="ai4factory-rerun-"))
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT / 'tools'}",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    py, t = sys.executable, "tools/"
    # (name, command, [(rerun output, stored file)], [in-place files whose hash must not change])
    steps = [
        ("Table 2: corrected replay", [py, t + "e1v2_corrected.py", "--out", str(out / "e1v2_corrected")],
         [(out / "e1v2_corrected/summary.json", R / "e1v2_corrected/summary.json"),
          (out / "e1v2_corrected/rows.csv", R / "e1v2_corrected/rows.csv")], []),
        ("Violation sources", [py, t + "e1v2_violation_sources.py", "--out", str(out / "vs.json")],
         [(out / "vs.json", R / "e1v2_violation_sources.json")], []),
        ("Log predicates", [py, t + "e1v2_log_checks.py", "--out", str(out / "log.json")],
         [(out / "log.json", R / "e1v2_log_checks.json")], []),
        ("First defect's effect (replay text)", [py, t + "e1v2_shadowing.py"],
         [], [R / "e1v2_shadowing.json"]),
        ("Figure 2: responses to abstention",
         [py, t + "e1v2_abstention_policies.py", "--out", str(out / "pol.json")],
         [(out / "pol.json", R / "e1v2_abstention_policies.json")], []),
        ("Figure 2: expected values", [py, t + "e1v2_abstention_expectations.py", "--out", str(out / "exp.json")],
         [(out / "exp.json", R / "e1v2_abstention_expectations.json")], []),
        ("Abstention reasons, truth cells, check counts", [py, t + "e1v2_abstention_reasons.py"],
         [], [R / "e1v2_abstention_reasons.json"]),
        ("Table 4: typed reader", [py, t + "laya_conformance.py", "analyse"],
         [], [R / "laya_conformance.json"]),
        ("Table 4: baselines", [py, t + "laya_baselines.py", "--out", str(out / "lb.json")],
         [(out / "lb.json", R / "laya_baselines.json")], []),
        ("Ablation", [py, t + "ft_ablation.py", "--out", str(out / "abl.json")],
         [(out / "abl.json", R / "fault_trial_r3_ablation.json")], []),
        ("One escape, traced", [py, t + "ft_r3_trace.py"], [], [R / "fault_trial_r3_trace.json"]),
    ]
    if not a.quick:
        steps.insert(10, ("Gate holds, round 3 (gated rerun)",
                         [py, t + "ft_r3_holds.py", "--faults", "data/fault_trial/round3/set-3/patches",
                          "--episodes", "data/fault_trial/round3/set-3/episodes",
                          "--out", str(out / "holds.json")],
                         [(out / "holds.json", R / "fault_trial_r3_holds.json")], []))
    if (PAPER / "verify_numbers.py").exists():
        steps.append(("Numbers in the paper", [py, str(PAPER / "verify_numbers.py")], [], []))
    failed = 0
    for name, cmd, compare, inplace in steps:
        before = {p: sha(p) for p in inplace}
        t0 = time.time()
        res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        ok = res.returncode == 0
        notes = [] if ok else [f"exit {res.returncode}: {res.stderr.strip().splitlines()[-1:]}"]
        for new, old in compare:
            if not new.exists() or not same(new, old):
                ok = False
                notes.append(f"differs: {old.relative_to(ROOT)}")
        for p in inplace:
            if sha(p) != before[p]:
                ok = False
                notes.append(f"changed in place: {p.relative_to(ROOT)}")
        if ok and name == "Numbers in the paper":
            notes.append(res.stdout.strip().splitlines()[-1])
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({time.time() - t0:.0f} s)  {'; '.join(notes)}")
    print(f"\n{len(steps) - failed} of {len(steps)} steps passed; reruns in {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
