"""Round 2 of the fault-injection evaluation with separated roles.

    uv run python tools/ft_round2.py --faults DIR --episodes DIR --negatives DIR \
        --manifest-sha SHA --out results/fault_trial_r2.json

Protocol: data/fault_trial/round2/PROTOCOL.md. Frozen before any round-2 fault existed. It
never opens the manifest. Steps, in order:
  1. Episode screening, before any patch runs: each generator episode is run on the clean
     pipeline; an episode whose pipeline cannot be built there is excluded and listed. The
     remaining episodes join the fixed worlds (the e1v2 replay and the twelve scenario
     episodes of round 1).
  2. The clean pipeline is run twice on all worlds; any difference stops the trial.
  3. Every patch: src/ is copied, the patch applied (round 1's ALLOWED rule), and the round-2
     harness run on all worlds; `ft_trial.compare` classifies it exactly as in round 1.
  4. The negative cases run on the clean pipeline; `ft_trial.negatives_report` as in round 1.
Scoring uses tools/ft_score.py unchanged.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_trial  # noqa: E402  (frozen in round 1, reused unchanged)

ROOT = ft_trial.ROOT
HARNESS = ROOT / "tools/ft_round2_harness.py"
INSTRUMENTS = ft_trial.INSTRUMENTS + (
    "tools/ft_trace_ext.py", "tools/ft_round2_harness.py", "tools/ft_round2.py",
    "tools/ft_sandbox2.py", "data/fault_trial/round2/PROTOCOL.md",
    "data/fault_trial/round2/FAULT_BRIEF.md", "data/fault_trial/round2/run_case.py",
    "data/fault_trial/round2/l5_monitors_stub.py")


def harness(src: Path, work: Path, extra: list[str]) -> tuple[dict | None, str]:
    out = work / "runs.json"
    env = {**os.environ, "PYTHONPATH": str(src)}
    probe = subprocess.run([sys.executable, "-c", "import dsoi; print(dsoi.__file__)"],
                           env=env, capture_output=True, text=True, cwd=work)
    if probe.returncode != 0:
        return None, probe.stderr[-2000:]
    if not probe.stdout.strip().startswith(str(src)):
        raise RuntimeError(f"the harness would not import the patched package: {probe.stdout}")
    run = subprocess.run([sys.executable, str(HARNESS), "--out", str(out), *extra],
                         env=env, capture_output=True, text=True, cwd=work)
    if run.returncode != 0:
        return None, run.stderr[-2000:]
    return json.loads(out.read_text()), ""


def in_copy(patch: Path | None, extra: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        shutil.copytree(ROOT / "src", work / "src")
        for name in ("configs", "scenarios", "data", "results"):
            (work / name).symlink_to(ROOT / name, target_is_directory=True)
        if patch is not None:
            bad = sorted(p for p in ft_trial.touched(patch) if p not in ft_trial.ALLOWED)
            if bad:
                return {"status": "rejected", "detail": f"touches files outside ALLOWED: {bad}"}
            applied = subprocess.run(["patch", "-p1", "-s", "--no-backup-if-mismatch", "-d",
                                      str(work), "-i", str(patch.resolve())],
                                     capture_output=True, text=True)
            if applied.returncode != 0:
                return {"status": "rejected", "detail": applied.stdout + applied.stderr}
        res, err = harness(work / "src", work, extra)
    if res is None:
        return {"status": "broken", "detail": err}
    return {"status": "ran", "result": res}


def dir_sha(d: Path) -> str:
    return hashlib.sha256(b"".join(ft_trial.sha256(f).encode()
                                   for f in sorted(d.rglob("*")) if f.is_file())).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--faults", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--negatives", type=Path)
    ap.add_argument("--manifest-sha", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results/fault_trial_r2.json")
    args = ap.parse_args(argv)
    # 1. screen the episodes on the clean pipeline
    screen = in_copy(None, ["--no-fixed", "--episodes", str(args.episodes.resolve())])
    if screen["status"] != "ran":
        raise SystemExit(f"episode screening failed: {screen['detail']}")
    built = {k.split("/", 1)[1] for k, v in screen["result"]["runs"].items() if not v["crashed"]}
    all_eps = sorted(p.name for p in args.episodes.iterdir() if (p / "scenario.yaml").exists())
    excluded = sorted(set(all_eps) - built)
    with tempfile.TemporaryDirectory() as d:
        kept = Path(d) / "episodes"
        kept.mkdir()
        for name in sorted(built):
            shutil.copytree(args.episodes / name, kept / name)
        extra = ["--episodes", str(kept)]
        # 2. clean pipeline twice
        first, second = in_copy(None, extra), in_copy(None, extra)
        if first["status"] != "ran" or second["status"] != "ran":
            raise SystemExit("the clean pipeline failed on the worlds")
        base = first["result"]
        if any(base["runs"][k]["trace"] != v["trace"]
               for k, v in second["result"]["runs"].items()):
            raise SystemExit("the clean pipeline is not deterministic on the worlds")
        if any(v["crashed"] for v in base["runs"].values()):
            raise SystemExit("a clean run could not be built after screening")
        # 3. every patch
        faults = {}
        for patch in sorted(args.faults.glob("*.patch")):
            t = in_copy(patch, extra)
            entry = {"sha256": ft_trial.sha256(patch)}
            if t["status"] == "ran":
                entry.update(ft_trial.compare(base, t["result"]))
            else:
                entry.update(status=t["status"], detail=t["detail"])
            faults[patch.name] = entry
            print(patch.name, entry["status"],
                  {c: entry[c]["detected"] for c in ft_trial.CONFIGS if c in entry}, flush=True)
    # 4. negatives on the clean pipeline
    negatives = {}
    if args.negatives:
        n = in_copy(None, ["--no-fixed", "--negatives", str(args.negatives.resolve())])
        if n["status"] != "ran":
            raise SystemExit(f"negative cases failed to run: {n['detail']}")
        negatives = ft_trial.negatives_report(n["result"])
        negatives["_sha256"] = {d.name: dir_sha(d) for d in sorted(args.negatives.iterdir())
                                if d.is_dir()}
    counts = {s: sum(f["status"] == s for f in faults.values())
              for s in ("effective", "inert", "broken", "rejected")}
    res = {"round": 2, "post_hoc": False, "fresh_model_calls": 0,
           "manifest_sha256": args.manifest_sha,
           "instruments_sha256": {p: ft_trial.sha256(ROOT / p) for p in INSTRUMENTS
                                  if (ROOT / p).exists()},
           "episodes": {"submitted": len(all_eps), "excluded_unbuildable": excluded,
                        "sha256": {e: dir_sha(args.episodes / e) for e in all_eps}},
           "clean": {"runs": len(base["runs"]),
                     "releases": sum(v["released"] for v in base["runs"].values()),
                     "violations": sum(v["violations"] for v in base["runs"].values()),
                     "fired": {k: sum(k in v["fired"] for v in base["runs"].values())
                               for k in base["checks"]}},
           "checks": base["checks"], "faults": faults, "negatives": negatives,
           "counts": counts}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"counts": counts, "excluded_episodes": excluded}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
