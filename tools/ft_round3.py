"""Fault-injection evaluation with the repaired configuration: regression and round 3.

    uv run python tools/ft_round3.py --faults DIR [--episodes DIR] [--negatives DIR ...] \
        [--label NAME] [--manifest-sha SHA] --out results/fault_trial_<label>.json

Written 13 Nov 2026 (seventh external review, step 4). Runs a fault set through
tools/ft_repaired.py, the frozen harness plus the repaired checks, twice per pipeline:
  monitor  the checks run after each run, as in rounds 1 and 2; detection is scored per
           configuration (solver < generic < full < repaired) with the classification of
           `ft_trial.compare`, unchanged except that it ranges over four configurations
  gating   the repaired configuration's absolute pre-execution checks hold a release that
           fails them, through the pipeline's L5 monitor hook (tools/ft_repaired.py); this
           measures containment: which harmful releases were in fact not executed
The steps follow tools/ft_round2.py: episode screening on the clean pipeline, the clean
pipeline twice (it must be deterministic), every patch in its own copy of src/ (round 1's
ALLOWED rule), the negative cases on the clean pipeline. The manifest is never opened.

Used post hoc on round 1 (set-1, fixed worlds) and round 2 (set-2, with its episodes) as
regression evidence, since the repaired checks were written after both were unblinded, and
prospectively on round 3 (data/fault_trial/round3/PROTOCOL.md).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_round2  # noqa: E402  (frozen: dir_sha)
import ft_trial  # noqa: E402  (frozen: ALLOWED, touched, sha256)

ROOT = ft_trial.ROOT
HARNESS = ROOT / "tools/ft_repaired.py"
CONFIGS = ("solver", "generic", "full", "repaired")
INSTRUMENTS = ft_round2.INSTRUMENTS + ("tools/ft_repaired.py", "tools/ft_round3.py")


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


def _rank(cfg: str) -> int:
    return CONFIGS.index(cfg)


def compare(base: dict, res: dict, gbase: dict, gres: dict) -> dict:
    """`ft_trial.compare` over four configurations, plus harm and containment.

    A harmful run has more violations of the true constraints than the clean run. It is
    flagged by a configuration when one of the configuration's checks newly fires on it, and
    contained when, with the gate on, its violations do not exceed the clean gated run's."""
    checks = base["checks"]
    runs = base["runs"]
    effective = any(res["runs"][k]["trace"] != v["trace"]
                    or res["runs"][k]["violations"] != v["violations"]
                    or res["runs"][k]["released"] != v["released"]
                    or res["runs"][k]["crashed"] != v["crashed"]
                    for k, v in runs.items())
    entry: dict = {"status": "effective" if effective else "inert",
                   "runs_changed": sum(res["runs"][k]["trace"] != v["trace"]
                                       for k, v in runs.items()),
                   "releases": [sum(v["released"] for v in runs.values()),
                                sum(v["released"] for v in res["runs"].values())],
                   "violations": [sum(v["violations"] for v in runs.values()),
                                  sum(v["violations"] for v in res["runs"].values())]}
    if not effective:
        return entry
    harmful = sorted(k for k, v in runs.items() if res["runs"][k]["violations"] > v["violations"])
    for cfg in CONFIGS:
        names = {n for n, c in checks.items() if _rank(c["config"]) <= _rank(cfg)}
        new_by_run, layers, pre = {}, set(), False
        for k, v in runs.items():
            r = res["runs"][k]
            new = sorted((set(r["fired"]) - set(v["fired"])) & names)
            if new:
                new_by_run[k] = new
                for n in new:
                    if n == "exception":
                        layers.add(r.get("exception_layer") or "unknown")
                    else:
                        layers.update(checks[n]["layers"])
                pre = pre or any(checks[n]["timing"] == "pre" for n in new)
        entry[cfg] = {"detected": bool(new_by_run), "detected_pre_execution": pre,
                      "checks": sorted({n for ns in new_by_run.values() for n in ns}),
                      "layers": sorted(layers), "runs_flagged": len(new_by_run),
                      "harmful_runs_flagged": sum(k in new_by_run for k in harmful),
                      "silent_violation_runs": sum(k not in new_by_run for k in harmful)}
    gated_harm = sorted(k for k in harmful
                        if gres["runs"][k]["violations"] > gbase["runs"][k]["violations"])
    entry["harm"] = {
        "harmful_runs": len(harmful),
        "extra_violations": sum(res["runs"][k]["violations"] - runs[k]["violations"]
                                for k in harmful),
        "harmful_runs_contained": len(harmful) - len(gated_harm),
        "harmful_runs_with_gate": len(gated_harm),
        "extra_violations_with_gate": sum(gres["runs"][k]["violations"]
                                          - gbase["runs"][k]["violations"] for k in gated_harm),
        "holds": sum(len(v["holds"]) for v in gres["runs"].values()),
        "runs_with_holds": sum(bool(v["holds"]) for v in gres["runs"].values()),
        "held_violating": sum(v["held_violating"] for v in gres["runs"].values()),
        "releases_with_gate": sum(v["released"] for v in gres["runs"].values()),
    }
    entry["fixed_or_generated"] = {
        "detected_fixed": any(k.split("/")[0] in ("e1v2", "scenarios")
                              and (set(res["runs"][k]["fired"]) - set(v["fired"]))
                              for k, v in runs.items()),
        "detected_generated": any(k.startswith("episodes")
                                  and (set(res["runs"][k]["fired"]) - set(v["fired"]))
                                  for k, v in runs.items())}
    return entry


def negatives_report(res: dict, gated: dict) -> dict:
    checks = res["checks"]
    out = {}
    for k, v in res["runs"].items():
        if not k.startswith("negatives"):
            continue
        fired = [n for n in v["fired"] if not checks[n]["relative"]]
        out[k] = {"fired": v["fired"], "crashed": v["crashed"], "released": v["released"],
                  "decisions": v["decisions"], "holds": gated["runs"][k]["holds"],
                  "released_with_gate": gated["runs"][k]["released"],
                  **{cfg: {"false_alarm": any(_rank(checks[n]["config"]) <= _rank(cfg)
                                              for n in fired),
                           "checks": sorted(n for n in fired
                                            if _rank(checks[n]["config"]) <= _rank(cfg))}
                     for cfg in CONFIGS}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--faults", type=Path, required=True)
    ap.add_argument("--episodes", type=Path)
    ap.add_argument("--negatives", type=Path, nargs="*", default=[])
    ap.add_argument("--label", required=True)
    ap.add_argument("--manifest-sha", default=None)
    ap.add_argument("--post-hoc", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    excluded, all_eps = [], []
    with tempfile.TemporaryDirectory() as d:
        extra = []
        if args.episodes:
            screen = in_copy(None, ["--no-fixed", "--episodes", str(args.episodes.resolve())])
            if screen["status"] != "ran":
                raise SystemExit(f"episode screening failed: {screen['detail']}")
            built = {k.split("/", 1)[1] for k, v in screen["result"]["runs"].items()
                     if not v["crashed"]}
            all_eps = sorted(p.name for p in args.episodes.iterdir()
                             if (p / "scenario.yaml").exists())
            excluded = sorted(set(all_eps) - built)
            kept = Path(d) / "episodes"
            kept.mkdir()
            for name in sorted(built):
                shutil.copytree(args.episodes / name, kept / name)
            extra = ["--episodes", str(kept)]
        first, second = in_copy(None, extra), in_copy(None, extra)
        gfirst = in_copy(None, extra + ["--gating"])
        if any(x["status"] != "ran" for x in (first, second, gfirst)):
            raise SystemExit("the clean pipeline failed on the worlds")
        base, gbase = first["result"], gfirst["result"]
        if any(base["runs"][k]["trace"] != v["trace"]
               for k, v in second["result"]["runs"].items()):
            raise SystemExit("the clean pipeline is not deterministic on the worlds")
        faults = {}
        for patch in sorted(args.faults.glob("*.patch")):
            t = in_copy(patch, extra)
            entry = {"sha256": ft_trial.sha256(patch)}
            if t["status"] != "ran":
                entry.update(status=t["status"], detail=t["detail"])
            else:
                g = in_copy(patch, extra + ["--gating"])
                if g["status"] != "ran":
                    raise SystemExit(f"{patch.name}: ran without the gate but not with it")
                entry.update(compare(base, t["result"], gbase, g["result"]))
                entry["per_run"] = {
                    k: {"fired_new": sorted(set(t["result"]["runs"][k]["fired"])
                                            - set(v["fired"])),
                        "violations": [v["violations"], t["result"]["runs"][k]["violations"],
                                       g["result"]["runs"][k]["violations"]],
                        "holds": g["result"]["runs"][k]["holds"]}
                    for k, v in base["runs"].items()
                    if t["result"]["runs"][k]["trace"] != v["trace"]
                    or t["result"]["runs"][k]["violations"] != v["violations"]
                    or set(t["result"]["runs"][k]["fired"]) != set(v["fired"])
                    or g["result"]["runs"][k]["holds"]}
            faults[patch.name] = entry
            print(patch.name, entry["status"],
                  {c: entry[c]["detected"] for c in CONFIGS if c in entry},
                  entry.get("harm", {}).get("harmful_runs"),
                  entry.get("harm", {}).get("harmful_runs_contained"), flush=True)
    negatives = {}
    if args.negatives:
        flags = ["--no-fixed", "--negatives", *[str(n.resolve()) for n in args.negatives]]
        n, ng = in_copy(None, flags), in_copy(None, flags + ["--gating"])
        if n["status"] != "ran" or ng["status"] != "ran":
            raise SystemExit("negative cases failed to run")
        negatives = negatives_report(n["result"], ng["result"])
    counts = {s: sum(f["status"] == s for f in faults.values())
              for s in ("effective", "inert", "broken", "rejected")}
    eff = [f for f in faults.values() if f["status"] == "effective"]
    summary = {cfg: {"detected": sum(f[cfg]["detected"] for f in eff),
                     "detected_pre": sum(f[cfg]["detected_pre_execution"] for f in eff),
                     "harmful_runs_flagged": sum(f[cfg]["harmful_runs_flagged"] for f in eff)}
               for cfg in CONFIGS}
    summary["harm"] = {k: sum(f["harm"][k] for f in eff) for k in eff[0]["harm"]} if eff else {}
    summary["harmful_faults"] = sum(f["harm"]["harmful_runs"] > 0 for f in eff)
    summary["harmful_faults_fully_contained"] = sum(
        f["harm"]["harmful_runs"] > 0 and f["harm"]["harmful_runs_with_gate"] == 0 for f in eff)
    clean = {"runs": len(base["runs"]),
             "releases": sum(v["released"] for v in base["runs"].values()),
             "violations": sum(v["violations"] for v in base["runs"].values()),
             "releases_with_gate": sum(v["released"] for v in gbase["runs"].values()),
             "violations_with_gate": sum(v["violations"] for v in gbase["runs"].values()),
             "holds_with_gate": {k: v["holds"] for k, v in gbase["runs"].items() if v["holds"]},
             "fired": {k: sum(k in v["fired"] for v in base["runs"].values())
                       for k in base["checks"]}}
    res = {"label": args.label, "post_hoc": args.post_hoc, "fresh_model_calls": 0,
           "manifest_sha256": args.manifest_sha,
           "instruments_sha256": {p: ft_trial.sha256(ROOT / p) for p in INSTRUMENTS
                                  if (ROOT / p).exists()},
           "episodes": {"submitted": len(all_eps), "excluded_unbuildable": excluded},
           "clean": clean, "checks": base["checks"], "faults": faults,
           "negatives": negatives, "counts": counts, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"counts": counts, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
