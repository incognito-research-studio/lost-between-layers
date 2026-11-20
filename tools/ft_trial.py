"""Fault-injection evaluation with separated roles: run the frozen checks on every patch.

    uv run python tools/ft_trial.py --faults DIR --negatives DIR --manifest-sha SHA \
        --out results/fault_trial_v2.json

Sixth external review (8 Nov 2026), experiment 1; protocol in data/fault_trial/PROTOCOL.md.
This tool never opens the fault manifest: it records the manifest's SHA-256 as given, so
`tools/ft_score.py` can check later that the manifest it scores against is the one sealed
before this run. For each patch it copies src/, applies the patch there (paths relative to
replication-package/, `patch -p1`), and runs tools/ft_harness.py against the patched copy in
a fresh interpreter. The working tree is never modified.

Per patch, against the unmodified pipeline (run twice first, to confirm it is deterministic):
  rejected   the patch does not apply, or touches a file outside ALLOWED
  broken     the patched package cannot be imported or the harness itself fails: loud, but
             not an integration fault; reported and excluded from rates
  inert      every run's effect trace and outcome equals the clean run's: the patch did
             nothing observable in these worlds; reported and excluded from rates
  effective  some run's trace or outcome differs
For an effective patch and each configuration (solver, generic, full): detected when some run
shows a check of that configuration firing that did not fire on the clean run; detected
before execution when such a check is a pre-execution check; the layers the new firings
point to (an exception points to the layer of its innermost dsoi frame); silent-violation
runs, whose violations exceed the clean run's while no check of that configuration newly
fires; and releases, against the clean count. Negative cases run on the unmodified pipeline:
a false alarm is a check of the configuration firing on one, apart from the relative checks,
which can fire on correct runs and count only as a change against clean.
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
HARNESS = ROOT / "tools/ft_harness.py"
INSTRUMENTS = ("tools/ft_harness.py", "tools/ft_trial.py", "tools/ft_score.py",
               "tools/ft_sandbox.py",
               "data/fault_trial/authority.yaml", "data/fault_trial/PROTOCOL.md",
               "data/fault_trial/FAULT_BRIEF.md", "data/fault_trial/NEGATIVE_FORMAT.md",
               "src/dsoi/properties.py")
ALLOWED = {f"src/dsoi/{p}" for p in (
    "canon.py", "constraints.py", "context.py", "store.py", "types.py", "pipeline.py",
    "registry.py", "layers/l1_interpreter.py", "layers/l1_claude.py",
    "layers/l2_procedural.py", "layers/l3_state.py", "layers/l4_decision.py",
    "layers/l5_gate.py", "layers/l6_audit.py", "experiments/e1v2_world.py",
    "experiments/e1v2_corrected.py")}
CONFIGS = ("solver", "generic", "full")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def touched(patch: Path) -> set[str]:
    """Files a unified diff names (paths after a/ and b/)."""
    out = set()
    for line in patch.read_text(errors="replace").splitlines():
        if line.startswith(("--- ", "+++ ")):
            path = line[4:].split("\t", 1)[0].strip()
            if path != "/dev/null":
                out.add(path.split("/", 1)[1] if path[:2] in ("a/", "b/") else path)
    return out


def harness(src: Path, work: Path, negatives: Path | None = None) -> tuple[dict | None, str]:
    out = work / "runs.json"
    env = {**os.environ, "PYTHONPATH": str(src)}
    probe = subprocess.run([sys.executable, "-c", "import dsoi; print(dsoi.__file__)"],
                           env=env, capture_output=True, text=True, cwd=work)
    if probe.returncode != 0:
        return None, probe.stderr[-2000:]
    if not probe.stdout.strip().startswith(str(src)):
        raise RuntimeError(f"the harness would not import the patched package: {probe.stdout}")
    cmd = [sys.executable, str(HARNESS), "--out", str(out)]
    if negatives is not None:
        cmd += ["--negatives", str(negatives.resolve()), "--worlds"]
    run = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=work)
    if run.returncode != 0:
        return None, run.stderr[-2000:]
    return json.loads(out.read_text()), ""


def in_copy(patch: Path | None, negatives: Path | None = None) -> dict:
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        shutil.copytree(ROOT / "src", work / "src")
        for name in ("configs", "scenarios", "data", "results"):
            (work / name).symlink_to(ROOT / name, target_is_directory=True)
        if patch is not None:
            bad = sorted(p for p in touched(patch) if p not in ALLOWED)
            if bad:
                return {"status": "rejected", "detail": f"touches files outside ALLOWED: {bad}"}
            applied = subprocess.run(["patch", "-p1", "-s", "--no-backup-if-mismatch", "-d",
                                      str(work), "-i", str(patch.resolve())],
                                     capture_output=True, text=True)
            if applied.returncode != 0:
                return {"status": "rejected", "detail": applied.stdout + applied.stderr}
        res, err = harness(work / "src", work, negatives)
    if res is None:
        return {"status": "broken", "detail": err}
    return {"status": "ran", "result": res}


def compare(base: dict, res: dict) -> dict:
    """Classify one effective-or-not patch against the clean runs, per configuration."""
    checks = base["checks"]
    effective = any(res["runs"][k]["trace"] != v["trace"]
                    or res["runs"][k]["violations"] != v["violations"]
                    or res["runs"][k]["released"] != v["released"]
                    or res["runs"][k]["crashed"] != v["crashed"]
                    for k, v in base["runs"].items())
    entry: dict = {"status": "effective" if effective else "inert",
                   "runs_changed": sum(res["runs"][k]["trace"] != v["trace"]
                                       for k, v in base["runs"].items()),
                   "releases": [sum(v["released"] for v in base["runs"].values()),
                                sum(v["released"] for v in res["runs"].values())],
                   "violations": [sum(v["violations"] for v in base["runs"].values()),
                                  sum(v["violations"] for v in res["runs"].values())]}
    if not effective:
        return entry
    for cfg in CONFIGS:
        names = {n for n, c in checks.items()
                 if CONFIGS.index(c["config"]) <= CONFIGS.index(cfg)}
        new_by_run, layers, pre, silent, contained = {}, set(), False, 0, 0
        for k, v in base["runs"].items():
            r = res["runs"][k]
            new = sorted((set(r["fired"]) - set(v["fired"])) & names)
            if new:
                new_by_run[k] = new
                for n in new:
                    if n == "exception":
                        layers.add(r.get("exception_layer") or "unknown")
                    else:
                        layers.update(checks[n]["layers"])
                if any(checks[n]["timing"] == "pre" for n in new):
                    pre = True
                    contained += r["released"]
            elif r["violations"] > v["violations"]:
                silent += 1
        entry[cfg] = {"detected": bool(new_by_run), "detected_pre_execution": pre,
                      "checks": sorted({n for ns in new_by_run.values() for n in ns}),
                      "layers": sorted(layers), "runs_flagged": len(new_by_run),
                      "silent_violation_runs": silent,
                      "releases_a_pre_check_would_hold": contained}
    return entry


def negatives_report(res: dict) -> dict:
    checks = res["checks"]
    out = {}
    for k, v in res["runs"].items():
        if not k.startswith("negatives/"):
            continue
        fired = [n for n in v["fired"] if not checks[n]["relative"]]
        out[k.split("/", 1)[1]] = {
            "fired": v["fired"], "crashed": v["crashed"], "exception": v["exception"],
            "released": v["released"], "decisions": v["decisions"],
            **{cfg: {"false_alarm": any(CONFIGS.index(checks[n]["config"])
                                        <= CONFIGS.index(cfg) for n in fired),
                     "checks": sorted(n for n in fired if CONFIGS.index(checks[n]["config"])
                                      <= CONFIGS.index(cfg))} for cfg in CONFIGS}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--faults", type=Path, required=True, help="directory of *.patch files")
    ap.add_argument("--negatives", type=Path, help="directory of negative-case scenarios")
    ap.add_argument("--manifest-sha", default=None,
                    help="SHA-256 of the sealed manifest, recorded, never checked here")
    ap.add_argument("--out", type=Path, default=ROOT / "results/fault_trial_v2.json")
    args = ap.parse_args(argv)
    first, second = in_copy(None), in_copy(None)
    if first["status"] != "ran" or second["status"] != "ran":
        detail = first.get("detail") or second.get("detail")
        raise SystemExit(f"unmodified pipeline failed: {detail}")
    base = first["result"]
    if any(base["runs"][k]["trace"] != v["trace"] for k, v in second["result"]["runs"].items()):
        raise SystemExit("the unmodified pipeline is not deterministic; fix the harness first")
    if any(v["crashed"] for v in base["runs"].values()):
        raise SystemExit("an unmodified run could not be built; fix the harness first")
    faults = {}
    for patch in sorted(args.faults.glob("*.patch")):
        t = in_copy(patch)
        entry = {"sha256": sha256(patch)}
        if t["status"] == "ran":
            entry.update(compare(base, t["result"]))
        else:
            entry.update(status=t["status"], detail=t["detail"])
        faults[patch.name] = entry
        print(patch.name, entry["status"],
              {c: entry[c]["detected"] for c in CONFIGS if c in entry}, flush=True)
    negatives = {}
    if args.negatives:
        n = in_copy(None, args.negatives)
        if n["status"] != "ran":
            raise SystemExit(f"negative cases failed to run: {n['detail']}")
        negatives = negatives_report(n["result"])
        negatives = {**negatives, "_sha256": {d.name: hashlib.sha256(b"".join(
            sha256(f).encode() for f in sorted(d.rglob("*")) if f.is_file())).hexdigest()
            for d in sorted(args.negatives.iterdir()) if d.is_dir()}}
    counts = {s: sum(f["status"] == s for f in faults.values())
              for s in ("effective", "inert", "broken", "rejected")}
    res = {"post_hoc": False, "fresh_model_calls": 0, "manifest_sha256": args.manifest_sha,
           "instruments_sha256": {p: sha256(ROOT / p) for p in INSTRUMENTS
                                  if (ROOT / p).exists()},
           "clean": {"runs": len(base["runs"]),
                     "releases": sum(v["released"] for v in base["runs"].values()),
                     "violations": sum(v["violations"] for v in base["runs"].values()),
                     "fired": {n: sum(n in v["fired"] for v in base["runs"].values())
                               for n in base["checks"]}},
           "checks": base["checks"], "faults": faults, "negatives": negatives,
           "counts": counts}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
