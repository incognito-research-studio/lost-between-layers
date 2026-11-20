"""Post-hoc per-run rerun of round 2 of the fault trial (13 Nov 2026).

    uv run python tools/ft_perrun.py [--only 01-orchid.patch ...] \
        --out results/fault_trial_r2_perrun.json

The frozen round-2 results (results/fault_trial_r2.json) keep one summary per fault and
configuration. This tool measures detection on the harmful run, before the harmful
release, on fixed against generator episodes, and for evidence that each patch exercised its
intended mechanism. This tool reruns round 2 with the same worlds, checks and classification
(`ft_trial.compare`, unchanged) and keeps, per run:
  hit        whether a line the patch added or changed executed in that run (sys.monitoring;
             a deletion-only hunk marks the lines around the deletion)
  changed    effect trace differs from the clean run's
  dviol      violations of the true constraints minus the clean run's
  new        per configuration, the checks that newly fire, with their timing tag
It first checks that `ft_trial.compare` on the rerun reproduces every frozen per-fault entry;
a mismatch is reported, not hidden. Nothing here is a check, a gate or a new classification:
the checks still run as monitors after each run, so `pre` remains a tag on the check, not
evidence that a release was held. Per-patch details: results/ft_perrun_r2/<patch>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_trial  # noqa: E402

ROOT = ft_trial.ROOT
HARNESS = ROOT / "tools/ft_perrun_harness.py"
SET2 = ROOT / "data/fault_trial/round2/set-2"
FROZEN = ROOT / "results/fault_trial_r2.json"
CONFIGS = ft_trial.CONFIGS


def patched_lines(patch: Path) -> dict[str, list[int]]:
    """New-file line numbers of each added line; around each deletion, its neighbours."""
    out: dict[str, set[int]] = {}
    path, new = None, 0
    for raw in patch.read_text().splitlines():
        if raw.startswith("+++ "):
            p = raw[4:].split("\t")[0].strip()
            p = p.split("/", 1)[1] if "/" in p else p          # -p1
            path = p[len("src/"):] if p.startswith("src/") else None
            continue
        if raw.startswith("--- "):
            continue
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            new = int(m.group(1))
            continue
        if path is None:
            continue
        if raw.startswith("+"):
            out.setdefault(path, set()).add(new)
            new += 1
        elif raw.startswith("-"):
            out.setdefault(path, set()).update({max(new - 1, 1), new})
        else:
            new += 1
    return {k: sorted(v) for k, v in out.items()}


def run_copy(patch: Path | None, episodes: Path) -> dict:
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        shutil.copytree(ROOT / "src", work / "src")
        for name in ("configs", "scenarios", "data", "results"):
            (work / name).symlink_to(ROOT / name, target_is_directory=True)
        extra = []
        if patch is not None:
            applied = subprocess.run(["patch", "-p1", "-s", "--no-backup-if-mismatch", "-d",
                                      str(work), "-i", str(patch.resolve())],
                                     capture_output=True, text=True)
            if applied.returncode != 0:
                return {"status": "rejected", "detail": applied.stdout + applied.stderr}
            hl = work / "hit_lines.json"
            hl.write_text(json.dumps(patched_lines(patch)))
            extra = ["--hit-lines", str(hl)]
        env = {**os.environ, "PYTHONPATH": str(work / "src")}
        out = work / "runs.json"
        run = subprocess.run([sys.executable, str(HARNESS), "--out", str(out),
                              "--episodes", str(episodes), *extra],
                             env=env, capture_output=True, text=True, cwd=work)
        if run.returncode != 0:
            return {"status": "broken", "detail": run.stderr[-2000:]}
        return {"status": "ran", "result": json.loads(out.read_text())}


def world(run_id: str) -> str:
    return "generator" if run_id.startswith("episodes/") else "fixed"


def per_run(base: dict, res: dict) -> dict:
    checks = base["checks"]
    rows = {}
    for k, v in base["runs"].items():
        r = res["runs"][k]
        changed = r["trace"] != v["trace"]
        differs = (changed or r["violations"] != v["violations"] or r["released"] != v["released"]
                   or r["crashed"] != v["crashed"] or set(r["fired"]) != set(v["fired"]))
        if not (differs or r.get("hits")):
            continue
        new = {}
        for cfg in CONFIGS:
            names = {n for n, c in checks.items()
                     if CONFIGS.index(c["config"]) <= CONFIGS.index(cfg)}
            new[cfg] = sorted((set(r["fired"]) - set(v["fired"])) & names)
        rows[k] = {"world": world(k), "hit": bool(r.get("hits")), "hits": r.get("hits", []),
                   "changed": changed, "dviol": r["violations"] - v["violations"],
                   "released": [v["released"], r["released"]], "crashed": r["crashed"],
                   "exception": r["exception"],
                   "new": new, "new_timing": {n: checks[n]["timing"] for n in new["full"]}}
        if "records" in r and r.get("records") != v.get("records"):
            rows[k]["records"] = {"clean": v.get("records"), "patched": r.get("records")}
    return rows


def summarize(rows: dict, import_hits: list, checks: dict) -> dict:
    def pre(names):
        return any(checks[n]["timing"] == "pre" for n in names)

    s = {"activated_at_import": bool(import_hits),
         "activated_runs": sum(r["hit"] for r in rows.values()),
         "changed_runs": sum(r["changed"] for r in rows.values()),
         "changed_runs_without_hit": sum(r["changed"] and not r["hit"] for r in rows.values()),
         "harmful_runs": sum(r["dviol"] > 0 for r in rows.values()),
         "harm_added": sum(max(r["dviol"], 0) for r in rows.values())}
    for cfg in CONFIGS:
        harmful = [r for r in rows.values() if r["dviol"] > 0]
        s[cfg] = {
            "harmful_runs_flagged": sum(bool(r["new"][cfg]) for r in harmful),
            "harmful_runs_flagged_pre": sum(pre(r["new"][cfg]) for r in harmful),
            "harmful_runs_silent": sum(not r["new"][cfg] for r in harmful),
            **{f"detected_{w}": any(r["new"][cfg] for r in rows.values() if r["world"] == w)
               for w in ("fixed", "generator")},
            **{f"harmful_runs_{w}": sum(r["dviol"] > 0 and r["world"] == w
                                        for r in rows.values())
               for w in ("fixed", "generator")}}
    return s


def _comparable(frozen_entry: dict, entry: dict) -> tuple[dict, dict]:
    """The frozen entry and the rerun's `ft_trial.compare` entry, each restricted to the keys
    both report (round 3's entries add a fourth configuration, harm and containment, and
    their own per-configuration fields)."""
    want, got = {}, {}
    for k, v in entry.items():
        if k not in frozen_entry or k == "sha256":
            continue
        w = frozen_entry[k]
        if isinstance(v, dict) and isinstance(w, dict):
            common = sorted(set(v) & set(w))
            want[k], got[k] = {c: w[c] for c in common}, {c: v[c] for c in common}
        else:
            want[k], got[k] = w, v
    return want, got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", type=Path, default=ROOT / "results/fault_trial_r2_perrun.json")
    ap.add_argument("--detail-dir", type=Path, default=ROOT / "results/ft_perrun_r2")
    ap.add_argument("--set", type=Path, default=SET2, help="fault set (round 3: round3/set-3)")
    ap.add_argument("--frozen", type=Path, default=FROZEN, help="the trial's primary result file")
    ap.add_argument("--round", type=int, default=2)
    args = ap.parse_args(argv)
    frozen = json.loads(args.frozen.read_text())
    fault_set = args.set
    args.detail_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clean = run_copy(None, fault_set / "episodes")
    if clean["status"] != "ran":
        raise SystemExit(f"clean pipeline failed: {clean['detail']}")
    base = clean["result"]
    if any(frozen["checks"].get(n, {}).get(k) != v for n, c in base["checks"].items()
           for k, v in c.items()):
        raise SystemExit("check metadata differs from the frozen run")
    if len(base["runs"]) != frozen["clean"]["runs"]:
        raise SystemExit("run count differs from the frozen round-2 run")
    print(f"clean: {len(base['runs'])} runs in {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for patch in sorted((fault_set / "patches").glob("*.patch")):
        if args.only and patch.name not in args.only:
            continue
        t1 = time.time()
        t = run_copy(patch, fault_set / "episodes")
        if t["status"] != "ran":
            summary[patch.name] = {"status": t["status"], "detail": t["detail"]}
            continue
        res = t["result"]
        entry = ft_trial.compare(base, res)
        want, entry = _comparable(frozen["faults"][patch.name], entry)
        rows = per_run(base, res)
        detail = {"patch": patch.name, "hit_lines": patched_lines(patch),
                  "import_hits": res["import_hits"], "between_hits": res["between_hits"],
                  "runs": rows}
        (args.detail_dir / f"{patch.stem}.json").write_text(
            json.dumps(detail, indent=1, sort_keys=True) + "\n")
        summary[patch.name] = {"reproduces_frozen": entry == want,
                               **summarize(rows, res["import_hits"], base["checks"])}
        if entry != want:
            summary[patch.name]["frozen_mismatch"] = {
                k: [want.get(k), entry.get(k)] for k in set(want) | set(entry)
                if want.get(k) != entry.get(k)}
        print(patch.name, f"{time.time() - t1:.0f}s", summary[patch.name]["reproduces_frozen"],
              {k: summary[patch.name][k] for k in ("activated_runs", "changed_runs",
                                                   "harmful_runs")}, flush=True)
    res = {"round": args.round, "post_hoc": True, "fresh_model_calls": 0,
           "frozen_source": str(args.frozen.resolve().relative_to(ROOT)),
           "note": "per-run rerun; checks still run as monitors after each run",
           "clean_runs": len(base["runs"]), "faults": summary}
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"reproduced": sum(v.get("reproduces_frozen", False)
                                        for v in summary.values()), "of": len(summary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
