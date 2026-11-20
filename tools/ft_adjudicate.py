"""Adjudication of round 2 of the fault trial (13 Nov 2026).

    python tools/ft_adjudicate.py   # after tools/ft_perrun.py

Merges, per effective fault: the sealed manifest (intended effect, family, layer); the frozen
scores (results/fault_trial_r2_scores.json); the post-hoc per-run rerun
(results/fault_trial_r2_perrun.json and results/ft_perrun_r2/); and the reader's verdicts
(data/fault_trial/round2/ADJUDICATION_READER.json, written by the architecture role after
unblinding). Deterministic columns are computed here; the reader supplies only `matches`,
`observed` and `localization`, and this tool refuses verdicts that contradict the data:
  - `missed` exactly when the frozen full configuration did not detect the fault;
  - `matches` only if the patched lines executed and every episode the manifest names changed;
  - `cause`, `diffuse` and `downstream` only where a full check fired on a run the manifest
    names, `other_run` only where none did.
Writes results/fault_trial_r2_adjudication.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SET2 = ROOT / "data/fault_trial/round2/set-2"


def main() -> int:
    manifest = json.loads((SET2 / "manifest.json").read_text())
    scores = json.loads((ROOT / "results/fault_trial_r2_scores.json").read_text())["per_fault"]
    perrun = json.loads((ROOT / "results/fault_trial_r2_perrun.json").read_text())["faults"]
    reader = json.loads((ROOT / "data/fault_trial/round2/ADJUDICATION_READER.json").read_text())
    rows, problems = {}, []
    for name in sorted(perrun):
        mf, sc, pr = manifest["faults"][name], scores[name], perrun[name]
        rd = reader["faults"][name]
        detail = json.loads((ROOT / "results/ft_perrun_r2" / f"{name[:-6]}.json").read_text())
        runs = detail["runs"]
        named = [n for n in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+){2,}", mf["intended_effect"])
                 if (SET2 / "episodes" / n).is_dir()]
        named_changed = {n: bool(runs.get(f"episodes/{n}", {}).get("changed")) for n in named}
        named_flagged = any(runs.get(f"episodes/{n}", {}).get("new", {}).get("full")
                            for n in named)
        detected = sc["full"]["detected"]
        if (rd["localization"] == "missed") == detected:
            problems.append(f"{name}: localization {rd['localization']} but detected={detected}")
        if rd["matches"] and not (pr["activated_runs"] > 0 and all(named_changed.values())):
            problems.append(f"{name}: matches, but activated={pr['activated_runs']}, "
                            f"named changed={named_changed}")
        if rd["localization"] in ("cause", "diffuse", "downstream") and named and not named_flagged:
            problems.append(f"{name}: {rd['localization']} but no full check fired on a named run")
        if rd["localization"] == "other_run" and named_flagged:
            problems.append(f"{name}: other_run but a full check fired on a named run")
        rows[name] = {
            "family": mf["family"], "layer": mf["layer"],
            "intended_effect": mf["intended_effect"],
            "observed": rd["observed"], "matches": rd["matches"],
            "localization": rd["localization"],
            "activated_runs": pr["activated_runs"], "changed_runs": pr["changed_runs"],
            "named_episodes_changed": named_changed,
            "detected_full": detected, "detected_generic": sc["generic"]["detected"],
            "checks_full": sc["full"]["checks"],
            "detected_fixed": pr["full"]["detected_fixed"],
            "detected_generator": pr["full"]["detected_generator"],
            "harmful_runs": pr["harmful_runs"],
            "harmful_runs_flagged_full": pr["full"]["harmful_runs_flagged"],
            "harmful_runs_flagged_pre_full": pr["full"]["harmful_runs_flagged_pre"],
            "harmful_runs_flagged_generic": pr["generic"]["harmful_runs_flagged"]}
    if problems:
        raise SystemExit("reader verdicts contradict the data:\n  " + "\n  ".join(problems))
    eff = list(rows.values())
    harmful = [r for r in eff if r["harmful_runs"]]
    det = [r for r in eff if r["detected_full"]]
    totals = {
        "effective": len(eff),
        "activated": sum(r["activated_runs"] > 0 for r in eff),
        "matches_intended": sum(r["matches"] for r in eff),
        "detected_full": len(det),
        "localization": {k: sum(r["localization"] == k for r in eff)
                         for k in ("cause", "diffuse", "other_run", "downstream", "missed")},
        "detected_full_fixed_worlds": sum(r["detected_fixed"] for r in eff),
        "detected_full_generator_episodes": sum(r["detected_generator"] for r in eff),
        "detected_full_only_generator": sum(r["detected_generator"] and not r["detected_fixed"]
                                            for r in eff),
        "harmful_faults": len(harmful),
        "harmful_runs": sum(r["harmful_runs"] for r in eff),
        "harmful_faults_flagged_on_harmful_run_full": sum(r["harmful_runs_flagged_full"] > 0
                                                          for r in harmful),
        "harmful_runs_flagged_full": sum(r["harmful_runs_flagged_full"] for r in eff),
        "harmful_runs_flagged_pre_full": sum(r["harmful_runs_flagged_pre_full"] for r in eff),
        "harmful_runs_flagged_generic": sum(r["harmful_runs_flagged_generic"] for r in eff),
        "harmful_faults_detected_elsewhere_only": sum(
            r["detected_full"] and r["harmful_runs_flagged_full"] == 0 for r in harmful)}
    out = {"round": 2, "post_hoc": True, "reader": reader["_reader"],
           "method": reader["_method"],
           "note": "the checks ran as monitors after each run; no release was held",
           "totals": totals, "faults": rows}
    path = ROOT / "results/fault_trial_r2_adjudication.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(totals, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
