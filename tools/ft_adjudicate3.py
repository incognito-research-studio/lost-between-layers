"""Adjudication of round 3 of the fault trial (13 Nov 2026).

    python tools/ft_adjudicate3.py   # after tools/ft_round3.py and tools/ft_perrun.py --round 3

Merges, per effective fault, the sealed manifest, the frozen round-3 results
(results/fault_trial_r3.json: four configurations, harm, containment, per-run firings), the
post-hoc per-run rerun (results/fault_trial_r3_perrun.json, results/ft_perrun_r3/) and the
reader's verdicts (data/fault_trial/round3/ADJUDICATION_READER.json). As in round 2 it refuses
verdicts that contradict the data, here for the repaired configuration:
  - `missed` exactly when the repaired configuration did not detect the fault;
  - `matches` only if the patched lines executed and every episode the fault exercises changed;
  - `cause` or `diffuse` only where a repaired check fired on an episode the fault exercises,
    `other_run` only where none did.
Writes results/fault_trial_r3_adjudication.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SET3 = ROOT / "data/fault_trial/round3/set-3"


def main() -> int:
    manifest = json.loads((SET3 / "manifest.json").read_text())
    r3 = json.loads((ROOT / "results/fault_trial_r3.json").read_text())
    perrun = json.loads((ROOT / "results/fault_trial_r3_perrun.json").read_text())["faults"]
    reader = json.loads((ROOT / "data/fault_trial/round3/ADJUDICATION_READER.json").read_text())
    rows, problems = {}, []
    for name, f in sorted(r3["faults"].items()):
        if f["status"] != "effective":
            continue
        mf, pr, rd = manifest["faults"][name], perrun[name], reader["faults"][name]
        runs = json.loads((ROOT / "results/ft_perrun_r3" / f"{name[:-6]}.json").read_text())["runs"]
        ex = [e for e, v in manifest["episodes"].items() if name in v["exercises"]]
        changed = {e: bool(runs.get(f"episodes/{e}", {}).get("changed")) for e in ex}
        fired_on_ex = any(f["per_run"].get(f"episodes/{e}", {}).get("fired_new")
                          and set(f["per_run"][f"episodes/{e}"]["fired_new"])
                          & set(f["repaired"]["checks"]) for e in ex)
        det = f["repaired"]["detected"]
        if (rd["localization"] == "missed") == det:
            problems.append(f"{name}: localization {rd['localization']} but detected={det}")
        if rd["matches"] and not (pr["activated_runs"] > 0 and all(changed.values())):
            problems.append(f"{name}: matches, but activated={pr['activated_runs']}, {changed}")
        if rd["localization"] in ("cause", "diffuse") and not fired_on_ex:
            problems.append(f"{name}: {rd['localization']} but no repaired check fired on its episodes")
        if rd["localization"] == "other_run" and fired_on_ex:
            problems.append(f"{name}: other_run but a repaired check fired on its episodes")
        rows[name] = {"family": mf["family"], "layer": mf["layer"],
                      "intended_effect": mf["intended_effect"], "observed": rd["observed"],
                      "matches": rd["matches"], "localization": rd["localization"],
                      "activated_runs": pr["activated_runs"], "changed_runs": pr["changed_runs"],
                      "exercised_episodes_changed": changed,
                      **{f"detected_{c}": f[c]["detected"] for c in ("solver", "generic", "full",
                                                                     "repaired")},
                      "checks_full": f["full"]["checks"], "checks_repaired": f["repaired"]["checks"],
                      "detected_pre_repaired": f["repaired"]["detected_pre_execution"],
                      "detected_fixed": f["fixed_or_generated"]["detected_fixed"],
                      "detected_generated": f["fixed_or_generated"]["detected_generated"],
                      "harmful_runs": f["harm"]["harmful_runs"],
                      "harmful_runs_flagged_full": f["full"]["harmful_runs_flagged"],
                      "harmful_runs_flagged_repaired": f["repaired"]["harmful_runs_flagged"],
                      "harmful_runs_contained": f["harm"]["harmful_runs_contained"],
                      "holds": f["harm"]["holds"]}
    if problems:
        raise SystemExit("reader verdicts contradict the data:\n  " + "\n  ".join(problems))
    eff = list(rows.values())
    harmful = [r for r in eff if r["harmful_runs"]]
    b = sum(r["detected_repaired"] and not r["detected_full"] for r in eff)
    c = sum(r["detected_full"] and not r["detected_repaired"] for r in eff)
    from math import comb
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
    totals = {"effective": len(eff), "activated": sum(r["activated_runs"] > 0 for r in eff),
              "matches_intended": sum(r["matches"] for r in eff),
              **{f"detected_{k}": sum(r[f"detected_{k}"] for r in eff)
                 for k in ("solver", "generic", "full", "repaired")},
              "repaired_only": b, "full_only": c, "mcnemar_exact_p": p,
              "detected_pre_repaired": sum(r["detected_pre_repaired"] for r in eff),
              "localization": {k: sum(r["localization"] == k for r in eff)
                               for k in ("cause", "diffuse", "other_run", "downstream", "missed")},
              "detected_repaired_fixed_worlds": sum(r["detected_repaired"] and r["detected_fixed"]
                                                    for r in eff),
              "detected_repaired_only_generated": sum(r["detected_repaired"]
                                                      and not r["detected_fixed"] for r in eff),
              "harmful_faults": len(harmful), "harmful_runs": sum(r["harmful_runs"] for r in eff),
              "harmful_runs_flagged_full": sum(r["harmful_runs_flagged_full"] for r in eff),
              "harmful_runs_flagged_repaired": sum(r["harmful_runs_flagged_repaired"] for r in eff),
              "harmful_runs_contained": sum(r["harmful_runs_contained"] for r in eff),
              "harmful_faults_fully_contained": sum(r["harmful_runs_contained"] == r["harmful_runs"]
                                                    for r in harmful)}
    out = {"round": 3, "post_hoc_reading": True, "reader": reader["_reader"],
           "method": reader["_method"], "totals": totals, "faults": rows}
    (ROOT / "results/fault_trial_r3_adjudication.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(totals, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
