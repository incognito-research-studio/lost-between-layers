"""Score a fault-injection run against the sealed manifest (predefined rules).

    uv run python tools/ft_score.py --results results/fault_trial_v2.json \
        --manifest path/to/manifest.json --out results/fault_trial_v2_scores.json

Protocol: data/fault_trial/PROTOCOL.md. Fixed before any fault was written. The manifest is
opened only here, after tools/ft_trial.py has written its results, and only if its SHA-256
equals the one the trial recorded. Manifest format (JSON):

  {"faults":    {"<patch file>": {"family": FAMILY, "layer": LAYER,
                                  "intended_effect": "<one or two sentences>"}},
   "negatives": {"<case dir>":   {"kind": NEGATIVE_KIND, "description": "..."}}}

Unit of analysis: the fault (the patch), not runs or variants of one patch. Rates are over
effective faults; broken, rejected and inert patches are reported and excluded. Per
configuration (solver, generic, full) and fault family: detection rate with a Wilson 95%
interval; detection before execution; faults with silent-violation runs; localization
accuracy among detected faults (the manifest's layer is among the layers the new firings
point to; a fault in `pipeline`, the integration code, is localized when any layer is named);
false alarms on negative cases; and an exact McNemar test between configurations on the
same effective faults. A fault family with fewer than three effective faults is reported
without an interval. The intended-effect check (does the observed change match what the
manifest says the patch does) is listed per fault for a reader; it is not automated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb, sqrt
from pathlib import Path

FAMILIES = ("precedence_supersession", "unit_calendar", "dropped_week_entity",
            "uncertainty_lost", "replay_observations", "audit_restart", "stale_approval",
            "other")
NEGATIVE_KINDS = ("authorized_master_override", "deliberate_version_rejection",
                  "valid_late_correction", "other")
LAYERS = ("L1", "L2", "L3", "L4", "L5", "L6", "registry", "pipeline", "types")
CONFIGS = ("solver", "generic", "full")


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def localized(entry: dict, layer: str) -> bool:
    named = set(entry["layers"])
    if layer == "pipeline":
        return bool(named)
    return layer in named


def score(results: dict, manifest: dict) -> dict:
    faults = results["faults"]
    meta = manifest["faults"]
    missing = sorted(set(faults) ^ set(meta))
    if missing:
        raise SystemExit(f"manifest and results disagree on these patches: {missing}")
    for name, m in meta.items():
        if m["family"] not in FAMILIES or m["layer"] not in LAYERS:
            raise SystemExit(f"{name}: family or layer outside the declared lists")
    eff = {n: f for n, f in faults.items() if f["status"] == "effective"}
    out: dict = {"faults_total": len(faults),
                 "status": {s: sum(f["status"] == s for f in faults.values())
                            for s in ("effective", "inert", "broken", "rejected")},
                 "configs": {}, "by_family": {}, "mcnemar": {}, "per_fault": {}}
    for cfg in CONFIGS:
        det = [n for n, f in eff.items() if f[cfg]["detected"]]
        pre = [n for n, f in eff.items() if f[cfg]["detected_pre_execution"]]
        silent = [n for n, f in eff.items() if f[cfg]["silent_violation_runs"]]
        loc = [n for n in det if localized(eff[n][cfg], meta[n]["layer"])]
        out["configs"][cfg] = {
            "effective": len(eff), "detected": len(det), "rate": len(det) / len(eff)
            if eff else None, "wilson95": wilson(len(det), len(eff)),
            "detected_pre_execution": len(pre), "pre_rate_wilson95": wilson(len(pre), len(eff)),
            "faults_with_silent_violations": len(silent),
            "localized": len(loc), "localization_rate": len(loc) / len(det) if det else None}
    for fam in FAMILIES:
        names = [n for n in eff if meta[n]["family"] == fam]
        if not names:
            continue
        row = {"effective": len(names)}
        for cfg in CONFIGS:
            k = sum(eff[n][cfg]["detected"] for n in names)
            row[cfg] = {"detected": k, "wilson95": wilson(k, len(names)) if len(names) >= 3
                        else None}
        out["by_family"][fam] = row
    for a, b in (("solver", "generic"), ("generic", "full"), ("solver", "full")):
        only_b = sum(eff[n][b]["detected"] and not eff[n][a]["detected"] for n in eff)
        only_a = sum(eff[n][a]["detected"] and not eff[n][b]["detected"] for n in eff)
        out["mcnemar"][f"{a}_vs_{b}"] = {"only_" + a: only_a, "only_" + b: only_b,
                                         "p_exact": mcnemar_exact(only_a, only_b)}
    negs = {k: v for k, v in results.get("negatives", {}).items() if not k.startswith("_")}
    nmeta = manifest.get("negatives", {})
    out["negatives"] = {"n": len(negs), **{cfg: {
        "false_alarms": sum(v[cfg]["false_alarm"] for v in negs.values()),
        "wilson95": wilson(sum(v[cfg]["false_alarm"] for v in negs.values()), len(negs))}
        for cfg in CONFIGS}, "per_case": {k: {"kind": nmeta.get(k, {}).get("kind"),
                                              **{c: v[c] for c in CONFIGS}}
                                          for k, v in negs.items()}}
    for n, f in faults.items():
        out["per_fault"][n] = {**meta[n], "status": f["status"],
                               **({c: {"detected": f[c]["detected"], "checks": f[c]["checks"],
                                       "layers": f[c]["layers"],
                                       "pre": f[c]["detected_pre_execution"],
                                       "silent_violation_runs": f[c]["silent_violation_runs"]}
                                   for c in CONFIGS} if f["status"] == "effective" else
                                  {"detail": f.get("detail", "")[:300]}),
                               "runs_changed": f.get("runs_changed"),
                               "violations": f.get("violations"),
                               "releases": f.get("releases")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    results = json.loads(args.results.read_text())
    sealed = results.get("manifest_sha256")
    actual = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if sealed != actual:
        raise SystemExit(f"manifest SHA-256 {actual} differs from the sealed {sealed}")
    out = score(results, json.loads(args.manifest.read_text()))
    out["manifest_sha256"] = actual
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["configs"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
