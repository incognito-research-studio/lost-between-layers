"""Round 3, post hoc (15 Nov 2026): detection by the repaired configuration with check families
removed, recomputed from the stored per-run firings of results/fault_trial_r3.json.

    uv run python tools/ft_ablation.py [--out results/fault_trial_r3_ablation.json]

No rerun is needed for detection: a configuration detects a fault when one of its checks newly
fires on some run (`ft_round3.compare`), and `per_run[...]["fired_new"]` lists every check that
newly fired on every run where anything changed. Removing checks therefore only removes names
from those lists. Containment is not recomputed: a hold changes the rest of its run, so it would
need gated reruns. Before reporting, the script checks that the full and repaired sets
reproduce the frozen per-fault detection and harmful-run flags.

Families (tables/repaired-checks.tex):
  second_impl   span_cover, commit_precedence, l1_reference: second implementations of L1
                parsing and L3 policy, written by the same author as the pipeline
  raw_reply     the checks that read the interpreter's raw reply, which the harness records
                outside the pipeline: the three above and blocked_read
  repaired_new  all eleven checks the repaired configuration adds (removing them gives full)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_overlap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RANK = {"solver": 0, "generic": 1, "full": 2, "repaired": 3}
SECOND_IMPL = ("span_cover", "commit_precedence", "l1_reference")
RAW_REPLY = SECOND_IMPL + ("blocked_read",)


def _names(checks: dict, cfg: str) -> set:
    return {n for n, c in checks.items() if RANK[c["config"]] <= RANK[cfg]}


def score(fault: dict, names: set) -> dict:
    flagged = {k for k, v in fault["per_run"].items() if set(v["fired_new"]) & names}
    harmful = {k for k, v in fault["per_run"].items() if v["violations"][1] > v["violations"][0]}
    return {"detected": bool(flagged), "harmful_runs_flagged": len(flagged & harmful),
            "harmful_runs": len(harmful)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trial", type=Path, default=ROOT / "results/fault_trial_r3.json")
    ap.add_argument("--overlap", type=Path, default=ROOT / "results/fault_trial_overlap.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/fault_trial_r3_ablation.json")
    args = ap.parse_args(argv)
    d = json.loads(args.trial.read_text())
    patches = json.loads(args.overlap.read_text())["round3_vs_rounds1_2"]["patches"]
    cls = {p + ".patch": v["class"] for p, v in patches.items()}
    checks = d["checks"]
    faults = {n: f for n, f in d["faults"].items() if f["status"] == "effective"}
    full, rep = _names(checks, "full"), _names(checks, "repaired")
    new = sorted(rep - full)
    for n, f in faults.items():
        for cfg, names in (("full", full), ("repaired", rep)):
            s = score(f, names)
            assert s["detected"] == f[cfg]["detected"], (n, cfg)
            assert s["harmful_runs_flagged"] == f[cfg]["harmful_runs_flagged"], (n, cfg)
    variants = {"repaired": rep,
                "without_second_impl": rep - set(SECOND_IMPL),
                "without_raw_reply": rep - set(RAW_REPLY),
                "full": full}
    variants.update({f"without_{c}": rep - {c} for c in new})
    variants.update({f"full_plus_{c}": full | {c} for c in new})
    classes = ("identical", "same_line", "none")
    out = {}
    for v, names in variants.items():
        per = {n: score(f, names) for n, f in faults.items()}
        row = {"detected": sum(s["detected"] for s in per.values()),
               "harmful_runs_flagged": sum(s["harmful_runs_flagged"] for s in per.values()),
               "by_class": {c: sum(per[n]["detected"] for n in faults if cls[n] == c)
                            for c in classes},
               "faults": sorted(n[:-6] for n, s in per.items() if s["detected"])}
        b = sum(per[n]["detected"] and not score(f, full)["detected"] for n, f in faults.items())
        c = sum(score(f, full)["detected"] and not per[n]["detected"] for n, f in faults.items())
        row["vs_full"] = {"only_this": b, "only_full": c, "mcnemar_p": ft_overlap.mcnemar(b, c)}
        out[v] = row
    sole = {}
    for n, f in faults.items():
        fired = sorted({x for v in f["per_run"].values() for x in v["fired_new"]} & rep)
        sole[n[:-6]] = {"class": cls[n], "checks": fired,
                        "only_second_impl": bool(fired) and set(fired) <= set(SECOND_IMPL),
                        "only_raw_reply": bool(fired) and set(fired) <= set(RAW_REPLY)}
    res = {"post_hoc": True, "fresh_model_calls": 0, "faults": len(faults),
           "class_sizes": {c: sum(cls[n] == c for n in faults) for c in classes},
           "families": {"second_impl": list(SECOND_IMPL), "raw_reply": list(RAW_REPLY),
                        "repaired_new": new},
           "variants": out, "per_fault": sole}
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for v in ("repaired", "without_second_impl", "without_raw_reply", "full"):
        r = out[v]
        print(f"{v:22s} {r['detected']:2d}/26 by class {r['by_class']} harmful flagged "
              f"{r['harmful_runs_flagged']} vs full {r['vs_full']}")
    for c in new:
        a, b = out[f"without_{c}"], out[f"full_plus_{c}"]
        print(f"  -{c:18s} {a['detected']:2d} none {a['by_class']['none']}   "
              f"full+{c}: {b['detected']:2d} none {b['by_class']['none']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
