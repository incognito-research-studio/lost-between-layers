"""Case-clustered uncertainty, paired arm differences and failure types for E1 v2 (R2, 3 Nov 2026).

    uv run python tools/e1v2_clustered.py

A reanalysis of the stored scores; no model is called and no stored file is changed. The
frozen scorer gives 95% Wilson intervals over case-repeats, which treat the two repeats of a
case as independent. Here each case is a cluster (both repeats, and every arm of a
case-repeat, stay together):

  intervals    Wilson intervals on the effective sample size n / deff, where the design
               effect deff is the cluster-robust variance of the rate divided by its binomial
               variance, bounded below by 1 and above by the largest cluster (2). A rate of 0
               or 1 has no estimable deff and takes the bound 2, i.e. one unit per case.
               A case bootstrap (percentile, 10,000 draws) is stored beside it for comparison.
  differences  paired differences between arms over the same case-repeats, in percentage
               points, with a case bootstrap interval and the number of discordant case-repeats.
  failures     each change-notice case-repeat put in exactly one class, in this order: model
               abstention, contract rejection, error cycle, no change committed, wrong plant,
               wrong weeks, value outside the stated interval, covered.
  categories   abstentions, coverage and violations per category and model.

Inputs: results/e1v2/{rows.csv,posthoc_rows.csv,replies.jsonl,summary.json} and
results/e1v2_diagnostics/case_repeats.csv. Output: results/e1v2_clustered.json, and with
--tables the two LaTeX tables for the paper. Everything here is post hoc.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dsoi.experiments import e1v2_score as sc
from dsoi.experiments.e1v2_notices import HORIZON, NOMINAL, load
from dsoi.experiments.e1v2_run import load_replies
from dsoi.experiments.e1v2_world import T_DECIDE, state_of
from dsoi.sim.intervals import wilson

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen2.5:7b-instruct", "llama3.1:8b", "phi4:latest")
NAMES = {
    "qwen2.5:7b-instruct": "Qwen2.5 7B",
    "llama3.1:8b": "Llama 3.1 8B",
    "phi4:latest": "Phi-4 14B",
}
SEED, DRAWS = 20261103, 10_000
FAILURES = (
    ("model", "Model abstention"),
    ("contract", "Contract rejection"),
    ("error", "Error cycle"),
    ("nochange", "Committed, not read"),
    ("entity", "Wrong plant"),
    ("scope", "Wrong weeks"),
    ("value", "Value outside interval"),
    ("covered", "Covered"),
)
QUIET = ("other_entity", "out_of_scope", "vague")


def _b(v: str) -> bool | None:
    return None if v == "" else v == "True"


def clustered(units: list[tuple[str, bool]], rng: np.random.Generator) -> dict | None:
    """Rate of a list of (case, outcome) units, with intervals that cluster by case."""
    if not units:
        return None
    groups: dict[str, list[bool]] = defaultdict(list)
    for case, v in units:
        groups[case].append(v)
    k_g = np.array([sum(v) for v in groups.values()], float)
    n_g = np.array([len(v) for v in groups.values()], float)
    k, n, G, m = k_g.sum(), n_g.sum(), len(groups), n_g.max()
    p = k / n
    if 0 < p < 1 and G > 1:
        var_c = G / (G - 1) * float(((k_g - p * n_g) ** 2).sum()) / n**2
        deff = min(max(var_c / (p * (1 - p) / n), 1.0), m)
    else:
        deff = m
    n_eff = n / deff
    lo, hi = wilson(p * n_eff, n_eff)
    idx = rng.integers(0, G, size=(DRAWS, G))
    boot = k_g[idx].sum(1) / n_g[idx].sum(1)
    return {
        "rate": p,
        "k": int(k),
        "n": int(n),
        "cases": G,
        "deff": round(deff, 4),
        "n_eff": round(n_eff, 2),
        "ci95": [round(lo, 4), round(hi, 4)],
        "ci95_wilson_iid": [round(x, 4) for x in wilson(k, n)],
        "ci95_bootstrap": [
            round(float(np.quantile(boot, 0.025)), 4),
            round(float(np.quantile(boot, 0.975)), 4),
        ],
    }


def paired(a: dict, b: dict, keys: list[tuple], rng: np.random.Generator) -> dict:
    """Mean of a - b over the same case-repeats, bootstrapped by case."""
    groups: dict[str, list[float]] = defaultdict(list)
    for key in keys:
        groups[key[1]].append(float(a[key]) - float(b[key]))
    s = np.array([sum(v) for v in groups.values()])
    n_g = np.array([len(v) for v in groups.values()], float)
    idx = rng.integers(0, len(s), size=(DRAWS, len(s)))
    boot = s[idx].sum(1) / n_g[idx].sum(1)
    return {
        "diff": float(s.sum() / n_g.sum()),
        "n": int(n_g.sum()),
        "cases": len(s),
        "discordant": sum(a[k] != b[k] for k in keys),
        "ci95_bootstrap": [
            round(float(np.quantile(boot, 0.025)), 4),
            round(float(np.quantile(boot, 0.975)), 4),
        ],
    }


def committed_changes(cases: dict, replies: dict, nu: dict, key: tuple) -> bool:
    """Whether the stated arm's state at decision time differs from the nominal state."""
    model, case_id, repeat = key
    _, p, _, _, _ = sc._run_arm(cases[case_id], "stated", replies[key], nu[model])
    state = state_of(p, T_DECIDE)
    return any(v != (NOMINAL, None, None) for ws in state.values() for v in ws.values()) or any(
        w not in ws for ws in state.values() for w in HORIZON
    )


def analyse(results: Path, diagnostics: Path) -> dict:
    rng = np.random.default_rng(SEED)
    rows = [
        r for r in csv.DictReader(open(results / "rows.csv", newline="")) if r["split"] == "test"
    ]
    post = {
        (r["model"], r["case_id"], int(r["repeat"])): r
        for r in csv.DictReader(open(results / "posthoc_rows.csv", newline=""))
    }
    src = {
        (r["model"], r["case_id"], int(r["repeat"])): r["sources"]
        for r in csv.DictReader(open(diagnostics / "case_repeats.csv", newline=""))
    }
    cases = {c.id: c for c in load(ROOT / "data" / "e1v2" / "cases.jsonl")}
    replies: dict[tuple, dict] = defaultdict(dict)
    nu: dict[str, str] = {}
    for (model, case_id, repeat, message_id), rep in load_replies(
        results / "replies.jsonl"
    ).items():
        replies[(model, case_id, repeat)][message_id] = rep
        nu[model] = rep["nu_i"]

    out: dict = {
        "post_hoc": True,
        "seed": SEED,
        "draws": DRAWS,
        "manifest": json.loads((results / "summary.json").read_text())["manifest"],
        "note": " ".join(__doc__.split("\n\n")[1:4]).replace("\n", " "),
        "models": {},
        "differences": {},
        "failures": {},
        "categories": {},
    }
    for model in MODELS:
        rs = [r for r in rows if r["model"] == model]
        key = {id(r): (model, r["case_id"], int(r["repeat"])) for r in rs}
        change = [r for r in rs if r["expect"] == "change"]
        quiet = [r for r in rs if r["expect"] != "change"]

        def u(sub, col, cond=lambda r: True):
            return [(r["case_id"], _b(r[col])) for r in sub if cond(r) and _b(r[col]) is not None]

        # Derived per-row outcomes for the two post hoc arms (as tools/e1v2_posthoc.py does).
        arm_v = {
            "point": {},
            "stated": {},
            "recheck": {},
            "calibrated": {},
            "cal_corrected": {},
            "stated_hold": {},
        }
        arm_r, arm_e = {a: {} for a in arm_v}, {a: {} for a in arm_v}
        for r in change:
            k = key[id(r)]
            for a in ("point", "stated", "recheck", "calibrated"):
                arm_v[a][k] = _b(r[f"violation_{a}"])
                arm_r[a][k] = _b(r[f"released_{a}"])
                arm_e[a][k] = _b(r[f"error_{a}"])
            ph = post[k]
            arm_v["cal_corrected"][k], arm_r["cal_corrected"][k], arm_e["cal_corrected"][k] = (
                _b(ph["violation"]),
                _b(ph["released"]),
                _b(ph["error"]),
            )
            held = "abstained" in r["outcomes"].split(",")
            arm_v["stated_hold"][k] = arm_v["stated"][k] and not held
            arm_r["stated_hold"][k] = arm_r["stated"][k] and not held
            arm_e["stated_hold"][k] = arm_e["stated"][k] and not held
        keys = [key[id(r)] for r in change]

        m = {
            "abstained": clustered(u(change, "abstained"), rng),
            "scope_exact": clustered(
                u(change, "scope_exact", lambda r: r["category"] != "other_entity"), rng
            ),
            "entity_ok": clustered(u(rs, "entity_ok"), rng),
            "quiet_ok": clustered(u(quiet, "quiet_ok"), rng),
            "coverage_stated": clustered(u(change, "covered_stated"), rng),
            "coverage_calibrated": clustered(u(change, "covered_calibrated"), rng),
            "coverage_cal_corrected": clustered(
                [(k[1], _b(post[k]["covered"])) for k in keys], rng
            ),
            "held_quiet": clustered(
                [(r["case_id"], "abstained" in r["outcomes"].split(",")) for r in quiet], rng
            ),
        }
        for a in arm_v:
            rel = [k for k in keys if arm_r[a][k]]
            m[a] = {
                "error": clustered([(k[1], arm_e[a][k]) for k in keys], rng),
                "release": clustered([(k[1], arm_r[a][k]) for k in keys], rng),
                "violation": clustered([(k[1], arm_v[a][k]) for k in keys], rng),
                "violation_given_release": clustered([(k[1], arm_v[a][k]) for k in rel], rng),
            }
        out["models"][model] = m
        out["differences"][model] = {
            f"violation: {a} - {b}": paired(arm_v[a], arm_v[b], keys, rng)
            for a, b in (
                ("stated", "point"),
                ("stated_hold", "stated"),
                ("cal_corrected", "stated"),
                ("recheck", "stated"),
            )
        }
        out["differences"][model]["release: stated_hold - stated"] = paired(
            arm_r["stated_hold"], arm_r["stated"], keys, rng
        )

        fail = Counter()
        for r in change:
            k = key[id(r)]
            s = src.get(k, "")
            if _b(r["abstained"]):
                fail["model" if "model" in s else "contract"] += 1
            elif _b(r["error_stated"]):
                fail["error"] += 1
            elif not committed_changes(cases, replies, nu, k):
                # The caption says why: every such reply only asserts (DEVIATIONS.md, item 6).
                ops = {
                    o["op"]
                    for rep in replies[k].values()
                    for o in json.loads(rep["text"]).get("ops", [])
                }
                assert ops == {"assert"}, (k, ops)
                fail["nochange"] += 1
            elif not _b(r["entity_ok"]):
                fail["entity"] += 1
            elif not _b(r["scope_exact"]):
                fail["scope"] += 1
            elif not _b(r["covered_stated"]):
                fail["value"] += 1
            else:
                fail["covered"] += 1
        assert sum(fail.values()) == len(change)
        out["failures"][model] = {c: fail[c] for c, _ in FAILURES}
        # Error cycles counted under an abstention because another message was abstained on.
        out["failures"][model]["error_in_abstention"] = sum(
            _b(r["abstained"]) and _b(r["error_stated"]) for r in change
        )

        cats = {}
        for cat in sorted({r["category"] for r in rs}):
            cr = [r for r in rs if r["category"] == cat]
            cats[cat] = {
                "n": len(cr),
                "abstained": sum(_b(r["abstained"]) for r in cr),
                "covered_or_kept": sum(
                    _b(r["quiet_ok"] if cat in QUIET else r["covered_stated"]) for r in cr
                ),
                "violation_stated": sum(_b(r["violation_stated"]) for r in cr),
            }
        out["categories"][model] = cats
    return out


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def failure_table(res: dict) -> str:
    n = sum(res["failures"][MODELS[0]][c] for c, _ in FAILURES)
    overlap = [
        (NAMES[m], res["failures"][m]["error_in_abstention"], res["failures"][m]["error"])
        for m in MODELS
        if res["failures"][m]["error_in_abstention"]
    ]
    note = "".join(
        f" Of {name}'s {k + e} error cycles, {k} occurred in a case-repeat with an abstention "
        f"and is counted there, so the table shows {e}."
        for name, k, e in overlap
    )
    lines = [
        "%% Generated by replication-package/tools/e1v2_clustered.py "
        "(post hoc); do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{\cdraft{Where each change-notice case-repeat of the live study "
        r"ended (test part, "
        f"{n} per model), counted once, in the order listed: a case-repeat is a model abstention "
        r"if the model declined any of its messages, a contract rejection if only L1's contract "
        r"did; an error cycle executed nothing. The remaining classes are answered readings, "
        r"checked in turn for any change to the capacity the planner reads, the plants, the "
        r"weeks of plant X, and whether the true "
        r"capacity lies in the stated intervals. Every case-repeat in ``committed, not read'' is "
        r"a reading whose operations all assert a new capacity, which the authenticated capacity "
        r"master outranked when the planner read the state (Section~\ref{sec:e1})."
        f"{note} Computed post hoc from the stored scores.}}}}"
        r"\label{tab:e1fail}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        "Outcome & " + " & ".join(NAMES[m] for m in MODELS) + r"\\\midrule",
    ]
    for c, label in FAILURES:
        lines.append(f"{label} & " + " & ".join(str(res["failures"][m][c]) for m in MODELS) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


CATNAMES = {
    "absolute": "absolute figure",
    "atleast_pct": "``at least''",
    "conflict": "conflicting sources",
    "correction": "in-message correction",
    "exact_pct": "exact percentage",
    "hedged_pct": "hedged (``about'')",
    "injection": "embedded instruction",
    "per_day": "per-day figure",
    "scope_relative": "relative weeks",
    "two_entities": "two plants",
    "upto_pct": "``up to''",
    "other_entity": "other plant only",
    "out_of_scope": "out of scope",
    "vague": "vague",
}


def category_table(res: dict) -> str:
    cats = list(res["categories"][MODELS[0]])
    order = [c for c in cats if c not in QUIET] + [c for c in QUIET if c in cats]
    lines = [
        "%% Generated by replication-package/tools/e1v2_clustered.py "
        "(post hoc); do not edit by hand.",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{\cdraft{Live study per category (test part; 8 cases, 16 case-repeats per model "
        r"and category). Each cell gives abstained / covered / violated: case-repeats with an "
        r"abstention, case-repeats whose stated intervals cover the true capacity (for the last "
        r"three, quiet, categories: plant X left unchanged), and case-repeats whose plan under "
        r"the box-robust gate on stated intervals violates the true capacity.}}\label{tab:e1cat}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        "Category & " + " & ".join(NAMES[m] for m in MODELS) + r"\\\midrule",
    ]
    for c in order:
        if c == QUIET[0]:
            lines.append(r"\midrule")
        cells = []
        for m in MODELS:
            x = res["categories"][m][c]
            cells.append(f"{x['abstained']} / {x['covered_or_kept']} / {x['violation_stated']}")
        lines.append(f"{CATNAMES.get(c, c)} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "e1v2")
    ap.add_argument("--diagnostics", type=Path, default=ROOT / "results" / "e1v2_diagnostics")
    ap.add_argument("--tables", type=Path, help="directory for e1-failures.tex, e1-categories.tex")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e1v2_clustered.json")
    args = ap.parse_args(argv)
    res = analyse(args.results, args.diagnostics)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    if args.tables:
        (args.tables / "e1-failures.tex").write_text(failure_table(res))
        (args.tables / "e1-categories.tex").write_text(category_table(res))
    for m in MODELS:
        print(NAMES[m], res["failures"][m])
        for d, v in res["differences"][m].items():
            print(
                f"  {d}: {100 * v['diff']:+.1f} pp {v['ci95_bootstrap']} "
                f"discordant {v['discordant']}/{v['n']}"
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
