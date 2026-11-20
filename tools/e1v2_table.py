"""Write the E1 v2 results table from the stored scores, so no number is typed by hand.

    uv run python tools/e1v2_table.py --out e1.tex

Reads results/e1v2/summary.json and rows.csv (written by `dsoi-e1v2 score`) and
posthoc.json (written by tools/e1v2_posthoc.py). Part A gives the interpretation measures
per model, part B the release, violation and cost of each arm. Every column is a
pre-registered measure (data/e1v2/PROTOCOL.md) except those marked post hoc: the corrected
calibrated arm and the hold-on-abstention arm (data/e1v2/DEVIATIONS.md, items 2 and 4)
and the cost of released plans that are feasible for the true state. The
pre-registered cost also averages over violating plans, which produce more than the plant
can and so look cheaper than the true optimum; it is negative when most releases violate.

Intervals: since 3 Nov 2026 the table prints the case-clustered intervals of
results/e1v2_clustered.json (tools/e1v2_clustered.py), not the scorer's Wilson intervals,
which treat the two repeats of a case as independent. Every rate, k and n is checked
against the scorer's before it is printed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["point", "stated", "calibrated", "recheck"]
NAMES = {
    "qwen2.5:7b-instruct": "Qwen2.5 7B",
    "llama3.1:8b": "Llama 3.1 8B",
    "phi4:latest": "Phi-4 14B",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
}
ORDER = list(NAMES)


def clustered(r: dict | None, c: dict | None) -> dict | None:
    """The case-clustered interval for the scorer's rate r, after checking k and n agree."""
    if r is None:
        assert c is None
        return None
    assert (r["k"], r["n"]) == (c["k"], c["n"]), (r, c)
    return c


def rate(r: dict | None) -> str:
    if r is None:
        return "--"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.0f} ({100 * lo:.0f}--{100 * hi:.0f})"


def cost(v: float | None) -> str:
    return "--" if v is None else f"${v:.0f}$"


def feasible_cost(rows: list[dict], model: str, arm: str) -> tuple[float | None, int]:
    """Post hoc: mean excess over the true optimum among released, non-violating test plans."""
    vals = [
        float(r[f"excess_{arm}"])
        for r in rows
        if r["model"] == model
        and r["split"] == "test"
        and r["expect"] == "change"
        and r[f"released_{arm}"] == "True"
        and r[f"violation_{arm}"] == "False"
    ]
    return (float(np.mean(vals)) if vals else None), len(vals)


def table(summary: dict, rows: list[dict], posthoc: dict, cl: dict) -> str:
    models = sorted(summary["models"], key=lambda m: (ORDER + [m]).index(m))
    first = summary["models"][models[0]]
    for m in models:  # replace every interval by the clustered one
        s, c, ph = summary["models"][m], cl["models"][m], posthoc["arms"]
        for key in (
            "abstained",
            "scope_exact",
            "entity_ok",
            "quiet_ok",
            "coverage_stated",
            "coverage_calibrated",
        ):
            s[key] = clustered(s[key], c[key])
        ph["calibrated_pinned"][m]["coverage"] = clustered(
            ph["calibrated_pinned"][m]["coverage"], c["coverage_cal_corrected"]
        )
        for arm, src, cm in [(a, s, a) for a in ARMS] + [
            ("calibrated_pinned", ph["calibrated_pinned"], "cal_corrected"),
            ("stated_hold", ph["stated_hold"], "stated_hold"),
        ]:
            d = src[arm] if src is s else src[m]
            for key in ("error", "release", "violation", "violation_given_release"):
                d[key] = clustered(d[key], c[cm][key])
    a = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Model & Abstained & Scope exact & Entity correct & Quiet kept & Coverage, stated"
        r" & Coverage, cal.$^{\ddagger}$ & Coverage, cal.\ corr.$^{\dagger}$ & $q$\\\midrule",
    ]
    for m in models:
        s = summary["models"][m]
        a.append(
            f"{NAMES.get(m, m)} & {rate(s['abstained'])} & {rate(s['scope_exact'])} & "
            f"{rate(s['entity_ok'])} & {rate(s['quiet_ok'])} & "
            f"{rate(s['coverage_stated'])} & {rate(s['coverage_calibrated'])} & "
            f"{rate(posthoc['arms']['calibrated_pinned'][m]['coverage'])} & "
            f"{summary['margin_q'][m]:.0f}\\\\"
        )
    a += [r"\bottomrule", r"\end{tabular}"]
    b = [
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Model & Arm & Error cycle & Released & $\Pr[V]$ & $\Pr[V\mid R]$ & Cost when released"
        r" & Cost, feasible releases$^{\dagger}$\\\midrule",
    ]
    for i, m in enumerate(models):
        s = summary["models"][m]
        lines = []
        for arm in ARMS:
            fc, n = feasible_cost(rows, m, arm)
            lines.append(
                (
                    arm + (r"$^{\ddagger}$" if arm == "calibrated" else ""),
                    s[arm],
                    s[arm]["mean_excess_when_released"],
                    fc,
                    n,
                )
            )
        for at, arm, label in (
            (3, "calibrated_pinned", r"cal.\ corrected$^{\dagger}$"),
            (2, "stated_hold", r"stated, hold$^{\dagger}$"),
        ):
            ph = posthoc["arms"][arm][m]
            lines.insert(
                at,
                (
                    label,
                    ph,
                    ph["mean_excess_when_released"],
                    ph["mean_excess_when_released_and_feasible"],
                    ph["released_and_feasible"],
                ),
            )
        for j, (label, r, c, fc, n) in enumerate(lines):
            b.append(
                f"{NAMES.get(m, m) if j == 0 else ''} & {label} & {rate(r['error'])} & "
                f"{rate(r['release'])} & {rate(r['violation'])} & "
                f"{rate(r['violation_given_release'])} & {cost(c)} & "
                f"{cost(fc)} ($n={n}$)\\\\"
            )
        if i < len(models) - 1:
            b.append(r"\addlinespace")
    b += [r"\bottomrule", r"\end{tabular}"]
    caption = (
        r"\cdraft{Live-interpreter study (Section~\ref{sec:e1}), test part: "
        f"{first['change_rows']} change-notice and {first['quiet_rows']} quiet-notice "
        r"case-repeats per model. Rates in percent with approximate 95\% cluster-adjusted "
        r"intervals: Wilson intervals on the case-repeats divided by a design effect for the two "
        r"repeats of a case (Appendix~\ref{app:supp}). (a)~Interpretation: "
        r"abstention, scope and coverage over change notices, entity over all notices with "
        r"a defined truth, quiet over quiet notices; $q$ is the split-conformal margin fitted "
        f"on the calibration part at $\\alpha={summary['alpha']}$, and $q=1000$, the nominal "
        r"capacity, widens every interval to the whole domain. (b)~Decisions per arm over "
        r"change notices; an error cycle stops before execution and releases nothing; cost "
        r"is the objective above the true optimum, negative when released plans exceed true "
        r"capacity, an apparent improvement caused by infeasible overproduction; "
        r"``stated, hold'' holds the decision when any message of the case was "
        r"abstained on, so a held case releases nothing. $^{\ddagger}$Frozen arm "
        r"with a defect: every widened reading is rejected, so it decides on the nominal "
        r"state. $^{\dagger}$Post hoc, not in the frozen protocol.}"
    )
    return "\n".join(
        [
            f"%% Generated by replication-package/tools/e1v2_table.py from results/e1v2 (manifest "
            f"{summary['manifest'][:16]}); do not edit by hand.",
            r"\begin{table}[t]",
            r"\begin{adjustwidth}{-\extralength}{0cm}",
            r"\centering\footnotesize",
            f"\\caption{{{caption}}}\\label{{tab:e1}}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\textit{(a)}\par\smallskip",
            *a,
            r"\par\bigskip\textit{(b)}\par\smallskip",
            *b,
            r"\end{adjustwidth}",
            r"\end{table}",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "e1v2")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    summary = json.loads((args.results / "summary.json").read_text())
    with open(args.results / "rows.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    posthoc = json.loads((args.results / "posthoc.json").read_text())
    cl = json.loads((args.results.parent / "e1v2_clustered.json").read_text())
    args.out.write_text(table(summary, rows, posthoc, cl))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
