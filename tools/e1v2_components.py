"""Separate effects of the two E1 v2 corrections, corrected categories, stratified intervals.

    uv run python tools/e1v2_components.py --tables ../mdpi-paper/tables

Review of 4 Nov 2026. A reanalysis of the stored replies; no model is called and no stored
result file is changed. Three outputs, all on the test split:

  factorial    the 2 x 2 of the two corrections in tools/e1v2_corrected.py, per case-repeat:
               source priority (historical: a single-source notice ranks no higher than the
               capacity master; corrected: it outranks the master) crossed with widening
               (historical: the widened reading is relabelled nu_I+cal<q>, so L1 rejects it
               whenever q > 0; corrected: the margin widens reported intervals inside L5,
               nu_I unchanged). The margin q is refitted under each priority setting with
               the historical selection rule. The stated arm depends on priority only.
               Paired differences against the historical cell use the case bootstrap.
  categories   abstentions, resolved-state coverage (quiet categories: plant X unchanged)
               and violations under the stated and widened box gates, per category and
               model, from the corrected replay.
  stratified   a sensitivity check on the intervals of the corrected replay: a percentile
               bootstrap that resamples cases within each category (the benchmark fixes
               eight test cases per category), beside the case-clustered interval.

The historical-priority widening cell under the corrected gate differs from the older
single-defect post hoc arm (tools/e1v2_posthoc.py), which widened inside L1 under a
widened label and therefore changed the state shown to the model's second message.

Output: results/e1v2_components.json; with --tables, e1-factorial.tex,
e1-categories-corrected.tex and e1-stratified.tex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from e1v2_clustered import CATNAMES, DRAWS, MODELS, NAMES, SEED, clustered, paired

from dsoi import properties as P
from dsoi.constraints import solve_lp
from dsoi.experiments import e1v2_score as old
from dsoi.experiments.e1v2_corrected import MarginGate, widen_state
from dsoi.experiments.e1v2_notices import HORIZON, NOMINAL, load
from dsoi.experiments.e1v2_run import load_replies
from dsoi.experiments.e1v2_world import T_DECIDE, messages, pipeline, scenario, state_of
from dsoi.layers.l5_gate import BoxRobustGate

ROOT = Path(__file__).resolve().parents[1]
PRIORITY = ("historical", "corrected")
WIDENING = ("relabel", "gate")
CELLS = (("historical", "relabel"), ("corrected", "relabel"),
         ("historical", "gate"), ("corrected", "gate"))
CELL_NAMES = {("historical", "relabel"): "original pipeline",
              ("corrected", "relabel"): "source-priority fix only",
              ("historical", "gate"): "widening fix only",
              ("corrected", "gate"): "both fixes (Table~\\ref{tab:e1corrected})"}
QUIET = ("other_entity", "out_of_scope", "vague")


def run(case, replies: dict, nu_i: str, priority: str, widening: str, q: float):
    """One case-repeat through one cell; q = 0 gives the stated box gate of that priority."""
    scn = scenario(case)
    relabel = widening == "relabel"
    replay = old.Replay(replies, nu_i, q if relabel else 0.0)
    gate = MarginGate(q) if (q and not relabel) else BoxRobustGate(mode="resolve")
    _, p = pipeline(case, gate, replay, nu_i, historical=priority == "historical")
    replay.p, replay.times = p, {m.message_id: t for m, t in messages(case)}
    outcomes = [p.ingest_message(m, t) for m, t in messages(case)]
    try:
        record = p.decide(T_DECIDE, scn.theta()).record
    except Exception:
        record = p.audit.records[-1]
        if record.outcome != "error":
            raise
    inst = p.formulation.build(scn.true_state(), scn.theta())
    best = inst.objective(solve_lp(inst.c, inst.system).x)
    executed = record.x_exec
    truth = case.true_capacity("plant:X")
    if q and not relabel:
        canon, _ = p.state.read(p.decision_class, T_DECIDE)
        x = {k[2]: (v, lo, hi) for k, v, lo, hi, _ in widen_state(canon, q)
             if k[0] == "plant:X"}
    else:
        x = state_of(p, T_DECIDE).get("plant:X", {})
    affected = {w for w in HORIZON if truth[w] != NOMINAL}
    missing = {w for w in HORIZON if w not in x}
    changed_x = {w for w, v in x.items() if v != (NOMINAL, None, None)} | missing
    lo, hi = old._bounds(x)
    hits = changed_x & affected
    return {
        "abstained": "abstained" in outcomes,
        "snapshot_ok": replay.snapshot_ok,
        "released": executed is not None,
        "error": record.outcome == "error",
        "violation": not P.k_exec_true(executed, scn.true_state(), scn.theta(), p.formulation),
        "excess": None if executed is None else inst.objective(np.array(executed)) - best,
        "covered": old._covered(x, truth),
        "quiet_ok": not changed_x,
        "cal_score": None if not hits else NOMINAL if hits & missing else
            max(max(lo[w] - truth[w], truth[w] - hi[w], 0.0) for w in hits),
    }


def compute(cases: list, replies: dict) -> tuple[dict, dict]:
    """Rows keyed by (model, case, repeat), with the stated and widened arms of every cell."""
    grouped, nu = defaultdict(dict), {}
    for (model, cid, repeat, mid), reply in replies.items():
        grouped[(model, cid, repeat)][mid] = reply
        nu[model] = reply["nu_i"]
    by_id = {c.id: c for c in cases}
    rows: dict[tuple, dict] = {}
    for key in sorted(grouped):
        model, cid, repeat = key
        case = by_id[cid]
        row = {"model": model, "case_id": cid, "repeat": repeat, "category": case.category,
               "split": case.split, "expect": case.expect}
        for priority in PRIORITY:
            row[f"stated:{priority}"] = run(case, grouped[key], nu[model], priority, "gate", 0.0)
        rows[key] = row
    margins = {}
    for priority in PRIORITY:
        margins[priority] = old.calibrate(
            [{"model": r["model"], "split": r["split"],
              "cal_score": r[f"stated:{priority}"]["cal_score"]} for r in rows.values()], 0.1)
    for key, row in rows.items():
        model, cid, _ = key
        for priority, widening in CELLS:
            row[f"widened:{priority}:{widening}"] = run(
                by_id[cid], grouped[key], nu[model], priority, widening,
                margins[priority][model])
    return rows, margins


def stratified(units: list[tuple[str, str, bool]], rng: np.random.Generator) -> list[float]:
    """Percentile interval when cases are resampled within their category."""
    by_cat: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for cat, case, v in units:
        by_cat[cat][case].append(v)
    k_total = np.zeros(DRAWS)
    n_total = np.zeros(DRAWS)
    for cat in sorted(by_cat):
        groups = by_cat[cat]
        k_g = np.array([sum(v) for v in groups.values()], float)
        n_g = np.array([len(v) for v in groups.values()], float)
        idx = rng.integers(0, len(k_g), size=(DRAWS, len(k_g)))
        k_total += k_g[idx].sum(1)
        n_total += n_g[idx].sum(1)
    boot = k_total / n_total
    return [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)]


def check_against_stored(rows: dict) -> dict:
    """The historical and corrected cells must reproduce the stored scores row by row."""
    import csv

    checks = {}
    stored = {
        "historical": (ROOT / "results/e1v2/rows.csv", "historical", "relabel"),
        "corrected": (ROOT / "results/e1v2_corrected/rows.csv", "corrected", "gate"),
    }
    for name, (path, priority, widening) in stored.items():
        bad = 0
        for r in csv.DictReader(open(path, newline="")):
            row = rows[(r["model"], r["case_id"], int(r["repeat"]))]
            s = row[f"stated:{priority}"]
            w = row[f"widened:{priority}:{widening}"]
            pairs = ((s["violation"], r["violation_stated"]),
                     (s["covered"], r["covered_stated"]),
                     (s["abstained"], r["abstained"]),
                     (w["violation"], r["violation_calibrated"]),
                     (w["covered"], r["covered_calibrated"]),
                     (w["released"], r["released_calibrated"]))
            bad += sum(str(a) != b for a, b in pairs)
        checks[name] = {"file": str(path.relative_to(ROOT)), "mismatched_values": bad}
    return checks


def analyse(rows: dict, margins: dict) -> dict:
    rng = np.random.default_rng(SEED)
    sens_rng = np.random.default_rng([SEED, 1])   # own stream: stored intervals unchanged
    out = {"post_hoc": True, "fresh_model_calls": 0, "seed": SEED, "draws": DRAWS,
           "note": " ".join(__doc__.split("\n\n")[1:3]).replace("\n", " "),
           "margins": margins, "factorial": {}, "categories": {}, "stratified": {}}
    for model in MODELS:
        test = [(k, r) for k, r in rows.items() if r["model"] == model and r["split"] == "test"]
        change = [(k, r) for k, r in test if r["expect"] == "change"]
        quiet = [(k, r) for k, r in test if r["expect"] != "change"]
        keys = [k for k, _ in change]

        def units(sub, col, field):
            return [(r["case_id"], bool(r[col][field])) for _, r in sub]

        cells = {}
        base = {k: r["widened:historical:relabel"]["violation"] for k, r in change}
        # Case-repeats where some relabel cell changes the state a later message was shown:
        # replaying the stored reply there is counterfactual. The sensitivity row drops them
        # from every cell, so the cells stay comparable.
        changed_input = {k for k, r in change for p in PRIORITY
                         if not r[f"widened:{p}:relabel"]["snapshot_ok"]}
        kept = [k for k in keys if k not in changed_input]
        for priority, widening in CELLS:
            col = f"widened:{priority}:{widening}"
            stated = f"stated:{priority}"
            vio = {k: r[col]["violation"] for k, r in change}
            cells[f"{priority}/{widening}"] = {
                "label": CELL_NAMES[(priority, widening)],
                "q": margins[priority][model],
                "state_coverage": clustered(units(change, stated, "covered"), rng),
                "stated_violation": clustered(units(change, stated, "violation"), rng),
                "widened_coverage": clustered(units(change, col, "covered"), rng),
                "widened_release": clustered(units(change, col, "released"), rng),
                "widened_violation": clustered(units(change, col, "violation"), rng),
                "widened_snapshot_mismatches": sum(not r[col]["snapshot_ok"] for _, r in change),
                "excluding_changed_inputs": {
                    "case_repeats": len(kept),
                    "stated_violation": sum(r[stated]["violation"] for k, r in change
                                            if k in kept),
                    "widened_violation": sum(vio[k] for k in kept),
                    "widened_minus_original": paired(vio, base, kept, sens_rng)},
                "widened_minus_original": paired(vio, base, keys, rng),
                "stated_minus_original_stated": paired(
                    {k: r[stated]["violation"] for k, r in change},
                    {k: r["stated:historical"]["violation"] for k, r in change}, keys, rng),
            }
        out["factorial"][model] = cells

        cats = {}
        for cat in sorted({r["category"] for _, r in test}):
            cr = [r for _, r in test if r["category"] == cat]
            s, w = "stated:corrected", "widened:corrected:gate"
            cats[cat] = {
                "n": len(cr),
                "abstained": sum(r[s]["abstained"] for r in cr),
                "covered_or_kept": sum(r[s]["quiet_ok" if cat in QUIET else "covered"]
                                       for r in cr),
                "violation_stated": sum(r[s]["violation"] for r in cr),
                "violation_widened": sum(r[w]["violation"] for r in cr),
                "released_widened": sum(r[w]["released"] for r in cr),
            }
        out["categories"][model] = cats

        strat = {}
        measures = (("abstained", "stated:corrected", "abstained"),
                    ("state_coverage", "stated:corrected", "covered"),
                    ("widened_coverage", "widened:corrected:gate", "covered"),
                    ("stated_violation", "stated:corrected", "violation"),
                    ("widened_violation", "widened:corrected:gate", "violation"))
        for name, col, field in measures:
            u = [(r["category"], r["case_id"], bool(r[col][field])) for _, r in change]
            c = clustered([(case, v) for _, case, v in u], rng)
            strat[name] = {"rate": c["rate"], "k": c["k"], "n": c["n"],
                           "clustered_ci95": c["ci95"], "case_bootstrap_ci95": c["ci95_bootstrap"],
                           "stratified_bootstrap_ci95": stratified(u, rng)}
        # Paired difference widened - stated under stratified case resampling.
        diffs = defaultdict(lambda: defaultdict(list))
        for k, r in change:
            diffs[r["category"]][k[1]].append(
                float(r["widened:corrected:gate"]["violation"])
                - float(r["stated:corrected"]["violation"]))
        tot_s, tot_n = np.zeros(DRAWS), np.zeros(DRAWS)
        for cat in sorted(diffs):
            s_g = np.array([sum(v) for v in diffs[cat].values()])
            n_g = np.array([len(v) for v in diffs[cat].values()], float)
            idx = rng.integers(0, len(s_g), size=(DRAWS, len(s_g)))
            tot_s += s_g[idx].sum(1)
            tot_n += n_g[idx].sum(1)
        boot = tot_s / tot_n
        unstrat = paired({k: r["widened:corrected:gate"]["violation"] for k, r in change},
                         {k: r["stated:corrected"]["violation"] for k, r in change}, keys, rng)
        strat["widened_minus_stated_violation"] = {
            "diff": unstrat["diff"], "case_bootstrap_ci95": unstrat["ci95_bootstrap"],
            "stratified_bootstrap_ci95": [round(float(np.quantile(boot, 0.025)), 4),
                                          round(float(np.quantile(boot, 0.975)), 4)]}
        out["stratified"][model] = strat
        out["factorial"][model]["quiet_rows"] = len(quiet)
    return out


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def ci(c: list[float]) -> str:
    return f"{pct(c[0])}--{pct(c[1])}"


def num(x: float) -> str:
    """Percentage points with one decimal and a typeset minus sign."""
    return f"${100 * x + 0.0:.1f}$"


def pp(d: float, c: list[float]) -> str:
    return f"{num(d)} [{num(c[0])}, {num(c[1])}]"


def kn(d: dict) -> str:
    return f"{d['k']}"


def factorial_table(res: dict) -> str:
    lines = [
        "%% Generated by replication-package/tools/e1v2_components.py (post hoc); "
        "do not edit by hand.",
        r"\begin{table}[htbp]", r"\begin{adjustwidth}{-\extralength}{0cm}",
        r"\centering\footnotesize",
        r"\caption{Effect of each correction separately on the fixed stored replies (test "
        r"part; 176 change-notice case-repeats per model; no fresh model calls). The table "
        r"measures what each pipeline does with these responses, not how a model would respond "
        r"to the inputs a changed pipeline would show it. Priority: whether a "
        r"single-source notice outranks the capacity master. Widening: whether the margin is "
        r"applied by relabelling the reading, which L1 then rejects, or inside the L5 gate. "
        r"$q$ is refitted under each priority setting. State coverage and stated-box "
        r"violations depend on priority only. $\Delta V$: change in widened-box violations "
        r"against the original pipeline, in percentage points, with a case-bootstrap 95\% "
        r"interval in brackets, whose coverage for paired differences has not been established. "
        r"Counts out of 176. In the relabel rows, rejecting a first reading "
        r"also changes the state shown with a later message, so these rows are counterfactual "
        r"for "
        + " and ".join(f"{res['factorial'][m]['historical/relabel']['widened_snapshot_mismatches']}"
                       f" {NAMES[m]}" for m in MODELS
                       if res["factorial"][m]["historical/relabel"]["widened_snapshot_mismatches"])
        + r" case-repeats; Section~\ref{sec:e1} repeats the comparison without them."
        r"}\label{tab:e1factorial}",
        r"\begin{tabular}{@{}llrrrrrr@{}}\toprule",
        r"Model & Pipeline & $q$ & State cov. & Stated $V$ & Widened cov. & Widened $V$ "
        r"& $\Delta V$ (pp)\\\midrule",
    ]
    for model in MODELS:
        cells = res["factorial"][model]
        for i, (priority, widening) in enumerate(CELLS):
            c = cells[f"{priority}/{widening}"]
            d = c["widened_minus_original"]
            delta = ("--" if (priority, widening) == CELLS[0] else
                     pp(d["diff"], d["ci95_bootstrap"]))
            lines.append(
                f"{NAMES[model] if i == 0 else ''} & {c['label']} & {c['q']:g} & "
                f"{kn(c['state_coverage'])} & {kn(c['stated_violation'])} & "
                f"{kn(c['widened_coverage'])} & {kn(c['widened_violation'])} & {delta}" + r"\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule\end{tabular}"
    lines += [r"\end{adjustwidth}", r"\end{table}", ""]
    return "\n".join(lines)


def category_table(res: dict) -> str:
    cats = list(res["categories"][MODELS[0]])
    order = [c for c in cats if c not in QUIET] + [c for c in QUIET if c in cats]
    lines = [
        "%% Generated by replication-package/tools/e1v2_components.py (post hoc); "
        "do not edit by hand.",
        r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Corrected replay per category (test part; 8 cases, 16 case-repeats per "
        r"model and category; no fresh model calls). Each cell gives abstained / covered / "
        r"violated (stated box) / violated (widened box): case-repeats with an abstention, "
        r"case-repeats whose resolved intervals cover the true capacity (for the last three, "
        r"quiet, categories: plant X left unchanged), and case-repeats whose plan violates the "
        r"true capacity under each gate. Supplementary Table S2 gives the historical "
        r"breakdown.}\label{tab:e1catcorr}",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        " & ".join(["Category"] + [NAMES[m] for m in MODELS]) + r"\\\midrule",
    ]
    for cat in order:
        if cat == QUIET[0]:
            lines.append(r"\midrule")
        cells = []
        for m in MODELS:
            c = res["categories"][m][cat]
            cells.append(f"{c['abstained']} / {c['covered_or_kept']} / "
                         f"{c['violation_stated']} / {c['violation_widened']}")
        lines.append(" & ".join([CATNAMES.get(cat, cat)] + cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def stratified_table(res: dict) -> str:
    labels = {"abstained": "Abstained", "state_coverage": "State coverage",
              "widened_coverage": "Widened coverage", "stated_violation": "Stated-box $V$",
              "widened_violation": "Widened-box $V$"}
    lines = [
        "%% Generated by replication-package/tools/e1v2_components.py (post hoc); "
        "do not edit by hand.",
        r"\begin{table}[htbp]", r"\centering\footnotesize",
        r"\caption{Sensitivity of the corrected-replay intervals to the resampling design "
        r"(176 change-notice case-repeats per model). Approximate 95\% intervals in percent: "
        r"case-clustered Wilson (as in Table~\ref{tab:e1corrected}), case bootstrap, and a "
        r"bootstrap that resamples the eight cases within each of the 11 change categories, "
        r"holding the benchmark's category composition fixed. The last row is the paired "
        r"difference widened minus stated box, in percentage points.}\label{tab:e1strat}",
        r"\begin{tabular}{@{}llrrrr@{}}\toprule",
        r"Model & Measure & Rate & Clustered & Case boot. & Stratified boot.\\\midrule",
    ]
    for model in MODELS:
        s = res["stratified"][model]
        for i, key in enumerate(labels):
            v = s[key]
            lines.append(f"{NAMES[model] if i == 0 else ''} & {labels[key]} & {pct(v['rate'])} & "
                         f"{ci(v['clustered_ci95'])} & {ci(v['case_bootstrap_ci95'])} & "
                         f"{ci(v['stratified_bootstrap_ci95'])}" + r"\\")
        d = s["widened_minus_stated_violation"]
        lines.append(f" & Widened $-$ stated $V$ (pp) & {num(d['diff'])} & -- & "
                     f"[{num(d['case_bootstrap_ci95'][0])}, {num(d['case_bootstrap_ci95'][1])}] & "
                     f"[{num(d['stratified_bootstrap_ci95'][0])}, "
                     f"{num(d['stratified_bootstrap_ci95'][1])}]" + r"\\\addlinespace")
    lines[-1] = lines[-1].replace(r"\\\addlinespace", r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/e1v2_components.json")
    ap.add_argument("--tables", type=Path)
    ap.add_argument("--cases", type=Path, default=ROOT / "data/e1v2/cases.jsonl")
    ap.add_argument("--replies", type=Path, default=ROOT / "results/e1v2/replies.jsonl")
    args = ap.parse_args(argv)
    rows, margins = compute(load(args.cases), load_replies(args.replies))
    res = analyse(rows, margins)
    res["reproduces_stored"] = check_against_stored(rows)
    res["inputs_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in (("cases", args.cases), ("replies", args.replies))}
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    if args.tables:
        (args.tables / "e1-factorial.tex").write_text(factorial_table(res))
        (args.tables / "e1-categories-corrected.tex").write_text(category_table(res))
        (args.tables / "e1-stratified.tex").write_text(stratified_table(res))
    print(json.dumps({"reproduces_stored": res["reproduces_stored"], "margins": margins}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
