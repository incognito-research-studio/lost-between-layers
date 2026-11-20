"""Exact expectations for the randomized abstention policies (post hoc, 16 Nov 2026).

    cd tools && uv run python e1v2_abstention_expectations.py \
        --table ../../manuscripts/arxiv-paper/tables/e1-abstention-policies.tex

DEVELOPMENT DATA, POST HOC, NO MODEL CALLS. e1v2_abstention_policies.py scores the policies
"hold to a deadline (q)" and "recheck, source stale with probability p" with one seeded draw
per case-repeat, so their reported rates are single draws. Both policies are mixtures of
deterministic outcomes per case-repeat with no reading:
  hold_deadline_q    = q * (recheck with the true source) + (1 - q) * (standing plan)
  recheck_stale_p    = (1 - p) * (recheck with the true source) + p * (recheck that returns
                       the value in force before the notice)
and every case-repeat with a reading is decided as under every other policy. This tool
computes the three deterministic outcomes for every case-repeat, with the same functions as
e1v2_abstention_policies.py, and reports the exact expected violations per change notice and
delivered shares. It also re-derives the seeded draws, as a check that the outcomes match the
stored policy results. Raised by the pre-submission review of the AI4Factory paper
(manuscripts/aaai-workshop/REVIEW-2026-11-16.md, row 10).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import e1v2_abstention_policies as A

from dsoi.experiments.e1v2_notices import NOMINAL, load
from dsoi.experiments.e1v2_run import load_replies
from dsoi.layers.l5_gate import BoxRobustGate, RecheckGate

ROOT = Path(__file__).resolve().parents[1]
QS = (0.5, 0.9)
PS = (0.2, 0.5)


def outcomes(case, replies, nu_i) -> dict:
    """Deterministic outcomes for one case-repeat: base (stated box gate, used whenever every
    message has a reading), and for a case-repeat without a reading the standing plan, the
    recheck with the true source and the recheck with a stale source."""
    scn, p, rec, rep = A.run(case, replies, nu_i, BoxRobustGate(mode="resolve"))
    base = A.outcome(case, scn, p, rec.x_exec)
    if not rep.no_reading:
        return {"no_reading": False, "base": base}
    standing = A.outcome(case, scn, p, A.standing_plan(case))
    s1, p1, r1, _ = A.run(case, replies, nu_i,
                          RecheckGate(scn.true_reader(), fallback="resolve"))
    s2, p2, r2, _ = A.run(case, replies, nu_i,
                          RecheckGate(lambda key: NOMINAL, fallback="resolve"))
    return {"no_reading": True, "base": base, "standing": standing,
            "recheck_true": A.outcome(case, s1, p1, r1.x_exec),
            "recheck_stale": A.outcome(case, s2, p2, r2.x_exec)}


def mix(o: dict, w_true: float, other: str) -> tuple[float, float, float]:
    """Expected (violation, delivered share, release) for one case-repeat."""
    if not o["no_reading"]:
        b = o["base"]
        return float(b["violation"]), b["delivered_share"], float(b["released"])
    a, b = o["recheck_true"], o[other]
    return tuple(w_true * float(a[k]) + (1 - w_true) * float(b[k])
                 for k in ("violation", "delivered_share", "released"))


def evaluate(cases_path: Path, replies_path: Path) -> dict:
    by_id = {c.id: c for c in load(cases_path)}
    grouped, nu = defaultdict(dict), {}
    for (model, cid, repeat, mid), reply in load_replies(replies_path).items():
        grouped[(model, cid, repeat)][mid] = reply
        nu[model] = reply["nu_i"]
    res = {}
    for model in A.MODELS:
        rows = []
        for (m, cid, rep) in sorted(grouped):
            if m != model or by_id[cid].split != "test":
                continue
            case = by_id[cid]
            o = outcomes(case, grouped[(m, cid, rep)], nu[m])
            o.update(case_id=cid, repeat=rep, change=case.expect == "change")
            rows.append(o)
        out = {}
        for subset in ("change", "quiet"):
            sub = [r for r in rows if r["change"] == (subset == "change")]
            pol = {}
            for q in QS:
                vals = [mix(r, q, "standing") for r in sub]
                drawn = [mix(r, 1.0 if A._draw(r["case_id"], r["repeat"], "deadline") < q
                             else 0.0, "standing") for r in sub]
                pol[f"hold_deadline_{q}"] = summary(vals, drawn)
            for p in PS:
                vals = [mix(r, 1 - p, "recheck_stale") for r in sub]
                drawn = [mix(r, 0.0 if A._draw(r["case_id"], r["repeat"], "stale") < p
                             else 1.0, "recheck_stale") for r in sub]
                pol[f"recheck_stale_{p}"] = summary(vals, drawn)
            out[subset] = {"case_repeats": len(sub), **pol}
        res[model] = out
    return res


def summary(vals, drawn) -> dict:
    n = len(vals)
    return {"expected_violations": round(sum(v[0] for v in vals), 3),
            "expected_violation_rate": round(sum(v[0] for v in vals) / n, 4),
            "expected_delivered_share": round(sum(v[1] for v in vals) / n, 4),
            "expected_release_rate": round(sum(v[2] for v in vals) / n, 4),
            "seeded_draw_violations": int(sum(v[0] for v in drawn)),
            "seeded_draw_delivered_share": round(sum(v[1] for v in drawn) / n, 4)}


NAMES = {"retain": "Retain old state", "hold_nothing": "Hold, nothing executes",
         "hold_standing": "Hold, standing plan runs",
         "hold_deadline_0.5": "Hold to deadline ($q=0.5$)",
         "hold_deadline_0.9": "Hold to deadline ($q=0.9$)",
         "bounds_inferred": "Domain bounds, inferred scope",
         "bounds_oracle": "Domain bounds, true scope (bound)",
         "recheck_stale_0": "Recheck, never stale", "recheck_stale_0.2": "Recheck, 20\\% stale",
         "recheck_stale_0.5": "Recheck, 50\\% stale"}


def table(stored: dict, res: dict) -> str:
    """The arXiv/IEEE Access table tab:e1abstpol: deterministic rows from the stored policy
    results, the four randomized rows as exact expectations from this tool."""
    def cell(model, pol):
        ch, qu = stored[model]["change"][pol], stored[model]["quiet"][pol]
        v, r, d = ch["violation"]["rate"], ch["released"]["rate"], ch["delivered_share_mean"]
        dq = qu["delivered_share_mean"]
        if pol in res[model]["change"]:
            e, eq = res[model]["change"][pol], res[model]["quiet"][pol]
            v, r, d = (e["expected_violation_rate"], e["expected_release_rate"],
                       e["expected_delivered_share"])
            dq = eq["expected_delivered_share"]
        return " / ".join(f"{100 * x:.0f}" for x in (v, r, d, dq))
    rows = [f"{NAMES[pol]} & " + " & ".join(cell(m, pol) for m in A.MODELS) + r" \\"
            for pol in A.POLICIES]
    return "\n".join([
        "%% Generated by replication-package/tools/e1v2_abstention_expectations.py; do not edit.",
        r"\begin{table*}[tp]", r"\centering\footnotesize",
        r"\caption{\cdraft{Abstention handling as a policy, on development data (the stored "
        r"e1v2 replies, test part; post hoc, no model calls; tools/e1v2\_abstention\_policies.py "
        r"and, for the four rows with a random draw, tools/e1v2\_abstention\_expectations.py). "
        r"A case-repeat in which every message has a reading is decided by the stated box gate "
        r"under every policy. Each cell: violations per change-notice opportunity / release rate "
        r"on change notices / delivered share of the best achievable output on change notices / "
        r"the same on quiet notices, all in percent. Delivered units are capped at the true "
        r"capacity, so over-release adds nothing; no release delivers nothing. The hold-to-deadline"
        r" and stale-recheck rows are exact expectations over the random draws, not one seeded "
        r"draw.}}\label{tab:e1abstpol}",
        r"\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Policy & Qwen2.5 7B & Llama 3.1 8B & Phi-4 14B\\\midrule",
        *rows, r"\bottomrule\end{tabular}", r"\end{table*}", ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=Path, default=ROOT / "data/e1v2/cases.jsonl")
    ap.add_argument("--replies", type=Path, default=ROOT / "results/e1v2/replies.jsonl")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/e1v2_abstention_expectations.json")
    ap.add_argument("--table", type=Path)
    args = ap.parse_args(argv)
    res = evaluate(args.cases, args.replies)
    stored = json.loads((ROOT / "results/e1v2_abstention_policies.json").read_text())["models"]
    checks = {}
    for model in A.MODELS:
        for subset in ("change", "quiet"):
            for pol, v in res[model][subset].items():
                if not isinstance(v, dict) or pol not in stored[model][subset]:
                    continue
                s = stored[model][subset][pol]
                checks[f"{model}/{subset}/{pol}"] = (
                    v["seeded_draw_violations"] == s["violation"]["k"]
                    and abs(v["seeded_draw_delivered_share"] - s["delivered_share_mean"]) < 1e-3)
    payload = {"development_data": True, "post_hoc": True, "fresh_model_calls": 0,
               "note": __doc__.split("\n\n")[2].replace("\n", " "),
               "seeded_draws_match_stored_policy_results": all(checks.values()),
               "check_detail": checks, "models": res}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    if args.table:
        args.table.write_text(table(stored, res))
    for model in A.MODELS:
        ch = res[model]["change"]
        print(model, {k: (v["expected_violation_rate"], v["expected_delivered_share"],
                          v["seeded_draw_violations"]) for k, v in ch.items()
                     if isinstance(v, dict)})
    print("seeded draws match stored:", all(checks.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
