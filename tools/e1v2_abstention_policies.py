"""Abstention handling as an architectural policy (sixth external review, experiment 3).

    cd tools && uv run python e1v2_abstention_policies.py \
        --table ../../mdpi-paper/tables/e1-abstention-policies.tex

DEVELOPMENT DATA. This runs on the stored e1v2 replies (test part, three models), which were
used to debug the pipeline; its numbers are post hoc and serve to develop and test the
policy code, which is to be applied unchanged to the fresh study. No model is called.

When L1 returns no reading for a message, either an explicit abstention or a reply the L1
contract rejects (counted apart), the decision class responds by one of these policies; a
case-repeat with a reading for every message is decided as in the corrected replay (the
stated-interval box gate) under every policy.
  retain               the reading changes nothing; the old state stays in force (default)
  hold_nothing         the decision is held and nothing executes
  hold_standing        the decision is held and the standing plan, made before the notice
                       (nominal capacity, full demand), keeps executing
  hold_deadline_q      held until an authoritative confirmation arrives; with probability q
                       it arrives before the deadline and the decision is made on it (as an
                       exact re-read), otherwise the standing plan executes (q = 0.5, 0.9)
  bounds_inferred      the affected values are replaced by the attribute's declared domain
                       [0, nominal] with the value in force as the point, and the box gate
                       decides; the scope is inferred from the message text without the
                       interpreter: entities by their names in the master data, weeks by a
                       generic pattern ("week 42", "weeks 42-44", "W42"), the whole horizon
                       when no week is found
  bounds_oracle        the same with the true affected entities and weeks: an upper bound,
                       not an operating policy
  recheck_stale_p      the box gate with an execution-time re-read of a simulated
                       authoritative source, which returns the value in force before the
                       notice (stale) with probability p (p = 0, 0.2, 0.5)
Random draws (deadlines, staleness) are seeded per case-repeat, so every policy sees the same
draws. Reported per model, over change and quiet notices separately: release rate, violations
per opportunity and among releases, the delivered share of the best achievable output
(units the true capacity allows, min(x_w, c*_w), over min(demand, c*_w); 0 for no release; a
total measure, unlike the conditional feasible-release cost), and the
case-repeats whose later message would have been shown a different state under the policy
(counterfactual inputs, as in Table 14), with the rates also given without them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import e1v2_clustered
import numpy as np

from dsoi import properties as P
from dsoi.experiments import e1v2_score as old
from dsoi.experiments.e1v2_notices import HORIZON, NAMES, NOMINAL, load
from dsoi.experiments.e1v2_run import load_replies
from dsoi.experiments.e1v2_world import T_DECIDE, T_LOAD, messages, pipeline, scenario, state_of
from dsoi.layers.l1_interpreter import InterpretResult, parse_output
from dsoi.layers.l5_gate import BoxRobustGate, RecheckGate
from dsoi.types import Delta

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen2.5:7b-instruct", "llama3.1:8b", "phi4:latest")
SEED = 20261108
POLICIES = ("retain", "hold_nothing", "hold_standing", "hold_deadline_0.5",
            "hold_deadline_0.9", "bounds_inferred", "bounds_oracle", "recheck_stale_0",
            "recheck_stale_0.2", "recheck_stale_0.5")
_WEEK_RANGE = re.compile(r"\b(?:weeks?\s*|W)(\d{1,2})\s*(?:-|to|through)\s*"
                         r"(?:weeks?\s*|W)?(\d{1,2})\b")
_WEEK_ONE = re.compile(r"\b(?:weeks?\s*|W)(\d{1,2})\b")


def infer_scope(text: str) -> dict[str, list[int]]:
    """Entities named in the text (master-data names) and weeks found by a generic pattern;
    every entity when none is named, the whole horizon when no week is found."""
    low = text.lower()
    ents = [e for e, (code, name) in NAMES.items()
            if code.lower() in low or name.lower().removeprefix("the ") in low] or list(NAMES)
    weeks: set[int] = set()
    for a, b in _WEEK_RANGE.findall(text):
        weeks |= set(range(int(a), int(b) + 1))
    for a in _WEEK_ONE.findall(text):
        weeks.add(int(a))
    weeks &= set(HORIZON)
    return {e: sorted(weeks or HORIZON) for e in ents}


def oracle_scope(case) -> dict[str, list[int]]:
    return {e: sorted(w for w, v in ws.items()) for e, ws in case.truth.items()}


class PolicyReplay(old.Replay):
    """The stored reply; when it gives no reading and the policy bounds the values, a hedged
    reading over the policy's scope instead (Definition 6), passed through the L1 contract."""

    def __init__(self, replies, nu_i, scope_fn=None):
        super().__init__(replies, nu_i)
        self.scope_fn, self.no_reading = scope_fn, []

    def interpret(self, message, schema):
        result = super().interpret(message, schema)
        if isinstance(result.output, Delta):
            return result
        reply = self.replies.get(message.message_id) or {}
        try:
            explicit = json.loads(reply.get("text") or "null").get("kind") == "abstain"
        except (ValueError, AttributeError):
            explicit = False
        self.no_reading.append("abstention" if explicit else "rejection")
        if self.scope_fn is None:
            return result
        state = state_of(self.p, self.times[message.message_id])
        ops = [{"op": "supersede", "subject": e, "predicate": "capacity",
                "value": state.get(e, {}).get(w, (NOMINAL,))[0], "lo": 0.0, "hi": NOMINAL,
                "unit": "units/week", "valid_from": w, "valid_to": w + 1,
                "span_text": message.text}
               for e, ws in self.scope_fn(message.text).items() for w in ws]
        if not ops:
            return result
        out = parse_output({"kind": "delta", "reason": "abstention policy: domain bounds",
                            "ops": ops}, message, schema, self.nu_i)
        return InterpretResult(out, nu_i=self.nu_i) if isinstance(out, Delta) else result


def _draw(case_id: str, repeat: int, what: str) -> float:
    h = hashlib.sha256(f"{SEED}-{case_id}-{repeat}-{what}".encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


def run(case, replies, nu_i, gate, scope_fn=None):
    scn = scenario(case)
    replay = PolicyReplay(replies, nu_i, scope_fn)
    _, p = pipeline(case, gate, replay, nu_i)
    replay.p, replay.times = p, {m.message_id: t for m, t in messages(case)}
    for m, t in messages(case):
        p.ingest_message(m, t)
    try:
        record = p.decide(T_DECIDE, scn.theta()).record
    except Exception:
        record = p.audit.records[-1]
        if record.outcome != "error":
            raise
    return scn, p, record, replay


def standing_plan(case) -> list[float]:
    """The plan in force before any notice: decided on the master data alone."""
    scn = scenario(case)
    _, p = pipeline(case, BoxRobustGate(mode="resolve"), None, "standing")
    return p.decide(T_LOAD + 0.5, scn.theta()).record.x_released


def outcome(case, scn, p, executed) -> dict:
    """Release, violation, and the delivered share: units that true capacity lets the plant
    make, min(x_w, c*_w), summed over weeks, as a share of the best achievable,
    min(demand, c*_w) summed (0 for no release; over-release delivers no extra units)."""
    names = list(p.formulation.build(scn.true_state(), scn.theta()).system.names)
    cap = case.true_capacity("plant:X")
    weeks = [int(float(n[2:-1])) for n in names]
    best = sum(min(NOMINAL, cap[w]) for w in weeks)
    delivered = 0.0 if executed is None else sum(min(float(x), cap[w])
                                                 for x, w in zip(executed, weeks, strict=True))
    return {"released": executed is not None,
            "violation": executed is not None and not P.k_exec_true(
                executed, scn.true_state(), scn.theta(), p.formulation),
            "delivered_share": delivered / best if best > 0 else 1.0}


def score_case(case, replies, nu_i, repeat) -> dict:
    base_scn, base_p, base_rec, base_replay = run(case, replies, nu_i,
                                                   BoxRobustGate(mode="resolve"))
    no_reading = list(base_replay.no_reading)
    retain = outcome(case, base_scn, base_p, base_rec.x_exec)
    row = {"case_id": case.id, "repeat": repeat, "category": case.category,
           "change": case.expect == "change", "no_reading": bool(no_reading),
           "abstentions": no_reading.count("abstention"),
           "rejections": no_reading.count("rejection"), "error": base_rec.outcome == "error"}
    if not no_reading:
        for pol in POLICIES:
            row[pol] = {**retain, "counterfactual": False}
        return row
    standing = standing_plan(case)
    hold_standing = outcome(case, base_scn, base_p, standing)
    for pol in POLICIES:
        cf = False
        if pol == "retain":
            res = retain
        elif pol == "hold_nothing":
            res = outcome(case, base_scn, base_p, None)
        elif pol == "hold_standing":
            res = hold_standing
        elif pol.startswith("hold_deadline_"):
            q = float(pol.rsplit("_", 1)[1])
            if _draw(case.id, repeat, "deadline") < q:
                scn, p, rec, rep = run(case, replies, nu_i,
                                       RecheckGate(base_scn.true_reader(), fallback="resolve"))
                res, cf = outcome(case, scn, p, rec.x_exec), not rep.snapshot_ok
            else:
                res = hold_standing
        elif pol.startswith("bounds_"):
            fn = infer_scope if pol == "bounds_inferred" else (lambda _t, c=case: oracle_scope(c))
            scn, p, rec, rep = run(case, replies, nu_i, BoxRobustGate(mode="resolve"), fn)
            res, cf = outcome(case, scn, p, rec.x_exec), not rep.snapshot_ok
        else:
            stale_p = float(pol.rsplit("_", 1)[1])
            true = base_scn.true_reader()
            stale = _draw(case.id, repeat, "stale") < stale_p
            gate = RecheckGate((lambda key: NOMINAL) if stale else true, fallback="resolve")
            scn, p, rec, rep = run(case, replies, nu_i, gate)
            res, cf = outcome(case, scn, p, rec.x_exec), not rep.snapshot_ok
        row[pol] = {**res, "counterfactual": cf}
    return row


def summarize(rows: list[dict], rng) -> dict:
    out = {}
    for subset, keep in (("change", lambda r: r["change"]), ("quiet", lambda r: not r["change"])):
        sub = [r for r in rows if keep(r)]
        out[subset] = {"case_repeats": len(sub),
                       "no_reading": sum(r["no_reading"] for r in sub),
                       "abstentions": sum(r["abstentions"] > 0 for r in sub),
                       "rejections_only": sum(r["rejections"] > 0 and r["abstentions"] == 0
                                              for r in sub)}
        for pol in POLICIES:
            def units(field, s=sub, p=pol):
                return [(r["case_id"], bool(r[p][field])) for r in s]
            rel = [r for r in sub if r[pol]["released"]]
            clean = [r for r in sub if not r[pol]["counterfactual"]]
            out[subset][pol] = {
                "released": e1v2_clustered.clustered(units("released"), rng),
                "violation": e1v2_clustered.clustered(units("violation"), rng),
                "violation_given_release": {"k": sum(r[pol]["violation"] for r in rel),
                                            "n": len(rel)},
                "delivered_share_mean": round(float(np.mean([r[pol]["delivered_share"]
                                                             for r in sub])), 4) if sub else None,
                "counterfactual": sum(r[pol]["counterfactual"] for r in sub),
                "violation_without_counterfactual": {
                    "k": sum(r[pol]["violation"] for r in clean), "n": len(clean)}}
    return out


def evaluate(cases_path: Path, replies_path: Path) -> dict:
    by_id = {c.id: c for c in load(cases_path)}
    grouped, nu = defaultdict(dict), {}
    for (model, cid, repeat, mid), reply in load_replies(replies_path).items():
        grouped[(model, cid, repeat)][mid] = reply
        nu[model] = reply["nu_i"]
    res = {}
    for model in MODELS:
        rows = [score_case(by_id[cid], grouped[(m, cid, rep)], nu[m], rep)
                for (m, cid, rep) in sorted(grouped) if m == model
                and by_id[cid].split == "test"]
        res[model] = summarize(rows, np.random.default_rng(SEED))
    return res


def pct(c: dict | None) -> str:
    return "--" if c is None else f"{100 * c['rate']:.0f}"


def table(res: dict) -> str:
    names = {"retain": "Retain old state", "hold_nothing": "Hold, nothing executes",
             "hold_standing": "Hold, standing plan runs", "hold_deadline_0.5":
             "Hold to deadline ($q=0.5$)", "hold_deadline_0.9": "Hold to deadline ($q=0.9$)",
             "bounds_inferred": "Domain bounds, inferred scope",
             "bounds_oracle": "Domain bounds, true scope (bound)",
             "recheck_stale_0": "Recheck, never stale", "recheck_stale_0.2": "Recheck, 20\\% stale",
             "recheck_stale_0.5": "Recheck, 50\\% stale"}
    rows = []
    for pol in POLICIES:
        cells = []
        for model in MODELS:
            ch, qu = res[model]["change"][pol], res[model]["quiet"][pol]
            cells.append(f"{pct(ch['violation'])} / {pct(ch['released'])} / "
                         f"{100 * ch['delivered_share_mean']:.0f} / "
                         f"{100 * qu['delivered_share_mean']:.0f}")
        rows.append(f"{names[pol]} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\begin{adjustwidth}{-\extralength}{0cm}",
        r"\centering\footnotesize",
        r"\caption{\cdraft{Abstention handling as a policy, on development data (the stored "
        r"e1v2 replies, test part; post hoc, no model calls; tools/e1v2\_abstention\_policies.py)."
        r" A case-repeat in which every message has a reading is decided by the stated box gate "
        r"under every policy. Each cell: violations per change-notice opportunity / release rate"
        r" on change notices / delivered share of the best achievable output on change notices"
        r" / the same on quiet notices, all in percent. Delivered units are capped at the true "
        r"capacity, so over-release adds nothing; no release delivers nothing.}}"
        r"\label{tab:e1abstpol}",
        r"\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Policy & Qwen2.5 7B & Llama 3.1 8B & Phi-4 14B\\\midrule",
        *rows,
        r"\bottomrule\end{tabular}",
        r"\end{adjustwidth}",
        r"\end{table}", ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=Path, default=ROOT / "data/e1v2/cases.jsonl")
    ap.add_argument("--replies", type=Path, default=ROOT / "results/e1v2/replies.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/e1v2_abstention_policies.json")
    ap.add_argument("--table", type=Path)
    args = ap.parse_args(argv)
    res = evaluate(args.cases, args.replies)
    payload = {"development_data": True, "post_hoc": True, "fresh_model_calls": 0,
               "seed": SEED, "policies": list(POLICIES),
               "note": __doc__.split("\n\n")[2].replace("\n", " "),
               "models": res}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    if args.table:
        args.table.write_text(table(res))
    for model in MODELS:
        print(model)
        for pol in POLICIES:
            ch, qu = res[model]["change"][pol], res[model]["quiet"][pol]
            print(f"  {pol:18s} viol {pct(ch['violation']):>3}  rel {pct(ch['released']):>3}  "
                  f"delivered {100 * ch['delivered_share_mean']:3.0f}  "
                  f"quiet delivered {100 * qu['delivered_share_mean']:3.0f}  "
                  f"cf {ch['counterfactual']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
