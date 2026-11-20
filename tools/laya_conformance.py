"""Typed-reader conformance (15 Nov 2026): does a typed-decision model meet the L1 contract?

    .venv-laya/bin/python tools/laya_conformance.py run      # model calls, writes replies
    .venv-laya/bin/python tools/laya_conformance.py analyse  # no model calls
    python tools/laya_conformance.py table \
        --out ../manuscripts/arxiv-paper/tables/laya-conformance.tex     # no model calls

Protocol: data/laya/PROTOCOL.md, frozen in data/laya/FREEZE.json before the first run. The
reader is Laya (ConvAI Innovations, Apache-2.0), package laya 0.4.1, checkpoint
convaiinnovations/laya, subfolder typed-decisions, revision 55cf4c4, zero-shot. It reads the
196 constructed notices of data/e1v2/cases.jsonl and answers three typed questions per case,
whose candidates the integration builds before the reader runs:

  start      Choice over "no change" (the value in force) and the ten planning weeks 40-49:
             the week in which the change to Plant X's weekly capacity begins
  end        Choice over the ten planning weeks: the week in which it ends
  cant_tell  companion Noul: is it impossible to tell which weeks of Plant X change? A yes
             (probability above 0.5) maps to abstention, the null of Definition 3

Every contiguous range of planning weeks is a (start, end) pair, so construction is complete
by design. A single Choice over all 56 ranges does not fit Laya's question budget; it is asked
once per case at the canonical order to record how many candidates stay distinguishable, and
its answers are not analysed.

The study measures contract properties, not reading accuracy: repeatability on one device,
agreement between CPU and Apple GPU, invariance to the order of the candidates (reversed and
five seeded shuffles), invariance to their names (neutral labels; names exchanged between
definitions), how often the null fires, and the parts of Remark 1 that a reader controls
(construction, and retention by the top choice and by the smallest set of probability mass
0.9 per question). Choices are compared by the definition chosen, not by the returned name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/e1v2/cases.jsonl"
FREEZE = ROOT / "data/laya/FREEZE.json"
OUT = ROOT / "results/laya_conformance"
REPO, SUBFOLDER = "convaiinnovations/laya", "typed-decisions"
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
WEEKS = tuple(range(40, 50))
NULL_THRESHOLD = 0.5
SET_MASS = 0.9
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)
BASE_SEED = 20261115
NO_CHANGE = "no change"
QS = ("start", "end")

PLANTS = "Plant X is the Northfield plant. Plant Y is the Riverside plant."
START_Q = "In which week does the change to Plant X's weekly capacity begin?"
END_Q = "In which week does the change to Plant X's weekly capacity end?"
WIDE_Q = "Which weeks of Plant X's weekly capacity do the messages change?"
NULL_Q = ("Is it impossible to tell from the messages which weeks of Plant X's weekly capacity "
          "change?")


def start_candidates() -> list[tuple[str, str]]:
    return [(NO_CHANGE, "The messages change no week of Plant X's capacity.")] + [
        (f"week {w}", f"The change to Plant X's capacity begins in week {w}.") for w in WEEKS]


def end_candidates() -> list[tuple[str, str]]:
    return [(f"week {w}", f"The change to Plant X's capacity ends in week {w}.") for w in WEEKS]


def wide_candidates() -> list[tuple[str, str]]:
    out = [(NO_CHANGE, "The messages change no week of Plant X's capacity.")]
    for a in WEEKS:
        for b in WEEKS:
            if b >= a:
                name = f"week {a}" if a == b else f"weeks {a}-{b}"
                out.append((name, f"Plant X's capacity changes in {name} only."))
    return out


def truth(case: dict) -> tuple[int, int | None] | None:
    """Indices of the right start and end candidates; None when abstention is right."""
    if case["expect"] == "abstain":
        return None
    weeks = sorted(int(w) for w in case["truth"].get("plant:X", {}))
    if not weeks:
        return 0, None
    return WEEKS.index(weeks[0]) + 1, WEEKS.index(weeks[-1])


def state_of(case: dict) -> dict:
    s = {"plants": PLANTS, "planning weeks": "weeks 40 to 49"}
    for i, (_, text) in enumerate(case["messages"], 1):
        s[f"message {i}"] = text
    return s


def _rng(seed: int, key: str):
    import numpy as np
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([BASE_SEED, seed, h])


def _derangement(n: int, rng) -> list[int]:
    while True:
        p = [int(x) for x in rng.permutation(n)]
        if all(p[i] != i for i in range(n)):
            return p


def presentations(cands, key: str) -> dict[str, list[tuple[int, str]]]:
    """Per condition, the presented options as (candidate index, name shown), in order."""
    n = len(cands)
    base = [(i, cands[i][0]) for i in range(n)]
    out = {"base": base, "reversed": base[::-1]}
    for s in SHUFFLE_SEEDS:
        out[f"shuffle{s}"] = [base[int(i)] for i in _rng(s, key).permutation(n)]
    labels = _rng(100, key).permutation(n)
    out["names_neutral"] = [(i, f"option {int(labels[i]) + 1:02d}") for i in range(n)]
    d = _derangement(n, _rng(200, key))
    out["names_swapped"] = [(i, cands[d[i]][0]) for i in range(n)]
    return out


CONDITIONS = ("base", "reversed", *(f"shuffle{s}" for s in SHUFFLE_SEEDS), "names_neutral",
              "names_swapped")


def _choice(options, cands, instructions) -> dict:
    return {"type": "choice", "instructions": instructions,
            "criteria": {name: cands[i][1] for i, name in options}}


def requests(case: dict) -> dict[str, tuple[dict, dict]]:
    """Condition -> (questions, the options presented per question)."""
    sc, ec = start_candidates(), end_candidates()
    ps, pe = presentations(sc, case["id"] + "/start"), presentations(ec, case["id"] + "/end")
    out = {}
    for c in CONDITIONS:
        qs = {"start": _choice(ps[c], sc, START_Q), "end": _choice(pe[c], ec, END_Q),
              "cant_tell": {"type": "noul", "instructions": NULL_Q}}
        out[c] = (qs, {"start": ps[c], "end": pe[c]})
    wc = wide_candidates()
    wide = [(i, wc[i][0]) for i in range(len(wc))]
    out["wide56"] = ({"wide": _choice(wide, wc, WIDE_Q)}, {"wide": wide})
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_cases() -> list[dict]:
    return [json.loads(line) for line in open(CASES)]


def run(args) -> int:
    import laya
    import torch
    freeze = json.loads(FREEZE.read_text())
    for rel, digest in freeze["files"].items():
        if sha256(ROOT / rel) != digest:
            raise SystemExit(f"{rel} differs from FREEZE.json; the protocol is frozen")
    if laya.__version__ != "0.4.1" or laya.PINNED_REVISIONS[REPO] != REVISION:
        raise SystemExit("laya version or pinned revision differs from the protocol")
    cases = _load_cases()
    OUT.mkdir(parents=True, exist_ok=True)
    passes = [("cpu", "r1"), ("cpu", "r2")]
    if torch.backends.mps.is_available():
        passes.append(("mps", "r1"))
    prov = {"laya": laya.__version__, "torch": torch.__version__, "repo": REPO,
            "subfolder": SUBFOLDER, "revision": REVISION, "passes": passes, "runs": {}}
    for device, rep in passes:
        agent = laya.load(REPO, subfolder=SUBFOLDER, revision=REVISION, device=device)
        path = OUT / f"replies_{device}_{rep}.jsonl"
        started, n = time.perf_counter(), 0
        with open(path, "w") as f:
            for case in cases:
                reqs = requests(case)
                if (device, rep) != ("cpu", "r1"):
                    reqs = {"base": reqs["base"]}
                for cname, (qs, options) in reqs.items():
                    r = agent.predict(state_of(case), qs)
                    n += 1
                    f.write(json.dumps({"case": case["id"], "condition": cname,
                                        "options": options, "reply": r}, sort_keys=True) + "\n")
        prov["runs"][f"{device}_{rep}"] = {"calls": n,
                                           "seconds": round(time.perf_counter() - started, 1),
                                           "sha256": sha256(path)}
        print(device, rep, n, "calls", flush=True)
    snap = Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots"
    prov["model_safetensors_sha256"] = sha256(snap / REVISION / SUBFOLDER / "model.safetensors")
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    return 0


def _read(path: Path) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["case"], r["condition"])] = r
    return out


def _decision(rec: dict, q: str) -> tuple[int, list[float]]:
    """Chosen candidate index and the probability of each candidate, by candidate index."""
    a = rec["reply"]["answers"][q]
    by_name = {name: i for i, name in rec["options"][q]}
    probs = [0.0] * len(rec["options"][q])
    for name, p in a["probabilities"].items():
        probs[by_name[name]] = float(p)
    return by_name[a["choice"]], probs


def _tv(p, q) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


def _retained(probs) -> set[int]:
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    kept, mass = set(), 0.0
    for i in order:
        kept.add(i)
        mass += probs[i]
        if mass >= SET_MASS:
            break
    return kept


def _ranges(starts: set[int], ends: set[int]) -> int:
    """Candidates (no change, or a contiguous range) that the two retained sets allow.

    Start index s > 0 is week WEEKS[s - 1]; end index e is week WEEKS[e]; a range needs e >= s - 1.
    """
    return (0 in starts) + sum(1 for s in starts if s > 0 for e in ends if e >= s - 1)


def analyse(args) -> int:
    cases = _load_cases()
    r1 = _read(OUT / "replies_cpu_r1.jsonl")
    r2 = _read(OUT / "replies_cpu_r2.jsonl")
    mps_path = OUT / "replies_mps_r1.jsonl"
    mps = _read(mps_path) if mps_path.exists() else None
    n = len(cases)
    res = {"post_hoc": False, "fresh_model_calls": 0, "cases": n,
           "null_threshold": NULL_THRESHOLD, "set_mass": SET_MASS}

    def key(rec):
        return (*(_decision(rec, q)[0] for q in QS),
                rec["reply"]["answers"]["cant_tell"]["noul"] > NULL_THRESHOLD)

    def probs(rec):
        return [p for q in QS for p in _decision(rec, q)[1]] + [
            rec["reply"]["answers"]["cant_tell"]["noul"]]

    def compare(other, label):
        same_bits = same = 0
        max_diff = 0.0
        for c in cases:
            a, b = r1[(c["id"], "base")], other[(c["id"], "base")]
            pa, pb = probs(a), probs(b)
            same_bits += pa == pb
            same += key(a) == key(b)
            max_diff = max(max_diff, *(abs(x - y) for x, y in zip(pa, pb, strict=True)))
        res[label] = {"identical_replies": same_bits, "same_decisions": same,
                      "max_abs_probability_difference": round(max_diff, 6)}

    compare(r2, "repeat_cpu")
    if mps is not None:
        compare(mps, "cpu_vs_mps")

    def changed(c, cond):
        b, o = r1[(c["id"], "base")], r1[(c["id"], cond)]
        return any(_decision(b, q)[0] != _decision(o, q)[0] for q in QS)

    inv = {}
    for cond in CONDITIONS[1:]:
        tvs = [_tv(_decision(r1[(c["id"], "base")], q)[1], _decision(r1[(c["id"], cond)], q)[1])
               for c in cases for q in QS]
        inv[cond] = {"flips": sum(changed(c, cond) for c in cases),
                     "mean_tv": round(sum(tvs) / len(tvs), 4)}
    res["invariance"] = inv
    orders = [k for k in CONDITIONS if k == "reversed" or k.startswith("shuffle")]
    res["any_order_flip"] = sum(any(changed(c, k) for k in orders) for c in cases)
    followed = 0
    for c in cases:
        b, s = r1[(c["id"], "base")], r1[(c["id"], "names_swapped")]
        followed += any(s["reply"]["answers"][q]["choice"] == b["reply"]["answers"][q]["choice"]
                        and _decision(s, q)[0] != _decision(b, q)[0] for q in QS)
    res["swapped_followed_name"] = followed
    wide = [r1[(c["id"], "wide56")]["reply"]["usage"].get("options", {}).get("wide")
            for c in cases]
    res["wide56"] = {"options": len(wide_candidates()),
                     "cases_collapsed": sum(w is not None for w in wide),
                     "distinct_min": min((w["distinct"] for w in wide if w), default=None),
                     "distinct_max": max((w["distinct"] for w in wide if w), default=None),
                     "tokens_per_option": sorted({w["tokens_per_option"] for w in wide if w})}
    res["collapsed_small_questions"] = sum(
        bool(r["reply"]["usage"].get("options")) for (_, cond), r in r1.items()
        if cond != "wide56")
    res["truncated_calls"] = sum(bool(r["reply"]["usage"].get("truncated")) for r in r1.values())
    by = {}
    for expect in ("change", "no_change", "abstain"):
        cs = [c for c in cases if c["expect"] == expect]
        row = {"cases": len(cs), "null": 0, "constructed": 0, "top_right": 0, "set_right": 0,
               "ranges_kept": 0}
        for c in cs:
            rec = r1[(c["id"], "base")]
            row["null"] += rec["reply"]["answers"]["cant_tell"]["noul"] > NULL_THRESHOLD
            (s, ps), (e, pe) = _decision(rec, "start"), _decision(rec, "end")
            ks, ke = _retained(ps), _retained(pe)
            row["ranges_kept"] += _ranges(ks, ke)
            t = truth(c)
            if t is None:
                continue
            ts, te = t
            row["constructed"] += 1
            row["top_right"] += s == ts and (te is None or e == te)
            row["set_right"] += ts in ks and (te is None or te in ke)
        row["mean_ranges_kept"] = round(row.pop("ranges_kept") / len(cs), 2)
        by[expect] = row
    res["by_expect"] = by
    (OUT.parent / "laya_conformance.json").write_text(json.dumps(res, indent=1,
                                                                 sort_keys=True) + "\n")
    print(json.dumps(res, indent=1))
    return 0


def table(args) -> int:
    r = json.loads((OUT.parent / "laya_conformance.json").read_text())
    n, inv, by = r["cases"], r["invariance"], r["by_expect"]
    ch, nc, ab = by["change"], by["no_change"], by["abstain"]
    sh = [inv[k]["flips"] for k in inv if k.startswith("shuffle")]
    m = n - ab["cases"]
    w = r["wide56"]
    kept = (f"{w['distinct_min']} to {w['distinct_max']} of {w['options']}"
            if w["distinct_min"] is not None else f"{w['options']} of {w['options']}")
    rows = [("Repeat on CPU: identical probabilities",
             f"{r['repeat_cpu']['identical_replies']}/{n}")]
    if "cpu_vs_mps" in r:
        g = r["cpu_vs_mps"]
        rows += [("CPU and Apple GPU: identical probabilities", f"{g['identical_replies']}/{n}"),
                 ("CPU and Apple GPU: same decisions", f"{g['same_decisions']}/{n}"),
                 ("CPU and Apple GPU: largest probability change",
                  f"{g['max_abs_probability_difference']:.4f}")]
    rows += [("Reversed order: changed choice", f"{inv['reversed']['flips']}/{n}"),
             ("Five shuffles: changed choice, per shuffle", f"{min(sh)} to {max(sh)} of {n}"),
             ("Any of the six orders: changed choice", f"{r['any_order_flip']}/{n}"),
             ("Neutral names: changed choice", f"{inv['names_neutral']['flips']}/{n}"),
             ("Names exchanged: changed choice", f"{inv['names_swapped']['flips']}/{n}"),
             ("Names exchanged: kept the name, not the definition",
              f"{r['swapped_followed_name']}/{n}"),
             ("Null fired: vague notices", f"{ab['null']}/{ab['cases']}"),
             ("Null fired: change notices", f"{ch['null']}/{ch['cases']}"),
             ("Null fired: no-change notices", f"{nc['null']}/{nc['cases']}"),
             ("One choice over all 56 ranges: options kept apart", kept),
             ("(i) Construction: right candidate offered",
              f"{ch['constructed'] + nc['constructed']}/{m}"),
             ("(ii) Retention, top choices", f"{ch['top_right'] + nc['top_right']}/{m}"),
             (f"(ii) Retention, sets of mass {r['set_mass']}",
              f"{ch['set_right'] + nc['set_right']}/{m}"),
             ("Mean candidates those sets keep",
              f"{ch['mean_ranges_kept']} / {nc['mean_ranges_kept']}")]
    lines = ["%% Generated by replication-package/tools/laya_conformance.py; do not edit by hand.",
             r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\caption{\cdraft{Conformance of a typed-decision reader to the L1 contract: Laya "
             r"0.4.1, zero-shot, on the " + str(n) + r" constructed notices (protocol frozen "
             r"before the first run, Appendix~\ref{app:laya}). A change is read as two choices, "
             r"its first and its last week, and choices are compared by the definition chosen. "
             r"Retention counts the change and no-change notices; candidates kept are given for "
             r"change / no-change notices, of 56.}}\label{tab:layaconf}",
             r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Property & Result\\\midrule"]
    lines += [f"{a} & {b}\\\\" for a, b in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path(args.out).write_text("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("run", "analyse", "table"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    return {"run": run, "analyse": analyse, "table": table}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
