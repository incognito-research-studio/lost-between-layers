"""Reference points for the Laya conformance results (post hoc, 16 Nov 2026; no model calls).

    python tools/laya_baselines.py

POST HOC. Not part of the frozen protocol (data/laya/PROTOCOL.md, FREEZE.json); logged as a
post hoc analysis in data/laya/DEVIATIONS.md. It reads the stored replies of the first CPU run
(results/laya_conformance/replies_cpu_r1.jsonl) and the frozen notices, and changes no frozen
file. Raised by the pre-submission review of the AI4Factory paper
(manuscripts/aaai-workshop/REVIEW-2026-11-16.md, row 7). It reports:
  A. retention of the true range by the 0.9-mass sets against random sets of the same size,
     under two nulls: a uniform set of ranges of the same size over the 56 candidate ranges,
     and independent uniform sets of the same sizes over the 11 start and 10 end options;
  B. a notice-blind prior: the k ranges most frequent among the other change notices
     (leave-one-out), with k the rounded mean size of the 0.9-mass sets;
  C. top-choice accuracy against a uniform guess over the 56 ranges;
  D. order flips and name-following stratified by the base reply's margin between its top two
     options (the smaller margin of the two questions);
  E. whether the 0.9-mass set, and its retention of the truth, changes with candidate order.
The definitions of the choice, the 0.9-mass set and a range follow tools/laya_conformance.py.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEEKS = list(range(40, 50))
QS = ("start", "end")
MASS = 0.9
ORDERS = ["reversed"] + [f"shuffle{i}" for i in range(5)]
MARGINS = (0.0, 0.05, 0.1, 0.2, 0.3)
ALL_RANGES = [("nc",)] + [(s, e) for s in range(1, 11) for e in range(10) if e >= s - 1]
assert len(ALL_RANGES) == 56


def truth(case):
    """(start option index, end option index) of plant X's true change; start 0 = no change."""
    w = sorted(int(x) for x in case["truth"].get("plant:X", {}))
    return (0, None) if not w else (WEEKS.index(w[0]) + 1, WEEKS.index(w[-1]))


def key(t):
    return ("nc",) if t[0] == 0 else t


def decision(rec, q):
    """(chosen option index, probabilities by option index) for question q."""
    by = {name: i for i, name in rec["options"][q]}
    p = [0.0] * len(by)
    for name, v in rec["reply"]["answers"][q]["probabilities"].items():
        p[by[name]] = float(v)
    return by[rec["reply"]["answers"][q]["choice"]], p


def kept(p):
    """Smallest set of options, by decreasing probability, whose mass reaches 0.9."""
    order = sorted(range(len(p)), key=lambda i: (-p[i], i))
    out, m = set(), 0.0
    for i in order:
        out.add(i)
        m += p[i]
        if m >= MASS:
            break
    return out


def ranges(ks, ke):
    out = {("nc",)} if 0 in ks else set()
    out |= {(s, e) for s in ks if s > 0 for e in ke if e >= s - 1}
    return out


def margin(rec, q):
    p = sorted(decision(rec, q)[1], reverse=True)
    return p[0] - p[1]


def evaluate(cases_path: Path, replies_path: Path) -> dict:
    cases = [json.loads(line) for line in cases_path.open()]
    recs = {}
    for line in replies_path.open():
        r = json.loads(line)
        recs[(r["case"], r["condition"])] = r
    change = [c for c in cases if c["expect"] == "change"]
    assert len(change) == 154 and all(truth(c)[0] > 0 for c in change)

    # A. retention against size-matched random sets
    held, sizes, e_uniform, e_product = 0, [], 0.0, 0.0
    for c in change:
        rec = recs[(c["id"], "base")]
        ks, ke = kept(decision(rec, "start")[1]), kept(decision(rec, "end")[1])
        rs = ranges(ks, ke)
        sizes.append(len(rs))
        held += key(truth(c)) in rs
        e_uniform += len(rs) / 56
        e_product += len(ks) / 11 * len(ke) / 10
    k = round(sum(sizes) / len(sizes))

    # B. notice-blind prior, leave-one-out
    freq = collections.Counter(key(truth(c)) for c in change)
    loo = 0
    for c in change:
        f = freq.copy()
        f[key(truth(c))] -= 1
        loo += key(truth(c)) in {r for r, _ in f.most_common(k)}

    # C. top choice
    top = sum(tuple(decision(recs[(c["id"], "base")], q)[0] for q in QS) == truth(c)
              for c in change)

    # D. flips by margin, over all 196 notices
    rows = []
    for c in cases:
        b = recs[(c["id"], "base")]
        m = min(margin(b, q) for q in QS)
        flip = any(decision(b, q)[0] != decision(recs[(c["id"], o)], q)[0]
                   for o in ORDERS for q in QS)
        sw = recs[(c["id"], "names_swapped")]
        follow = any(sw["reply"]["answers"][q]["choice"] == b["reply"]["answers"][q]["choice"]
                     and decision(sw, q)[0] != decision(b, q)[0] for q in QS)
        rows.append((m, flip, follow))
    strata = {f"margin_ge_{t}": {"notices": sum(r[0] >= t for r in rows),
                                 "order_changed_choice": sum(r[1] for r in rows if r[0] >= t),
                                 "followed_name": sum(r[2] for r in rows if r[0] >= t)}
              for t in MARGINS}
    ms = sorted(r[0] for r in rows)

    # E. set-level order sensitivity on change notices
    set_changed, retention_changed = 0, 0
    for c in change:
        t = key(truth(c))

        def state(cond, cid=c["id"]):
            r = recs[(cid, cond)]
            ks, ke = kept(decision(r, "start")[1]), kept(decision(r, "end")[1])
            return frozenset(ranges(ks, ke))

        base = state("base")
        others = [state(o) for o in ORDERS]
        set_changed += any(s != base for s in others)
        retention_changed += any((t in s) != (t in base) for s in others)

    return {
        "post_hoc": True, "fresh_model_calls": 0, "replies": replies_path.name,
        "change_notices": len(change), "distinct_true_ranges": len(freq),
        "A_retention": {"held": held, "mean_set_size": round(sum(sizes) / len(sizes), 2),
                        "random_uniform_over_56_expected": round(e_uniform, 1),
                        "random_per_question_expected": round(e_product, 1)},
        "B_notice_blind_prior": {"k": k, "held_leave_one_out": loo},
        "C_top_choice": {"right": top, "uniform_guess_expected": round(len(change) / 56, 1)},
        "D_margin": {"median": round(ms[len(ms) // 2], 3), "max": round(ms[-1], 3),
                     "strata": strata},
        "E_order_sets": {"set_changed": set_changed, "retention_changed": retention_changed},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=Path, default=ROOT / "data/e1v2/cases.jsonl")
    ap.add_argument("--replies", type=Path,
                    default=ROOT / "results/laya_conformance/replies_cpu_r1.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/laya_baselines.json")
    args = ap.parse_args(argv)
    res = evaluate(args.cases, args.replies)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
