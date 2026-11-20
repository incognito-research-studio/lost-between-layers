"""Overlap of each round's fault patches with the earlier rounds' (post hoc, 13 Nov 2026).

    uv run python tools/ft_overlap.py --out results/fault_trial_overlap.json

Written after a check of round 3 found that some round-3 patches repeat
patches of rounds 1 and 2, on which the repaired checks were developed. Nothing in the frozen
instruments changes; this script only reads the patches and the stored results.

Classification of a patch P against an earlier patch Q (both unified diffs of src/):
  identical  P and Q change the same file with the same removed and added lines, after
             stripping whitespace and trailing comments;
  same line  not identical, but they change the same file and remove at least one common
             line, or (for patches that only add lines) insert after the same context line;
  none       otherwise.
A patch's class is the closest it reaches against any earlier patch. Round 2 is compared with
round 1; round 3 with rounds 1 and 2.

For round 3, the detection counts, the exact McNemar test of `repaired` against `full`, the
harmful runs and their containment, and the fixed-world/generated-episode split are reported
for each class, with and without deviation 1 (results/fault_trial_r3_dev1.json).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "data/fault_trial"
SETS = {1: FT / "set-1/patches", 2: FT / "round2/set-2/patches", 3: FT / "round3/set-3/patches"}
RANK = {"identical": 2, "same_line": 1, "none": 0}


def _norm(line: str) -> str:
    line = re.sub(r"\s+#.*$", "", line)
    return re.sub(r"\s+", "", line)


def parse(path: Path) -> dict:
    """Per file: removed lines, added lines and, for each hunk, the context line before it."""
    files: dict[str, dict] = {}
    cur = None
    prev_ctx = None
    for raw in path.read_text().splitlines():
        if raw.startswith("+++ "):
            name = raw[4:].removeprefix("b/")
            cur = files.setdefault(name, {"rm": Counter(), "add": Counter(), "anchors": set()})
            prev_ctx = None
        elif raw.startswith(("--- ", "diff ", "index ")):
            continue
        elif raw.startswith("@@"):
            prev_ctx = None
        elif cur is None:
            continue
        elif raw.startswith("-"):
            n = _norm(raw[1:])
            if n:
                cur["rm"][n] += 1
        elif raw.startswith("+"):
            n = _norm(raw[1:])
            if n:
                cur["add"][n] += 1
                if prev_ctx is not None:
                    cur["anchors"].add(prev_ctx)
        else:
            n = _norm(raw[1:] if raw.startswith(" ") else raw)
            if n:
                prev_ctx = n
    return files


def relation(p: dict, q: dict) -> str:
    if set(p) == set(q) and all(p[f]["rm"] == q[f]["rm"] and p[f]["add"] == q[f]["add"]
                                for f in p):
        return "identical"
    for f in set(p) & set(q):
        if set(p[f]["rm"]) & set(q[f]["rm"]):
            return "same_line"
        if not p[f]["rm"] and p[f]["anchors"] & (q[f]["anchors"] | set(q[f]["rm"])):
            return "same_line"
    return "none"


def classify(round_no: int) -> dict:
    earlier = {f"r{r}:{q.stem}": parse(q) for r in range(1, round_no)
               for q in sorted(SETS[r].glob("*.patch"))}
    out = {}
    for p in sorted(SETS[round_no].glob("*.patch")):
        pp = parse(p)
        rel = {k: relation(pp, q) for k, q in earlier.items()}
        best = max(rel.values(), key=RANK.get, default="none")
        out[p.stem] = {"class": best,
                       "counterparts": sorted(k for k, v in rel.items() if v == best != "none")}
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p, den = k / n, 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)]


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


CONFIGS = ("solver", "generic", "full", "repaired")


def split(entry: dict, checks: dict, cfg: str) -> tuple[bool, bool]:
    """Whether a check of `cfg` (or a smaller configuration) newly fired on a fixed world
    (e1v2 replay, scenarios) and on a generated episode, from the stored per-run detail."""
    fixed = gen = False
    for run, v in entry["per_run"].items():
        if not any(CONFIGS.index(checks[c]["config"]) <= CONFIGS.index(cfg)
                   for c in v.get("fired_new", [])):
            continue
        if run.split("/")[0] in ("e1v2", "scenarios"):
            fixed = True
        elif run.startswith("episodes"):
            gen = True
    return fixed, gen


def summarize(names: list[str], faults: dict, dev1: dict, checks: dict) -> dict:
    def det(src, n, cfg):
        return bool(src[n + ".patch"][cfg]["detected"])
    full = [det(faults, n, "full") for n in names]
    rep = [det(faults, n, "repaired") for n in names]
    rep1 = [det(dev1, n, "repaired") for n in names]
    k = len(names)
    harm = [faults[n + ".patch"]["harm"] for n in names]
    out = {"faults": k, "names": names,
           "full": {"detected": sum(full), "wilson": wilson(sum(full), k)},
           "repaired": {"detected": sum(rep), "wilson": wilson(sum(rep), k)},
           "repaired_dev1": {"detected": sum(rep1), "wilson": wilson(sum(rep1), k)}}
    for key, r in (("repaired_vs_full", rep), ("repaired_dev1_vs_full", rep1)):
        b = sum(x and not y for x, y in zip(r, full))
        c = sum(y and not x for x, y in zip(r, full))
        out[key] = {"repaired_only": b, "full_only": c, "mcnemar_p": round(mcnemar(b, c), 4)}
    out["harm"] = {"harmful_faults": sum(h["harmful_runs"] > 0 for h in harm),
                   "harmful_runs": sum(h["harmful_runs"] for h in harm),
                   "contained": sum(h["harmful_runs_contained"] for h in harm),
                   "fully_contained_faults": sum(0 < h["harmful_runs"] == h["harmful_runs_contained"]
                                                 for h in harm)}
    for cfg in ("full", "repaired"):
        sp = [split(faults[n + ".patch"], checks, cfg) for n in names]
        out[cfg]["detected_on_fixed_worlds"] = sum(fx for fx, _ in sp)
        out[cfg]["detected_only_on_generated"] = sum(gen and not fx for fx, gen in sp)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    r2, r3 = classify(2), classify(3)
    res = json.loads((ROOT / "results/fault_trial_r3.json").read_text())
    dev1 = json.loads((ROOT / "results/fault_trial_r3_dev1.json").read_text())["faults"]
    faults = res["faults"]
    by_class = {c: sorted(n for n, v in r3.items() if v["class"] == c) for c in RANK}
    out = {"post_hoc": True,
           "method": __doc__.split("Classification", 1)[1].split("For round 3", 1)[0].strip(),
           "round2_vs_round1": {"counts": dict(Counter(v["class"] for v in r2.values())),
                                "patches": r2},
           "round3_vs_rounds1_2": {"counts": dict(Counter(v["class"] for v in r3.values())),
                                   "patches": r3},
           "round3_by_class": {c: summarize(ns, faults, dev1, res["checks"]) for c, ns in by_class.items()},
           "round3_all": summarize(sorted(r3), faults, dev1, res["checks"])}
    for name in sorted(r3):
        v = faults[name + ".patch"]
        r3[name].update(full=v["full"]["detected"], repaired=v["repaired"]["detected"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"round2": out["round2_vs_round1"]["counts"],
                      "round3": out["round3_vs_rounds1_2"]["counts"],
                      **{c: {k: s[k] for k in ("full", "repaired", "repaired_dev1",
                                               "repaired_vs_full", "repaired_dev1_vs_full",
                                               "harm")}
                         for c, s in out["round3_by_class"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
