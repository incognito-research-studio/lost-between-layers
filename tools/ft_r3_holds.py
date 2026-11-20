"""Round 3, post hoc (15 Nov 2026): what each hold of the gated repaired configuration stopped.

    uv run python tools/ft_r3_holds.py --faults DIR --episodes DIR \
        --out results/fault_trial_r3_holds.json

Runs the clean pipeline and every round-3 patch once with the gate on, through
tools/ft_repaired_holds.py (the frozen harness with per-decision output), and classifies every
hold by three facts:
  violating   whether the held release would have violated the true constraints
  vs_clean    whether the held release equals what the clean gated pipeline released at the
              same decision ("same") or differs from it ("changed"), whether the clean gated
              pipeline held that decision too ("clean_held"), or whether the clean run made no
              such decision ("no_clean")
  checks      which gating checks fired, and the kind of each (KIND below)
Before writing, it checks that the holds, their checks and the held-violating counts equal the
frozen results/fault_trial_r3.json, so the classification is of the frozen run's holds.

KIND sorts the gating checks by what they compare. "integrity" checks test a requirement on the
log, approvals, schema or versions that holds whatever the true constraints are; "state"
checks compare the store or the decision's inputs with the sources or the replies; "release"
checks test the released action against the stored intervals.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_round3  # noqa: E402  (frozen for round 3)

ROOT = ft_round3.ROOT
ft_round3.HARNESS = ROOT / "tools/ft_repaired_holds.py"

KIND = {
    "version": "integrity", "version_pinned": "integrity", "admission": "integrity",
    "l2_authority": "integrity", "schema": "integrity", "blocked_read": "integrity",
    "C1": "integrity", "C2": "integrity", "G1": "integrity", "G2": "integrity",
    "R2": "integrity", "audit": "integrity", "fields": "integrity",
    "l1_reference": "state", "precedence": "state", "commit_precedence": "state",
    "span_cover": "state", "interval_store": "state", "restart": "state",
    "restart_content": "state", "replay_store": "state", "unit_declared": "state",
    "committed_unread": "state",
    "robust_release": "release", "recheck_complete": "release", "kprop": "release",
    "exec_fidelity": "release", "action": "release",
}


def _key_decisions(decided):
    """(t, class, occurrence) -> record, so a decision is matched to the clean run's."""
    seen, out = Counter(), {}
    for d in decided:
        k = (d["t"], d["class"])
        out[(*k, seen[k])] = d
        seen[k] += 1
    return out


def _close(a, b):
    return a is not None and b is not None and len(a) == len(b) and all(
        abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y)) for x, y in zip(a, b, strict=True))


def classify(clean: dict, faulty: dict) -> list[dict]:
    rows = []
    for run_id, run in faulty["runs"].items():
        if not run.get("decided"):
            continue
        base = _key_decisions(clean["runs"].get(run_id, {}).get("decided") or [])
        held = [(k, d) for k, d in _key_decisions(run["decided"]).items() if d["held"]]
        if len(held) != len(run["holds"]):
            raise SystemExit(f"{run_id}: {len(held)} held records, {len(run['holds'])} holds")
        for (k, d), h in zip(held, run["holds"], strict=True):
            if d["t"] != h["t"]:
                raise SystemExit(f"{run_id}: held record at {d['t']}, hold at {h['t']}")
            c = base.get(k)
            if c is not None and c["held"]:
                vs = "clean_held"
            elif c is None or c["x_released"] is None:
                vs = "no_clean"
            else:
                vs = "same" if _close(c["x_released"], d["x_released"]) else "changed"
            kinds = sorted({KIND.get(x, "unknown") for x in h["checks"]})
            rows.append({"run": run_id, "t": d["t"], "class": d["class"], "checks": h["checks"],
                         "kinds": kinds, "violating": d["violating"], "vs_clean": vs})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--faults", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, default=ROOT / "results/fault_trial_r3.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    frozen = json.loads(args.frozen.read_text())
    with tempfile.TemporaryDirectory() as d:
        kept = Path(d) / "episodes"
        kept.mkdir()
        bad = set(frozen["episodes"]["excluded_unbuildable"])
        for p in sorted(args.episodes.iterdir()):
            if (p / "scenario.yaml").exists() and p.name not in bad:
                shutil.copytree(p, kept / p.name)
        extra = ["--episodes", str(kept), "--gating"]
        g0 = ft_round3.in_copy(None, extra)
        if g0["status"] != "ran":
            raise SystemExit("the clean pipeline failed")
        clean = g0["result"]
        frozen_clean = frozen["clean"]["holds_with_gate"]
        got_clean = {k: v["holds"] for k, v in clean["runs"].items() if v["holds"]}
        if got_clean != frozen_clean:
            raise SystemExit("clean gated holds differ from the frozen run")
        out, problems = {}, []
        for patch in sorted(args.faults.glob("*.patch")):
            g = ft_round3.in_copy(patch, extra)
            if g["status"] != "ran":
                raise SystemExit(f"{patch.name}: {g['status']}")
            f = frozen["faults"][patch.name]
            got = {k: v["holds"] for k, v in g["result"]["runs"].items() if v["holds"]}
            want = {k: v["holds"] for k, v in f["per_run"].items() if v["holds"]}
            if got != want:
                problems.append(f"{patch.name}: holds differ")
            hv = sum(v["held_violating"] for v in g["result"]["runs"].values())
            if hv != f["harm"]["held_violating"]:
                want_hv = f["harm"]["held_violating"]
                problems.append(f"{patch.name}: held_violating {hv} != {want_hv}")
            out[patch.name] = classify(clean, g["result"])
            print(patch.name, len(out[patch.name]), flush=True)
    if problems:
        raise SystemExit("does not reproduce the frozen holds:\n" + "\n".join(problems))
    rows = [{"patch": p, **r} for p, rs in out.items() for r in rs]
    tally = defaultdict(Counter)
    for r in rows:
        tally["violating"][str(r["violating"])] += 1
        tally["vs_clean"][f"{r['violating']}/{r['vs_clean']}"] += 1
        tally["kinds"][f"{r['violating']}/{'+'.join(r['kinds'])}"] += 1
    res = {"post_hoc": True, "fresh_model_calls": 0, "reproduces_frozen_holds": True,
           "kind": KIND, "summary": {k: dict(sorted(v.items())) for k, v in tally.items()},
           "holds": rows}
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
