"""One escaped fault, traced from notice to execution (post hoc, 17 Nov 2026; no model calls).

    python tools/ft_r3_trace.py

Round 3's one new harmful fault, 06-orchid (L3: when a supersede closes a longer fact, the
right-hand remainder starts one week early), was not contained by the repaired gate. This
tool replays its harmful corrected-replay run, Phi-4's stored reply to notice
atleast_pct-008, repeat 2, twice: on a clean copy of src/ and on a copy with the patch
applied, each in a fresh interpreter as tools/ft_trial.py does. For both it records the
notice, the truth, the stored reply, the plant-X capacity facts after L3, the outcome of the
decision cycle, the released and executed plan, and whether execution violates the true
capacity. It also reads which checks fired on that run and whether the gate held it from
results/fault_trial_r3.json. Writes results/fault_trial_r3_trace.json.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "data/fault_trial/round3/set-3/patches/06-orchid.patch"
CASE, MODEL, REPEAT = "atleast_pct-008", "phi4:latest", "2"
RUN = f"e1v2/{MODEL}/{CASE}/r{REPEAT}"

CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import ft_harness as H
for name, go in H.e1v2_runs():
    if name != sys.argv[2]:
        continue
    run = go(); p = run.p
    facts = [{"vt": list(f.vt), "value": float(f.o.value)} for f in p.state.store.facts()
             if f.s == "plant:X" and hasattr(f.o, "value")]
    recs = p.audit.decisions()
    r = recs[-1] if recs else None
    out = {"exception": run.exception,
           "ingests": [[i.message_id, i.outcome] for i in p.audit.ingests],
           "plant_x_facts": facts,
           "outcome": p.audit.records[-1].outcome if p.audit.records else None,
           "names": run.names(r) if r else None,
           "x_released": None if r is None or r.x_released is None else [float(v) for v in r.x_released],
           "x_exec": None if r is None or r.x_exec is None else [float(v) for v in r.x_exec],
           "violates_true": None if r is None or r.x_exec is None else bool(run.truth(r.x_exec))}
    print(json.dumps(out))
'''


def run_copy(patched: bool) -> dict:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for sub in ("src", "configs", "scenarios"):
            if (ROOT / sub).exists():
                shutil.copytree(ROOT / sub, d / sub)
        if patched:
            subprocess.run(["patch", "-p1", "-s", "-i", str(PATCH)], cwd=d, check=True)
        res = subprocess.run([sys.executable, "-I", "-c",
                              f"import sys; sys.path.insert(0, {str(d / 'src')!r});" + CHILD,
                              str(ROOT / "tools"), RUN],
                             capture_output=True, text=True, check=True, cwd=ROOT)
        return json.loads(res.stdout.strip().splitlines()[-1])


def main() -> int:
    case = next(json.loads(line) for line in open(ROOT / "data/e1v2/cases.jsonl")
                if json.loads(line)["id"] == CASE)
    reply = next(json.loads(line) for line in open(ROOT / "results/e1v2/replies.jsonl")
                 if (lambda x: x["case_id"] == CASE and x["model"] == MODEL
                     and str(x["repeat"]) == REPEAT)(json.loads(line)))
    r3 = json.loads((ROOT / "results/fault_trial_r3.json").read_text())
    fault = r3["faults"]["06-orchid.patch"]
    clean, patched = run_copy(False), run_copy(True)
    week = int(next(iter(case["truth"]["plant:X"])))
    truth = case["truth"]["plant:X"][str(week)]
    i = patched["names"].index(f"x[{week}]")
    out = {"post_hoc": True, "fresh_model_calls": 0, "run": RUN, "patch": PATCH.name,
           "notice": case["messages"][0][1], "true_week": week, "true_capacity": truth,
           "reply": json.loads(reply["text"]), "clean": clean, "patched": patched,
           "executed_in_week": patched["x_exec"][i],
           "excess_units": round(patched["x_exec"][i] - truth, 2),
           "checks_fired_on_run": fault["per_run"][RUN]["fired_new"],
           "gate_holds_on_run": fault["per_run"][RUN]["holds"],
           "violations_clean_patched_gated": fault["per_run"][RUN]["violations"],
           "committed_unread_is_relative": r3["checks"]["committed_unread"]["relative"]}
    (ROOT / "results/fault_trial_r3_trace.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("notice", "true_capacity", "executed_in_week",
                                          "excess_units", "checks_fired_on_run",
                                          "gate_holds_on_run")}, indent=1))
    print("clean:", clean["exception"], clean["outcome"], "| patched:", patched["outcome"],
          patched["violates_true"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
