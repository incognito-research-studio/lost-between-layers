"""Round 3, deviation 2 (data/fault_trial/round3/DEVIATIONS.md): the repaired harness with
`l1_reference` sorting by key only, nothing else changed. Post hoc.

The frozen `c_l1_reference` sorts (key, fact) pairs; when two facts share a key, Python
compares the facts themselves and raises TypeError, and `_fires` counts a raising check as
firing. Here the sort uses the key alone (stable, so ties keep store order). Every exception a
check raises is also logged, one JSON line per exception, to the file named by FT_EXC_LOG (set
by tools/ft_r3_deviation2.py), so that firings by exception can be told apart from firings by
a failed condition. A raising check still counts as firing, as in the frozen harness.
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_repaired as R  # noqa: E402  (frozen for round 3)


def c_l1_reference(run):
    store, raw = run.p.state.store, R._raw(run)
    facts = store.facts()
    for rec in R._ingests(run, "committed"):
        cap = raw.get(rec.message_id) or {}
        ops = R.reference_ops(cap.get("payload"))
        if ops is None:
            continue
        want = sorted((((o["s"], o["p"], o["vt"]), o) for o in ops if o["kind"] != "retract"),
                      key=lambda kv: kv[0])
        got = sorted((((f.s, f.p, tuple(f.vt)), f) for f in facts
                      if f.prov.span is not None and f.prov.span.message_id == rec.message_id
                      and f.tt[0] == rec.t and f.prov.component == "L1"),
                     key=lambda kv: kv[0])
        if [k for k, _ in want] != [k for k, _ in got]:
            return True
        for (_, o), (_, f) in zip(want, got, strict=True):
            if cap.get("conf") is not None and o["p"] in R.RESTRICTION_PREDICATES:
                continue
            if not hasattr(f.o, "value"):
                return True
            if ((R._q(o["value"]), R._q(o["lo"]), R._q(o["hi"]), o["unit"])
                    != (R._q(f.o.value), R._q(f.o.lo), R._q(f.o.hi), f.o.unit)):
                return True


def _swap(checks):
    return tuple(dataclasses.replace(c, fn=c_l1_reference) if c.name == "l1_reference" else c
                 for c in checks)


R.NEW_CHECKS, R.CHECKS, R.GATING = _swap(R.NEW_CHECKS), _swap(R.CHECKS), _swap(R.GATING)
R.c_l1_reference = c_l1_reference

_LOG = os.environ.get("FT_EXC_LOG")
_GATING_FLAG = "--gating" in sys.argv


def _fires(run, checks) -> set:
    out = set()
    for c in checks:
        try:
            if c.fn(run):
                out.add(c.name)
        except Exception as exc:
            out.add(c.name)
            if _LOG:
                with open(_LOG, "a") as fh:
                    fh.write(json.dumps({"patch": os.environ.get("FT_PATCH"),
                                         "gating": _GATING_FLAG, "run": getattr(run, "id", None),
                                         "check": c.name, "gate_shadow": checks is R.GATING,
                                         "exc": f"{type(exc).__name__}: {exc}"[:200]}) + "\n")
    return out


R._fires = _fires

if __name__ == "__main__":
    raise SystemExit(R.main())
