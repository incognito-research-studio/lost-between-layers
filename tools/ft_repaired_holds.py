"""Round 3, post hoc (15 Nov 2026): the frozen repaired harness, with each gated run also
reporting every decided record (time, class, released action, whether the gate held it and,
for a held release, whether it would have violated the true constraints).

Nothing in the checks, the gate or the runs changes: `evaluate` below is the frozen
`ft_repaired.evaluate` with one extra output field, `decided`. tools/ft_r3_holds.py runs it and
classifies the holds; the frozen results (results/fault_trial_r3.json) stay primary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_repaired as R  # noqa: E402  (frozen for round 3)

H = R.H


def evaluate(make, gating: bool = False) -> dict:
    R._CAPTURE, R._GATE_ON = {}, gating
    try:
        run = R._make_bound(make)
    except Exception as exc:
        R._CAPTURE, R._GATE_ON = None, False
        return {"crashed": True, "exception": f"{type(exc).__name__}: {exc}"[:300],
                "exception_layer": H._layer_of(exc.__traceback__), "fired": ["exception"],
                "trace": None, "released": 0, "violations": 0, "decisions": 0, "holds": [],
                "decided": []}
    run.raw = R._CAPTURE
    fired = sorted(R._fires(run, R.CHECKS), key=[c.name for c in R.CHECKS].index)
    decided = run.p.audit.decisions()
    executed = [r for r in decided if r.x_exec is not None]
    viol = 0
    for r in executed:
        try:
            viol += bool(run.truth(r.x_exec))
        except Exception:
            pass
    held_harm = 0
    detail = []
    for r in decided:
        held = (r.x_exec is None and r.x_released is not None
                and (r.admission or {}).get("monitor") == "ft-gate")
        violating = None
        if held:
            try:
                violating = bool(run.truth(r.x_released))
                held_harm += violating
            except Exception:
                violating = None
        detail.append({"t": float(r.t), "class": r.decision_class,
                       "x_released": None if r.x_released is None
                       else [float(v) for v in r.x_released],
                       "held": held, "violating": violating})
    out = {"crashed": False, "exception": run.exception, "exception_layer": run.exception_layer,
           "fired": fired, "trace": H.trace(run), "released": len(executed),
           "violations": viol, "decisions": len(decided), "holds": getattr(run, "holds", []),
           "held_violating": held_harm, "decided": detail}
    R._CAPTURE, R._GATE_ON = None, False
    return out


R.evaluate = evaluate

if __name__ == "__main__":
    raise SystemExit(R.main())
