"""dsoi-trace: print a scenario walk-through layer by layer, one row per gate rung."""

from __future__ import annotations

import argparse

from dsoi import properties as P
from dsoi.layers.l5_gate import BoxRobustGate, NominalGate, RecheckGate
from dsoi.scenarios import Scenario, run_fixture


def fmt(xs) -> str:
    return "-" if xs is None else " ".join(f"{v:6.0f}" for v in xs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsoi-trace", description=__doc__)
    ap.add_argument("scenario", nargs="?", default="S1", choices=["S1"])
    ap.add_argument("--fixture", default="correct", help="fixture name in scenario.yaml")
    args = ap.parse_args(argv)

    scn = Scenario.load(args.scenario)
    truth, theta = scn.true_state(), scn.theta()
    gates = [NominalGate(), BoxRobustGate(), RecheckGate(scn.true_reader())]
    p, first = run_fixture(scn, gates[0], args.fixture)

    print(f"{scn.spec['id']}  {scn.spec['title']}  (fixture: {args.fixture})")
    print(f"message  {scn.spec['stimulus']['text']!r}")
    for ing in p.audit.ingests:
        print(f"ingest   {ing.message_id:<22} {ing.outcome:<10} nu_I={ing.nu_i} {ing.reason}")
    weeks = " ".join(f"{w:6.0f}" for w in scn.horizon)
    state = {k[2]: (v, lo, hi) for k, v, lo, hi, _ in first.context.state}
    print(f"L3 state week   {weeks}")
    print(f"         point  {fmt(s[0] for s in state.values())}")
    print(f"         lo     {fmt(s[1] if s[1] is not None else s[0] for s in state.values())}")
    print(f"         true   {fmt(v for _, v, *_ in truth)}")
    active = first.decision.certificate["active_rows"]
    print(f"L4 x            {fmt(first.decision.x)}   active: {active}")
    print(f"L5 {'rung':<11} {'released':<{len(weeks)}}  K-prop K-exec  outcome")
    for gate in gates:
        p, r = run_fixture(scn, gate, args.fixture)
        kp = P.k_prop(r.record, p.formulation)
        ke = P.k_exec_true(r.record.x_exec, truth, theta, p.formulation)
        print(f"   {gate.rung:<11} {fmt(r.record.x_exec)}  {kp!s:<6} {ke!s:<6}  {r.gate.outcome}")
        if not ke:
            differs = len(P.interpretation_differs(r.record.state, truth))
            missed = len(P.coverage_failures(r.record.state, truth))
            print(f"   {'':<11} interpretation differs on {differs} key(s);"
                  f" interval misses truth on {missed}")
    print(f"L6 chain verifies: {p.audit.verify()}  replay reproduces: {P.c1_replay(p, 1)}"
          f"  G1: {P.g1(r.record, p.audit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
