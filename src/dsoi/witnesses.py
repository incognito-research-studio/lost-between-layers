"""Prop. 3 (formerly 5(d)): G1, R2, C1 and K-prop are mutually independent.

Product construction without a gate. A system is assembled from four components,
each with two variants, on the S1 world (capacity 700 < demand 1000 in weeks 42-44,
so demand exceeds capacity by DELTA_GAP = 300 > SHIFT_MAX):

  component   variant where the property holds   variant where it fails
  store       history-preserving      (R2)        mutable: deletes superseded history
  provenance  traced                  (G1)        untraced: lineage stripped on commit
  randomness  deterministic           (C1)        shifted down by u in [0, SHIFT_MAX)
                                                  drawn from an undeclared RNG
  plan        optimum = min(demand, capacity)     demand
              (K-prop)

The policy's output is the released action (no gate), so K-prop, which R4 states
for the policy's output and the released action, is decided by the plan alone.

Each property depends on its own component only:

  G1      Lineage is fixed at commit by the provenance component. No other
          component writes provenance, the mutable store deletes only superseded
          facts (never the current facts a decision reads), and L6 logs rho(x)
          in every variant.
  R2      (i) depends only on whether superseded facts are closed or deleted.
          Every variant commits each accepted message at once and logs an
          ingest record (ii), and every update is deterministic (iii).
          Stripping provenance changes no (s, p, o, VT, TT) field.
  C1      A deterministic policy is a function of its context, so C1 holds
          whichever plan it follows. A shifted policy subtracts a fresh draw
          u in [0, SHIFT_MAX) from every week, so its output varies on a fixed
          context and C1 fails.
  K-prop  The optimal plan lies in F(S_hat, C), and subtracting u (floored at 0)
          keeps it there. The demand plan exceeds capacity in weeks 42-44, and
          subtracting u < SHIFT_MAX < DELTA_GAP keeps it above capacity.

So all 16 combinations are realizable. What the code adds: for every property
that should fail, `observe` exhibits a concrete counterexample run (a refutation);
for every property that should hold, it checks a battery of decisions and replays
(evidence, not proof). The "holds" side rests on the argument above.
Under C1-d (equal output distributions) the shifted policies would satisfy
consistency; the witness concerns C1 as stated.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from dsoi import properties as P
from dsoi.constraints import solve_lp
from dsoi.context import VersionVector
from dsoi.layers.l1_interpreter import FixtureInterpreter
from dsoi.layers.l4_decision import Decision
from dsoi.layers.l5_gate import GateResult
from dsoi.scenarios import VERSIONS_FILE, Scenario, build_pipeline
from dsoi.store import BitemporalStore
from dsoi.types import Delta, Provenance

PROPERTIES = ("G1", "R2", "C1", "K-prop")
COMPONENT = {  # property -> (component, variant where it holds, variant where it fails)
    "G1": ("provenance", "traced", "untraced"),
    "R2": ("store", "history_preserving", "mutable"),
    "C1": ("randomness", "deterministic", "shifted"),
    "K-prop": ("plan", "optimal", "demand"),
}
SHIFT_MAX = 50.0     # u in [0, SHIFT_MAX); demand - capacity = 300 in weeks 42-44
PAIRWISE_WITNESSES = [("C1", "K-prop"), ("G1", "K-prop"), ("R2", "C1"), ("G1", "C1"),
                      ("R2", "K-prop"), ("G1", "R2")]

UNTRACED = Provenance("unknown", "unknown", "unknown", authenticated=False)


# -- components ------------------------------------------------------------------


class WitnessStore(BitemporalStore):
    """The L3 store in its four variants: history-preserving or mutable, traced or untraced."""

    def __init__(self, mutable: bool, traced: bool):
        super().__init__()
        self.mutable = mutable
        self.traced = traced

    def commit(self, delta: Delta, t: float) -> float:
        if not self.traced:
            delta = Delta(tuple(replace(op, prov=UNTRACED) for op in delta.ops))
        out = super().commit(delta, t)
        if self.mutable:  # superseded versions are deleted instead of closed
            self._facts = [f for f in self._facts if f.current]
        return out


class WitnessPolicy:
    """Plans to the optimum min(demand, capacity) or to demand (blind to C), either
    deterministically or shifted down by u in [0, SHIFT_MAX) from an undeclared RNG."""

    def __init__(self, formulation, plan: str, rng: np.random.Generator | None):
        self.formulation = formulation
        self.plan = plan
        self.rng = rng

    def decide(self, ctx):
        inst = self.formulation.build(ctx.state, ctx.theta_dict())
        if self.plan == "optimal":
            res = solve_lp(inst.c, inst.system)
            assert res.status == 0
            x = np.asarray(res.x, dtype=float)
        else:
            x = np.array(inst.system.upper, dtype=float)
        if self.rng is not None:
            x = np.maximum(x - self.rng.uniform(0.0, SHIFT_MAX), 0.0)
        return Decision(x, "optimal", inst.objective(x), {"active_rows": []}), inst


class NoGate:
    rung = "none"

    def apply(self, x, inst, ctx, formulation) -> GateResult:
        return GateResult(self.rung, "pass" if x is not None else "rejected", x)


# -- configurations ----------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    store: str
    provenance: str
    randomness: str
    plan: str

    def expected(self) -> dict[str, bool]:
        return {p: getattr(self, comp) == ok for p, (comp, ok, _) in COMPONENT.items()}

    @property
    def label(self) -> str:
        return f"{self.store}/{self.provenance}/{self.randomness}/{self.plan}"


def all_configs() -> list[Config]:
    choices = [(ok, bad) for _, ok, bad in (COMPONENT[p] for p in ("R2", "G1", "C1", "K-prop"))]
    return [Config(*combo) for combo in itertools.product(*choices)]


def witness_pair(p: str, q: str) -> tuple[Config, Config]:
    """Witness pair (p, q): a system with p but not q, and one with q but not p.

    The other two properties hold in both, so each witness differs from the
    all-properties system in exactly one component.
    """
    def only_fails(prop: str) -> Config:
        (cfg,) = [c for c in all_configs()
                  if [k for k, v in c.expected().items() if not v] == [prop]]
        return cfg
    return only_fails(q), only_fails(p)


@dataclass
class Observation:
    config: Config
    holds: dict[str, bool]
    counterexamples: dict[str, str] = field(default_factory=dict)

    @property
    def matches(self) -> bool:
        return self.holds == self.config.expected()


def _fmt(x) -> str:
    return "-" if x is None else "[" + ", ".join(f"{v:.0f}" for v in x) + "]"


def observe(cfg: Config, seed: int = 0, decisions: int = 6, replays: int = 2) -> Observation:
    """Run the configured system on S1 and check each property on its log."""
    scn = Scenario.load("S1")
    fx, msg = scn.fixture("correct"), scn.message()
    versions = VersionVector.load(VERSIONS_FILE).bump(interpreter=fx["interpreter_version"])
    store = WitnessStore(mutable=cfg.store == "mutable", traced=cfg.provenance == "traced")
    p = build_pipeline(scn, NoGate(), FixtureInterpreter({msg.message_id: fx}), versions,
                       store=store)
    rng = np.random.default_rng(seed) if cfg.randomness == "shifted" else None
    p.policy = WitnessPolicy(p.formulation, cfg.plan, rng)

    probes = {(w, 1.5): store.snapshot(w, 1.5) for w in scn.horizon}  # baseline at t = 1
    p.ingest_message(msg, scn.time("message"))
    records = [p.decide(3 + k, scn.theta()).record for k in range(decisions)]
    log, cx = p.audit, {}

    bad = [r.id for r in records if not P.g1(r, log)]
    if bad:
        cx["G1"] = f"record {bad[0]}: a supporting fact has no authenticated source or span"
    changed = [k for k, snap in probes.items() if store.snapshot(*k) != snap]
    if changed or not P.r2_ii(log, [msg.message_id]):
        cx["R2"] = f"as-of view (v, t) = {changed[0]} changed after a later commit"
    for r in records:
        for _ in range(replays):
            again = p.replay(r.id)
            if again is None or not np.allclose(again, r.x_released):
                cx["C1"] = (f"record {r.id}: same logged context released {_fmt(r.x_released)}, "
                            f"replay released {_fmt(again)}")
                break
        if "C1" in cx:
            break
    bad = [r for r in records if not P.k_prop(r, p.formulation)]
    if bad:
        cx["K-prop"] = (f"record {bad[0].id}: policy output {_fmt(bad[0].x)} and release "
                        f"{_fmt(bad[0].x_released)} exceed capacity")
    return Observation(cfg, {k: k not in cx for k in PROPERTIES}, cx)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsoi-witnesses", description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("results") / "witnesses")
    args = ap.parse_args(argv)
    obs = [observe(c) for c in all_configs()]

    mark = {True: "yes", False: "no "}
    print(f"{'store':<12}{'provenance':<11}{'randomness':<14}{'plan':<11}"
          + "".join(f"{p:<8}" for p in PROPERTIES) + "as constructed")
    for o in obs:
        c = o.config
        print(f"{c.store:<12}{c.provenance:<11}{c.randomness:<14}{c.plan:<11}"
              + "".join(f"{mark[o.holds[p]]:<8}" for p in PROPERTIES) + mark[o.matches])
    combos = {tuple(o.holds[p] for p in PROPERTIES) for o in obs}
    print(f"\ndistinct property combinations realized: {len(combos)} of 16")
    print("\nPairwise witnesses (p without q | q without p):")
    for p_, q in PAIRWISE_WITNESSES:
        a, b = witness_pair(p_, q)
        print(f"  {p_:>6} / {q:<6}  {a.label:<44} | {b.label}")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "combinations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["store", "provenance", "randomness", "plan", *PROPERTIES, "counterexamples"])
        for o in obs:
            c = o.config
            w.writerow([c.store, c.provenance, c.randomness, c.plan,
                        *(int(o.holds[p]) for p in PROPERTIES),
                        " | ".join(f"{k}: {v}" for k, v in o.counterexamples.items())])
    print(f"\nwrote {args.out}/combinations.csv")
    return 0 if all(o.matches for o in obs) and len(combos) == 16 else 1


if __name__ == "__main__":
    raise SystemExit(main())
