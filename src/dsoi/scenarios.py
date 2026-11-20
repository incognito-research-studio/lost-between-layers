"""Load ATAM-style scenarios (Table IV) and run them through a seeded pipeline.

A scenario names a decision model (`model:` in scenario.yaml) and either a single
stimulus (S1) or a list of timed `events`: messages through L1, authenticated
records through L2, and decisions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from dsoi.canon import CanonicalState, DecisionClass, canonicalize
from dsoi.context import VersionVector
from dsoi.layers.l1_interpreter import FixtureInterpreter, Interpreter, InterpreterSchema
from dsoi.layers.l2_procedural import RESTRICTION_PREDICATES, ProceduralLayer, Procedure
from dsoi.layers.l3_state import Signature, StateLayer, UnresolvedConflict
from dsoi.layers.l4_decision import (
    CapacityPlanFormulation,
    LotAllocationFormulation,
    SupplierMixFormulation,
    WarehouseFlowFormulation,
)
from dsoi.layers.l5_gate import Gate
from dsoi.pipeline import CycleResult, Executor, Pipeline
from dsoi.store import BitemporalStore
from dsoi.types import Delta, Fact, Message, Op, OpKind, Provenance, Quantity

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = Path(os.environ.get("DSOI_SCENARIOS", REPO_ROOT / "scenarios"))
VERSIONS_FILE = Path(os.environ.get("DSOI_VERSIONS", REPO_ROOT / "configs" / "versions.yaml"))

SCENARIO_DIRS = {
    "S1": "s1_capacity_cut",
    "S2": "s2_quality_hold",
    "S3": "s3_supplier_suspension",
    "S4": "s4_contradictory_notices",
    "S5": "s5_injected_approval",
    "S6": "s6_model_update",
    "S7": "s7_late_correction",
    "S8": "s8_pallet_conversion",  # artifact extension: coefficient uncertainty
}

TRUTH = Provenance("truth", "world", "-", True)


def _erp(source_id: str) -> Provenance:
    return Provenance(source_id, "erp", "erp-connector@1", True)


# -- decision models -------------------------------------------------------------


class CapacityModel:
    """Weekly production plan under plant capacity (S1, S4, S6, S7)."""

    def __init__(self, world: dict):
        self.w = world
        self.subject = f"plant:{world['plant']}"
        self.horizon = tuple(float(v) for v in world["horizon"])
        self.formulation = CapacityPlanFormulation(world["plant"], self.horizon)

    def decision_class(self) -> DecisionClass:
        return DecisionClass("weekly-production-plan", frozenset({"capacity"}),
                             frozenset({self.subject}), self.horizon)

    def schema(self) -> InterpreterSchema:
        return InterpreterSchema(frozenset({"capacity"}), frozenset({self.subject}))

    def theta(self) -> dict:
        return {"demand": {w: float(self.w["demand_per_week"]) for w in self.horizon}}

    def baseline_delta(self) -> Delta:
        vt = (min(self.horizon), max(self.horizon) + 1)
        q = Quantity(float(self.w["nominal_capacity"]), unit="units/week")
        return Delta((Op(OpKind.ASSERT, self.subject, "capacity", vt,
                         _erp(self.w["erp_source_id"]), q),))

    def true_capacity(self, w: float) -> float:
        true = {float(k): float(v) for k, v in self.w.get("true_capacity", {}).items()}
        return true.get(w, float(self.w["nominal_capacity"]))

    def true_values(self) -> dict[tuple, float]:
        return {(self.subject, "capacity", w): self.true_capacity(w) for w in self.horizon}

    def system_of_record(self, key: tuple) -> float:
        return self.true_capacity(float(key[2]))


class _PeriodModel:
    """Shared logic for single-period models whose entities carry attributes."""

    entity: str          # "lot" | "supplier"
    collection: str      # world key holding the entities
    attrs: tuple[str, ...]

    def __init__(self, world: dict):
        self.w = world
        self.period = float(world["period"])
        self.names = tuple(sorted(world[self.collection]))
        self.subjects = frozenset(f"{self.entity}:{n}" for n in self.names)

    def baseline_delta(self) -> Delta:
        vt = (self.period, self.period + 1)
        prov = _erp(self.w["erp_source_id"])
        return Delta(tuple(
            Op(OpKind.ASSERT, f"{self.entity}:{n}", a, vt, prov,
               Quantity(float(self.w[self.collection][n][a])))
            for n in self.names for a in self.attrs
        ))

    def true_values(self) -> dict[tuple, float]:
        true = self.w.get("truth", {})  # not "true": YAML 1.1 reads that key as a bool
        return {
            (f"{self.entity}:{n}", a, self.period):
                float(true.get(n, {}).get(a, self.w[self.collection][n][a]))
            for n in self.names for a in self.attrs
        }

    def system_of_record(self, key: tuple) -> float:
        return self.true_values()[key]


class LotModel(_PeriodModel):
    """Lot allocation under quality holds (S2, S5)."""

    entity, collection, attrs = "lot", "lots", ("qty", "hold")

    def __init__(self, world: dict):
        super().__init__(world)
        self.formulation = LotAllocationFormulation(self.names, self.period)

    def decision_class(self) -> DecisionClass:
        return DecisionClass("lot-allocation", frozenset(self.attrs), self.subjects,
                             (self.period,))

    def schema(self) -> InterpreterSchema:
        return InterpreterSchema(frozenset(self.attrs), self.subjects)

    def theta(self) -> dict:
        return {"demand": float(self.w["demand"])}


class SupplierModel(_PeriodModel):
    """Supplier mix under suspensions (S3). The vendor master may be stale."""

    entity, collection, attrs = "supplier", "suppliers", ("capacity", "suspended")

    def __init__(self, world: dict):
        super().__init__(world)
        self.formulation = SupplierMixFormulation(self.names, self.period)

    def decision_class(self) -> DecisionClass:
        return DecisionClass("supplier-mix", frozenset(self.attrs), self.subjects,
                             (self.period,))

    def schema(self) -> InterpreterSchema:
        return InterpreterSchema(frozenset({"suspended"}), self.subjects)

    def theta(self) -> dict:
        return {"demand": float(self.w["demand"]), "penalty": float(self.w["penalty"]),
                "cost": {n: float(self.w["suppliers"][n]["cost"]) for n in self.names}}

    def system_of_record(self, key: tuple) -> float:
        """A stale vendor master still shows the baseline suspension flags (A-ORACLE fails)."""
        if self.w.get("master_stale") and key[1] == "suspended":
            name = key[0].split(":", 1)[1]
            return float(self.w["suppliers"][name]["suspended"])
        return self.true_values()[key]


class WarehouseModel:
    """Pallet flows at a DC under an interpreted units-per-pallet factor (S8)."""

    attrs = ("units_per_pallet", "free_space")

    def __init__(self, world: dict):
        self.w = world
        self.period = float(world["period"])
        self.subject = f"dc:{world['dc']}"
        self.formulation = WarehouseFlowFormulation(world["dc"], self.period,
                                                    exact_fill=world.get("space_rule") == "exactly")

    def decision_class(self) -> DecisionClass:
        return DecisionClass("dc-flow-plan", frozenset(self.attrs), frozenset({self.subject}),
                             (self.period,))

    def schema(self) -> InterpreterSchema:
        return InterpreterSchema(frozenset({"units_per_pallet"}), frozenset({self.subject}))

    def theta(self) -> dict:
        return {k: float(self.w[k]) for k in ("in_max", "out_max", "store_demand")}

    def baseline_delta(self) -> Delta:
        vt = (self.period, self.period + 1)
        prov = _erp(self.w["erp_source_id"])
        return Delta(tuple(Op(OpKind.ASSERT, self.subject, a, vt, prov,
                              Quantity(float(self.w[a]))) for a in self.attrs))

    def true_values(self) -> dict[tuple, float]:
        truth = self.w.get("truth", {})
        return {(self.subject, a, self.period): float(truth.get(a, self.w[a]))
                for a in self.attrs}

    def system_of_record(self, key: tuple) -> float:
        return self.true_values()[key]


MODELS = {"capacity_plan": CapacityModel, "lot_allocation": LotModel,
          "supplier_mix": SupplierModel, "warehouse_flow": WarehouseModel}


# -- scenarios -------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    root: Path
    spec: dict

    @classmethod
    def load(cls, scenario_id: str) -> Scenario:
        root = SCENARIO_ROOT / SCENARIO_DIRS[scenario_id]
        with open(root / "scenario.yaml") as fh:
            return cls(root, yaml.safe_load(fh))

    def with_world(self, **updates) -> Scenario:
        """A copy with world entries replaced (e.g. a simulated true_capacity)."""
        return Scenario(self.root, {**self.spec, "world": {**self.spec["world"], **updates}})

    @property
    def world(self) -> dict:
        return self.spec["world"]

    @property
    def model(self):
        return MODELS[self.spec.get("model", "capacity_plan")](self.world)

    # Capacity-model conveniences used by S1 code and tests.
    @property
    def horizon(self) -> tuple[float, ...]:
        return self.model.horizon

    @property
    def subject(self) -> str:
        return self.model.subject

    def true_capacity(self, w: float) -> float:
        return self.model.true_capacity(w)

    # Generic interface.
    def time(self, name: str) -> float:
        return float(self.spec["times"][name])

    def theta(self) -> dict:
        return self.model.theta()

    def decision_class(self) -> DecisionClass:
        return self.model.decision_class()

    def schema(self) -> InterpreterSchema:
        return self.model.schema()

    def baseline_delta(self) -> Delta:
        return self.model.baseline_delta()

    def message(self, text: str | None = None) -> Message:
        st = self.spec["stimulus"]
        return Message(st["message_id"], st["source_id"], text or st["text"])

    def true_state(self) -> CanonicalState:
        """S* restricted to D, exact values (no intervals)."""
        facts = {key: Fact(key[0], key[1], Quantity(v), (key[2], key[2] + 1),
                           (0, float("inf")), TRUTH)
                 for key, v in self.model.true_values().items()}
        return canonicalize(facts, self.decision_class())

    def true_reader(self) -> Callable[[tuple], float]:
        """System-of-record re-read for the recheck gate (A-ORACLE, unless it is stale)."""
        model = self.model
        return model.system_of_record

    def fixture(self, name: str) -> dict:
        with open(self.root / self.spec["fixtures"][name]) as fh:
            return json.load(fh)

    def load_json(self, relpath: str) -> dict:
        with open(self.root / relpath) as fh:
            return json.load(fh)


def build_pipeline(scn: Scenario, gate: Gate, interpreter: Interpreter | None = None,
                   versions: VersionVector | None = None,
                   resolution: str | None = None,
                   store: BitemporalStore | None = None,
                   executor: Executor | None = None,
                   threshold_rules: tuple = ()) -> Pipeline:
    """A pipeline whose store already holds the authenticated baseline (t = erp_load).

    `store` swaps the L3 store implementation (used by the Prop. 3 witnesses);
    `executor` swaps the execution systems (execution-fidelity studies);
    `threshold_rules` makes L2 hold back threshold actions without an approval event.
    """
    model = scn.model
    procedures = tuple(
        Procedure(s["name"], {(p, float(v)): node for p, v, node in s["steps"]},
                  dict(s.get("requires", {})))
        for s in scn.spec.get("sop", []))
    versions = (versions or VersionVector.load(VERSIONS_FILE)).bump(
        formulation=model.formulation.version)
    if resolution is not None:
        versions = versions.bump(resolution=f"{resolution}@1")
    p = Pipeline(
        interpreter=interpreter or FixtureInterpreter({}),
        schema=model.schema(),
        procedural=ProceduralLayer(versions.sop, threshold_rules=threshold_rules,
                                   lift_issuers={k: frozenset(v) for k, v in
                                                 scn.spec.get("lift_issuers", {}).items()},
                                   procedures=procedures),
        state=StateLayer(store=store, source_priority=scn.spec.get("source_priority", {}),
                         resolution=versions.resolution.split("@")[0],
                         signature=Signature.from_model(
                             model.baseline_delta(), model.decision_class().reads,
                             RESTRICTION_PREDICATES | {p for proc in procedures
                                                       for p, _ in proc.steps})),
        formulation=model.formulation,
        gate=gate,
        versions=versions,
        decision_class=model.decision_class(),
        executor=executor,
    )
    p.ingest_record(model.baseline_delta(), scn.time("erp_load"), scn.world["erp_source_id"])
    return p


def run_fixture(scn: Scenario, gate: Gate, fixture: str = "correct",
                executor: Executor | None = None) -> tuple[Pipeline, CycleResult]:
    """S1-style: ingest the stimulus with a recorded interpreter output, then decide once."""
    fx = scn.fixture(fixture)
    message = scn.message()
    base = VersionVector.load(VERSIONS_FILE)
    versions = base.bump(interpreter=fx["interpreter_version"])  # pin nu_I to the fixture
    p = build_pipeline(scn, gate, FixtureInterpreter({message.message_id: fx}), versions,
                       executor=executor)
    p.ingest_message(message, scn.time("message"))
    return p, p.decide(scn.time("decide"), scn.theta())


@dataclass(frozen=True)
class Deferred:
    """A decision deferred because L3 holds an unresolved conflict (S4, block policy)."""

    t: float
    reason: str


def delta_from_spec(ops: list[dict], source_id: str) -> Delta:
    """An authenticated structured record (ERP update, approval event)."""
    prov = Provenance(source_id, "record", "connector@1", True)
    out = []
    for o in ops:
        kind = OpKind(o["op"])
        value = None if kind is OpKind.RETRACT else Quantity(
            float(o["value"]), o.get("lo"), o.get("hi"), o.get("unit", ""))
        out.append(Op(kind, o["subject"], o["predicate"],
                      (float(o["valid_from"]), float(o["valid_to"])), prov, value))
    return Delta(tuple(out))


def run_events(scn: Scenario, gate: Gate, fixtures: dict[str, str] | None = None,
               times: dict[str, float] | None = None, resolution: str | None = None,
               interpreter_version: str | None = None, payloads: dict[str, dict] | None = None,
               texts: dict[str, str] | None = None,
               ) -> tuple[Pipeline, list[CycleResult | Deferred]]:
    """Run the scenario's event list.

    Per event id: `fixtures` swaps the fixture file, `payloads` supplies a generated
    interpreter output directly (simulation), `texts` replaces a message's text,
    and `times` moves the event.
    """
    fixtures, times, given, texts = fixtures or {}, times or {}, payloads or {}, texts or {}
    events = [{**e, "t": float(times.get(e.get("id", ""), e["t"]))} for e in scn.spec["events"]]
    payloads = {e["id"]: given.get(e["id"]) or scn.load_json(fixtures.get(e["id"], e["fixture"]))
                for e in events if e["kind"] == "message"}
    base = VersionVector.load(VERSIONS_FILE).bump(
        interpreter=interpreter_version or scn.spec["interpreter_version"])
    p = build_pipeline(scn, gate, FixtureInterpreter(payloads), base, resolution)
    results: list[CycleResult | Deferred] = []
    for e in sorted(events, key=lambda e: e["t"]):
        if e["kind"] == "message":
            p.ingest_message(Message(e["id"], e["source"], texts.get(e["id"], e["text"])), e["t"])
        elif e["kind"] == "record":
            p.ingest_record(delta_from_spec(e["ops"], e["source"]), e["t"], e["source"])
        elif e["kind"] == "decide":
            try:
                results.append(p.decide(e["t"], scn.theta()))
            except UnresolvedConflict as exc:
                results.append(Deferred(e["t"], str(exc)))
        else:
            raise ValueError(f"unknown event kind {e['kind']!r}")
    return p, results
