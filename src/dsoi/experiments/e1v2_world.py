"""E1 v2 world: the pipeline one case runs through, shared by the runner and the scorer.

Plant X is planned; plant Y is master data the interpreter may name. Both plants enter
L3 from the authenticated capacity master, so a notice about Y commits to Y and leaves
X's plan alone, and a reading that puts X's cut on Y leaves X at nominal (a coverage
miss for X). Conflicting notices are resolved by L3's span policy, supplier portal
ranked above planner e-mail. L1 receives a text snapshot of the current state.
"""

from __future__ import annotations

import hashlib

from dsoi.context import VersionVector
from dsoi.experiments.e1v2_notices import (
    EMAIL,
    HORIZON,
    NAMES,
    NOMINAL,
    PLANNER,
    SUPPLIER,
    Case,
    snapshot,
)
from dsoi.layers.l1_interpreter import InterpreterSchema
from dsoi.scenarios import REPO_ROOT, VERSIONS_FILE, Scenario, build_pipeline, delta_from_spec
from dsoi.types import Message, Quantity

MASTER = "erp:capacity-master"
T_LOAD, T_FIRST, T_STEP, T_DECIDE = 1.0, 2.0, 0.5, 3.0


def scenario(case: Case) -> Scenario:
    spec = {
        "id": "E1v2", "model": "capacity_plan",
        "world": {"plant": "X", "horizon": list(HORIZON), "nominal_capacity": NOMINAL,
                  "demand_per_week": NOMINAL, "erp_source_id": MASTER,
                  "true_capacity": case.true_capacity("plant:X")},
        "source_priority": {SUPPLIER: 2, PLANNER: 1},
        "times": {"erp_load": T_LOAD, "decide": T_DECIDE},
    }
    return Scenario(REPO_ROOT / "scenarios", spec)


def schema() -> InterpreterSchema:
    return InterpreterSchema(frozenset({"capacity"}), frozenset(NAMES))


def messages(case: Case) -> list[tuple[Message, float]]:
    return [(Message(f"{case.id}-m{i}", source, text), T_FIRST + i * T_STEP)
            for i, (source, text) in enumerate(case.messages)]


def pipeline(case: Case, gate, interpreter, nu_i: str, *, historical: bool = False):
    """Case pipeline; the known source-priority defect needs explicit historical opt-in."""
    scn = scenario(case)
    versions = VersionVector.load(VERSIONS_FILE).bump(interpreter=nu_i)
    p = build_pipeline(scn, gate, interpreter, versions, resolution="span")
    if not historical:
        p.state.source_priority = {**p.state.source_priority, EMAIL: 1}
    y = [{"op": "assert", "subject": "plant:Y", "predicate": "capacity", "value": NOMINAL,
          "unit": "units/week", "valid_from": HORIZON[0], "valid_to": HORIZON[-1] + 1}]
    p.ingest_record(delta_from_spec(y, MASTER), T_LOAD + 0.25, MASTER)
    p.schema = schema()
    return scn, p


def state_of(p, t: float) -> dict[str, dict[int, tuple[float, float | None, float | None]]]:
    """Each plant's resolved capacity per week as known at t: (value, lo, hi)."""
    out: dict[str, dict[int, tuple]] = {}
    for entity in NAMES:
        for w in HORIZON:
            facts = [f for f in p.state.store.asof(w, t, ["capacity"]) if f.s == entity]
            if facts:
                (f,) = p.state.resolve(facts)
                q = f.o if isinstance(f.o, Quantity) else Quantity(NOMINAL)
                out.setdefault(entity, {})[w] = (q.value, q.lo, q.hi)
    return out


def snapshot_text(p, t: float) -> str:
    """No current week: a message that needs one (scope_relative) states it."""
    return snapshot(state_of(p, t))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
