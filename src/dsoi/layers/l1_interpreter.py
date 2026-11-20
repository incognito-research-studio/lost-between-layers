"""L1 Interpreter: I(m, S, Sigma, schema_D) -> (delta or bottom, spans, conf, nu_I).

Contract (Table I):
  Pre:  the message has a source id; nu_I is pinned.
  Post: delta is well typed; every operation carries a span; abstention is always available.
  Invariant: L1 cannot assert authority-bearing facts (approvals, releases, identity).

Typing gives structural guarantees only: a delta that parses can still be false.
The postconditions are enforced here, in `parse_output`, not left to the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from dsoi.types import Abstain, Delta, Message, Op, OpKind, Provenance, Quantity, Span


@dataclass(frozen=True)
class InterpreterSchema:
    """schema_D: which (subject, predicate) pairs L1 may assert for this decision class."""

    predicates: frozenset[str]
    subjects: frozenset[str]
    authority_predicates: frozenset[str] = frozenset({"approval", "release", "identity"})

    def __post_init__(self) -> None:
        clash = self.predicates & self.authority_predicates
        if clash:
            raise ValueError(f"L1 invariant: schema may not expose authority predicates {clash}")


@dataclass(frozen=True)
class InterpretResult:
    output: Delta | Abstain
    nu_i: str                     # the interpreter version that actually produced the output
    conf: float | None = None
    raw: dict = field(default_factory=dict, compare=False)


def apply_confidence(result: InterpretResult, thresholds: str,
                     flags: frozenset[str]) -> InterpretResult:
    """Def. 5 and Sec. 7: a yes/no reading (an op on a flag in `flags`) whose confidence lies
    in [lo, hi) becomes the interval [0, 1] around the restrictive point 1; below lo the
    reading is an abstention; at or above hi it stands. `thresholds` is nu's entry "lo/hi"
    ("none" turns this off). Readings without a confidence, or without flags, pass unchanged."""
    if thresholds == "none" or result.conf is None or not isinstance(result.output, Delta):
        return result
    lo, hi = (float(v) for v in thresholds.split("/"))
    yes_no = [op for op in result.output.ops if op.p in flags and isinstance(op.o, Quantity)]
    if not yes_no or result.conf >= hi:
        return result
    if result.conf < lo:
        return replace(result, output=Abstain(f"confidence {result.conf:g} below {lo:g}"))
    hedged = tuple(replace(op, o=Quantity(1.0, 0.0, 1.0, op.o.unit)) if op in yes_no else op
                   for op in result.output.ops)
    return replace(result, output=Delta(hedged))


class Interpreter(Protocol):
    def interpret(self, message: Message, schema: InterpreterSchema) -> InterpretResult: ...


def parse_output(
    payload: dict, message: Message, schema: InterpreterSchema, nu_i: str
) -> Delta | Abstain:
    """Validate a raw interpreter payload against the L1 contract.

    Any violation becomes an explicit abstention with a reason, so an
    out-of-schema output (including an attempted approval) is a type error,
    never a committed fact.
    """
    if not isinstance(payload, dict):
        return Abstain(f"malformed output: {payload!r}")
    if payload.get("kind") == "abstain":
        return Abstain(payload.get("reason") or "interpreter abstained")
    if payload.get("kind") != "delta":
        return Abstain(f"unknown output kind: {payload.get('kind')!r}")
    ops: list[Op] = []
    for raw in payload.get("ops", []):
        try:
            op = _parse_op(raw, message, schema, nu_i)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return Abstain(f"malformed operation {raw!r}: {exc}")
        if isinstance(op, Abstain):
            return op
        ops.append(op)
    if not ops:
        return Abstain("empty delta")
    return Delta(tuple(ops))


def _parse_op(raw: dict, message: Message, schema: InterpreterSchema, nu_i: str) -> Op | Abstain:
    """One operation of a delta, or an abstention naming the violated condition."""
    predicate, subject = raw.get("predicate"), raw.get("subject")
    if predicate in schema.authority_predicates:
        return Abstain(f"L1 invariant: cannot assert authority-bearing '{predicate}'")
    if predicate not in schema.predicates or subject not in schema.subjects:
        return Abstain(f"out-of-schema assertion: ({subject}, {predicate})")
    span_text = raw.get("span_text") or ""
    start = message.text.find(span_text) if span_text else -1
    if start < 0:
        return Abstain(f"postcondition: span not found in message: {span_text!r}")
    prov = Provenance(
        source_id=message.source_id,
        component="L1",
        component_version=nu_i,
        authenticated=False,  # interpreted facts are never authenticated
        span=Span(message.message_id, start, start + len(span_text)),
    )
    kind = OpKind(raw["op"])
    vt = (float(raw["valid_from"]), float(raw["valid_to"]))
    if not (math.isfinite(vt[0]) and vt[0] < vt[1]):
        return Abstain(f"postcondition: empty or invalid valid time: {raw}")
    value = None
    if kind is not OpKind.RETRACT:
        lo, hi = raw.get("lo"), raw.get("hi")
        if (lo is None) != (hi is None):
            return Abstain(f"postcondition: interval needs both bounds: {raw}")
        value = Quantity(float(raw["value"]), None if lo is None else float(lo),
                         None if hi is None else float(hi), raw.get("unit", ""))
        bounds = [b for b in (value.value, value.lo, value.hi) if b is not None]
        if not all(math.isfinite(b) for b in bounds):
            return Abstain(f"postcondition: non-finite value: {raw}")
        if value.lo is not None and not value.lo <= value.value <= value.hi:
            return Abstain(f"postcondition: point outside reported interval: {raw}")
    return Op(kind, subject, predicate, vt, prov, value)


class FixtureInterpreter:
    """Replays recorded interpreter outputs, keyed by message id.

    The default L1 for tests, scenarios and replay: deterministic and needs no API key.
    """

    def __init__(self, fixtures: dict[str, dict]):
        self._fixtures = fixtures

    @classmethod
    def from_file(cls, message_id: str, path: str | Path) -> FixtureInterpreter:
        with open(path) as fh:
            return cls({message_id: json.load(fh)})

    def interpret(self, message: Message, schema: InterpreterSchema) -> InterpretResult:
        fx = self._fixtures.get(message.message_id)
        if fx is None:
            return InterpretResult(Abstain("no fixture for message"), nu_i="fixture:none")
        nu_i = fx["interpreter_version"]
        out = parse_output(fx["output"], message, schema, nu_i)
        return InterpretResult(out, nu_i=nu_i, conf=fx.get("conf"), raw=fx)
