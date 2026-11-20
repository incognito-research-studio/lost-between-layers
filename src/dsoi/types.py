"""Core value types: quantities, provenance, facts and deltas (Defs. 1-3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

# Valid time and transaction time are half-open intervals [start, end).
# Transaction-time end is math.inf while a fact is current.
Interval = tuple[float, float]


def overlaps(a: Interval, b: Interval) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def contains(iv: Interval, point: float) -> bool:
    return iv[0] <= point < iv[1]


@dataclass(frozen=True)
class Quantity:
    """A numeric attribute value.

    `lo`/`hi` carry the interval an interpreter reports instead of a point
    (Sec. VI). Both are None for exact values, such as authenticated ERP records.
    """

    value: float
    lo: float | None = None
    hi: float | None = None
    unit: str = ""

    def interval(self) -> Interval:
        lo = self.value if self.lo is None else self.lo
        hi = self.value if self.hi is None else self.hi
        return (lo, hi)

    @property
    def is_exact(self) -> bool:
        return self.lo is None and self.hi is None


@dataclass(frozen=True)
class Span:
    """A character span of a logged message: the where-provenance of an asserted fact."""

    message_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Provenance:
    """prov in Def. 2: source, message span, producing component and its version."""

    source_id: str
    component: str
    component_version: str
    authenticated: bool
    span: Span | None = None


@dataclass(frozen=True)
class Fact:
    """f = (s, p, o, VT, TT, prov) of Def. 2."""

    s: str
    p: str
    o: Quantity | str
    vt: Interval
    tt: Interval
    prov: Provenance

    @property
    def current(self) -> bool:
        return math.isinf(self.tt[1])


class OpKind(StrEnum):
    ASSERT = "assert"
    RETRACT = "retract"
    SUPERSEDE = "supersede"


@dataclass(frozen=True)
class Op:
    kind: OpKind
    s: str
    p: str
    vt: Interval
    prov: Provenance
    o: Quantity | str | None = None  # None for RETRACT


@dataclass(frozen=True)
class Delta:
    """A well-typed finite set of operations (Def. 3)."""

    ops: tuple[Op, ...]


@dataclass(frozen=True)
class Abstain:
    """Explicit abstention, the bottom element of Def. 3."""

    reason: str


@dataclass(frozen=True)
class Message:
    """An inbound signal. Unstructured messages pass through L1."""

    message_id: str
    source_id: str
    text: str
    authenticated: bool = False
    meta: dict = field(default_factory=dict, compare=False)
