"""Decision-relevant restriction and canonicalization (Def. 4, Lemma 1(a)).

Under A-KEY every decision-relevant entity has a stable business key, so the
canonical form is computed by normalizing values and sorting (Lemma 1(a)).
Renaming anonymous nodes (Lemma 1(b), GI-hard) is deliberately unsupported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from dsoi.types import Fact, Quantity

# One canonical entry: ((subject, predicate, valid_time), value, lo, hi, unit)
CanonEntry = tuple[tuple[str, str, float], float, float | None, float | None, str]
CanonicalState = tuple[CanonEntry, ...]


@dataclass(frozen=True)
class DecisionClass:
    """A decision class D: what it reads and over which valid times."""

    name: str
    reads: frozenset[str]           # predicates D reads (the pi_D restriction)
    subjects: frozenset[str]        # subjects D reads
    horizon: tuple[float, ...]      # valid times D plans over
    decimals: int = 6               # declared value normalization


def restrict(facts_by_time: dict[float, list[Fact]], d: DecisionClass) -> dict[tuple, Fact]:
    """pi_D: keep only the facts D reads. Expects at most one fact per key (resolved by L3)."""
    out: dict[tuple, Fact] = {}
    for v, facts in facts_by_time.items():
        for f in facts:
            if f.p in d.reads and f.s in d.subjects:
                key = (f.s, f.p, v)
                if key in out:
                    raise ValueError(f"unresolved conflict for {key}; resolve in L3 first")
                out[key] = f
    return out


def _norm(x: float | None, decimals: int) -> float | None:
    return None if x is None else round(float(x), decimals)


def canonicalize(restricted: dict[tuple, Fact], d: DecisionClass) -> CanonicalState:
    """kappa_D: normalize and sort. Equal outputs iff the states are D-equivalent."""
    entries: list[CanonEntry] = []
    for key, f in restricted.items():
        if not isinstance(f.o, Quantity):
            raise TypeError(f"canon@1 handles numeric quantities only: {key}")
        q = f.o
        entries.append(
            (key, _norm(q.value, d.decimals), _norm(q.lo, d.decimals), _norm(q.hi, d.decimals),
             q.unit)
        )
    return tuple(sorted(entries))


def state_hash(state: CanonicalState) -> str:
    """hash(kappa_D(S)) for the audit record. Collision resistance is assumed (audit item 24)."""
    payload = json.dumps(state, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def lookup(state: CanonicalState) -> dict[tuple, Quantity]:
    """Rebuild a key -> Quantity map from a canonical state (used by L4/L5 and replay)."""
    return {key: Quantity(value, lo, hi, unit) for key, value, lo, hi, unit in state}
