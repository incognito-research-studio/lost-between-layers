"""Append-only bitemporal store (Def. 2; requirement R2).

Corrections never delete. A correction closes the transaction-time end of the
old fact and inserts new facts, so every past as-of view stays retrievable
(R2(i)).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import replace

from dsoi.types import Delta, Fact, Interval, OpKind, contains, overlaps


class TransactionTimeError(ValueError):
    pass


class BitemporalStore:
    def __init__(self) -> None:
        self._facts: list[Fact] = []
        self._last_tt: float = -math.inf
        # called with (delta, t) after every successful commit; L6 logs the deltas there,
        # so that re-applying them can be checked to reproduce the store (R2(iii))
        self.on_commit: list[Callable[[Delta, float], None]] = []

    # -- writes -----------------------------------------------------------------

    def commit(self, delta: Delta, t: float) -> float:
        """Apply `delta` atomically at transaction time `t` (the update U of Def. 3).

        U is deterministic in (store, delta, t): R2(iii).
        """
        if t <= self._last_tt:
            raise TransactionTimeError(
                f"transaction time must increase: got {t}, last was {self._last_tt}"
            )
        for op in delta.ops:  # validate first, so a rejected delta changes nothing
            if op.kind in (OpKind.ASSERT, OpKind.SUPERSEDE) and op.o is None:
                raise ValueError(f"{op.kind.value} needs a value: {op}")
        for op in delta.ops:
            if op.kind in (OpKind.RETRACT, OpKind.SUPERSEDE):
                self._close(op.s, op.p, op.vt, t)
            if op.kind in (OpKind.ASSERT, OpKind.SUPERSEDE):
                self._facts.append(Fact(op.s, op.p, op.o, op.vt, (t, math.inf), op.prov))
        self._last_tt = t
        for listener in self.on_commit:
            listener(delta, t)
        return t

    def _close(self, s: str, p: str, vt: Interval, t: float) -> None:
        """Close current facts of (s, p) overlapping `vt`; re-insert the uncovered remainder."""
        remainder: list[Fact] = []
        for i, f in enumerate(self._facts):
            if f.s == s and f.p == p and f.current and overlaps(f.vt, vt):
                self._facts[i] = replace(f, tt=(f.tt[0], t))
                if f.vt[0] < vt[0]:
                    remainder.append(replace(f, vt=(f.vt[0], vt[0]), tt=(t, math.inf)))
                if vt[1] < f.vt[1]:
                    remainder.append(replace(f, vt=(vt[1], f.vt[1]), tt=(t, math.inf)))
        self._facts.extend(remainder)

    # -- reads ------------------------------------------------------------------

    @property
    def last_transaction_time(self) -> float:
        return self._last_tt

    def facts(self) -> tuple[Fact, ...]:
        return tuple(self._facts)

    def asof(self, v: float, t: float, predicates: Iterable[str] | None = None) -> list[Fact]:
        """Facts valid at `v` as known at transaction time `t`."""
        wanted = None if predicates is None else set(predicates)
        return [
            f
            for f in self._facts
            if contains(f.vt, v)
            and contains(f.tt, t)
            and (wanted is None or f.p in wanted)
        ]

    def snapshot(self, v: float, t: float) -> frozenset[tuple]:
        """S(v, t) of Def. 2: the set of (s, p, o) triples."""
        return frozenset((f.s, f.p, f.o) for f in self.asof(v, t))

    def conflicts(self, v: float, t: float, functional: Iterable[str]) -> dict[tuple, list[Fact]]:
        """Functional relations with more than one value at (v, t) (L3 conflicts(D))."""
        groups: dict[tuple, list[Fact]] = {}
        for f in self.asof(v, t, functional):
            groups.setdefault((f.s, f.p), []).append(f)
        return {k: fs for k, fs in groups.items() if len(fs) > 1}
