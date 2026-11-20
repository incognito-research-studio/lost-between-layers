"""L3 State: bitemporal typed store with deterministic conflict resolution.

Contract (Table I):
  Atomic commit(delta, t); asof(v, t) returns S(v, t); conflicts(D) lists
  contradictory functional facts.
  Invariants: append-only transaction time; deterministic resolution (source
  priority, then authenticated over unauthenticated, then latest transaction
  time) or a blocking flag; schema validity and referential integrity against a
  `Signature` (a delta that breaks either is rejected whole, before any write).
  Guarantees R2(i)-(iii). Does not guarantee S_hat = S*.

Contradictory claims (S4) are unauthenticated assertions from different sources
with different values over overlapping valid time. The policy decides what is
committed; it changes decisions, so it is part of nu (`resolution`).
  latest    the newer claim supersedes the older one
  priority  the claim from the higher-ranked source wins, whatever the arrival order
  span      point from the priority winner, interval spanning both claims (Sec. VI)
  block     nothing is committed; decisions reading the keys are deferred until an
            authenticated record covers them

Reads resolve each (s, p) to one fact by rank: source priority, then authenticated over
unauthenticated, then latest transaction time. A supersession rule may override the rank;
it also changes decisions, so it enters nu with the source ranks (`priority_label`).
  none                  rank only (the default, and the rule of every study before phase 3)
  authenticated_latest  an authenticated record newer than the ranked winner supersedes it:
                        a master correction displaces an earlier notice, even one whose
                        source ranks higher, while a later notice still outranks an earlier
                        master value as before (Section 9.5, "When a notice may outrank
                        the master")
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from dsoi.canon import CanonicalState, DecisionClass, canonicalize, restrict
from dsoi.store import BitemporalStore
from dsoi.types import Delta, Fact, Op, OpKind, Quantity, overlaps

POLICIES = ("latest", "priority", "span", "block")
SUPERSESSION = ("none", "authenticated_latest")


class SchemaViolation(ValueError):
    """A delta that breaks schema validity or referential integrity; nothing is committed."""


@dataclass(frozen=True)
class Signature:
    """Sigma for L3 (Def. 1): the predicates facts may use, the type of their values, and
    the entities facts may refer to. Entities come from master data: a subject is known if
    the signature declares it or an authenticated fact about it is already stored."""

    predicates: dict[str, type]   # predicate -> Quantity or str
    entities: frozenset[str]

    @classmethod
    def from_model(cls, baseline: Delta, reads: frozenset[str],
                   extra: frozenset[str] = frozenset(),
                   text: frozenset[str] = frozenset({"sop_state"})) -> Signature:
        """Numeric predicates read or loaded by a decision model (canon@1 is numeric) plus
        `extra`, the text-valued `text` predicates (procedure state), and the entities its
        master-data baseline introduces."""
        names = {op.p for op in baseline.ops} | set(reads) | set(extra)
        return cls({**{p: Quantity for p in sorted(names)}, **{p: str for p in sorted(text)}},
                   frozenset(op.s for op in baseline.ops))


class UnresolvedConflict(RuntimeError):
    """A decision reads a key with a blocked contradiction: defer it."""


@dataclass(frozen=True)
class CommitOutcome:
    outcome: str          # committed | resolved | overridden | conflict
    reason: str = ""


class StateLayer:
    def __init__(self, store: BitemporalStore | None = None,
                 source_priority: dict[str, int] | None = None,
                 resolution: str = "latest", signature: Signature | None = None,
                 supersession: str = "none"):
        if resolution not in POLICIES:
            raise ValueError(f"unknown resolution policy {resolution!r}")
        if supersession not in SUPERSESSION:
            raise ValueError(f"unknown supersession rule {supersession!r}")
        self.store = store or BitemporalStore()
        self.source_priority = source_priority or {}
        self.resolution = resolution
        self.supersession = supersession
        self.signature = signature                         # None: no schema checks
        self.blocked: list[tuple[str, str, tuple]] = []   # (s, p, vt) awaiting resolution

    @property
    def priority_label(self) -> str:
        """The source ranks as an entry of nu: they decide conflicts, so they change decisions.
        A non-default supersession rule is appended; the default adds nothing, so the label,
        and every nu recorded before the rule existed, stay as they were."""
        ranks = ";".join(f"{k}={v}" for k, v in sorted(self.source_priority.items())) or "none"
        if self.supersession == "none":
            return ranks
        return f"{ranks};supersede={self.supersession}"

    # -- writes -----------------------------------------------------------------

    def validate(self, delta: Delta) -> None:
        """Schema validity and referential integrity; raises SchemaViolation, writes nothing.
        An authenticated op may introduce its subject (master data); others must refer to a
        known entity."""
        sig = self.signature
        if sig is None:
            return
        known = set(sig.entities) | {f.s for f in self.store.facts() if f.prov.authenticated}
        for op in delta.ops:
            if op.p not in sig.predicates:
                raise SchemaViolation(f"unknown predicate '{op.p}' on {op.s}")
            if op.o is not None and not isinstance(op.o, sig.predicates[op.p]):
                raise SchemaViolation(f"'{op.p}' on {op.s} takes "
                                      f"{sig.predicates[op.p].__name__}, got {op.o!r}")
            if op.s not in known and not op.prov.authenticated:
                raise SchemaViolation(f"unknown entity {op.s} (from {op.prov.source_id})")

    def commit(self, delta: Delta, t: float) -> float:
        """Commit as given. Authenticated records also clear blocks they cover."""
        self.validate(delta)
        for op in delta.ops:
            if op.prov.authenticated:
                self.blocked = [b for b in self.blocked
                                if not (b[0] == op.s and b[1] == op.p and overlaps(b[2], op.vt))]
        return self.store.commit(delta, t)

    def _priority(self, source_id: str) -> int:
        return max((p for prefix, p in self.source_priority.items()
                    if source_id.startswith(prefix)), default=0)

    def _contradicted(self, op: Op, t: float) -> list[Fact]:
        if op.prov.authenticated or op.kind is OpKind.RETRACT:
            return []
        return [
            f for f in self.store.facts()
            if f.current and f.s == op.s and f.p == op.p and overlaps(f.vt, op.vt)
            and not f.prov.authenticated and f.prov.source_id != op.prov.source_id
            and f.o != op.o
        ]

    def commit_interpreted(self, delta: Delta, t: float) -> CommitOutcome:
        """Commit an interpreted delta, applying the conflict policy to contradictory claims."""
        if self.resolution == "latest":
            self.commit(delta, t)
            return CommitOutcome("committed")
        ops: list[Op] = []
        notes: list[str] = []
        for op in delta.ops:
            rivals = self._contradicted(op, t)
            if not rivals:
                ops.append(op)
                continue
            rival = max(rivals, key=lambda f: (self._priority(f.prov.source_id), f.tt[0]))
            incoming_wins = self._priority(op.prov.source_id) >= self._priority(
                rival.prov.source_id)
            notes.append(f"{op.s}/{op.p} {op.prov.source_id} vs {rival.prov.source_id}")
            if self.resolution == "block":
                self.blocked.append((op.s, op.p, op.vt))
                return CommitOutcome("conflict", "blocked: " + "; ".join(notes))
            if self.resolution == "priority":
                if incoming_wins:
                    ops.append(op)
                continue
            # span
            winner_q, winner_prov = (op.o, op.prov) if incoming_wins else (rival.o, rival.prov)
            lows = [q.interval()[0] for q in (op.o, rival.o)]
            highs = [q.interval()[1] for q in (op.o, rival.o)]
            spanned = Quantity(winner_q.value, min(lows), max(highs), winner_q.unit)
            ops.append(replace(op, kind=OpKind.SUPERSEDE, o=spanned, prov=winner_prov))
        if not ops:
            return CommitOutcome("overridden", "lower-priority claim: " + "; ".join(notes))
        self.commit(Delta(tuple(ops)), t)
        return CommitOutcome("resolved" if notes else "committed", "; ".join(notes))

    # -- reads ------------------------------------------------------------------

    def _rank(self, f: Fact) -> tuple:
        return (self._priority(f.prov.source_id), f.prov.authenticated, f.tt[0])

    def resolve(self, facts: list[Fact]) -> list[Fact]:
        """One fact per (s, p): the highest-ranked, unless the supersession rule displaces it
        (deterministic either way)."""
        groups: dict[tuple, list[Fact]] = {}
        for f in facts:
            groups.setdefault((f.s, f.p), []).append(f)
        return [self._winner(fs) for _, fs in sorted(groups.items())]

    def _winner(self, facts: list[Fact]) -> Fact:
        top = max(facts, key=self._rank)
        if self.supersession == "authenticated_latest":
            newer = [f for f in facts if f.prov.authenticated and f.tt[0] > top.tt[0]]
            if newer:
                return max(newer, key=lambda f: (f.tt[0], self._rank(f)))
        return top

    def conflicts(self, d: DecisionClass, t: float) -> dict[tuple, list[Fact]]:
        """conflicts(D) (Table 2): the functional facts D reads that hold more than one current
        value at a valid time of D's horizon, as known at t, keyed (s, p, v). `resolve` picks
        one of each; the decision record lists them, so the choice can be audited."""
        out: dict[tuple, list[Fact]] = {}
        for v in d.horizon:
            for (s, p), facts in self.store.conflicts(v, t, d.reads).items():
                if s in d.subjects:
                    out[(s, p, v)] = facts
        return out

    def read(self, d: DecisionClass, t: float) -> tuple[CanonicalState, dict[tuple, Fact]]:
        """kappa_D(S(v, t)) over D's horizon, plus the facts it was built from (for rho)."""
        for s, p, vt in self.blocked:
            if s in d.subjects and p in d.reads and any(vt[0] <= v < vt[1] for v in d.horizon):
                raise UnresolvedConflict(f"{s}/{p} over {vt} is blocked pending resolution")
        by_time = {v: self.resolve(self.store.asof(v, t, d.reads)) for v in d.horizon}
        used = restrict(by_time, d)
        return canonicalize(used, d), used
