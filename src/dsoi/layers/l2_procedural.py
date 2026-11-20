"""L2 Procedural: Adm(delta or x, S, role, SOP_nu) -> admit | queue(approval) | reject(reason).

Contract (Table 2):
  Post: committed deltas satisfy node preconditions; threshold actions link to an
        authenticated approval event.
  Design rule 2: authority never flows through text. Authority-bearing facts are
  admitted only from authenticated sources, and a threshold action is executed only
  after `admit_action` links it to an authenticated approval event (Pipeline.decide).
  Approvals are bound to the actions they admit: an event is consumed by use (in full
  if it has no cap, up to its cap otherwise), its approver must hold a role the rule
  accepts, and lifting a restriction that is in force needs a source authorized for
  that restriction, not just an authenticated one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from dsoi.types import Delta, Interval, Op, OpKind, Provenance, Quantity

AUTHORITY_PREDICATES = frozenset({"approval", "release", "identity"})
RESTRICTION_PREDICATES = frozenset({"hold", "suspended"})


class Verdict(StrEnum):
    ADMIT = "admit"
    QUEUE = "queue"
    REJECT = "reject"


@dataclass(frozen=True)
class Admission:
    verdict: Verdict
    reason: str = ""
    approvals: tuple[str, ...] = ()   # ids of the approval events the admission rests on
    uses: Mapping[str, float] = field(default_factory=dict)  # event id -> quantity it covers
    effects: tuple[Op, ...] = ()      # procedure-state ops to commit with the admitted delta


@dataclass(frozen=True)
class Procedure:
    """An SOP graph over the cases of one entity type (Table 2, L2: node preconditions).

    `steps` maps a (predicate, value) that an op sets to the node it moves the case to,
    and `requires` maps a node to the node the case must already be in. A case enters
    the procedure at a node without a prerequisite; before that, other steps do not
    apply to it (a baseline load with hold 0 releases nothing). A retract sets the
    value 0. The case's node is persisted in L3 as the fact `sop_state` =
    "<name>/<node>", committed in the same delta as the step that moves it."""

    name: str
    steps: Mapping[tuple[str, float], str]
    requires: Mapping[str, str] = field(default_factory=dict)

    def step(self, op: Op) -> str | None:
        if op.kind is OpKind.RETRACT:
            value = 0.0
        elif isinstance(op.o, Quantity):
            value = float(op.o.value)
        else:
            return None
        return self.steps.get((op.p, value))


@dataclass(frozen=True)
class ThresholdRule:
    """An action is a threshold action of `kind` when it sets `variable` above `above`
    (a lot release, a large order). Executing it needs an approval event of that kind
    for that variable from an approver holding one of `approvers` (the delegation)."""

    kind: str
    variable: str
    approvers: frozenset[str]
    above: float = 0.0


@dataclass(frozen=True)
class ApprovalEvent:
    """An approval issued by the identity/approval service (Fig. 1). The service sets
    `authenticated`; L1 cannot produce approval events, so text never yields one.

    `expires_at` ends the event's validity (exclusive); `max_quantity` caps the total
    quantity it covers over all the actions it admits. None means no expiry, or an event
    that covers one action of any quantity. Use is recorded in the audit log, which
    `admit_action` reads back as `consumed` (Pipeline.decide)."""

    id: str
    kind: str
    variable: str
    approver: str
    source_id: str
    authenticated: bool
    t: float
    expires_at: float | None = None
    max_quantity: float | None = None

    def covers(self, quantity: float, t: float, tol: float = 1e-9,
               used: tuple[float, ...] = ()) -> str | None:
        """None if the event covers `quantity` at time t, else the reason it does not.
        `used` lists the quantities of the actions it already admitted."""
        if not self.authenticated:
            return "unauthenticated"
        if self.t > t:
            return "issued later"
        if self.expires_at is not None and t >= self.expires_at:
            return f"expired at {self.expires_at:g}"
        if self.max_quantity is None:
            return "already used" if used else None
        if sum(used) + quantity > self.max_quantity + tol:
            return (f"approved up to {self.max_quantity:g}, used {sum(used):g}, "
                    f"released {quantity:g}")
        return None


class ProceduralLayer:
    def __init__(
        self,
        sop_version: str,
        authority_predicates: frozenset[str] = AUTHORITY_PREDICATES,
        approval_required: frozenset[str] = frozenset(),
        restrictions: frozenset[str] = RESTRICTION_PREDICATES,
        threshold_rules: tuple[ThresholdRule, ...] = (),
        lift_issuers: Mapping[str, frozenset[str]] | None = None,
        procedures: tuple[Procedure, ...] = (),
    ):
        self.sop_version = sop_version
        self.authority_predicates = authority_predicates
        self.approval_required = approval_required
        self.restrictions = restrictions
        self.threshold_rules = threshold_rules
        # restriction -> sources authorized to lift it while it is in force; none by default
        self.lift_issuers = dict(lift_issuers or {})
        self.procedures = procedures
        self.provenance = Provenance("l2:procedure", "L2", sop_version, True)

    @property
    def rules_label(self) -> str:
        """The rules beyond `sop_version` that change decisions, as an entry of nu: a hash of
        the threshold rules with their approver roles and of the lift issuers."""
        if not self.threshold_rules and not self.lift_issuers and not self.procedures:
            return "none"
        spec = {"thresholds": sorted([r.kind, r.variable, sorted(r.approvers), r.above]
                                     for r in self.threshold_rules),
                "lift_issuers": {k: sorted(v) for k, v in sorted(self.lift_issuers.items())},
                "procedures": sorted([p.name, sorted([*k, v] for k, v in p.steps.items()),
                                      sorted(p.requires.items())] for p in self.procedures)}
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        return f"sha256:{digest[:16]}"

    def lifts_restriction(self, op: Op) -> bool:
        """Lifting a hold or suspension, wholly or partly, is authority-bearing (S2, S5).

        A restriction flag is 1 when the restriction applies in full. A retract, or any
        point value below 1, loosens it, because L4 scales capacity by (1 - flag) and the
        canonicalizer rounds; so only a value of exactly 1 is restrictive and admitted
        from any source. A hedged reading (1 in [0, 1]) keeps the restrictive point.
        """
        if op.p not in self.restrictions:
            return False
        if op.kind is OpKind.RETRACT:
            return True
        return not (isinstance(op.o, Quantity) and op.o.value >= 1)

    def flag_in_range(self, op: Op) -> bool:
        """Restriction flags, with their interval bounds, must lie in [0, 1]."""
        if op.p not in self.restrictions or not isinstance(op.o, Quantity):
            return True
        return all(0 <= b <= 1 for b in (op.o.value, op.o.lo, op.o.hi) if b is not None)

    def admit_delta(self, delta: Delta, role: str = "system",
                    in_force: Callable[[Op], bool] | None = None,
                    sop_state: Callable[[str, str, Interval], str | None] | None = None,
                    ) -> Admission:
        """Adm(delta, S, ...). `in_force(op)` tells whether the restriction an op would
        loosen currently applies in S; without it, every restriction counts as in force.
        `sop_state(subject, procedure, vt)` returns the case's current node in S, if any.
        A delta that moves a case to a node whose required node the case is not in is
        queued; an admitted delta carries the procedure-state ops to commit with it."""
        for op in delta.ops:
            if not self.flag_in_range(op):
                return Admission(Verdict.REJECT, f"restriction flag '{op.p}' on {op.s} "
                                                 f"outside [0, 1]: {op.o}")
            if op.p in self.authority_predicates and not op.prov.authenticated:
                return Admission(Verdict.REJECT, f"authority-bearing '{op.p}' from "
                                                 f"unauthenticated source {op.prov.source_id}")
            if self.lifts_restriction(op) and not op.prov.authenticated:
                return Admission(Verdict.REJECT, f"lifting '{op.p}' on {op.s} needs an "
                                                 f"authenticated approval; got {op.prov.source_id}")
            if (self.lifts_restriction(op) and (in_force is None or in_force(op))
                    and op.prov.source_id not in self.lift_issuers.get(op.p, frozenset())):
                return Admission(Verdict.REJECT, f"{op.prov.source_id} is not authorized to "
                                                 f"lift '{op.p}' on {op.s}")
            if op.p in self.approval_required:
                return Admission(Verdict.QUEUE,
                                 f"'{op.p}' requires approval (SOP {self.sop_version})")
        effects, moved = [], {}
        for op in delta.ops:
            for proc in self.procedures:
                node = proc.step(op)
                if node is None:
                    continue
                key = (op.s, proc.name)
                current = moved[key] if key in moved else (
                    sop_state(op.s, proc.name, op.vt) if sop_state else None)
                need = proc.requires.get(node)
                if current is None and need is not None:
                    continue    # a case enters only at a node without a prerequisite
                if need is not None and current != need:
                    return Admission(Verdict.QUEUE, f"SOP {proc.name}: '{node}' on {op.s} needs "
                                                    f"the case at '{need}', it is at "
                                                    f"'{current or 'start'}'")
                moved[key] = node
                effects.append(Op(OpKind.SUPERSEDE, op.s, "sop_state", op.vt, self.provenance,
                                  f"{proc.name}/{node}"))
        return Admission(Verdict.ADMIT, effects=tuple(effects))

    def admit_action(self, action: Mapping[str, float], approvals: Sequence[ApprovalEvent],
                     t: float, tol: float = 1e-9,
                     consumed: Mapping[str, tuple[float, ...]] | None = None) -> Admission:
        """Adm(x, ...): admit a released action, or queue it for approval.

        `action` maps variable names to released quantities. Every threshold rule the
        action triggers needs an authenticated approval event of the rule's kind for the
        rule's variable, from an approver the rule accepts, issued at or before t, not
        expired at t, and covering the released quantity on top of what it already
        admitted (`consumed`: event id -> quantities of earlier admitted actions);
        otherwise the action is queued and not executed. Admission names the events it
        rests on and the quantity it takes from each, for the audit record.

        Not modelled: SOP graph nodes and their preconditions beyond holds and
        thresholds, and procedure state persisted in L3 (Table 2, L2 contract).
        """
        consumed = consumed or {}
        linked, missing, uses = [], [], {}
        for rule in self.threshold_rules:
            if action and rule.variable not in action:
                missing.append(f"{rule.kind} on unknown variable {rule.variable}")
                continue
            quantity = action.get(rule.variable, 0.0)
            if quantity <= rule.above + tol:
                continue
            events = [e for e in approvals if e.kind == rule.kind and e.variable == rule.variable]
            why = {e.id: (e.covers(quantity, t, tol, consumed.get(e.id, ()))
                          if e.approver in rule.approvers
                          else f"approver {e.approver} not authorized for {rule.kind}")
                   for e in events}
            valid = [e for e in events if why[e.id] is None]
            if valid:
                linked.append(valid[-1].id)
                uses[valid[-1].id] = quantity
            else:
                stale = [f"{i}: {r}" for i, r in why.items() if r not in ("unauthenticated",
                                                                          "issued later")]
                missing.append(f"{rule.kind} on {rule.variable}"
                               + (f" ({'; '.join(stale)})" if stale else ""))
        if missing:
            return Admission(Verdict.QUEUE, "awaits authenticated approval: " + "; ".join(missing))
        return Admission(Verdict.ADMIT, "", tuple(linked), uses)
