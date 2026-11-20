"""One decision cycle through L1-L6 (Fig. 1).

Design rule 1: stochasticity is confined to L1. Everything after `ingest_message`
is deterministic in its declared inputs, so Cor. 1 applies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsoi.canon import DecisionClass, state_hash
from dsoi.context import UNSET, Context, VersionVector
from dsoi.layers.l1_interpreter import Interpreter, InterpreterSchema, apply_confidence
from dsoi.layers.l2_procedural import Admission, ApprovalEvent, ProceduralLayer, Verdict
from dsoi.layers.l3_state import SchemaViolation, StateLayer, UnresolvedConflict
from dsoi.layers.l4_decision import (
    Decision,
    Formulation,
    LPInstance,
    LPPolicy,
    encoding,
    encoding_ids,
)
from dsoi.layers.l5_gate import Gate, GateResult, RecheckGate, gate_label
from dsoi.layers.l5_monitors import Monitor, monitors_label
from dsoi.layers.l6_audit import AuditLog, AuditRecord, IngestRecord, logged_tuple
from dsoi.types import Abstain, Delta, Message, Op, Quantity, overlaps


class Executor:
    """Execution systems. Returns (x^e, exec-ack), where x^e is what execution reports
    doing. This one is faithful: x^e = x. The subclasses below break execution fidelity,
    the goal the GSN pattern needs besides A1-A3 (audit item 5)."""

    def __init__(self) -> None:
        self._n = 0

    def _ack(self) -> str:
        self._n += 1
        return f"ack-{self._n}"

    def execute(self, x: np.ndarray | None) -> tuple[np.ndarray | None, str | None]:
        if x is None:
            return None, None
        return np.array(x, copy=True), self._ack()


class LotSizeExecutor(Executor):
    """Rounds every quantity up to a whole lot, as a downstream MES might."""

    def __init__(self, lot: float):
        super().__init__()
        self.lot = lot

    def execute(self, x):
        if x is None:
            return None, None
        return np.ceil(np.asarray(x) / self.lot - 1e-9) * self.lot, self._ack()


class NoisyExecutor(Executor):
    """With probability `p`, execution deviates from the release by a factor in
    [1 - max_dev, 1 + max_dev] per position (overrun or underrun)."""

    def __init__(self, p: float, max_dev: float, rng: np.random.Generator):
        super().__init__()
        self.p, self.max_dev, self.rng = p, max_dev, rng

    def execute(self, x):
        if x is None:
            return None, None
        x = np.asarray(x, dtype=float)
        if self.rng.random() < self.p:
            x = x * (1 + self.rng.uniform(-self.max_dev, self.max_dev, size=x.shape))
        return x, self._ack()


@dataclass(frozen=True, eq=False)
class CycleResult:
    record: AuditRecord
    context: Context
    decision: Decision
    gate: GateResult
    instance: LPInstance

    def released(self, name: str) -> float | None:
        """Released quantity of one decision variable, by its formulation name."""
        if self.record.x_released is None:
            return None
        return float(self.record.x_released[self.instance.system.names.index(name)])

    def executed(self, name: str) -> float | None:
        """Executed quantity (x^e) of one decision variable."""
        if self.record.x_exec is None:
            return None
        return float(self.record.x_exec[self.instance.system.names.index(name)])


class Pipeline:
    def __init__(self, *, interpreter: Interpreter, schema: InterpreterSchema,
                 procedural: ProceduralLayer, state: StateLayer, formulation: Formulation,
                 gate: Gate, versions: VersionVector, decision_class: DecisionClass,
                 audit: AuditLog | None = None, executor: Executor | None = None,
                 strict_versions: bool = True, monitors: tuple[Monitor, ...] = ()):
        self.interpreter = interpreter
        self.schema = schema
        self.procedural = procedural
        self.state = state
        self.formulation = formulation
        self.policy = LPPolicy(formulation)
        self.gate = gate
        self.versions = versions
        self.decision_class = decision_class
        self.audit = audit or AuditLog()
        self.executor = executor or Executor()
        self.strict_versions = strict_versions
        self.monitors = monitors
        self.approvals: list[ApprovalEvent] = []
        # L6 logs every delta L3 commits; looked up at call time, so a swapped log receives them
        self.state.store.on_commit.append(lambda delta, t: self.audit.log_delta(delta, t))

    @property
    def nu(self) -> VersionVector:
        """nu as run: the declared vector, with the gate (rung and repair mode) and the L3
        source ranks filled in from the components (design rule 3). A pinned entry that
        disagrees with its component raises, since the log would misattribute decisions."""
        actual = {"gate": gate_label(self.gate), "source_priority": self.state.priority_label,
                  "sop_rules": self.procedural.rules_label,
                  "monitors": monitors_label(self.monitors)}
        for name, value in actual.items():
            pinned = getattr(self.versions, name)
            if pinned not in (UNSET, value):
                raise ValueError(f"nu.{name} pins {pinned!r}, but the pipeline runs {value!r}")
        policy = self.versions.resolution.split("@")[0]
        if policy != self.state.resolution:
            raise ValueError(f"nu.resolution pins {policy!r}, but L3 runs "
                             f"{self.state.resolution!r}")
        return self.versions.bump(**actual)

    # -- inputs -----------------------------------------------------------------

    def ingest_record(self, delta: Delta, t: float, source_id: str) -> str:
        """Structured, authenticated input (ERP/EDI). Enters at L2, bypassing L1."""
        adm = self.procedural.admit_delta(delta, in_force=self._in_force,
                                          sop_state=self._sop_state)
        outcome, reason = adm.verdict.value, adm.reason
        if adm.verdict is Verdict.ADMIT:
            try:
                self.state.commit(Delta(delta.ops + adm.effects), t)
            except SchemaViolation as exc:
                outcome, reason = "rejected", f"schema: {exc}"
        self.audit.log_ingest(IngestRecord(source_id, t, outcome, reason, "n/a"))
        return outcome

    def ingest_message(self, message: Message, t: float) -> str:
        """Unstructured input through L1 -> L2 -> L3. Always leaves an ingest record (R2(ii))."""
        self.audit.log_message(message.message_id, message.source_id, message.text, t,
                               message.meta.get("received_at"))
        result = self.interpreter.interpret(message, self.schema)
        if self.strict_versions and result.nu_i != self.versions.interpreter:
            # S6: a model other than the pinned nu_I answered (alias update, fallback).
            rec = IngestRecord(message.message_id, t, "abstained",
                               f"nu_I mismatch: pinned {self.versions.interpreter}, "
                               f"served {result.nu_i}", result.nu_i)
            self.audit.log_ingest(rec)
            return rec.outcome
        result = apply_confidence(result, self.versions.confidence, self.procedural.restrictions)
        if isinstance(result.output, Abstain):
            rec = IngestRecord(message.message_id, t, "abstained", result.output.reason,
                               result.nu_i)
            self.audit.log_ingest(rec)
            return rec.outcome
        adm = self.procedural.admit_delta(result.output, in_force=self._in_force,
                                          sop_state=self._sop_state)
        reason = adm.reason
        if adm.verdict is Verdict.ADMIT:
            try:
                committed = self.state.commit_interpreted(
                    Delta(result.output.ops + adm.effects), t)
                outcome, reason = committed.outcome, committed.reason
            except SchemaViolation as exc:
                outcome, reason = "rejected", f"schema: {exc}"
        else:
            outcome = "rejected" if adm.verdict is Verdict.REJECT else "queued"
        self.audit.log_ingest(IngestRecord(message.message_id, t, outcome, reason, result.nu_i))
        return outcome

    def record_approval(self, event: ApprovalEvent) -> None:
        """An event from the identity/approval service (dashed input to L2 in Fig. 1).

        Every event is logged; L2 counts only authenticated ones. Nothing on the
        message path (L1) calls this, so text cannot create an approval."""
        self.approvals.append(event)
        self.audit.log_approval(event)

    def _in_force(self, op: Op) -> bool:
        """Whether the restriction `op` would loosen applies now: a current fact of the same
        subject and predicate over an overlapping valid time with a value, or an interval
        top, above zero. L2 asks this before letting a source lift a restriction."""
        for f in self.state.store.facts():
            if f.current and f.s == op.s and f.p == op.p and overlaps(f.vt, op.vt):
                q = f.o
                if isinstance(q, Quantity) and max(q.value, q.value if q.hi is None else q.hi) > 0:
                    return True
        return False

    def _sop_state(self, subject: str, procedure: str, vt: tuple) -> str | None:
        """The case's current node in `procedure`, from the procedure-state facts in L3."""
        prefix = procedure + "/"
        facts = [f for f in self.state.store.facts()
                 if f.current and f.s == subject and f.p == "sop_state"
                 and isinstance(f.o, str) and f.o.startswith(prefix) and overlaps(f.vt, vt)]
        return max(facts, key=lambda f: f.tt[0]).o[len(prefix):] if facts else None

    def consumed_approvals(self) -> dict[str, tuple[float, ...]]:
        """Approval id -> quantities of the executed actions it admitted, read from the audit
        log, so consumption survives a restart from a stored log (L6 is append-only)."""
        used: dict[str, list[float]] = {}
        for rec in self.audit.records:
            if rec.decided and rec.exec_ack is not None and rec.admission:
                for approval_id, quantity in rec.admission.get("uses", {}).items():
                    used.setdefault(approval_id, []).append(float(quantity))
        return {k: tuple(v) for k, v in used.items()}

    # -- decisions --------------------------------------------------------------

    def context_at(self, t_asof: float, theta: dict, nu: VersionVector | None = None):
        canon, used = self.state.read(self.decision_class, t_asof)
        inst_ids = self.formulation.build(canon, theta)
        ctx = Context(canon, Context.freeze_theta(theta), inst_ids.constraints_id,
                      inst_ids.objective_id, nu or self.nu)
        return ctx, used

    def decide(self, t: float, theta: dict, t_asof: float | None = None) -> CycleResult:
        """One decision cycle. Every cycle leaves exactly one audit record (L6 invariant):
        a cycle that L3 defers (a blocked conflict) or that ends in an error before
        execution is logged with outcome "deferred" or "error", nothing executes, and
        the exception propagates to the caller."""
        t_asof = t if t_asof is None else t_asof
        nu = self.nu
        ctx = None
        try:
            ctx, used = self.context_at(t_asof, theta, nu)
            self.audit.log_encoding((ctx.constraints_id, ctx.objective_id),
                                    encoding(self.formulation))
            decision, inst = self.policy.decide(ctx)
            gate = self.gate.apply(decision.x, inst, ctx, self.formulation)
            blocked = self._monitor(gate, inst, ctx, t)
            # L2 admits the released action before execution: a threshold action without
            # an authenticated approval event is queued, not executed (Table 2, L2; S2).
            action = ({} if gate.released is None else
                      dict(zip(inst.system.names, (float(v) for v in gate.released),
                               strict=True)))
            adm = (Admission(Verdict.REJECT, f"monitor {blocked[0]}: {blocked[1]}") if blocked
                   else self.procedural.admit_action(action, self.approvals, t,
                                                     consumed=self.consumed_approvals()))
            rho = self._provenance(used, decision, inst)
            rho["conflicts"] = {"|".join(map(str, k)): sorted(f.prov.source_id for f in fs)
                                for k, fs in sorted(self.state.conflicts(
                                    self.decision_class, t_asof).items())}
        except Exception as exc:
            self._log_unfinished(t, t_asof, theta, nu, ctx, exc)
            raise
        x_exec, ack = self.executor.execute(gate.released if adm.verdict is Verdict.ADMIT
                                            else None)
        record = self.audit.append(
            t=t, decision_class=self.decision_class.name, state_hash=state_hash(ctx.state),
            state=ctx.state, t_asof=t_asof, theta=ctx.theta,
            constraints_id=ctx.constraints_id, objective_id=ctx.objective_id,
            nu=nu.as_dict(),
            x=None if decision.x is None else [float(v) for v in decision.x],
            cert=decision.certificate, rho=rho, gate_rung=gate.rung,
            gate_outcome=gate.outcome, gate_detail=gate.detail,
            x_released=None if gate.released is None else [float(v) for v in gate.released],
            x_exec=None if x_exec is None else [float(v) for v in x_exec], exec_ack=ack,
            admission={"verdict": adm.verdict.value, "reason": adm.reason,
                       "approvals": list(adm.approvals), "uses": dict(adm.uses),
                       **({"monitor": blocked[0]} if blocked else {})},
        )
        return CycleResult(record, ctx, decision, gate, inst)

    def _monitor(self, gate: GateResult, inst: LPInstance, ctx: Context,
                 t: float) -> tuple[str, str] | None:
        """L5 trace monitors on the released action: (rule, reason) for the first violated rule,
        which holds the action back from execution. The release itself is left as the gate made
        it: a rule over the trace reads inputs outside the logged tuple, so letting it change
        the release would break C2; holding back execution does not, as with L2's queue."""
        if gate.released is None or not self.monitors:
            return None
        names = tuple(inst.system.names)
        released = dict(zip(names, (float(v) for v in gate.released), strict=True))
        logged = logged_tuple(state_hash(ctx.state), ctx.theta, ctx.constraints_id,
                              ctx.objective_id, ctx.nu.as_dict(),
                              observations=gate.detail.get("reread", {}))
        for m in self.monitors:
            reason = m.check(released, t, logged, names, self.audit)
            if reason is not None:
                return m.label, reason
        return None

    def _log_unfinished(self, t: float, t_asof: float, theta: dict, nu: VersionVector,
                        ctx: Context | None, exc: Exception) -> None:
        """The record of a cycle that ended before execution: what it had read, and why."""
        self.audit.append(
            t=t, decision_class=self.decision_class.name,
            state_hash=None if ctx is None else state_hash(ctx.state),
            state=None if ctx is None else ctx.state, t_asof=t_asof,
            theta=Context.freeze_theta(theta) if ctx is None else ctx.theta,
            constraints_id=None if ctx is None else ctx.constraints_id,
            objective_id=None if ctx is None else ctx.objective_id,
            nu=nu.as_dict(), x=None, cert={}, rho={}, gate_rung=self.gate.rung,
            gate_outcome="not run", gate_detail={"error": type(exc).__name__,
                                                 "reason": str(exc)},
            x_released=None, x_exec=None, exec_ack=None, admission=None,
            outcome="deferred" if isinstance(exc, UnresolvedConflict) else "error",
        )

    def _provenance(self, used: dict, decision: Decision, inst: LPInstance) -> dict:
        """rho(x): the facts read, the active rows, the support and the objective (R1)."""
        return {
            "facts": {
                "|".join(map(str, k)): {
                    "source_id": f.prov.source_id,
                    "authenticated": f.prov.authenticated,
                    "component_version": f.prov.component_version,
                    "span": None if f.prov.span is None else
                    [f.prov.span.message_id, f.prov.span.start, f.prov.span.end],
                }
                for k, f in sorted(used.items())
            },
            "constraints": decision.certificate.get("active_rows", []),
            # F_x of G2: the facts behind the active rows (Prop. 4(b))
            "support": sorted({"|".join(map(str, k)) for r in inst.system.rows
                               if r.label in decision.certificate.get("active_rows", [])
                               for k in r.keys}),
            "objective": inst.objective_id,
        }

    def replay(self, record_id: int, from_store: bool = False) -> np.ndarray | None:
        """Re-run L4-L5 on logged inputs (Prop. 4(a)).

        from_store=False uses the logged pi_D(S_hat); True re-reads the store as of
        t_asof, which relies on R2(i). Requires the logged nu to equal the active nu,
        and the logged C-id and J-id to be the content hashes of the active formulation.
        """
        rec = self.audit.get(record_id)
        if not rec.decided:
            raise ValueError(f"record {record_id} is {rec.outcome}: no decision to replay")
        nu = self.nu
        if rec.nu != nu.as_dict():
            raise ValueError("logged nu differs from the active version vector")
        if (rec.constraints_id, rec.objective_id) != encoding_ids(self.formulation):
            raise ValueError("the encoded constraints or objective differ from the logged "
                             "content hashes")
        theta = dict(rec.theta)
        if from_store:
            ctx, _ = self.context_at(rec.t_asof, theta, nu)
        else:
            ctx = Context(rec.state, rec.theta, rec.constraints_id, rec.objective_id, nu)
        decision, inst = self.policy.decide(ctx)
        gate = self.gate
        if isinstance(gate, RecheckGate):
            observed = rec.gate_detail.get("reread")
            if observed is None:
                raise ValueError("recheck replay requires the recorded observations")

            def recorded_value(key):
                name = "|".join(map(str, key))
                if name not in observed:
                    raise ValueError(f"recheck replay is missing the observation for {name}")
                return observed[name]

            gate = RecheckGate(recorded_value, fallback=gate.fallback, decimals=gate.decimals)
        return gate.apply(decision.x, inst, ctx, self.formulation).released
