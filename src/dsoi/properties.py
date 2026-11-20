"""Checkable predicates for the properties of Sec. IV (Table II).

Log predicates (G1, R2(ii), C2, K-prop, audit completeness) run over L6; those about
decisions skip the records of deferred or error-terminated cycles.
Replay checks (C1, G2) re-execute logged records. K-exec cannot be established
from the log alone: `k_exec_argued` checks the gate construction recorded in the
log, and `k_exec_true` needs the true state, which only a simulation has.
"""

from __future__ import annotations

import numpy as np

from dsoi.canon import CanonicalState
from dsoi.constraints import LinearSystem, solve_lp
from dsoi.layers.l4_decision import Formulation
from dsoi.layers.l6_audit import AuditLog, AuditRecord
from dsoi.pipeline import Pipeline
from dsoi.store import BitemporalStore


def g1(record: AuditRecord, log: AuditLog) -> bool:
    """Every supporting fact has an authenticated source or a span of a logged message."""
    for fact in record.rho["facts"].values():
        if fact["authenticated"]:
            continue
        span = fact["span"]
        if span is None:
            return False
        message_id, start, end = span
        msg = log.messages.get(message_id)
        if msg is None or not 0 <= start < end <= len(msg["text"]):
            return False
    return True


def _support_system(record: AuditRecord, formulation: Formulation):
    """The LP restricted to the logged support: active rows and active bounds only."""
    inst = formulation.build(record.state, dict(record.theta))
    rows = tuple(r for r in inst.system.rows if r.label in record.cert.get("active_rows", []))
    lower, upper = np.full(inst.system.n, -np.inf), np.full(inst.system.n, np.inf)
    for name, side in record.cert.get("active_bounds", []):
        j = inst.system.names.index(name)
        if side == "upper":
            upper[j] = inst.system.upper[j]
        else:
            lower[j] = inst.system.lower[j]
    return inst, LinearSystem(rows, lower, upper, inst.system.names)


def support_rank(record: AuditRecord, formulation: Formulation) -> tuple[int, int]:
    """(rank of the active constraints, number of variables): Prop. 4(b)'s condition."""
    inst, sup = _support_system(record, formulation)
    eye = np.eye(sup.n)
    a = [r.a_bar for r in sup.rows] + [eye[j] for j in range(sup.n)
                                       if np.isfinite(sup.lower[j]) or np.isfinite(sup.upper[j])]
    return (int(np.linalg.matrix_rank(np.array(a))) if a else 0), sup.n


def g2(record: AuditRecord, formulation: Formulation, tol: float = 1e-6) -> bool:
    """G2 (sufficiency): the support alone reproduces the decision.

    Re-solves the LP with only the active rows and bounds kept (F_x and C_x of
    Prop. 4(b)). When their rank is below n the optimum is not determined by the
    support, and this check can fail even for a correct decision: the support
    must then also record a basis or tie-breaking rule.
    """
    if record.x is None:
        return True
    inst, sup = _support_system(record, formulation)
    res = solve_lp(inst.c, sup)
    return res.status == 0 and bool(np.allclose(res.x, record.x, atol=tol))


def r2_i(store: BitemporalStore, probes: dict[tuple[float, float], frozenset]) -> bool:
    """No as-of view recorded earlier has changed: probes map (v, t) -> snapshot then."""
    return all(store.snapshot(v, t) == snap for (v, t), snap in probes.items())


def r2_ii(log: AuditLog, message_ids: list[str], delta_d: float | None = None) -> bool:
    """Every accepted change-asserting message yields a commit, abstention or conflict
    record, within `delta_d` of its arrival when a bound is given (bounded disposition
    delay). It bounds how long a message waits for an outcome, not how current the state is."""
    first = {}
    for r in log.ingests:
        first.setdefault(r.message_id, r.t)
    for m in message_ids:
        if m not in first:
            return False
        if delta_d is not None:
            arrived = log.messages.get(m, {}).get("received_at", first[m])
            if first[m] - arrived > delta_d:
                return False
    return True


def r2_iii(log: AuditLog, store: BitemporalStore) -> bool:
    """R2(iii) (Table 3): re-applying the logged deltas, at their transaction times, to an
    empty store reproduces the store. A store that deletes history fails it."""
    fresh = BitemporalStore()
    for t, delta in log.deltas:
        fresh.commit(delta, t)
    return fresh.facts() == store.facts()


def c1_replay(pipeline: Pipeline, record_id: int, from_store: bool = False) -> bool:
    """Replay releases the logged action."""
    rec = pipeline.audit.get(record_id)
    again = pipeline.replay(record_id, from_store=from_store)
    if rec.x_released is None or again is None:
        return rec.x_released is None and again is None
    return bool(np.allclose(again, rec.x_released))


def c2_violations(log: AuditLog, include_nu: bool = True) -> list[tuple[int, int]]:
    """Pairs of records of one class with equal logged tuples but different decisions.

    include_nu=False models a system that does not log versions (Prop. 2).
    """
    seen: dict[tuple, AuditRecord] = {}
    bad = []
    for rec in log.decisions():
        key = (rec.decision_class,) + rec.logged_tuple(include_nu)
        prev = seen.get(key)
        if prev is not None and prev.x_released != rec.x_released:
            bad.append((prev.id, rec.id))
        seen.setdefault(key, rec)
    return bad


def state_with_reread(state: tuple, reread: dict) -> tuple:
    """The logged state with the values an execution-time re-read logged (exact values)."""
    return tuple((key, reread["|".join(map(str, key))], None, None, unit)
                 if "|".join(map(str, key)) in reread else (key, v, lo, hi, unit)
                 for key, v, lo, hi, unit in state)


def k_prop(record: AuditRecord, formulation: Formulation) -> bool:
    """x in F(S_hat, C) or noop, for the policy's output and the released action (R4),
    re-evaluated from the logged canonical state. For the released action, S_hat includes
    any execution-time re-read that L5 logged (design rule 1: the re-read is an input).

    Counting noop as compliant follows R4; noop is not always feasible for the
    true constraints (audit item 4)."""
    inst = formulation.build(record.state, dict(record.theta))
    if record.x is not None and not inst.system.nominal_feasible(np.array(record.x)):
        return False
    if record.x_released is None:
        return True
    reread = (record.gate_detail or {}).get("reread")
    if reread:
        inst = formulation.build(state_with_reread(record.state, reread), dict(record.theta))
    return inst.system.nominal_feasible(np.array(record.x_released))


def execution_fidelity(record: AuditRecord, tol: float = 1e-6) -> bool:
    """x^e = x: execution did what the gate released (the GSN goal missing beside A1-A3).

    Checkable on the log because L6 records both the release and the execution report.
    It certifies the report, not the physical world: a faulty report passes it."""
    if (record.admission or {}).get("verdict") in ("queue", "reject"):  # held back (L2, L5)
        return record.x_exec is None and record.exec_ack is None
    if record.x_released is None or record.x_exec is None:
        return record.x_released is None and record.x_exec is None
    return bool(np.allclose(record.x_exec, record.x_released, atol=tol, rtol=0))


def k_exec_argued(record: AuditRecord, formulation: Formulation) -> bool:
    """Table 3: the released action lies in F_box(o) and execution reported doing it, or
    x^e is feasible for a re-read logged with the record.

    The first branch argues K-exec only under A-SPEC and A-COVER (the GSN specification
    and coverage goals); the second only under A-SPEC and A-ORACLE. Neither trusts the
    logged rung label alone."""
    if record.x_exec is None:
        return True
    inst = formulation.build(record.state, dict(record.theta))
    if record.gate_rung == "box_robust":
        return (record.x_released is not None and execution_fidelity(record)
                and inst.system.robust_feasible(np.array(record.x_released)))
    reread = (record.gate_detail or {}).get("reread")
    if record.gate_rung == "recheck" and reread:
        true_inst = formulation.build(state_with_reread(record.state, reread),
                                      dict(record.theta))
        return true_inst.system.nominal_feasible(np.array(record.x_exec))
    return False


def r5(registry) -> bool:
    """R5 on the version registry (Table 3): every transition is logged, and every
    promotion changes the version vector and passed every Inv check."""
    return all(t.new != t.old and all(t.results.values())
               for t in registry.history if t.promoted)


def k_exec_true(x_exec, true_state: CanonicalState, theta: dict,
                formulation: Formulation) -> bool:
    """x^e in F(S*, C*), assuming A-SPEC (C = C*). Simulation only."""
    if x_exec is None:
        return True
    inst = formulation.build(true_state, theta)
    return inst.system.nominal_feasible(np.asarray(x_exec, dtype=float))


def _values(state: CanonicalState) -> dict[tuple, tuple]:
    return {key: (v, lo, hi) for key, v, lo, hi, _unit in state}


def interpretation_differs(state_hat: CanonicalState, state_true: CanonicalState) -> list[tuple]:
    """Keys where pi_R(C)(S_hat) != pi_R(C)(S*): the interpretation cause in Prop. 3(c)."""
    hat, true = _values(state_hat), _values(state_true)
    return [k for k in sorted(true) if k not in hat or abs(hat[k][0] - true[k][0]) > 1e-9]


def coverage_failures(state_hat: CanonicalState, state_true: CanonicalState) -> list[tuple]:
    """Keys whose true value lies outside the reported interval: S* not in Gamma_hat(o).

    This is GSN goal A2 failing. An exact value counts as a degenerate interval,
    which is how a scope misread shows up (the key never received an interval).
    """
    hat, true = _values(state_hat), _values(state_true)
    out = []
    for k in sorted(true):
        if k not in hat:
            out.append(k)
            continue
        v, lo, hi = hat[k]
        lo = v if lo is None else lo
        hi = v if hi is None else hi
        if not lo - 1e-9 <= true[k][0] <= hi + 1e-9:
            out.append(k)
    return out


def unsafe_coverage_failures(state_hat: CanonicalState,
                             state_true: CanonicalState) -> list[tuple]:
    """Coverage failures on the unsafe side: the true value is below the reported lower bound.

    For upper-bound constraints such as capacity, only these can make the box gate
    release an infeasible action; a true value above the interval is harmless.
    """
    hat, true = _values(state_hat), _values(state_true)
    out = []
    for k in sorted(true):
        v, lo, _ = hat.get(k, (float("inf"), None, None))
        lo = v if lo is None else lo
        if true[k][0] < lo - 1e-9:
            out.append(k)
    return out


def inv_report(pipeline: Pipeline, message_ids: list[str]) -> dict[str, bool]:
    """Inv = {G1, R2, C1, K-prop, audit completeness} over the log (R5(ii)).

    G1, C1 and K-prop concern decisions, so they run over the decided records; deferred
    and error records count only for audit completeness. R2 is checked through R2(ii)
    and R2(iii) here; R2(i) needs earlier snapshots (see `r2_i`).
    """
    log = pipeline.audit
    decided = log.decisions()
    return {
        "G1": all(g1(r, log) for r in decided),
        "R2": r2_ii(log, message_ids) and r2_iii(log, pipeline.state.store),
        # replay from the logged state: C1 must not depend on R2(i), which the store checks
        "C1": all(c1_replay(pipeline, r.id) for r in decided),
        "K-prop": all(k_prop(r, pipeline.formulation) for r in decided),
        "audit": audit_complete(log),
    }


def audit_complete(log: AuditLog) -> bool:
    """Hash chain verifies; one record per execution acknowledgement; no deferred or
    error record shows an execution; every logged C-id and J-id resolves to an encoding.

    One record per decision cycle is enforced where records are written (Pipeline.decide):
    a log cannot show a cycle that left no record."""
    acks = [r.exec_ack for r in log.records if r.exec_ack is not None]
    executed = [r for r in log.records if r.x_exec is not None]
    unfinished_ok = all(r.x_exec is None and r.exec_ack is None
                        for r in log.records if not r.decided)
    resolvable = all(i in log.encodings for r in log.decisions()
                     for i in (r.constraints_id, r.objective_id))
    return (log.verify() and len(acks) == len(set(acks)) == len(executed)
            and unfinished_ok and resolvable)
