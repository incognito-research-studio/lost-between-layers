"""Fault-injection evaluation, repaired configuration (seventh external review, step 4).

    PYTHONPATH=<patched src> python tools/ft_repaired.py --out runs.json \
        [--episodes DIR ...] [--negatives DIR ...] [--no-fixed] [--gating]

Written 13 Nov 2026, after round 2 was unblinded, by the architecture role. Round 2 showed
that no frozen check compares the intervals a decision used with those the store holds, that
the restart check compares the number of reloaded deltas rather than their content, and that
unit, calendar, authority and version faults passed unnoticed. This module adds checks for
those obligations as a fourth configuration, `repaired` (solver < generic < full < repaired).
It imports the frozen harness (tools/ft_harness.py) and the extended effect trace
(tools/ft_trace_ext.py) unchanged; the frozen checks, worlds and evaluation stay as they were,
and the frozen results stay reported as they are. Because the checks were designed after
seeing round 2's faults, round 2 (and round 1) can only give regression evidence for them;
evidence that they generalize must come from faults written after they were frozen (round 3,
data/fault_trial/round3/PROTOCOL.md).

New checks (each reads the log, the store, the interpreter's raw reply, declared policy kept as
data, or the formulation evaluated at points of the reported state; none reads the truth):

  interval_store   L3, L4   pre   each quantity a decision used (its logged canonical state)
                                  equals the quantity of the stored fact it names in rho, that
                                  fact selected here with half-open valid and transaction times
  robust_release   L4, L5   pre   a release by a gate that claims robustness (box_robust,
                                  vertex) satisfies every row of the formulation built at every
                                  vertex, over that row's uncertain keys, of the box the gate
                                  claims: the reported intervals, widened by any margin it logs
  recheck_complete L5       pre   a recheck release logs a re-read observation for every state
                                  key its formulation's rows read
  span_cover       L3       pre   under the declared span policy, the committed interval covers
                                  both the incoming claim and the rival claim it replaced
  commit_precedence L3      pre   what L3 commits from a message is what the declared ranks and
                                  conflict policy say it commits (the frozen precedence check
                                  sees only facts that were committed and read)
  blocked_read     L3       pre   under the declared block policy, no decision reads a key whose
                                  contradiction is still blocked (no authenticated record since)
  l1_reference     L1       pre   the facts committed from a message have the scope (subject,
                                  predicate, valid time) and the values of an independent reading
                                  of the interpreter's raw reply (half-open valid times as the
                                  reply states them)
  l2_authority     L2       pre   no stored fact from an unauthenticated source sets an
                                  authority-bearing predicate or loosens a restriction flag
                                  (a value below 1); design rule 2, declared below as data
  version_pinned   L1, reg. pre   every message that got past the version check was answered by
                                  exactly the pinned interpreter version
  unit_declared    L1, L3   pre   an interpreted quantity carries the unit the system of record
                                  uses for its predicate, where the record declares one
  restart_content  L6       post  the reloaded log has the same committed deltas, by content,
                                  and the same encodings as the log that was saved

Gating (--gating): the repaired configuration's pre-execution checks that never need a clean
run for comparison (relative=False) run before every release as an L5 hold, through the
pipeline's own monitor hook: each cycle is first decided on a deep copy of the pipeline, the
checks are evaluated on the copy's log, and if any fires the real cycle is decided with the
release held (nothing executes, the record names the checks). The gate fails closed: a check
that fires on the log holds every later release of the run too, since the log is append-only.
This measures containment directly instead of counting releases a check could have held.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_harness as H  # noqa: E402  (frozen, unchanged)
import ft_trace_ext  # noqa: E402,F401  (installs the extended trace on ft_harness)

ROOT = H.ROOT
CONFIGS = ("solver", "generic", "full", "repaired")
# Design rule 2 and the L2 contract as data (Table 2): authority-bearing predicates and
# restriction flags, whose value 1 means the restriction applies in full.
AUTHORITY_PREDICATES = frozenset({"approval", "release", "identity"})
RESTRICTION_PREDICATES = frozenset({"hold", "suspended"})
# Declared L3 conflict policies: the e1v2 world's (its benchmark declares span, Section 9.5)
# and the deployment default of configs/versions.yaml for episodes that declare none.
E1V2_RESOLUTION = "span"
DEFAULT_RESOLUTION = yaml.safe_load((ROOT / "configs/versions.yaml").read_text())[
    "resolution"].split("@")[0]
ROBUST_RUNGS = ("box_robust", "vertex")
MAX_ROW_KEYS = 12
TOL = 1e-6

# -- capture of the interpreter's raw replies and of the declared trial --------------------
# The raw reply is what the model returned before L1 parsed it; a deployment that logs it
# can re-read it. The harness records it by wrapping the interpreters' `interpret` (the
# fixture interpreter and the e1v2 replay), not by trusting the code under test to log it.

_CAPTURE: dict | None = None


def _install_capture() -> None:
    from dsoi.experiments import e1v2_score
    from dsoi.layers import l1_interpreter

    fixture_interpret = l1_interpreter.FixtureInterpreter.interpret
    replay_interpret = e1v2_score.Replay.interpret

    def fixture_wrap(self, message, schema):
        if _CAPTURE is not None:
            fx = getattr(self, "_fixtures", {}).get(message.message_id)
            if isinstance(fx, dict):
                _CAPTURE[message.message_id] = {"payload": fx.get("output"),
                                                "conf": fx.get("conf"),
                                                "source": message.source_id}
        return fixture_interpret(self, message, schema)

    def replay_wrap(self, message, schema):
        if _CAPTURE is not None:
            reply = getattr(self, "replies", {}).get(message.message_id)
            if isinstance(reply, dict):
                _CAPTURE[message.message_id] = {"payload": _json_payload(reply.get("text")),
                                                "conf": None, "source": message.source_id}
        return replay_interpret(self, message, schema)

    l1_interpreter.FixtureInterpreter.interpret = fixture_wrap
    e1v2_score.Replay.interpret = replay_wrap


def _json_payload(text):
    """The JSON object in a model's reply text, or None (a reply this reader cannot read is
    left to L1; the checks that use it then say nothing about that message)."""
    if not isinstance(text, str):
        return None
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


_frozen_episode = H.episode


_PENDING_RESOLUTION: str | None = None


def _episode(scn, trial, run_id):
    """The frozen episode builder, with the trial's declared conflict policy handed to the run
    as soon as it exists (gating evaluates the checks while the events run)."""
    global _PENDING_RESOLUTION
    _PENDING_RESOLUTION = trial.get("resolution") or DEFAULT_RESOLUTION
    try:
        run = _frozen_episode(scn, trial, run_id)
    finally:
        _PENDING_RESOLUTION = None
    run.declared_resolution = trial.get("resolution") or DEFAULT_RESOLUTION
    return run


H.episode = _episode     # the frozen scenario and episode builders look `episode` up here


def _declared_resolution(run) -> str:
    return getattr(run, "declared_resolution", None) or (
        E1V2_RESOLUTION if run.id.startswith("e1v2/") else DEFAULT_RESOLUTION)


def _raw(run) -> dict:
    return getattr(run, "raw", {}) or {}


# -- reference reading of a raw reply ---------------------------------------------------------

def reference_ops(payload) -> list[dict] | None:
    """The operations of a raw reply as it states them: subject, predicate, kind, half-open
    valid time [valid_from, valid_to), value, interval and unit. None if it is not a delta."""
    if not isinstance(payload, dict) or payload.get("kind") != "delta":
        return None
    ops = []
    try:
        for raw in payload.get("ops", []):
            kind = raw["op"]
            op = {"s": raw["subject"], "p": raw["predicate"], "kind": kind,
                  "vt": (float(raw["valid_from"]), float(raw["valid_to"]))}
            if kind != "retract":
                lo, hi = raw.get("lo"), raw.get("hi")
                op.update(value=float(raw["value"]), lo=None if lo is None else float(lo),
                          hi=None if hi is None else float(hi), unit=raw.get("unit", ""))
            ops.append(op)
    except (KeyError, TypeError, ValueError):
        return None
    return ops


def _q(x):
    return None if x is None else round(float(x), 6)


def _holds_at(f, v: float, t: float) -> bool:
    """A fact valid at v and known at t, with half-open valid and transaction times, read
    here rather than through the store's own selection, which is code under test."""
    return f.vt[0] <= v < f.vt[1] and f.tt[0] <= t < f.tt[1]


def _overlaps(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _ingests(run, outcome: str):
    ids = set(run.message_ids)
    return [r for r in run.p.audit.ingests if r.message_id in ids and r.outcome == outcome]


# -- the new checks ---------------------------------------------------------------------------

def c_interval_store(run):
    store = run.p.state.store
    for r in run.p.audit.decisions():
        state = {tuple(k): (v, lo, hi, unit) for k, v, lo, hi, unit in (r.state or ())}
        for key, read in (r.rho or {}).get("facts", {}).items():
            s, pred, v = key.split("|")
            entry = state.get((s, pred, float(v)))
            if entry is None:
                return True
            span = read.get("span")
            want = H._ident(read["source_id"], read["authenticated"],
                            None if not span else [span[0]])
            matches = []
            for f in store.facts():      # selected here, half-open in both times, not by asof
                if (f.s != s or f.p != pred or not hasattr(f.o, "value")
                        or not _holds_at(f, float(v), r.t_asof)):
                    continue
                fspan = None if f.prov.span is None else [f.prov.span.message_id]
                if H._ident(f.prov.source_id, f.prov.authenticated, fspan) == want:
                    matches.append((_q(f.o.value), _q(f.o.lo), _q(f.o.hi), f.o.unit))
            got = (_q(entry[0]), _q(entry[1]), _q(entry[2]), entry[3])
            if got not in matches:
                return True
    return False


def claimed_box(key, lo, hi, detail: dict) -> tuple:
    """The interval a gate claims to protect for one key: the reported one, widened by the
    margin the gate logged (the e1v2 benchmark's declared rule: capacity intervals widened by
    the margin within [0, nominal capacity])."""
    margin = detail.get("reported_margin")
    if margin is None or key[1] != "capacity":
        return (lo, hi)
    from dsoi.experiments.e1v2_notices import NOMINAL
    return (max(0.0, lo - margin), min(NOMINAL, hi + margin))


def c_robust_release(run):
    from dsoi.constraints import LinearSystem
    form = run.p.formulation
    for r in run.p.audit.decisions():
        if r.gate_rung not in ROBUST_RUNGS or r.x_released is None or not r.state:
            continue
        x = [float(v) for v in r.x_released]
        theta = dict(r.theta)
        base = form.build(r.state, theta).system
        uncertain = {tuple(k): claimed_box(k, lo, hi, r.gate_detail or {})
                     for k, v, lo, hi, unit in r.state if lo is not None and hi is not None}
        uncertain = {k: b for k, b in uncertain.items() if b[1] - b[0] > 1e-12}
        for row in base.rows:
            keys = [tuple(k) for k in row.keys if tuple(k) in uncertain]
            if not keys:
                continue
            if len(keys) > MAX_ROW_KEYS:
                keys = keys[:MAX_ROW_KEYS]
            for corner in itertools.product(*[uncertain[k] for k in keys]):
                at = dict(zip(keys, corner, strict=True))
                state = tuple((k, at[tuple(k)], at[tuple(k)], at[tuple(k)], unit)
                              if tuple(k) in at else (k, v, lo, hi, unit)
                              for k, v, lo, hi, unit in r.state)
                sysv = form.build(state, theta).system
                rows = [w for w in sysv.rows if w.label == row.label]
                if not rows:
                    return True
                one = LinearSystem(tuple(rows), sysv.lower, sysv.upper, sysv.names)
                if not one.nominal_feasible(x):
                    return True
    return False


def c_recheck_complete(run):
    """A release by the recheck gate logs a re-read observation for every state key that a
    row of its formulation reads (C2 and replay need all of them; Prop. 2(a))."""
    form = run.p.formulation
    for r in run.p.audit.decisions():
        if r.gate_rung != "recheck" or r.x_released is None or not r.state:
            continue
        seen = set((r.gate_detail or {}).get("reread", {}))
        rows = form.build(r.state, dict(r.theta)).system.rows
        need = {"|".join(map(str, k)) for row in rows for k in row.keys}
        if not need <= seen:
            return True
    return False


def c_span_cover(run):
    if _declared_resolution(run) != "span":
        return False
    store, raw = run.p.state.store, _raw(run)
    facts = store.facts()
    for rec in _ingests(run, "resolved"):
        ops = reference_ops((raw.get(rec.message_id) or {}).get("payload"))
        if ops is None:
            continue
        src = (raw.get(rec.message_id) or {}).get("source")
        for op in ops:
            if op["kind"] == "retract":
                continue
            claims = [(op["value"] if op["lo"] is None else op["lo"],
                       op["value"] if op["hi"] is None else op["hi"])]
            claims += [g.o.interval() for g in facts
                       if g.s == op["s"] and g.p == op["p"] and g.tt[1] == rec.t
                       and _overlaps(g.vt, op["vt"]) and not g.prov.authenticated
                       and g.prov.source_id != src and hasattr(g.o, "interval")]
            lo, hi = min(c[0] for c in claims), max(c[1] for c in claims)
            for f in facts:
                if (f.s == op["s"] and f.p == op["p"] and f.tt[0] == rec.t
                        and _overlaps(f.vt, op["vt"]) and hasattr(f.o, "interval")):
                    flo, fhi = f.o.interval()
                    if flo > lo + TOL or fhi < hi - TOL:
                        return True
    return False


def _rank_of(run, source: str) -> int:
    return max((v for k, v in run.policy.priority.items() if source.startswith(k)), default=0)


def _same_quantity(o, q) -> bool:
    return (hasattr(q, "value") and (_q(o["value"]), _q(o["lo"]), _q(o["hi"]), o["unit"])
            == (_q(q.value), _q(q.lo), _q(q.hi), q.unit))


def c_commit_precedence(run):
    """What L3 commits from a message is what the declared policy says it commits: a claim
    with no rival is committed; under the priority policy a claim is committed exactly when its
    source ranks at least as high as the highest-ranked rival (ties go to the newer claim);
    under the span policy the committed point is the winner's. Rivals are current,
    unauthenticated facts of the same subject and predicate from another source, over
    overlapping valid time, with a different value. Ranks are the declared ones (data)."""
    policy = _declared_resolution(run)
    if policy not in ("latest", "priority", "span"):
        return False
    store, raw = run.p.state.store, _raw(run)
    facts = store.facts()
    ids = set(run.message_ids)
    for rec in run.p.audit.ingests:
        if rec.message_id not in ids or rec.outcome not in ("committed", "resolved",
                                                             "overridden"):
            continue
        cap = raw.get(rec.message_id) or {}
        ops = reference_ops(cap.get("payload"))
        if ops is None:
            continue
        src = cap.get("source")
        for op in ops:
            if op["kind"] == "retract":
                continue
            rivals = [f for f in facts
                      if f.s == op["s"] and f.p == op["p"] and _overlaps(f.vt, op["vt"])
                      and not f.prov.authenticated and f.prov.source_id != src
                      and f.tt[0] < rec.t <= f.tt[1] and not _same_quantity(op, f.o)]
            new = [f for f in facts
                   if f.s == op["s"] and f.p == op["p"] and f.tt[0] == rec.t
                   and _overlaps(f.vt, op["vt"]) and not f.prov.authenticated]
            mine = [f for f in new if f.prov.span is not None
                    and f.prov.span.message_id == rec.message_id]
            if not rivals or policy == "latest":
                if not mine:
                    return True
                continue
            rival = max(rivals, key=lambda f: (_rank_of(run, f.prov.source_id), f.tt[0]))
            wins = _rank_of(run, src) >= _rank_of(run, rival.prov.source_id)
            if policy == "priority" and wins != bool(mine):
                return True
            if policy == "span":
                point = op["value"] if wins else rival.o.value
                if not new or any(abs(f.o.value - point) > TOL for f in new
                                  if hasattr(f.o, "value") and f.vt[0] >= op["vt"][0]
                                  and f.vt[1] <= op["vt"][1]):
                    return True
    return False


def c_blocked_read(run):
    if _declared_resolution(run) != "block":
        return False
    store, raw = run.p.state.store, _raw(run)
    blocks = []
    for rec in _ingests(run, "conflict"):
        for op in reference_ops((raw.get(rec.message_id) or {}).get("payload")) or []:
            blocks.append((rec.t, op["s"], op["p"], op["vt"]))
    if not blocks:
        return False
    auth = [f for f in store.facts() if f.prov.authenticated]
    for r in run.p.audit.decisions():
        for key in (r.rho or {}).get("facts", {}):
            s, pred, v = key.split("|")
            v = float(v)
            for t_b, bs, bp, vt in blocks:
                if (bs, bp) != (s, pred) or not vt[0] <= v < vt[1] or r.t_asof < t_b:
                    continue
                cleared = any(f.s == s and f.p == pred and _overlaps(f.vt, vt)
                              and t_b < f.tt[0] <= r.t_asof for f in auth)
                if not cleared:
                    return True
    return False


def c_l1_reference(run):
    store, raw = run.p.state.store, _raw(run)
    facts = store.facts()
    for rec in _ingests(run, "committed"):
        cap = raw.get(rec.message_id) or {}
        ops = reference_ops(cap.get("payload"))
        if ops is None:
            continue
        want = sorted(((o["s"], o["p"], o["vt"]), o) for o in ops if o["kind"] != "retract")
        got = sorted(((f.s, f.p, tuple(f.vt)), f) for f in facts
                     if f.prov.span is not None and f.prov.span.message_id == rec.message_id
                     and f.tt[0] == rec.t and f.prov.component == "L1")
        if [k for k, _ in want] != [k for k, _ in got]:
            return True
        for (_, o), (_, f) in zip(want, got, strict=True):
            if cap.get("conf") is not None and o["p"] in RESTRICTION_PREDICATES:
                continue            # a hedged yes/no reading is changed on purpose (Def. 5)
            if not hasattr(f.o, "value"):
                return True
            if ((_q(o["value"]), _q(o["lo"]), _q(o["hi"]), o["unit"])
                    != (_q(f.o.value), _q(f.o.lo), _q(f.o.hi), f.o.unit)):
                return True
    return False


def c_l2_authority(run):
    for f in run.p.state.store.facts():
        if f.prov.authenticated:
            continue
        if f.p in AUTHORITY_PREDICATES:
            return True
        if f.p in RESTRICTION_PREDICATES and hasattr(f.o, "value") and f.o.value < 1 - 1e-12:
            return True
    return False


def c_version_pinned(run):
    pinned = run.policy.declared_versions[-1]
    ids = set(run.message_ids)
    return any(r.message_id in ids and r.outcome != "abstained" and r.nu_i != pinned
               for r in run.p.audit.ingests)


def unit_key(unit: str) -> str:
    """A unit's spelling reduced to a comparable form: lower case, "per" read as "/",
    underscores and spaces dropped ("units per week" and "units_per_week" are "units/week";
    "units/day" stays different). Fixed 13 Nov 2026 after the clean e1v2 replies showed the
    models spell the record's unit both ways."""
    u = unit.lower().replace("_", " ")
    u = re.sub(r"\s+per\s+", "/", u)
    return re.sub(r"\s+", "", u)


def c_unit_declared(run):
    facts = run.p.state.store.facts()
    declared: dict = {}
    for f in facts:
        if f.prov.authenticated and hasattr(f.o, "unit") and f.o.unit:
            declared.setdefault(f.p, set()).add(unit_key(f.o.unit))
    return any(f.prov.component == "L1" and hasattr(f.o, "unit") and f.p in declared
               and unit_key(f.o.unit) not in declared[f.p] for f in facts)


def _delta_rows(deltas):
    return [[float(t), repr(d)] for t, d in deltas]


def c_restart_content(run):
    from dsoi.layers.l6_audit import AuditLog
    p = run.p
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "audit.jsonl"
        head = p.audit.to_jsonl(path)
        loaded = AuditLog.from_jsonl(path, expected_head=head)
    return (_delta_rows(loaded.deltas) != _delta_rows(p.audit.deltas)
            or json.dumps(loaded.encodings, sort_keys=True, default=str)
            != json.dumps(p.audit.encodings, sort_keys=True, default=str))


NEW_CHECKS = (
    H.Check("interval_store", "repaired", ("L3", "L4"), "pre", fn=c_interval_store),
    H.Check("robust_release", "repaired", ("L4", "L5"), "pre", fn=c_robust_release),
    H.Check("recheck_complete", "repaired", ("L5",), "pre", fn=c_recheck_complete),
    H.Check("span_cover", "repaired", ("L3",), "pre", fn=c_span_cover),
    H.Check("commit_precedence", "repaired", ("L3",), "pre", fn=c_commit_precedence),
    H.Check("blocked_read", "repaired", ("L3",), "pre", fn=c_blocked_read),
    H.Check("l1_reference", "repaired", ("L1",), "pre", fn=c_l1_reference),
    H.Check("l2_authority", "repaired", ("L2",), "pre", fn=c_l2_authority),
    H.Check("version_pinned", "repaired", ("L1", "registry"), "pre", fn=c_version_pinned),
    H.Check("unit_declared", "repaired", ("L1", "L3"), "pre", fn=c_unit_declared),
    H.Check("restart_content", "repaired", ("L6",), "post", fn=c_restart_content),
)
CHECKS = H.CHECKS + NEW_CHECKS
GATING = tuple(c for c in CHECKS if c.timing == "pre" and not c.relative
               and c.name != "exception")


# -- gating through the pipeline's monitor hook ------------------------------------------------

_GATE_ON = False


def _fires(run, checks) -> set:
    out = set()
    for c in checks:
        try:
            if c.fn(run):
                out.add(c.name)
        except Exception:
            out.add(c.name)
    return out


def _install_gating() -> None:
    from dsoi.pipeline import Pipeline
    decide, monitor = Pipeline.decide, Pipeline._monitor

    def gated_monitor(self, gate, inst, ctx, t):
        held = getattr(self, "_ft_hold", None)
        if held:
            self._ft_hold = None
            return ("ft-gate", "check(s) " + ", ".join(sorted(held)) + " fired before release")
        return monitor(self, gate, inst, ctx, t)

    def gated_decide(self, t, theta, t_asof=None):
        view = getattr(self, "_ft_run", None)
        if not _GATE_ON or view is None or getattr(self, "_ft_shadow", False):
            return decide(self, t, theta, t_asof)
        self._ft_run = None                     # do not copy the run (truth lives there)
        try:
            shadow = copy.deepcopy(self)
        finally:
            self._ft_run = view
        shadow._ft_shadow = True
        try:
            decide(shadow, t, theta, t_asof)
        except Exception:
            return decide(self, t, theta, t_asof)    # deferred or failed: nothing executes
        rec = shadow.audit.records[-1]
        if rec.x_released is None:
            return decide(self, t, theta, t_asof)
        sview = copy.copy(view)
        sview.p, sview.inv = shadow, None
        new = _fires(sview, GATING)
        if new:
            self._ft_hold = new
            view.holds.append({"t": float(t), "checks": sorted(new)})
        return decide(self, t, theta, t_asof)

    Pipeline._monitor = gated_monitor
    Pipeline.decide = gated_decide


# -- evaluation (the frozen evaluate, with the capture, the new checks and gating) ---------------

def _bind(run) -> None:
    """Give the pipeline a view of its run, so gating can evaluate the checks mid-run."""
    run.raw = _CAPTURE
    run.holds = []
    if _PENDING_RESOLUTION is not None:
        run.declared_resolution = _PENDING_RESOLUTION
    run.p._ft_run = run


def evaluate(make, gating: bool = False) -> dict:
    global _CAPTURE, _GATE_ON
    _CAPTURE, _GATE_ON = {}, gating
    try:
        run = _make_bound(make)
    except Exception as exc:
        _CAPTURE, _GATE_ON = None, False
        return {"crashed": True, "exception": f"{type(exc).__name__}: {exc}"[:300],
                "exception_layer": H._layer_of(exc.__traceback__), "fired": ["exception"],
                "trace": None, "released": 0, "violations": 0, "decisions": 0, "holds": []}
    run.raw = _CAPTURE
    fired = sorted(_fires(run, CHECKS), key=[c.name for c in CHECKS].index)
    decided = run.p.audit.decisions()
    executed = [r for r in decided if r.x_exec is not None]
    viol = 0
    for r in executed:
        try:
            viol += bool(run.truth(r.x_exec))
        except Exception:
            pass
    held_harm = 0
    for r in decided:
        if r.x_exec is None and r.x_released is not None and (r.admission or {}).get(
                "monitor") == "ft-gate":
            try:
                held_harm += bool(run.truth(r.x_released))
            except Exception:
                pass
    out = {"crashed": False, "exception": run.exception, "exception_layer": run.exception_layer,
           "fired": fired, "trace": H.trace(run), "released": len(executed),
           "violations": viol, "decisions": len(decided), "holds": getattr(run, "holds", []),
           "held_violating": held_harm}
    _CAPTURE, _GATE_ON = None, False
    return out


def _make_bound(make):
    """Run `make`, binding the pipeline to its run as soon as the run object exists, so that
    gating sees it from the first decision. The frozen builders create the Run before the
    first event, so wrapping the Run constructor is enough."""
    original = H.Run.__init__

    def init(self, *a, **k):
        original(self, *a, **k)
        _bind(self)
    H.Run.__init__ = init
    try:
        return make()
    finally:
        H.Run.__init__ = original


def dir_runs(root: Path, prefix: str):
    from dsoi.scenarios import Scenario
    for d in sorted(p for p in root.iterdir() if (p / "scenario.yaml").exists()):
        spec = yaml.safe_load((d / "scenario.yaml").read_text())
        trial = spec.pop("trial", {})
        run_id = f"{prefix}/{d.name}"
        yield run_id, (lambda d=d, spec=spec, trial=trial, run_id=run_id:
                       H.episode(Scenario(d, spec), trial, run_id))


def check_meta() -> dict:
    return {c.name: {"config": c.config, "layers": list(c.layers), "timing": c.timing,
                     "relative": c.relative, "gating": c in GATING} for c in CHECKS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, nargs="*", default=[])
    ap.add_argument("--negatives", type=Path, nargs="*", default=[])
    ap.add_argument("--no-fixed", action="store_true")
    ap.add_argument("--gating", action="store_true")
    args = ap.parse_args(argv)
    _install_capture()
    _install_gating()
    gens = [] if args.no_fixed else [H.e1v2_runs(), H.scenario_runs()]
    for i, d in enumerate(args.episodes):
        gens.append(dir_runs(d, "episodes" if i == 0 else f"episodes{i + 1}"))
    for i, d in enumerate(args.negatives):
        gens.append(dir_runs(d, "negatives" if i == 0 else f"negatives{i + 1}"))
    runs = {}
    for gen in gens:
        for run_id, go in gen:
            runs[run_id] = evaluate(go, gating=args.gating)
    args.out.write_text(json.dumps({"checks": check_meta(), "gating": args.gating,
                                    "runs": runs}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
