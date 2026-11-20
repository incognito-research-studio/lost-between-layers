"""Fault-injection evaluation, harness side: worlds, checks and the effect trace.

    uv run python tools/ft_harness.py --out runs.json [--negatives DIR]

Sixth external review (8 Nov 2026), experiment 1. `tools/ft_trial.py` runs this module in
a fresh interpreter against a copy of src/ with one fault patch applied, so every `dsoi`
import below may be faulty; the checks here and `dsoi.properties`, which no patch may
touch, are the instruments. Protocol: data/fault_trial/PROTOCOL.md.

Worlds (each a list of runs; a run is one pipeline taken through its events):
  e1v2       the corrected replay of the stored e1v2 replies (test part, three models,
             widened box gate at the stored margins): 672 runs
  scenarios  fixed episodes of the scenario suite S1-S8 (EPISODES), which exercise the
             recheck and its recorded observations, approvals and holds, conflicts, a
             declared model update and a late correction
  negatives  legitimate but unusual inputs written for the trial (scenario directories,
             each with a `trial:` block), run on the unmodified pipeline only

Checks (CHECKS) carry the configuration they belong to (solver < generic < full), the
layers they point to, and whether they can run before execution (pre) or only afterwards
(post). A check that raises counts as firing: the log could not support it. Every check
reads the log, the store, declared policy kept as data (data/fault_trial/authority.yaml,
the scenario files) or a replay of logged inputs; none reads the benchmark's truth.

Each run also yields an effect trace (a digest of the store, the records, the ingests and
the stored log) and outcome counts against the truth; the trial uses them only to tell
whether a patch changed anything and whether a change produced violations, never to
detect it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from dsoi import properties as P

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "data/fault_trial/authority.yaml"
LAYER_OF_MODULE = {
    "layers/l1_interpreter": "L1", "layers/l1_claude": "L1", "layers/l1_ollama": "L1",
    "layers/l2_procedural": "L2", "layers/l3_state": "L3", "store": "L3", "canon": "L4",
    "layers/l4_decision": "L4", "constraints": "L4", "layers/l5_gate": "L5",
    "layers/l5_monitors": "L5", "layers/l6_audit": "L6", "context": "registry",
    "registry": "registry", "pipeline": "pipeline", "types": "types",
}


# -- declared policy, independent of the code under test -----------------------------

@dataclass(frozen=True)
class Policy:
    """Declared L3 authority: source ranks by prefix, the conflict policy, the supersession
    rule, and the interpreter versions a deployment has declared (whose refusal is
    expected). Kept as data, so a fault in the code cannot rewrite it."""

    priority: dict
    supersession: str = "none"
    declared_versions: tuple = ()

    def rank(self, source: str, authenticated: bool, tt: float) -> tuple:
        pr = max((v for k, v in self.priority.items() if source.startswith(k)), default=0)
        return (pr, authenticated, tt)


def declared_winner(facts: list, policy: Policy):
    """The fact the declared policy selects among the current facts of one key."""
    if not facts:
        return None
    top = max(facts, key=lambda f: policy.rank(f.prov.source_id, f.prov.authenticated,
                                               f.tt[0]))
    if policy.supersession == "authenticated_latest":
        newer = [f for f in facts if f.prov.authenticated and f.tt[0] > top.tt[0]]
        if newer:
            top = max(newer, key=lambda f: f.tt[0])
    return top


def _ident(source, authenticated, span) -> tuple:
    return (source, bool(authenticated), None if not span else span[0])


# -- checks ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    name: str
    config: str                 # solver | generic | full: the smallest configuration with it
    layers: tuple
    timing: str                 # pre | post
    relative: bool = False      # can fire on a correct run: only a change against clean counts
    fn: Callable = None


def _finite(x) -> bool:
    return x is not None and all(math.isfinite(float(v)) for v in x)


def c_kprop(run):
    return not all(P.k_prop(r, run.p.formulation) for r in run.p.audit.decisions())


def c_exception(run):
    return run.exception is not None


def c_schema(run):
    """Generic validation of committed facts: finite non-negative values, lo <= value <= hi,
    a non-empty valid time."""
    for f in run.p.state.store.facts():
        o = f.o
        if f.vt[0] >= f.vt[1]:
            return True
        if hasattr(o, "value"):
            lo, hi = o.interval()
            if not all(math.isfinite(v) for v in (o.value, lo, hi)):
                return True
            if o.value < 0 or not (lo - 1e-9 <= o.value <= hi + 1e-9):
                return True
    return False


def c_action(run):
    """Generic validation of decisions: finite, non-negative plans, releases and executions."""
    for r in run.p.audit.decisions():
        for x in (r.x, r.x_released, r.x_exec):
            if x is not None and (not _finite(x) or min(x) < -1e-9):
                return True
    return False


def c_fields(run):
    """Generic validation of decision records: the fields every record must carry."""
    return any(r.state_hash is None or not r.nu or r.constraints_id is None
               or r.objective_id is None for r in run.p.audit.decisions())


def _inv(run):
    if run.inv is None:
        try:
            run.inv = P.inv_report(run.p, run.message_ids)
        except Exception:
            run.inv = dict.fromkeys(("G1", "R2", "C1", "K-prop", "audit"), False)
    return run.inv


def c_g1(run):
    return not _inv(run)["G1"]


def c_r2(run):
    return not _inv(run)["R2"]


def c_c1(run):
    return not _inv(run)["C1"]


def c_audit(run):
    return not _inv(run)["audit"]


def c_g2(run):
    return not all(P.g2(r, run.p.formulation) for r in run.p.audit.decisions())


def c_exec(run):
    return not all(P.execution_fidelity(r) for r in run.p.audit.decisions())


def c_precedence(run):
    """Each fact a decision read is the one the declared policy selects among the facts
    current for that key as of the decision's t_asof (Section 10.3, read against declared
    authority and supersession rules)."""
    store = run.p.state.store
    for r in run.p.audit.decisions():
        for key, read in (r.rho or {}).get("facts", {}).items():
            s, pred, v = key.split("|")
            facts = [f for f in store.asof(float(v), r.t_asof, [pred]) if f.s == s]
            w = declared_winner(facts, run.policy)
            if w is None:
                continue
            span = None if w.prov.span is None else [w.prov.span.message_id]
            if _ident(w.prov.source_id, w.prov.authenticated, span) != _ident(
                    read["source_id"], read["authenticated"], read.get("span")):
                return True
    return False


def c_version(run):
    """An ingest refused for an interpreter-version mismatch whose served version the
    deployment has not declared (Section 10.3: a refusal is a defect only when no declared
    version change explains it)."""
    for rec in run.p.audit.ingests:
        if rec.reason.startswith("nu_I mismatch"):
            served = rec.reason.rsplit("served ", 1)[-1]
            if served not in run.policy.declared_versions:
                return True
    return False


def c_unread(run):
    """A committed message that contributes no fact to the last decision record."""
    recs = run.p.audit.decisions()
    if not recs:
        return False
    facts = (recs[-1].rho or {}).get("facts", {})
    spans = {f["span"][0] for f in facts.values() if f.get("span")}
    committed = {r.message_id for r in run.p.audit.ingests if r.outcome == "committed"}
    return bool(committed - spans)


def c_c2(run):
    return bool(P.c2_violations(run.p.audit))


def c_replay_store(run):
    """Every decision replays from the store as of its t_asof (R2(i), Prop. 1(d))."""
    return not all(P.c1_replay(run.p, r.id, from_store=True) for r in run.p.audit.decisions())


def c_restart(run):
    """The stored log reloads with its chain verified and keeps every message, ingest,
    approval and record; each decision replays from the reloaded log without live reads."""
    from dsoi.layers.l5_gate import RecheckGate
    from dsoi.layers.l6_audit import AuditLog
    p = run.p
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "audit.jsonl"
        head = p.audit.to_jsonl(path)
        loaded = AuditLog.from_jsonl(path, expected_head=head)
    if (loaded.messages != p.audit.messages or loaded.ingests != p.audit.ingests
            or loaded.approvals != p.audit.approvals
            or [r.record_hash for r in loaded.records]
            != [r.record_hash for r in p.audit.records]
            or len(loaded.deltas) != len(p.audit.deltas)):
        return True
    live, gate = p.audit, p.gate
    try:
        p.audit = loaded
        if isinstance(gate, RecheckGate):
            def unavailable(key):
                raise RuntimeError("replay contacted the live source")
            p.gate = RecheckGate(unavailable, fallback=gate.fallback, decimals=gate.decimals)
        for r in loaded.decisions():
            again = p.replay(r.id)
            if (r.x_released is None) != (again is None) or (
                    again is not None and not np.allclose(again, r.x_released)):
                return True
    finally:
        p.audit, p.gate = live, gate
    return False


def c_admission(run):
    """L2: every executed threshold action is covered by an authenticated approval event of
    the rule's kind for its variable, from an approver the rule names, issued at or before
    the decision and not expired; an event without a quantity cap covers one action, one
    with a cap covers actions up to it (Table 2, L2). Uses are read back from the log."""
    rules = getattr(run.p.procedural, "threshold_rules", ()) or ()
    approvals = {a["id"]: a for a in run.p.audit.approvals}
    used: dict = {}
    for r in run.p.audit.decisions():
        if r.x_exec is None:
            continue
        names = run.names(r)
        for rule in rules:
            if rule.variable not in names:
                continue
            q = r.x_exec[names.index(rule.variable)]
            if q <= rule.above + 1e-9:
                continue
            ok = None
            for i in (r.admission or {}).get("approvals", []):
                a = approvals.get(i)
                if not a or not a["authenticated"] or a["kind"] != rule.kind \
                        or a["variable"] != rule.variable or a["approver"] not in rule.approvers \
                        or a["t"] > r.t or (a.get("expires_at") is not None
                                            and r.t >= a["expires_at"]):
                    continue
                prior = used.get(i, [])
                cap = a.get("max_quantity")
                if (cap is None and not prior) or (cap is not None
                                                   and sum(prior) + q <= cap + 1e-9):
                    ok = i
                    break
            if ok is None:
                return True
            used.setdefault(ok, []).append(q)
    return False


CHECKS = (
    Check("kprop", "solver", ("L4", "L5"), "pre", fn=c_kprop),
    Check("exception", "generic", ("by-traceback",), "pre", fn=c_exception),
    Check("schema", "generic", ("L1", "L3"), "pre", fn=c_schema),
    Check("action", "generic", ("L4", "L5"), "pre", fn=c_action),
    Check("fields", "generic", ("L6",), "post", fn=c_fields),
    Check("G1", "full", ("L1", "L3"), "pre", fn=c_g1),
    Check("R2", "full", ("L3", "L6"), "pre", fn=c_r2),
    Check("C1", "full", ("L4", "L5"), "pre", fn=c_c1),
    Check("audit", "full", ("L6",), "post", fn=c_audit),
    Check("G2", "full", ("L4",), "pre", relative=True, fn=c_g2),
    Check("exec_fidelity", "full", ("L5", "pipeline"), "post", fn=c_exec),
    Check("precedence", "full", ("L3",), "pre", fn=c_precedence),
    Check("version", "full", ("L1", "registry"), "pre", fn=c_version),
    Check("committed_unread", "full", ("L3", "L4"), "pre", relative=True, fn=c_unread),
    Check("C2", "full", ("L5", "L6", "registry"), "post", fn=c_c2),
    Check("replay_store", "full", ("L3", "L4"), "post", fn=c_replay_store),
    Check("restart", "full", ("L6",), "post", fn=c_restart),
    Check("admission", "full", ("L2",), "pre", fn=c_admission),
)
CONFIGS = ("solver", "generic", "full")


def in_config(check: Check, config: str) -> bool:
    return CONFIGS.index(check.config) <= CONFIGS.index(config)


# -- runs ------------------------------------------------------------------------------

@dataclass
class Run:
    id: str
    p: object
    message_ids: list
    policy: Policy
    truth: Callable            # x_exec -> True when it violates the true constraints
    names: Callable            # record -> variable names of its decision
    exception: str | None = None
    exception_layer: str | None = None
    inv: dict | None = None


def _layer_of(tb) -> str | None:
    """The layer of the innermost dsoi frame of a traceback."""
    layer = None
    for frame in traceback.extract_tb(tb):
        path = frame.filename.replace("\\", "/")
        if "/dsoi/" in path:
            mod = path.split("/dsoi/", 1)[1].removesuffix(".py")
            layer = LAYER_OF_MODULE.get(mod, mod)
    return layer


def _names_from(p):
    def names(record):
        return list(p.formulation.build(record.state, dict(record.theta)).system.names)
    return names


def _authority() -> dict:
    return yaml.safe_load(AUTHORITY.read_text())


def e1v2_runs():
    """The corrected replay of the stored e1v2 replies, test part, all three models."""
    from collections import defaultdict

    from dsoi.experiments import e1v2_score as old
    from dsoi.experiments.e1v2_corrected import MarginGate
    from dsoi.experiments.e1v2_notices import load
    from dsoi.experiments.e1v2_run import load_replies
    from dsoi.experiments.e1v2_world import T_DECIDE, messages, pipeline, scenario
    auth = _authority()["e1v2"]
    margins = json.loads((ROOT / "results/e1v2_components.json").read_text())["margins"][
        "corrected"]
    by_id = {c.id: c for c in load(ROOT / "data/e1v2/cases.jsonl")}
    grouped, nu = defaultdict(dict), {}
    for (model, cid, repeat, mid), reply in load_replies(ROOT / "results/e1v2/replies.jsonl"
                                                         ).items():
        grouped[(model, cid, repeat)][mid] = reply
        nu[model] = reply["nu_i"]
    for key in sorted(k for k in grouped if by_id[k[1]].split == "test"):
        model, cid, repeat = key
        case = by_id[cid]
        policy = Policy(auth["priority"], auth.get("supersession", "none"),
                        tuple(auth.get("declared_versions", [])) + (nu[model],))

        def go(case=case, key=key, model=model, cid=cid, repeat=repeat, policy=policy):
            scn = scenario(case)
            replay = old.Replay(grouped[key], nu[model])
            _, p = pipeline(case, MarginGate(margins[model]), replay, nu[model])
            replay.p, replay.times = p, {m.message_id: t for m, t in messages(case)}
            ids = []
            run = Run(f"e1v2/{model}/{cid}/r{repeat}", p, ids, policy,
                      lambda x, s=scn, p=p: not P.k_exec_true(x, s.true_state(), s.theta(),
                                                              p.formulation),
                      _names_from(p))
            try:
                for m, t in messages(case):
                    ids.append(m.message_id)
                    p.ingest_message(m, t)
                p.decide(T_DECIDE, scn.theta())
            except Exception as exc:
                _record(run, exc)
            return run
        yield f"e1v2/{model}/{cid}/r{repeat}", go


def _gate(spec: dict, scn):
    from dsoi.layers import l5_gate as G
    kind = spec.get("gate", "nominal")
    if kind == "nominal":
        return G.NominalGate()
    if kind == "box_robust":
        return G.BoxRobustGate()
    if kind == "vertex":
        return G.StateVertexGate()
    if kind == "recheck":
        return G.RecheckGate(scn.true_reader(), fallback=spec.get("fallback", "resolve"))
    raise ValueError(f"unknown gate {kind!r}")


def _label(key: str) -> str:
    """A state key as 'subject|predicate|week', the week in its shortest form."""
    s, p, v = str(key).split("|")
    return f"{s}|{p}|{float(v):g}"


def _stimulus_events(scn) -> list:
    """A single-stimulus scenario (S1) as an event list: the message, then one decision."""
    st = scn.spec["stimulus"]
    return [{"id": st["message_id"], "t": scn.time("message"), "kind": "message",
             "source": st["source_id"], "text": st["text"],
             "fixture": scn.spec["fixtures"]["correct"]},
            {"t": scn.time("decide"), "kind": "decide"}]


def episode(scn, trial: dict, run_id: str) -> Run:
    """A scenario's events through a pipeline built as `trial` declares: gate, conflict
    policy, supersession rule, threshold rules, approvals and declared versions. Event kinds:
    message, record, approval, decide, and sor (a change in the system of record)."""
    from dsoi.context import VersionVector
    from dsoi.layers.l1_interpreter import FixtureInterpreter
    from dsoi.layers.l2_procedural import ThresholdRule
    from dsoi.layers.l5_gate import RecheckGate
    from dsoi.layers.l5_monitors import SameInputsSameRelease
    from dsoi.scenarios import VERSIONS_FILE, build_pipeline
    events = sorted(scn.spec.get("events") or _stimulus_events(scn), key=lambda e: float(e["t"]))
    payloads = {e["id"]: scn.load_json(e["fixture"]) for e in events if e["kind"] == "message"}
    payloads.update({k: scn.load_json(v) for k, v in trial.get("fixture_override", {}).items()})
    sor = {_label(k): float(v) for k, v in trial.get("system_of_record", {}).items()}
    gate = _gate(trial, scn)
    if isinstance(gate, RecheckGate):   # reads the system of record, as changed by sor events
        base_reader = scn.true_reader()

        def reader(key, base=base_reader):
            label = _label("|".join(map(str, key)))
            return sor[label] if label in sor else base(key)
        gate = RecheckGate(reader, fallback=trial.get("fallback", "resolve"))
    rules = tuple(ThresholdRule(r["kind"], r["variable"], frozenset(r["approvers"]),
                                float(r.get("above", 0))) for r in trial.get("threshold_rules", []))
    version = (trial.get("interpreter_version") or scn.spec.get("interpreter_version")
               or scn.fixture("correct")["interpreter_version"])
    versions = VersionVector.load(VERSIONS_FILE).bump(interpreter=version)
    p = build_pipeline(scn, gate, FixtureInterpreter(payloads), versions,
                       trial.get("resolution"), threshold_rules=rules)
    p.state.supersession = trial.get("supersession", "none")
    p._trial_theta = scn.theta()
    if trial.get("c1_monitor"):
        p.monitors = (SameInputsSameRelease(),)
    policy = Policy(scn.spec.get("source_priority", {}), trial.get("supersession", "none"),
                    tuple(trial.get("declared_versions", [])) + (versions.interpreter,))
    ids: list = []
    run = Run(run_id, p, ids, policy,
              lambda x: not P.k_exec_true(x, scn.true_state(), scn.theta(), p.formulation),
              _names_from(p))
    try:
        _events(p, events, ids, sor)
    except Exception as exc:
        _record(run, exc)
    return run


def _record(run: Run, exc: Exception) -> None:
    run.exception = f"{type(exc).__name__}: {exc}"[:300]
    run.exception_layer = _layer_of(exc.__traceback__)


def _events(p, events, ids, sor) -> None:
    from dsoi.layers.l2_procedural import ApprovalEvent
    from dsoi.layers.l3_state import UnresolvedConflict
    from dsoi.scenarios import delta_from_spec
    from dsoi.types import Message
    theta = p._trial_theta
    for e in events:
        k, t = e["kind"], float(e["t"])
        if k == "message":
            ids.append(e["id"])
            p.ingest_message(Message(e["id"], e["source"], e["text"]), t)
        elif k == "record":
            p.ingest_record(delta_from_spec(e["ops"], e["source"]), t, e["source"])
        elif k == "approval":
            p.record_approval(ApprovalEvent(e["approval_id"], e["approval_kind"], e["variable"],
                                            e["approver"], e["source"],
                                            bool(e.get("authenticated", True)), t,
                                            **({"expires_at": float(e["expires_at"])}
                                               if "expires_at" in e else {})))
        elif k == "sor":
            sor.update({_label(a): float(b) for a, b in e["values"].items()})
        elif k == "decide":
            try:
                p.decide(t, theta)
            except UnresolvedConflict:
                pass
        else:
            raise ValueError(f"unknown event kind {k!r}")


# Fixed episodes of the scenario suite (frozen with this file).
EPISODES = (
    ("S1", {"gate": "nominal"}), ("S1", {"gate": "box_robust"}),
    ("S1", {"gate": "recheck", "c1_monitor": True, "events_extra": "s1_sor_change"}),
    ("S2", {"gate": "nominal", "threshold_rules": [
        {"kind": "lot-release", "variable": "x[L1]", "approvers": ["qa.manager"]}],
        "events_extra": "s2_approval"}),
    ("S3", {"gate": "recheck"}), ("S3", {"gate": "nominal"}),
    ("S4", {"gate": "box_robust", "resolution": "span"}),
    ("S4", {"gate": "nominal", "resolution": "priority"}),
    ("S5", {"gate": "nominal"}), ("S6", {"gate": "nominal"}),
    ("S7", {"gate": "nominal"}), ("S8", {"gate": "vertex"}),
)
EXTRA_EVENTS = {
    # S1 has a single stimulus; decide twice with the system of record changing between
    # the two decisions (the 3 Nov regression: both releases explained, both replayable).
    "s1_sor_change": lambda scn: [
        {"id": scn.spec["stimulus"]["message_id"], "t": scn.time("message"), "kind": "message",
         "source": scn.spec["stimulus"]["source_id"], "text": scn.spec["stimulus"]["text"],
         "fixture": scn.spec["fixtures"]["correct"]},
        {"t": 4, "kind": "sor", "values": {"plant:X|capacity|42": 600, "plant:X|capacity|43": 600,
                                           "plant:X|capacity|44": 600}},
        {"t": 4.5, "kind": "decide"},
        {"t": 5, "kind": "sor", "values": {"plant:X|capacity|42": 500, "plant:X|capacity|43": 500,
                                           "plant:X|capacity|44": 500}},
        {"t": 5.5, "kind": "decide"}],
    # S2 with an approval for the threshold action, used once, then a second decision.
    "s2_approval": lambda scn: [
        {"t": 1.5, "kind": "approval", "approval_id": "apr-1", "approval_kind": "lot-release",
         "variable": "x[L1]", "approver": "qa.manager", "source": "idp:qa-approvals",
         "expires_at": 10}],
}


def scenario_runs():
    from dsoi.scenarios import Scenario
    for k, (sid, trial) in enumerate(EPISODES):
        run_id = f"scenarios/{k:02d}-{sid}-{trial['gate']}"

        def go(sid=sid, trial=trial, run_id=run_id):
            scn = Scenario.load(sid)
            extra = trial.get("events_extra")
            if extra:
                events = EXTRA_EVENTS[extra](scn)
                base = [] if extra == "s1_sor_change" else list(scn.spec.get("events", []))
                scn = Scenario(scn.root, {**scn.spec, "events": base + events})
            return episode(scn, trial, run_id)
        yield run_id, go


def negative_runs(root: Path):
    from dsoi.scenarios import Scenario
    for d in sorted(p for p in root.iterdir() if (p / "scenario.yaml").exists()):
        spec = yaml.safe_load((d / "scenario.yaml").read_text())
        trial = spec.pop("trial", {})
        run_id = f"negatives/{d.name}"
        yield run_id, (lambda d=d, spec=spec, trial=trial, run_id=run_id:
                       episode(Scenario(d, spec), trial, run_id))


# -- evaluation --------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset, tuple)):
        return sorted(map(str, x)) if isinstance(x, (set, frozenset)) else list(x)
    return str(x)


def trace(run: Run) -> str:
    """Digest of what the run did: current and closed facts, records, ingests, approvals."""
    p = run.p
    body = {
        "facts": sorted(json.dumps([f.s, f.p, repr(f.o), f.vt, f.tt, f.prov.source_id,
                                    f.prov.authenticated], default=_jsonable)
                        for f in p.state.store.facts()),
        "records": [json.dumps(asdict(r), sort_keys=True, default=_jsonable)
                    for r in p.audit.records],
        "ingests": [json.dumps(asdict(r), sort_keys=True, default=_jsonable)
                    for r in p.audit.ingests],
        "approvals": json.dumps(p.audit.approvals, sort_keys=True, default=_jsonable),
        "exception": run.exception,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def evaluate(make) -> dict:
    """One run: build it (an exception is recorded, not raised), then every check."""
    try:
        run = make()
    except Exception as exc:          # the pipeline could not even be built
        return {"crashed": True, "exception": f"{type(exc).__name__}: {exc}"[:300],
                "exception_layer": _layer_of(exc.__traceback__), "fired": ["exception"],
                "trace": None, "released": 0, "violations": 0, "decisions": 0}
    fired = []
    for c in CHECKS:
        try:
            if c.fn(run):
                fired.append(c.name)
        except Exception:
            fired.append(c.name)
    decided = run.p.audit.decisions()
    executed = [r for r in decided if r.x_exec is not None]
    viol = 0
    for r in executed:
        try:
            viol += bool(run.truth(r.x_exec))
        except Exception:
            pass
    return {"crashed": False, "exception": run.exception, "exception_layer": run.exception_layer,
            "fired": fired, "trace": trace(run), "released": len(executed),
            "violations": viol, "decisions": len(decided)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--negatives", type=Path, help="a directory of negative-case scenarios")
    ap.add_argument("--worlds", nargs="*", default=["e1v2", "scenarios"])
    args = ap.parse_args(argv)
    gens = {"e1v2": e1v2_runs, "scenarios": scenario_runs}
    runs = {}
    for w in args.worlds:
        for run_id, go in gens[w]():
            runs[run_id] = evaluate(go)
    if args.negatives:
        for run_id, go in negative_runs(args.negatives):
            runs[run_id] = evaluate(go)
    meta = {c.name: {"config": c.config, "layers": list(c.layers), "timing": c.timing,
                     "relative": c.relative} for c in CHECKS}
    args.out.write_text(json.dumps({"checks": meta, "runs": runs}, indent=1,
                                   sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
