"""Run one case directory and print what the system did (fault trial, round 2, sandbox tool).

    PYTHONPATH=src .venv/bin/python run_case.py episodes/<name>

Behaviour only: the decisions, releases and executions per decision, the disposition of every
input, the actions that replaying each decision returns (from the logged state and from the
store), and whether the audit log reloads to the same content after it is saved. It runs a
case exactly as the evaluation does (same event kinds and `trial:` keys as
NEGATIVE_FORMAT.md, except `c1_monitor`, which the sandbox cannot run). Compare its output
on a clean tree (`git checkout -- src`) and with a patch applied to see what the patch
changes on that case.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml

from dsoi.context import VersionVector
from dsoi.layers import l5_gate as G
from dsoi.layers.l1_interpreter import FixtureInterpreter
from dsoi.layers.l2_procedural import ApprovalEvent, ThresholdRule
from dsoi.layers.l3_state import UnresolvedConflict
from dsoi.layers.l6_audit import AuditLog
from dsoi.scenarios import VERSIONS_FILE, Scenario, build_pipeline, delta_from_spec
from dsoi.types import Message


def _label(key: str) -> str:
    s, p, v = str(key).split("|")
    return f"{s}|{p}|{float(v):g}"


def _stimulus_events(scn) -> list:
    st = scn.spec["stimulus"]
    return [{"id": st["message_id"], "t": scn.time("message"), "kind": "message",
             "source": st["source_id"], "text": st["text"],
             "fixture": scn.spec["fixtures"]["correct"]},
            {"t": scn.time("decide"), "kind": "decide"}]


def build(scn, trial: dict):
    events = sorted(scn.spec.get("events") or _stimulus_events(scn), key=lambda e: float(e["t"]))
    payloads = {e["id"]: scn.load_json(e["fixture"]) for e in events if e["kind"] == "message"}
    payloads.update({k: scn.load_json(v) for k, v in trial.get("fixture_override", {}).items()})
    sor = {_label(k): float(v) for k, v in trial.get("system_of_record", {}).items()}
    kind = trial.get("gate", "nominal")
    gate = {"nominal": G.NominalGate, "box_robust": G.BoxRobustGate,
            "vertex": G.StateVertexGate}.get(kind)
    if kind == "recheck":
        base = scn.true_reader()

        def reader(key, base=base):
            label = _label("|".join(map(str, key)))
            return sor[label] if label in sor else base(key)
        gate = G.RecheckGate(reader, fallback=trial.get("fallback", "resolve"))
    elif gate is None:
        raise ValueError(f"unknown gate {kind!r}")
    else:
        gate = gate()
    rules = tuple(ThresholdRule(r["kind"], r["variable"], frozenset(r["approvers"]),
                                float(r.get("above", 0))) for r in trial.get("threshold_rules", []))
    version = (trial.get("interpreter_version") or scn.spec.get("interpreter_version")
               or scn.fixture("correct")["interpreter_version"])
    p = build_pipeline(scn, gate, FixtureInterpreter(payloads),
                       VersionVector.load(VERSIONS_FILE).bump(interpreter=version),
                       trial.get("resolution"), threshold_rules=rules)
    p.state.supersession = trial.get("supersession", "none")
    return p, events, sor


def run(case_dir: Path) -> dict:
    spec = yaml.safe_load((case_dir / "scenario.yaml").read_text())
    trial = spec.pop("trial", {})
    scn = Scenario(case_dir, spec)
    p, events, sor = build(scn, trial)
    error = None
    try:
        for e in events:
            k, t = e["kind"], float(e["t"])
            if k == "message":
                p.ingest_message(Message(e["id"], e["source"], e["text"]), t)
            elif k == "record":
                p.ingest_record(delta_from_spec(e["ops"], e["source"]), t, e["source"])
            elif k == "approval":
                p.record_approval(ApprovalEvent(
                    e["approval_id"], e["approval_kind"], e["variable"], e["approver"],
                    e["source"], bool(e.get("authenticated", True)), t,
                    **({"expires_at": float(e["expires_at"])} if "expires_at" in e else {})))
            elif k == "sor":
                sor.update({_label(a): float(b) for a, b in e["values"].items()})
            elif k == "decide":
                try:
                    p.decide(t, scn.theta())
                except UnresolvedConflict:
                    pass
            else:
                raise ValueError(f"unknown event kind {k!r}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    def attempt(fn):
        try:
            v = fn()
            return None if v is None else [round(float(x), 6) for x in v]
        except Exception as exc:
            return f"raised {type(exc).__name__}: {exc}"[:160]

    decisions = [{"t": r.t, "outcome": r.outcome, "x": r.x, "released": r.x_released,
                  "executed": r.x_exec, "gate": r.gate_outcome,
                  "admission": (r.admission or {}).get("verdict"),
                  "replay": attempt(lambda r=r: p.replay(r.id)),
                  "replay_from_store": attempt(lambda r=r: p.replay(r.id, from_store=True))}
                 for r in p.audit.records]
    try:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            head = p.audit.to_jsonl(path)
            loaded = AuditLog.from_jsonl(path, expected_head=head)
        reload = {"same_messages": loaded.messages == p.audit.messages,
                  "same_ingests": loaded.ingests == p.audit.ingests,
                  "same_approvals": loaded.approvals == p.audit.approvals,
                  "same_records": [r.record_hash for r in loaded.records]
                  == [r.record_hash for r in p.audit.records],
                  "same_deltas": [(t, repr(d)) for t, d in loaded.deltas]
                  == [(t, repr(d)) for t, d in p.audit.deltas]}
    except Exception as exc:
        reload = f"raised {type(exc).__name__}: {exc}"[:160]
    return {"error": error, "decisions": decisions,
            "inputs": [asdict(i) for i in p.audit.ingests],
            "approvals": p.audit.approvals, "reload": reload}


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1])), indent=1, default=str))
