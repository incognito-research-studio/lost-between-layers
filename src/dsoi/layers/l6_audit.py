"""L6 Audit: append-only, hash-chained log with one record per decision (Def. 8).

Contract (Table I):
  append(record) with hash chain; replay(id).
  Invariant: one complete record per decision cycle, with its execution ack if any.
  A cycle that L3 defers or that ends in an error before execution also leaves a
  record (`outcome` "deferred" or "error"), so nonexecution is logged too.
  Does not certify the truth of logged inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dsoi.types import Delta, Op, OpKind, Provenance, Quantity, Span

GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + payload).encode()).hexdigest()


def logged_tuple(state_hash, theta, constraints_id, objective_id, nu: dict,
                 include_nu: bool = True, observations: dict | None = None) -> tuple:
    """Complete release inputs, including execution-time observations z.

    z is empty for a gate without a re-read. Recording a re-read only as diagnostic
    detail is insufficient: C2 and replay must use it as an input too.
    """
    base = (state_hash, json.dumps(theta, default=str), constraints_id, objective_id,
            json.dumps(observations or {}, sort_keys=True))
    return base + ((json.dumps(nu, sort_keys=True),) if include_nu else ())


@dataclass(frozen=True)
class AuditRecord:
    id: int
    t: float
    decision_class: str
    state_hash: str | None         # None only when the cycle ended before reading the state
    state: tuple | None            # logged pi_D(S_hat): replay without the store (Prop. 4(a))
    t_asof: float
    theta: tuple
    constraints_id: str | None     # content hashes (A-LOG); the encodings are in AuditLog
    objective_id: str | None
    nu: dict
    x: list | None
    cert: dict
    rho: dict                      # provenance map rho(x)
    gate_rung: str
    gate_outcome: str
    gate_detail: dict
    x_released: list | None        # what the gate released (K-prop, C1 and C2 concern this)
    x_exec: list | None            # what execution reported doing (K-exec concerns this)
    exec_ack: str | None
    admission: dict | None = None  # L2 admission of the released action, with approval ids
    outcome: str = "decided"       # decided | deferred | error; only "decided" ran the gate
    prev_hash: str = GENESIS
    record_hash: str = ""

    @property
    def decided(self) -> bool:
        return self.outcome == "decided"

    def body(self) -> dict:
        d = asdict(self)
        d.pop("record_hash")
        return d

    def logged_tuple(self, include_nu: bool = True) -> tuple:
        """(hash kappa_D, theta, C-id, J-id, z, nu): the tuple C2 compares."""
        return logged_tuple(self.state_hash, self.theta, self.constraints_id,
                            self.objective_id, self.nu, include_nu,
                            self.gate_detail.get("reread", {}))


@dataclass(frozen=True)
class IngestRecord:
    """R2(ii): each accepted change-asserting message yields a commit, abstention or conflict."""

    message_id: str
    t: float
    outcome: str                   # committed | abstained | rejected | queued
    reason: str
    nu_i: str


@dataclass
class AuditLog:
    records: list[AuditRecord] = field(default_factory=list)
    ingests: list[IngestRecord] = field(default_factory=list)
    messages: dict[str, dict] = field(default_factory=dict)   # message_id -> logged message
    approvals: list[dict] = field(default_factory=list)       # approval events as received
    encodings: dict[str, dict] = field(default_factory=dict)  # C-id / J-id -> encoding
    deltas: list[tuple[float, Delta]] = field(default_factory=list)  # every committed delta

    def log_message(self, message_id: str, source_id: str, text: str, t: float,
                    received_at: float | None = None) -> None:
        """Log a message as processed at `t`; `received_at` is when it arrived (R2(ii))."""
        self.messages[message_id] = {"source_id": source_id, "text": text, "t": t,
                                     "received_at": t if received_at is None else received_at,
                                     "sha256": hashlib.sha256(text.encode()).hexdigest()}

    def log_ingest(self, rec: IngestRecord) -> None:
        self.ingests.append(rec)

    def log_encoding(self, ids: tuple[str, str], encoding: dict) -> None:
        """Store the encoded constraints and objective under their content hashes, so that
        each record's C-id and J-id can be resolved (Prop. 1(d))."""
        for content_id in ids:
            self.encodings.setdefault(content_id, encoding)

    def log_delta(self, delta: Delta, t: float) -> None:
        """Log a delta the store committed at transaction time t (R2(iii) replays these)."""
        self.deltas.append((t, delta))

    def log_approval(self, event) -> None:
        """Log an approval event from the identity/approval service, authenticated or not."""
        self.approvals.append(asdict(event))

    def append(self, **fields) -> AuditRecord:
        prev = self.records[-1].record_hash if self.records else GENESIS
        draft = AuditRecord(id=len(self.records) + 1, prev_hash=prev, **fields)
        rec = AuditRecord(**{**asdict(draft), "record_hash": _digest(prev, draft.body())})
        self.records.append(rec)
        return rec

    def decisions(self) -> list[AuditRecord]:
        """Records of cycles that produced a decision (not deferred, no error)."""
        return [r for r in self.records if r.decided]

    def get(self, record_id: int) -> AuditRecord:
        return self.records[record_id - 1]

    def verify(self) -> bool:
        """Verify the decision-record chain; not a claim of completeness or immutability."""
        prev = GENESIS
        for rec in self.records:
            if rec.prev_hash != prev or rec.record_hash != _digest(prev, rec.body()):
                return False
            prev = rec.record_hash
        return True

    def to_jsonl(self, path: str | Path) -> str:
        """Atomically save all audit evidence in a version-2 snapshot; return its head.

        Every entry is hash-chained, including messages, dispositions, approvals,
        encodings and deltas. A final checkpoint detects an incomplete file. Store the
        returned head independently to detect a replacement by an older valid snapshot.
        This file is not a write-once store and cannot establish A-LOG by itself.
        """
        path = Path(path)
        entries = [{"record": asdict(rec)} for rec in self.records]
        entries += [{"encoding": key, "value": value}
                    for key, value in sorted(self.encodings.items())]
        entries += [{"delta": t, "ops": [asdict(op) for op in delta.ops]}
                    for t, delta in self.deltas]
        entries += [{"message": key, "value": value}
                    for key, value in sorted(self.messages.items())]
        entries += [{"ingest": asdict(rec)} for rec in self.ingests]
        entries += [{"approval": event} for event in self.approvals]
        head = GENESIS
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as fh:
                temporary = fh.name
                for entry in entries:
                    digest = _digest(head, entry)
                    fh.write(json.dumps({**entry, "entry_prev": head, "entry_hash": digest},
                                        default=str) + "\n")
                    head = digest
                fh.write(json.dumps({"audit_format": 2, "entries": len(entries),
                                     "head": head}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
        return head

    @classmethod
    def from_jsonl(cls, path: str | Path, *, expected_head: str | None = None,
                   allow_legacy: bool = False) -> AuditLog:
        """Verify and load all evidence; optionally check an independently stored head.

        Legacy snapshots require explicit opt-in: they did not persist message,
        disposition or approval evidence and cannot support the corresponding checks.
        An unanchored valid snapshot can still be maliciously rewritten or rolled back.
        """
        log = cls()
        with open(path) as fh:
            items = [json.loads(line) for line in fh]
        modern = bool(items and items[-1].get("audit_format") == 2)
        if modern:
            checkpoint = items.pop()
            head = GENESIS
            for item in items:
                previous, digest = item.pop("entry_prev", None), item.pop("entry_hash", None)
                if previous != head or digest != _digest(head, item):
                    raise ValueError(f"{path}: the audit entry hash chain does not verify")
                head = digest
            if checkpoint["entries"] != len(items) or checkpoint["head"] != head:
                raise ValueError(f"{path}: the audit checkpoint does not verify")
            if expected_head is not None and head != expected_head:
                raise ValueError(f"{path}: the audit head differs from the trusted checkpoint")
        elif not allow_legacy or expected_head is not None or any(
                "entry_hash" in item or "audit_format" in item for item in items):
            raise ValueError(f"{path}: missing version-2 audit checkpoint")
        for item in items:
            if "record" in item:
                fields = item["record"]
                for name in ("state", "theta"):
                    fields[name] = _tuples(fields[name])
                log.records.append(AuditRecord(**fields))
            elif "delta" in item:
                log.deltas.append((item["delta"], Delta(tuple(_op(d) for d in item["ops"]))))
            elif "encoding" in item:
                log.encodings[item["encoding"]] = item["value"]
            elif "message" in item:
                log.messages[item["message"]] = item["value"]
            elif "ingest" in item:
                log.ingests.append(IngestRecord(**item["ingest"]))
            elif "approval" in item:
                log.approvals.append(item["approval"])
            else:
                raise ValueError(f"{path}: unknown audit entry")
        if not log.verify():
            raise ValueError(f"{path}: the hash chain does not verify")
        return log


def _op(d: dict) -> Op:
    """An Op back from its JSON form (`asdict`), for the logged deltas."""
    prov = d["prov"]
    span = prov["span"]
    o = d["o"]
    return Op(OpKind(d["kind"]), d["s"], d["p"], tuple(d["vt"]),
              Provenance(**{**prov, "span": None if span is None else Span(**span)}),
              Quantity(**o) if isinstance(o, dict) else o)


def _tuples(v):
    """JSON lists back to the tuples that canonical states and theta are built from."""
    return tuple(_tuples(x) for x in v) if isinstance(v, list) else v
