"""Phase 3 runner: interpreter arms, the LLM-decides baseline and the equivalent-input
variants, for Claude models through the Message Batches API and local models through Ollama.

Every request is one of
  interp  one message of a case through L1, under one interface arm (e1v3_world.ARMS)
  decide  a whole case for the LLM-decides baseline (e1v3_decide), one request per case
and is keyed (kind, model, arm, case_id, repeat, message_id). A job names which cases go
to which arms; the runner sends, per model and round, every pending request of every job
in one batch, so the arms of a model share batches and a silent model update cannot line
up with one arm. A conflict or amending case's later message waits for the earlier reply
(its snapshot depends on it), so a model needs as many rounds as a case has messages.

A reply whose last stored attempt errored (errored or expired batch request, failed call)
is requested again, up to `max_attempts`; the store is append-only and the last attempt
counts. A refusal is stored with its stop reason and category, and scoring counts it
apart from abstention. No fallback model is configured: each model answers as itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dsoi.experiments import e1v3_decide
from dsoi.experiments.e1v2_run import BudgetExceeded, ClaudeBatch, seed_for
from dsoi.experiments.e1v3_world import ARMS, messages, pipeline, sha, snapshot_text
from dsoi.layers import l1_ollama
from dsoi.layers.l1_claude import build_cached_params, interpret_reply
from dsoi.layers.l5_gate import NominalGate

__all__ = ["BudgetExceeded", "ClaudeV3", "Job", "load_replies", "pending", "run_claude",
           "run_ollama"]


@dataclass(frozen=True)
class Job:
    kind: str                     # interp | decide
    cases: tuple                  # Case objects
    arms: tuple[str, ...]         # for decide: ("decide",)
    repeats: dict                 # model -> repeats


def key(r: dict) -> tuple:
    return (r["kind"], r["model"], r["arm"], r["case_id"], r["repeat"], r["message_id"])


def load_replies(store: Path) -> tuple[dict, dict]:
    """The last stored attempt per key, and the number of attempts per key."""
    last, attempts = {}, {}
    if store.exists():
        with open(store) as fh:
            for r in map(json.loads, fh):
                k = key(r)
                last[k] = r
                attempts[k] = attempts.get(k, 0) + 1
    return last, attempts


def is_error(reply: dict) -> bool:
    return str(reply.get("stop_reason", "")).startswith("error:")


def settled(k, last, attempts, max_attempts) -> bool:
    return k in last and (not is_error(last[k]) or attempts.get(k, 0) >= max_attempts)


def nu_i(backend_name: str, model: str, effort_or_temp: str, arm: str) -> str:
    """The interpreter version: model, effort (or sampling) and the interface arm."""
    return f"{backend_name}:{model}@{effort_or_temp}#{arm}"


class _Stop(Exception):
    pass


class _Collector:
    """The L1 the pipeline calls while planning a round: a settled reply is replayed; the
    first unsettled message of a case is recorded with the snapshot it will see."""

    def __init__(self, model, arm, case, repeat, nu, last, ok, out):
        self.model, self.arm, self.case, self.repeat, self.nu = model, arm, case, repeat, nu
        self.last, self.ok, self.out, self.p = last, ok, out, None
        self.times = {m.message_id: t for m, t in messages(case)}

    def interpret(self, message, sch):
        k = ("interp", self.model, self.arm.name, self.case.id, self.repeat, message.message_id)
        if self.ok(k):
            r = self.last[k]
            return interpret_reply(self.nu, r.get("stop_reason"), r.get("text"), message, sch,
                                   r.get("refusal_category"))
        self.out.append({"kind": "interp", "model": self.model, "arm": self.arm.name,
                         "case": self.case, "repeat": self.repeat, "message": message,
                         "schema": sch, "system": self.arm.system,
                         "snapshot": snapshot_text(self.p, self.times[message.message_id],
                                                   self.arm),
                         "message_id": message.message_id, "nu_i": self.nu})
        raise _Stop


def _initial_snapshot(case) -> str:
    """The state before the first message, as the reference arm shows it."""
    _, p = pipeline(case, NominalGate(), None, "decide")
    first_t = messages(case)[0][1]
    return snapshot_text(p, first_t, ARMS["ref"])


def pending(jobs: list[Job], model: str, backend_label, last: dict, attempts: dict,
            max_attempts: int = 3) -> list[dict]:
    """Every request of `model` that can be sent now."""
    def ok(k):
        return settled(k, last, attempts, max_attempts)

    out: list[dict] = []
    for job in jobs:
        for repeat in range(1, job.repeats.get(model, 0) + 1):
            for case in job.cases:
                if job.kind == "decide":
                    mid = f"{case.id}-plan"
                    k = ("decide", model, "decide", case.id, repeat, mid)
                    if not ok(k):
                        out.append({"kind": "decide", "model": model, "arm": "decide",
                                    "case": case, "repeat": repeat, "message_id": mid,
                                    "snapshot": _initial_snapshot(case),
                                    "messages": list(case.messages),
                                    "nu_i": backend_label(model, "decide")})
                    continue
                for arm_name in job.arms:
                    arm = ARMS[arm_name]
                    nu = backend_label(model, arm_name)
                    col = _Collector(model, arm, case, repeat, nu, last, ok, out)
                    _, p = pipeline(case, NominalGate(), col, nu)
                    col.p = p
                    try:
                        for message, t in messages(case):
                            p.ingest_message(message, t)
                    except _Stop:
                        pass
    return out


def custom_id(item: dict) -> str:
    short = item["model"].replace("claude-", "").split("-")[0]
    cid = f"{short}-{item['arm']}-{item['message_id']}-r{item['repeat']}"
    if len(cid) > 64:
        raise ValueError(f"custom_id too long: {cid}")
    return cid


@dataclass
class ClaudeV3(ClaudeBatch):
    """ClaudeBatch's prices, cost and worst case, with phase 3 request parameters."""

    name: str = "claude"

    def label(self, model: str, arm: str) -> str:
        return nu_i(self.name, model, self.effort, arm)

    def params(self, item: dict) -> dict:
        if item["kind"] == "decide":
            return e1v3_decide.build_cached_params(item["snapshot"], item["messages"],
                                                   item["model"], self.effort,
                                                   self.max_tokens, self.cache_ttl)
        return build_cached_params(item["message"], item["schema"], item["model"],
                                   self.effort, self.max_tokens, item["snapshot"],
                                   self.cache_ttl, system=item["system"])


def _record(item: dict, backend_name: str, manifest: str, reply: dict) -> dict:
    return {"backend": backend_name, "kind": item["kind"], "model": item["model"],
            "arm": item["arm"], "nu_i": item["nu_i"], "case_id": item["case"].id,
            "repeat": item["repeat"], "message_id": item["message_id"],
            "seed": seed_for(item["case"].id, item["repeat"]),
            "snapshot_sha": sha(item["snapshot"]), "manifest": manifest, **reply}


def run_claude(jobs: list[Job], backend: ClaudeV3, models: list[str], store: Path,
               manifest: str, budget: float, ledger_path: Path, client,
               poll_seconds: float = 30.0, progress=print, sleep=None,
               max_attempts: int = 3, batch_size: int = 500) -> float:
    """Rounds of batches per model; returns the total spent. A round is sent in batches of
    at most `batch_size` requests, one after another, and a batch is submitted only if its
    worst case (full max_tokens on every request) fits in the budget left after what the
    earlier batches actually cost. The guard therefore never lets the ledger pass the
    budget, and it stops on real spend rather than on one round's worst case."""
    import time as _time
    sleep = sleep or _time.sleep
    store.parent.mkdir(parents=True, exist_ok=True)
    ledger = (json.loads(ledger_path.read_text()) if ledger_path.exists()
              else {"budget_usd": budget, "spent_usd": 0.0, "batches": []})
    last, attempts = load_replies(store)
    for model in models:
        while round_items := pending(jobs, model, backend.label, last, attempts,
                                     max_attempts):
            items = round_items[:batch_size]
            worst = backend.worst_case(model, len(items))
            if ledger["spent_usd"] + worst > budget:
                raise BudgetExceeded(f"{model}: {len(items)} requests could cost up to "
                                     f"${worst:.2f}; ${budget - ledger['spent_usd']:.2f} left")
            by_id = {custom_id(it): it for it in items}
            batch = client.messages.batches.create(requests=[
                {"custom_id": cid, "params": backend.params(it)} for cid, it in by_id.items()])
            entry = {"id": batch.id, "model": model, "requests": len(items),
                     "worst_case_usd": round(worst, 4), "cost_usd": 0.0}
            ledger["batches"].append(entry)
            ledger_path.write_text(json.dumps(ledger, indent=1) + "\n")
            progress(f"{model}: batch {batch.id} with {len(items)} requests submitted")
            while client.messages.batches.retrieve(batch.id).processing_status != "ended":
                sleep(poll_seconds)
            for result in client.messages.batches.results(batch.id):
                it = by_id[result.custom_id]
                reply = {"text": None, "stop_reason": f"error:{result.result.type}",
                         "usage": {}, "cost_usd": 0.0}
                if result.result.type == "succeeded":
                    msg = result.result.message
                    usage = msg.usage.to_dict()
                    details = getattr(msg, "stop_details", None)
                    reply = {"text": next((b.text for b in msg.content if b.type == "text"),
                                          None),
                             "stop_reason": msg.stop_reason, "served_model": msg.model,
                             "refusal_category": getattr(details, "category", None),
                             "usage": usage, "cost_usd": backend.cost(model, usage)}
                rec = _record(it, backend.name, manifest, {"batch_id": batch.id, **reply})
                with open(store, "a") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
                k = key(rec)
                last[k] = rec
                attempts[k] = attempts.get(k, 0) + 1
                entry["cost_usd"] = round(entry["cost_usd"] + reply["cost_usd"], 6)
            ledger["spent_usd"] = round(sum(b["cost_usd"] for b in ledger["batches"]), 6)
            ledger_path.write_text(json.dumps(ledger, indent=1) + "\n")
            progress(f"{model}: batch {batch.id} done, ${entry['cost_usd']:.4f}; "
                     f"total ${ledger['spent_usd']:.4f}")
    return ledger["spent_usd"]


@dataclass
class OllamaV3:
    """Local models at temperature 0 (the pre-registered setting for local runs), pinned by
    weight digest, sent one request at a time."""

    temperature: float = 0.0
    url: str = l1_ollama.OLLAMA_URL
    name: str = "ollama"
    max_tokens: int = 1024

    def prepare(self, models):
        self.metadata = {m: l1_ollama.model_metadata(m, self.url) for m in models}

    def label(self, model: str, arm: str) -> str:
        return nu_i(self.name, f"{model}@{self.metadata[model]['model_digest']}",
                    f"t{self.temperature:g}", arm)

    def request(self, item: dict, seed: int) -> dict:
        if item["kind"] == "decide":
            prompt = e1v3_decide.user_prompt(item["snapshot"], item["messages"])
            return {"model": item["model"], "stream": False,
                    "format": e1v3_decide.output_schema(),
                    "options": {"temperature": self.temperature, "seed": seed,
                                "num_predict": self.max_tokens},
                    "messages": [{"role": "system", "content": e1v3_decide.SYSTEM_PROMPT},
                                 {"role": "user", "content": prompt}]}
        return l1_ollama.build_request(item["message"], item["schema"], item["model"],
                                       item["snapshot"], self.temperature, seed,
                                       self.max_tokens, system=item["system"])


def run_ollama(jobs: list[Job], backend: OllamaV3, models: list[str], store: Path,
               manifest: str, progress=print, max_attempts: int = 3) -> int:
    """Request every pending local reply, one at a time; return how many were requested."""
    store.parent.mkdir(parents=True, exist_ok=True)
    backend.prepare(models)
    last, attempts = load_replies(store)
    sent = 0
    for model in models:
        pinned = backend.metadata[model]["model_digest"]
        while items := pending(jobs, model, backend.label, last, attempts, max_attempts):
            for it in items:
                seed = seed_for(it["case"].id, it["repeat"])
                if l1_ollama.model_metadata(model, backend.url)["model_digest"] != pinned:
                    raise ValueError(f"{model}: model digest changed during the run")
                try:
                    reply = l1_ollama.call(backend.request(it, seed), backend.url)
                except Exception as exc:     # stored, and retried up to max_attempts
                    reply = {"text": None, "stop_reason": f"error:{type(exc).__name__}",
                             "error": str(exc)}
                rec = _record(it, backend.name, manifest,
                              {**reply, "runtime_metadata": backend.metadata[model]})
                with open(store, "a") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
                k = key(rec)
                last[k] = rec
                attempts[k] = attempts.get(k, 0) + 1
                sent += 1
            progress(f"{model}: {sent} requests so far")
    return sent
