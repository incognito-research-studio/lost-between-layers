"""E1 v2 runner: every case's messages to each interpreter, with the raw replies stored.

For each model, case and repeat, the case's messages go through the case pipeline in
arrival order. Before each message, L1 receives the snapshot of the state committed so
far; the reply is stored raw with the snapshot's hash, the sampling seed and the hash of
the frozen protocol (manifest). Scoring replays the stored replies offline
(`e1v2_score`), so the same replies always give the same scores. The run is resumable:
a reply already stored is not requested again.

Backends: `ollama` (local models, no cost) and `mock` (dry runs and tests). A Claude
backend uses `l1_claude` with the E1 budget ledger and is added only after the budget is
approved.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dsoi.experiments.e1v2_notices import NOMINAL, Case
from dsoi.experiments.e1v2_world import messages, pipeline, sha, snapshot_text
from dsoi.layers import l1_ollama
from dsoi.layers.l1_claude import build_cached_params, interpret_reply
from dsoi.layers.l5_gate import NominalGate


def seed_for(case_id: str, repeat: int) -> int:
    return int(hashlib.sha256(f"{case_id}-r{repeat}".encode()).hexdigest()[:8], 16)


@dataclass
class OllamaBackend:
    temperature: float = 0.7
    url: str = l1_ollama.OLLAMA_URL
    name: str = "ollama"
    metadata: dict = field(default_factory=dict)

    def prepare(self, models):
        self.metadata = {model: l1_ollama.model_metadata(model, self.url) for model in models}

    def nu_i(self, model: str) -> str:
        digest = self.metadata[model]["model_digest"]
        return f"ollama:{model}@{digest}@t{self.temperature:g}"

    def send(self, message, schema, snapshot, model, seed, case) -> dict:
        req = l1_ollama.build_request(message, schema, model, snapshot, self.temperature, seed)
        pinned = self.metadata[model]
        before = l1_ollama.model_metadata(model, self.url)
        if before["model_digest"] != pinned["model_digest"]:
            raise ValueError(f"{model}: model digest changed during the run")
        reply = l1_ollama.call(req, self.url)
        after = l1_ollama.model_metadata(model, self.url)
        if after["model_digest"] != pinned["model_digest"]:
            raise ValueError(f"{model}: model digest changed during inference")
        return {**reply, "runtime_metadata": pinned, "sampling_options": req["options"]}


@dataclass
class MockBackend:
    """A stand-in interpreter that reads the case's truth and adds error: a point error of
    `point_sd` units, an interval of +-`half_width` units, a scope error (last affected
    week dropped) with probability `scope_error`, and abstention with `abstain`. For dry
    runs and tests only; it cheats by construction."""

    point_sd: float = 30.0
    half_width: float = 50.0
    scope_error: float = 0.1
    abstain: float = 0.05
    name: str = "mock"

    def nu_i(self, model: str) -> str:
        return f"mock:{model}"

    def send(self, message, schema, snapshot, model, seed, case) -> dict:
        rng = np.random.default_rng(seed + int(message.message_id[-1]))
        if case.expect != "change" or rng.random() < self.abstain:
            text = json.dumps({"kind": "abstain", "reason": "mock", "ops": []})
            return {"text": text, "stop_reason": "end_turn", "served_model": model}
        ops = []
        for entity, weeks in case.truth.items():
            ws = sorted(weeks)
            if len(ws) > 1 and rng.random() < self.scope_error:
                ws = ws[:-1]
            value = float(np.clip(weeks[ws[0]] + rng.normal(0, self.point_sd), 0, NOMINAL))
            ops.append({"op": "supersede", "subject": entity, "predicate": "capacity",
                        "value": value, "lo": max(0.0, value - self.half_width),
                        "hi": min(NOMINAL, value + self.half_width), "unit": "units/week",
                        "valid_from": ws[0], "valid_to": ws[-1] + 1,
                        "span_text": message.text[: min(len(message.text), 20)]})
        text = json.dumps({"kind": "delta", "reason": "mock", "ops": ops})
        return {"text": text, "stop_reason": "end_turn", "served_model": model}


def parse(reply: dict, nu_i: str, message, schema):
    """A stored reply under the L1 contract (both backends use the same checks)."""
    return interpret_reply(nu_i, reply.get("stop_reason"), reply.get("text"), message, schema)


class Recorder:
    """The L1 the pipeline calls during a run: it asks the backend, stores the reply with
    the snapshot it was given, and returns the parsed result."""

    def __init__(self, backend, model, case, repeat, store, manifest, done):
        self.backend, self.model, self.case, self.repeat = backend, model, case, repeat
        self.store, self.manifest, self.done = store, manifest, done
        self.p = None
        self.times = {m.message_id: t for m, t in messages(case)}

    def interpret(self, message, schema):
        snap = snapshot_text(self.p, self.times[message.message_id])
        key = (self.model, self.case.id, self.repeat, message.message_id)
        nu_i = self.backend.nu_i(self.model)
        if key in self.done:
            reply = self.done[key]
        else:
            seed = seed_for(self.case.id, self.repeat)
            try:
                reply = self.backend.send(message, schema, snap, self.model, seed, self.case)
            except Exception as exc:          # a failed call is stored, not retried silently
                reply = {"text": None, "stop_reason": f"error:{type(exc).__name__}",
                         "error": str(exc)}
            reply = {"backend": self.backend.name, "model": self.model, "nu_i": nu_i,
                     "case_id": self.case.id, "repeat": self.repeat,
                     "message_id": message.message_id, "seed": seed,
                     "snapshot_sha": sha(snap), "manifest": self.manifest, **reply}
            with WRITE_LOCK, open(self.store, "a") as fh:
                fh.write(json.dumps(reply, sort_keys=True) + "\n")
            self.done[key] = reply
        return parse(reply, nu_i, message, schema)


def load_replies(store: Path) -> dict[tuple, dict]:
    if not store.exists():
        return {}
    out = {}
    with open(store) as fh:
        for r in map(json.loads, fh):
            out[(r["model"], r["case_id"], r["repeat"], r["message_id"])] = r
    return out


WRITE_LOCK = threading.Lock()


def _one(backend, model, case, repeats, store, manifest, done) -> None:
    for repeat in range(1, repeats + 1):
        rec = Recorder(backend, model, case, repeat, store, manifest, done)
        _, p = pipeline(case, NominalGate(), rec, backend.nu_i(model))
        rec.p = p
        for message, t in messages(case):        # a case's messages stay in order
            p.ingest_message(message, t)


def run(cases: list[Case], backend, models: list[str], repeats: int, store: Path,
        manifest: str, progress=None, workers: int = 1) -> int:
    """Request every missing reply; return how many were requested. Cases are independent,
    so `workers` of them run at once (the server batches them); stored replies, and so the
    scores, do not depend on the order in which cases finish."""
    store.parent.mkdir(parents=True, exist_ok=True)
    done = load_replies(store)
    if isinstance(backend, OllamaBackend):
        backend.prepare(models)
        for (model, *_), reply in done.items():
            if model in models and reply["nu_i"] != backend.nu_i(model):
                raise ValueError(f"{model}: stored replies use different or unrecorded weights; "
                                 "start a new result store rather than mixing runs")
    before = len(done)
    for model in models:
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(_one, backend, model, c, repeats, store, manifest, done)
                       for c in cases]
            for k, f in enumerate(futures):
                f.result()
                if progress:
                    progress(f"{model}: {k + 1}/{len(cases)} cases")
    return len(done) - before



# -- Claude models through the Message Batches API ------------------------------------
#
# A batch cannot hold a message whose snapshot depends on a reply still in the same batch
# (the second message of a conflict case), so a run proceeds in rounds: each round sends
# every message whose earlier messages already have stored replies. Requests mark the
# shared prefix for prompt caching; batches bill every token at half price. No fallback
# model is configured: each model is measured as itself, and a refusal is an abstention.


class _Stop(Exception):
    pass


class Collector:
    """The L1 for planning a round: stored replies are replayed; the first message of a case
    without one is recorded, with the snapshot it will be given, and the case stops there."""

    def __init__(self, model, nu_i, case, repeat, done, out):
        self.model, self.nu_i, self.case, self.repeat = model, nu_i, case, repeat
        self.done, self.out, self.p = done, out, None
        self.times = {m.message_id: t for m, t in messages(case)}

    def interpret(self, message, schema):
        key = (self.model, self.case.id, self.repeat, message.message_id)
        if key in self.done:
            return parse(self.done[key], self.nu_i, message, schema)
        self.out.append({"model": self.model, "case": self.case, "repeat": self.repeat,
                         "message": message, "schema": schema,
                         "snapshot": snapshot_text(self.p, self.times[message.message_id])})
        raise _Stop


def pending(cases: list[Case], model: str, repeats: int, done: dict, nu_i: str) -> list[dict]:
    out: list[dict] = []
    for case in cases:
        for repeat in range(1, repeats + 1):
            col = Collector(model, nu_i, case, repeat, done, out)
            _, p = pipeline(case, NominalGate(), col, nu_i)
            col.p = p
            try:
                for message, t in messages(case):
                    p.ingest_message(message, t)
            except _Stop:
                pass
    return out


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ClaudeBatch:
    """Batched Claude calls with prompt caching. `prices` maps a model to its input, output
    and cache-read prices in USD per million tokens; cache writes cost 2x input for the 1-hour
    TTL, and the batch discount halves everything."""

    prices: dict
    effort: str = "medium"
    max_tokens: int = 4096
    cache_ttl: str = "1h"
    batch_discount: float = 0.5
    worst_input_tokens: int = 1500
    name: str = "claude"

    def nu_i(self, model: str) -> str:
        return f"claude:{model}@{self.effort}"

    def params(self, item: dict) -> dict:
        return build_cached_params(item["message"], item["schema"], item["model"], self.effort,
                                   self.max_tokens, item["snapshot"], self.cache_ttl)

    def cost(self, model: str, usage: dict) -> float:
        p = self.prices[model]
        write = 2.0 if self.cache_ttl == "1h" else 1.25
        usd = (usage.get("input_tokens", 0) * p["input"]
               + (usage.get("cache_creation_input_tokens") or 0) * p["input"] * write
               + (usage.get("cache_read_input_tokens") or 0) * p["cache_read"]
               + usage.get("output_tokens", 0) * p["output"]) / 1e6
        return usd * self.batch_discount

    def worst_case(self, model: str, n: int) -> float:
        """Every request at the worst input and the full max_tokens of output."""
        return n * self.cost(model, {"input_tokens": self.worst_input_tokens,
                                     "output_tokens": self.max_tokens})


def custom_id(model: str, message_id: str, repeat: int) -> str:
    short = model.replace("claude-", "").split("-")[0]
    return f"{short}-{message_id}-r{repeat}"


def run_claude(cases: list[Case], backend: ClaudeBatch, models: list[str], repeats: int,
               store: Path, manifest: str, budget: float, ledger_path: Path, client,
               poll_seconds: float = 30.0, progress=print, sleep=None) -> float:
    """Rounds of batches per model, cheapest model first; returns the total spent. Before a
    batch is submitted, its worst case (full max_tokens on every request) must fit in the
    budget left; the ledger records every batch and the cost billed from its usage."""
    import time as _time
    sleep = sleep or _time.sleep
    store.parent.mkdir(parents=True, exist_ok=True)
    ledger = (json.loads(ledger_path.read_text()) if ledger_path.exists()
              else {"budget_usd": budget, "spent_usd": 0.0, "batches": []})
    done = load_replies(store)
    for model in models:
        nu_i = backend.nu_i(model)
        while items := pending(cases, model, repeats, done, nu_i):
            worst = backend.worst_case(model, len(items))
            if ledger["spent_usd"] + worst > budget:
                raise BudgetExceeded(f"{model}: {len(items)} requests could cost up to "
                                     f"${worst:.2f}; ${budget - ledger['spent_usd']:.2f} left")
            by_id = {custom_id(model, it["message"].message_id, it["repeat"]): it
                     for it in items}
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
                message = it["message"]
                reply = {"backend": backend.name, "model": model, "nu_i": nu_i,
                         "case_id": it["case"].id, "repeat": it["repeat"],
                         "message_id": message.message_id,
                         "seed": seed_for(it["case"].id, it["repeat"]),
                         "snapshot_sha": sha(it["snapshot"]), "manifest": manifest,
                         "batch_id": batch.id, **reply}
                with open(store, "a") as fh:
                    fh.write(json.dumps(reply, sort_keys=True) + "\n")
                done[(model, it["case"].id, it["repeat"], message.message_id)] = reply
                entry["cost_usd"] += reply["cost_usd"]
                ledger["spent_usd"] += reply["cost_usd"]
            ledger_path.write_text(json.dumps(ledger, indent=1) + "\n")
            progress(f"{model}: batch {batch.id} done, ${entry['cost_usd']:.4f}; "
                     f"${ledger['spent_usd']:.4f} spent of ${budget:.2f}")
    return ledger["spent_usd"]
