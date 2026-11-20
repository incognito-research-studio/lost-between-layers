"""E1 harness: live pilot, batched main run, a budget ledger, and the response store.

Every API reply is stored raw (text, stop reason, usage, served model) so that
scoring is offline and repeatable: the same replies always give the same scores.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dsoi.experiments.e1_notices import Notice
from dsoi.layers.l1_claude import build_params, response_text
from dsoi.layers.l1_interpreter import InterpreterSchema
from dsoi.types import Message

SOURCE = "email:e1"
WORST_CASE_INPUT_TOKENS = 2000


def load_env(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines from a git-ignored .env without overriding the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def short(model: str) -> str:
    return model.replace("claude-", "").split("-")[0]


def custom_id(model: str, notice_id: str, repeat: int) -> str:
    return f"{short(model)}-{notice_id}-r{repeat}"


def message_for(notice: Notice) -> Message:
    return Message(notice.id, SOURCE, notice.text)


def cost_usd(model: str, usage: dict, rates: dict, batch: bool, discount: float) -> float:
    r = rates[model]
    cost = (usage.get("input_tokens", 0) * r["input"]
            + usage.get("output_tokens", 0) * r["output"]) / 1e6
    return cost * (discount if batch else 1.0)


# -- ledger --------------------------------------------------------------------------


@dataclass
class Ledger:
    path: Path
    budget: float
    spent: float = 0.0
    entries: list = field(default_factory=list)
    batches: list = field(default_factory=list)

    @classmethod
    def open(cls, path: Path, budget: float) -> Ledger:
        if path.exists():
            d = json.loads(path.read_text())
            return cls(path, budget, d["spent"], d["entries"], d["batches"])
        return cls(path, budget)

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    def charge(self, phase: str, model: str, n: int, cost: float) -> None:
        self.spent += cost
        self.entries.append({"phase": phase, "model": model, "n": n, "cost": round(cost, 6),
                             "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"budget": self.budget, "spent": self.spent,
                                         "entries": self.entries, "batches": self.batches},
                                        indent=2) + "\n")


class BudgetExceeded(RuntimeError):
    pass


# -- response store --------------------------------------------------------------------


def append_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def record_from_message(msg, model: str, notice: Notice, repeat: int, phase: str,
                        cost: float, extra: dict) -> dict:
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    details = getattr(msg, "stop_details", None)
    return {"custom_id": custom_id(model, notice.id, repeat), "phase": phase,
            "model": model, "served_model": msg.model, "notice_id": notice.id,
            "repeat": repeat, "stop_reason": msg.stop_reason,
            "refusal_category": getattr(details, "category", None),
            "text": response_text(msg), "usage": usage, "cost_usd": cost, **extra}


# -- live pilot --------------------------------------------------------------------------


def run_pilot(client, config: dict, notices: list[Notice], schema: InterpreterSchema,
              ledger: Ledger, store: Path) -> list[dict]:
    """One live call per pilot notice and model, each checked against the worst-case cost."""
    out = []
    rates, max_tokens = config["rates_usd_per_mtok"], config["max_tokens"]
    for model in config["models"]:
        worst = cost_usd(model, {"input_tokens": WORST_CASE_INPUT_TOKENS,
                                 "output_tokens": max_tokens}, rates, False, 1.0)
        for notice in notices:
            if ledger.spent + worst > ledger.budget:
                raise BudgetExceeded(f"pilot stopped before {model}/{notice.id}: "
                                     f"${ledger.spent:.2f} spent, worst case ${worst:.2f}")
            params = build_params(message_for(notice), schema, model, config["effort"],
                                  max_tokens)
            msg = client.messages.create(**params)
            usage = {"input_tokens": msg.usage.input_tokens,
                     "output_tokens": msg.usage.output_tokens}
            cost = cost_usd(model, usage, rates, False, 1.0)
            rec = record_from_message(msg, model, notice, 0, "pilot", cost,
                                      {"request_id": msg._request_id})
            append_records(store, [rec])
            ledger.charge("pilot", model, 1, cost)
            out.append(rec)
    return out


# -- planning and batches -------------------------------------------------------------------


def plan_main(config: dict, notices: list[Notice], pilot: list[dict], ledger: Ledger) -> dict:
    """Size the main run to the remaining budget from the pilot's measured costs.

    Repeats are cut first (down to 1), then the notice set (keeping category balance).
    """
    disc, safety = config["batch_discount"], config["safety_factor"]
    per_req = {}
    for model in config["models"]:
        costs = [r["cost_usd"] for r in pilot if r["model"] == model]
        if not costs:
            raise ValueError(f"no pilot data for {model}; run the pilot first")
        per_req[model] = sum(costs) / len(costs) * disc * safety
    unit = sum(per_req.values())               # one request per model
    repeats, ids = config["repeats"], [n.id for n in notices]
    while repeats > 1 and unit * len(ids) * repeats > ledger.remaining:
        repeats -= 1
    if unit * len(ids) * repeats > ledger.remaining:
        keep = int(ledger.remaining / (unit * repeats))
        ids = ids[::max(1, -(-len(ids) // max(keep, 1)))]
    return {"repeats": repeats, "notice_ids": ids,
            "per_request_estimate": per_req,
            "estimate_usd": unit * len(ids) * repeats,
            "remaining_usd": ledger.remaining}


def submit_and_collect(client, config: dict, plan: dict, by_id: dict[str, Notice],
                       schema: InterpreterSchema, ledger: Ledger, store: Path,
                       poll_seconds: int = 30, timeout_seconds: int = 4 * 3600) -> None:
    """One batch per model, cheapest first; each is re-checked against the ledger."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    done = {r["custom_id"] for r in load_records(store) if r["phase"] == "main"}
    for model in config["models"]:
        todo = [(nid, k) for nid in plan["notice_ids"] for k in range(plan["repeats"])
                if custom_id(model, nid, k) not in done]
        if not todo:
            continue
        estimate = plan["per_request_estimate"][model] * len(todo)
        if estimate > ledger.remaining:
            raise BudgetExceeded(f"{model}: estimate ${estimate:.2f} exceeds remaining "
                                 f"${ledger.remaining:.2f}")
        requests = [Request(custom_id=custom_id(model, nid, k),
                            params=MessageCreateParamsNonStreaming(
                                **build_params(message_for(by_id[nid]), schema, model,
                                               config["effort"], config["max_tokens"])))
                    for nid, k in todo]
        batch = client.messages.batches.create(requests=requests)
        ledger.batches.append({"id": batch.id, "model": model, "n": len(todo),
                               "estimate": round(estimate, 4)})
        ledger.save()
        collect_batch(client, batch.id, model, by_id, config, ledger, store,
                      poll_seconds, timeout_seconds)


def collect_batch(client, batch_id: str, model: str, by_id: dict[str, Notice], config: dict,
                  ledger: Ledger, store: Path, poll_seconds: int = 30,
                  timeout_seconds: int = 4 * 3600) -> None:
    started = time.time()
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"batch {batch_id} still running; resume with `collect`")
        time.sleep(poll_seconds)
    rates, disc = config["rates_usd_per_mtok"], config["batch_discount"]
    records, total = [], 0.0
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        nid, rep = cid.split("-", 1)[1].rsplit("-r", 1)
        notice = by_id[nid]
        if result.result.type == "succeeded":
            msg = result.result.message
            usage = {"input_tokens": msg.usage.input_tokens,
                     "output_tokens": msg.usage.output_tokens}
            cost = cost_usd(model, usage, rates, True, disc)
            total += cost
            records.append(record_from_message(msg, model, notice, int(rep), "main", cost,
                                               {"batch_id": batch_id}))
        else:
            records.append({"custom_id": cid, "phase": "main", "model": model,
                            "notice_id": nid, "repeat": int(rep), "batch_id": batch_id,
                            "error": result.result.type, "cost_usd": 0.0})
    append_records(store, records)
    ledger.charge("main", model, len(records), total)
