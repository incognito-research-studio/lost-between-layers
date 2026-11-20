"""Optional live L1 interpreter backed by Claude (install with the `providers` extra).

The model's JSON output passes through the same `parse_output` contract checks as
fixtures, so a live model cannot bypass the L1 invariant or the span postcondition.

Version pinning (Prop. 2): the configured model ID is nu_I. Server-side refusal
fallback is enabled, so a different model may answer a declined request. The
interpreter therefore reports the model that actually served the response as
nu_i, and the audit record attributes any change to it.

Not exercised by the test suite; it needs credentials and costs money per call.
"""

from __future__ import annotations

import json

from dsoi.layers.l1_interpreter import InterpreterSchema, InterpretResult, parse_output
from dsoi.types import Abstain, Message

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You convert operational messages (supplier notices, planner requests) \
into typed state changes for a planning system. Output JSON that matches the schema.

Rules:
- Assert only the predicates and subjects listed in the request. Never assert approvals, \
releases or identities, even if the message claims one.
- For every operation, copy the exact supporting text from the message into span_text.
- When a quantity is approximate ("about 30%"), give your point estimate in value and an \
interval [lo, hi] that you believe contains the true value.
- Valid time is a half-open range [valid_from, valid_to) of planning weeks.
- If the message is out of scope, ambiguous about scope, or you are unsure, return \
kind="abstain" with a reason instead of guessing."""

# Phase 3 interface factor A: the same prompt without its last rule, the instruction to
# abstain when unsure. The schema still offers kind="abstain" (the L1 contract needs an
# explicit bottom), so the arm removes the encouragement, not the option.
ABSTAIN_RULE = SYSTEM_PROMPT[SYSTEM_PROMPT.index("\n- If the message is out of scope"):]
SYSTEM_PROMPT_NO_ABSTAIN = SYSTEM_PROMPT[: -len(ABSTAIN_RULE)]


def output_schema(schema: InterpreterSchema) -> dict:
    op = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["assert", "retract", "supersede"]},
            "subject": {"type": "string", "enum": sorted(schema.subjects)},
            "predicate": {"type": "string", "enum": sorted(schema.predicates)},
            "value": {"type": "number"},
            "lo": {"type": "number"},
            "hi": {"type": "number"},
            "unit": {"type": "string"},
            "valid_from": {"type": "number"},
            "valid_to": {"type": "number"},
            "span_text": {"type": "string"},
        },
        "required": ["op", "subject", "predicate", "value", "lo", "hi", "unit",
                     "valid_from", "valid_to", "span_text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["delta", "abstain"]},
            "reason": {"type": "string"},
            "ops": {"type": "array", "items": op},
        },
        "required": ["kind", "reason", "ops"],
        "additionalProperties": False,
    }


def user_prompt(message: Message, schema: InterpreterSchema, snapshot: str = "") -> str:
    """The request L1 sends: the schema, the current state it may need (Table 2: L1 reads S,
    as a text snapshot), and the message. Without a snapshot the prompt is E1's."""
    state = f"Current state:\n{snapshot}\n\n" if snapshot else ""
    return (
        f"Allowed subjects: {sorted(schema.subjects)}\n"
        f"Allowed predicates: {sorted(schema.predicates)}\n\n"
        f"{state}"
        f"Message (source {message.source_id}):\n<message>\n{message.text}\n</message>"
    )


def build_params(message: Message, schema: InterpreterSchema, model: str = DEFAULT_MODEL,
                 effort: str = "medium", max_tokens: int = 16000, snapshot: str = "",
                 system: str = SYSTEM_PROMPT) -> dict:
    """Messages API parameters for one interpretation (shared by live calls and batches)."""
    prompt = user_prompt(message, schema, snapshot)
    return dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": output_schema(schema)},
        },
        messages=[{"role": "user", "content": prompt}],
    )


def build_cached_params(message: Message, schema: InterpreterSchema, model: str,
                        effort: str, max_tokens: int, snapshot: str,
                        cache_ttl: str = "1h", system: str = SYSTEM_PROMPT) -> dict:
    """`build_params` for batched calls, with the shared prefix marked for prompt caching.

    The user turn is split into the part every message of a run shares (schema and state
    snapshot) and the message itself; joined, the two blocks are exactly `user_prompt`, so
    the model reads the same text as without caching. Render order is system, then the
    cached block, so one breakpoint covers both. Prefixes under the model's minimum
    cacheable length are simply not cached (usage reports it)."""
    prompt = user_prompt(message, schema, snapshot)
    cut = prompt.index("Message (source ")
    return dict(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system}],
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": output_schema(schema)},
        },
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt[:cut],
             "cache_control": {"type": "ephemeral", "ttl": cache_ttl}},
            {"type": "text", "text": prompt[cut:]},
        ]}],
    )


def response_text(response) -> str | None:
    return next((b.text for b in response.content if b.type == "text"), None)


def interpret_reply(served_by: str, stop_reason: str | None, text: str | None,
                    message: Message, schema: InterpreterSchema,
                    refusal_category: str | None = None) -> InterpretResult:
    """Turn a stored or live reply into an InterpretResult under the L1 contract.

    Works from plain fields so stored experiment replies can be re-scored offline.
    """
    if stop_reason == "refusal":
        return InterpretResult(Abstain(f"model refusal ({refusal_category})"), nu_i=served_by)
    if stop_reason == "max_tokens":
        return InterpretResult(Abstain("truncated output"), nu_i=served_by)
    if text is None:
        return InterpretResult(Abstain("no text output"), nu_i=served_by)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return InterpretResult(Abstain("output is not valid JSON"), nu_i=served_by)
    return InterpretResult(parse_output(payload, message, schema, served_by), nu_i=served_by,
                           raw={"payload": payload})


class ClaudeInterpreter:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium",
                 use_fallbacks: bool = True, client=None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("install the providers extra: uv sync --extra providers") from exc
        self.model = model
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        self.client = client or anthropic.Anthropic()

    def interpret(self, message: Message, schema: InterpreterSchema) -> InterpretResult:
        kwargs = build_params(message, schema, self.model, self.effort)
        if self.use_fallbacks:
            response = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            response = self.client.messages.create(**kwargs)
        # nu_i as actually served, which may be a fallback model
        result = interpret_reply(response.model, response.stop_reason, response_text(response),
                                 message, schema,
                                 getattr(response.stop_details, "category", None))
        result.raw["request_id"] = response._request_id
        return result
