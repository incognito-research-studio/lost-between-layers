"""Phase 3 baseline: the language model decides (WS4).

The designs of the coverage mapping in which the LLM decides (Section 9.1, F4) read the
raw state and the messages and output the action. This baseline gives the model the same
world, the state before any message and every message of the case in arrival order, and
asks for the production plan of plant X directly. No L1 contract, L3 store, solver or gate
stands between the reply and execution; scoring compares the plan with the true capacity,
as for the architecture's arms.
"""

from __future__ import annotations

import json

from dsoi.experiments.e1v2_notices import HORIZON, NOMINAL

SYSTEM_PROMPT = f"""You plan weekly production at plant:X for a planning system. Demand is \
{NOMINAL:g} units in every planning week {HORIZON[0]}-{HORIZON[-1]}. Production in a week \
must not exceed plant:X's capacity in that week. Read the current state and the messages, \
then return the production plan: the units to produce in each week, as much of the demand \
as capacity allows. Output JSON that matches the schema."""

WEEK_KEYS = tuple(f"week_{w}" for w in HORIZON)


def output_schema() -> dict:
    return {
        "type": "object",
        "properties": {"reason": {"type": "string"},
                       **{k: {"type": "number"} for k in WEEK_KEYS}},
        "required": ["reason", *WEEK_KEYS],
        "additionalProperties": False,
    }


def user_prompt(snapshot: str, messages: list[tuple[str, str]]) -> str:
    """The state before the first message, then every message in arrival order."""
    body = "\n\n".join(f"Message {i + 1} (source {source}):\n<message>\n{text}\n</message>"
                       for i, (source, text) in enumerate(messages))
    return f"Current state:\n{snapshot}\n\n{body}"


def build_cached_params(snapshot: str, messages: list[tuple[str, str]], model: str,
                        effort: str, max_tokens: int, cache_ttl: str = "1h") -> dict:
    prompt = user_prompt(snapshot, messages)
    cut = prompt.index("Message 1 (source ")
    return dict(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": SYSTEM_PROMPT}],
        output_config={"effort": effort,
                       "format": {"type": "json_schema", "schema": output_schema()}},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt[:cut],
             "cache_control": {"type": "ephemeral", "ttl": cache_ttl}},
            {"type": "text", "text": prompt[cut:]},
        ]}],
    )


def parse_plan(text: str | None) -> dict[int, float] | None:
    """The plan, or None for a reply that is not a complete plan (counted as an error)."""
    if text is None:
        return None
    try:
        payload = json.loads(text)
        return {w: float(payload[k]) for w, k in zip(HORIZON, WEEK_KEYS, strict=True)}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
