"""A mock interpreter for E1 dry runs: replies in the stored-API format, at no cost.

It reads the stated figure at face value (so "up to" and "at least" are taken
literally, a plausible model error), adds reading noise, reports a symmetric
interval, and misreads scope with a set probability. Its numbers say nothing
about real models; it exists to test the harness and scoring end to end.
"""

from __future__ import annotations

import json
import re

import numpy as np

from dsoi.experiments.e1_harness import custom_id
from dsoi.experiments.e1_notices import NOMINAL, Notice

PROFILES = {
    "mock-careful": {"scope_error": 0.02, "point_sd": 15.0, "half_width": 50.0,
                     "abstain_vague": 0.9, "over_interpret": 0.02},
    "mock-sloppy": {"scope_error": 0.15, "point_sd": 40.0, "half_width": 20.0,
                    "abstain_vague": 0.4, "over_interpret": 0.15},
}


def stated_capacity(notice: Notice) -> float:
    number = float(re.search(r"(\d+)", notice.statement).group(1))
    if notice.category == "absolute":
        return number
    if notice.category == "per_day":
        return NOMINAL - 5 * number
    return NOMINAL * (1 - number / 100)


def _op(a: int, b: int, value: float, lo: float, hi: float, span: str) -> dict:
    return {"op": "supersede", "subject": "plant:X", "predicate": "capacity",
            "value": value, "lo": lo, "hi": hi, "unit": "units/week",
            "valid_from": a, "valid_to": b + 1, "span_text": span}


def mock_reply(notice: Notice, profile: dict, rng) -> dict:
    """Reply fields as the harness stores them (text holds the JSON output)."""
    if notice.expect != "change":
        wrong = rng.random() < (profile["over_interpret"] if notice.expect == "no_change"
                                else 1 - profile["abstain_vague"])
        if not wrong:
            return {"kind": "abstain", "reason": "no capacity change stated", "ops": []}
        return {"kind": "delta", "reason": "", "ops": [
            _op(41, 43, 900.0, 800.0, 1000.0, notice.text)]}
    weeks = list(notice.weeks)
    if len(weeks) > 1 and rng.random() < profile["scope_error"]:
        weeks = weeks[:-1]
    point = float(np.clip(stated_capacity(notice) + profile["point_sd"] * rng.standard_normal(),
                          0, NOMINAL))
    hw = profile["half_width"]
    return {"kind": "delta", "reason": "", "ops": [
        _op(weeks[0], weeks[-1], point, max(point - hw, 0.0), min(point + hw, NOMINAL),
            notice.text)]}


def dry_run_records(notices: list[Notice], repeats: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    for model, profile in PROFILES.items():
        for notice in notices:
            for k in range(repeats):
                records.append({
                    "custom_id": custom_id(model, notice.id, k), "phase": "dryrun",
                    "model": model, "served_model": model, "notice_id": notice.id,
                    "repeat": k, "stop_reason": "end_turn", "refusal_category": None,
                    "text": json.dumps(mock_reply(notice, profile, rng)),
                    "usage": {"input_tokens": 0, "output_tokens": 0}, "cost_usd": 0.0})
    return records
