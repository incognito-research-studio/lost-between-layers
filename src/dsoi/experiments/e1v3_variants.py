"""Phase 3 equivalent-input variants (WS4): rule-based rewrites that keep a notice's meaning.

Each transform rewrites the message text of a single-message test case so that, under the
case's truth rules, the true capacity is unchanged. Proposition 1 predicts that the
architecture's decision changes only when the canonical interpretation does; the "LLM
decides" baseline has no such guarantee. Comparing each variant's decision with the
original's measures that difference with real models. The rewrites are deterministic
string rules, not generated paraphrases, so no model writes the inputs another reads.

  name_swap      "Plant X" <-> "the Northfield plant" (the snapshot lists both names)
  range_format   "weeks a-b" <-> "weeks a to b"; "week a" <-> "Wa"
  hedge_synonym  about -> roughly -> approximately -> around -> about (same truth rule)
  unit_restate   "at most c units per week" -> "at most c/5 units per working day (five
                 working days per week)"; "drops by d units per working day ..." -> "drops
                 by 5d units per week"; "is reduced by p%" -> "will run at (100 - p)% of
                 normal capacity"
"""

from __future__ import annotations

import re
from dataclasses import replace

from dsoi.experiments.e1_notices import HEDGES
from dsoi.experiments.e1v2_notices import Case

CATEGORIES = ("hedged_pct", "exact_pct", "upto_pct", "atleast_pct", "absolute", "per_day")


def name_swap(text: str) -> str | None:
    if "Plant X" in text:
        out = re.sub(r"(^|[.:;!?]\s+)Plant X", lambda m: m.group(1) + "The Northfield plant",
                     text)
        return out.replace("Plant X", "the Northfield plant")
    if "orthfield plant" in text:
        return text.replace("The Northfield plant", "Plant X").replace(
            "the Northfield plant", "Plant X")
    return None


_RANGES = [
    (re.compile(r"\bweeks (\d+)-(\d+)\b"), lambda a, b: f"weeks {a} to {b}"),
    (re.compile(r"\bweeks (\d+) (?:to|through) (\d+)\b"), lambda a, b: f"weeks {a}-{b}"),
    (re.compile(r"\bW(\d+)-W(\d+)\b"), lambda a, b: f"weeks {a}-{b}"),
    (re.compile(r"\bthe (?:two|three|four) weeks starting week (\d+)\b"), None),
    (re.compile(r"\bweek (\d+)\b"), lambda a: f"W{a}"),
    (re.compile(r"\bW(\d+)\b"), lambda a: f"week {a}"),
]


def range_format(text: str) -> str | None:
    for pattern, render in _RANGES:
        m = pattern.search(text)
        if m:
            if render is None:      # the spelled-out form: rewrite to weeks a-b
                n = {"two": 2, "three": 3, "four": 4}[m.group(0).split()[1]]
                a = int(m.group(1))
                return text[:m.start()] + f"weeks {a}-{a + n - 1}" + text[m.end():]
            return text[:m.start()] + render(*m.groups()) + text[m.end():]
    return None


def hedge_synonym(text: str) -> str | None:
    for i, h in enumerate(HEDGES):
        m = re.search(rf"\b{h}\b", text)
        if m:
            nxt = HEDGES[(i + 1) % len(HEDGES)]
            return text[:m.start()] + nxt + text[m.end():]
    return None


def unit_restate(text: str) -> str | None:
    m = re.search(r"at most (\d+) units per week", text)
    if m and int(m.group(1)) % 5 == 0:
        return text.replace(m.group(0), f"at most {int(m.group(1)) // 5} units per working "
                                         f"day (five working days per week)")
    m = re.search(r"drops by (\d+) units per working day for (.+?) \(five working days per "
                  r"week\)", text)
    if m:
        return text.replace(m.group(0), f"drops by {5 * int(m.group(1))} units per week for "
                                         f"{m.group(2)}")
    m = re.search(r"is reduced by (\d+)%", text)
    if m:
        return text.replace(m.group(0), f"will run at {100 - int(m.group(1))}% of normal "
                                         f"capacity")
    return None


TRANSFORMS = {"name_swap": name_swap, "range_format": range_format,
              "hedge_synonym": hedge_synonym, "unit_restate": unit_restate}


def variants(cases: list[Case]) -> list[Case]:
    """Every applicable variant of the single-message test cases of CATEGORIES, with the
    original's truth; ids are '<case id>-v-<transform>'."""
    out = []
    for c in cases:
        if c.split != "test" or c.category not in CATEGORIES or len(c.messages) != 1:
            continue
        (source, text), = c.messages
        for name, fn in TRANSFORMS.items():
            new = fn(text)
            if new is not None and new != text:
                out.append(replace(c, id=f"{c.id}-v-{name}", messages=((source, new),),
                                   statement=f"{c.statement} [{name}]"))
    return out


def original_id(variant_id: str) -> str:
    return variant_id.split("-v-", 1)[0]
