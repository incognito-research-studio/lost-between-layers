"""Phase 3 world: e1v2's case pipeline with the corrected source ranks and the phase 3
supersession rule, and the four interface arms of the 2 x 2.

Arms (factor A: the abstain instruction; factor B: the state snapshot sent with each
message). `ref` is the original prompt and interface of e1v2.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsoi.experiments import e1v2_world
from dsoi.experiments.e1v3_cases import glossary
from dsoi.layers.l1_claude import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_ABSTAIN

SUPERSESSION = "authenticated_latest"   # decided 8 Nov 2026 (later transaction wins)
messages, schema, sha, state_of = (e1v2_world.messages, e1v2_world.schema, e1v2_world.sha,
                                   e1v2_world.state_of)


@dataclass(frozen=True)
class Arm:
    name: str
    abstain_instruction: bool
    state_snapshot: bool

    @property
    def system(self) -> str:
        return SYSTEM_PROMPT if self.abstain_instruction else SYSTEM_PROMPT_NO_ABSTAIN


ARMS = {a.name: a for a in (
    Arm("ref", True, True),          # the original interface (reference)
    Arm("noabs", False, True),       # no abstain instruction
    Arm("nosnap", True, False),      # state withheld
    Arm("neither", False, False),
)}


def pipeline(case, gate, interpreter, nu_i: str):
    """The corrected e1v2 pipeline (notices outrank the baseline master) with the phase 3
    rule that a newer authenticated record supersedes an earlier notice."""
    scn, p = e1v2_world.pipeline(case, gate, interpreter, nu_i)
    p.state.supersession = SUPERSESSION
    return scn, p


def snapshot_text(p, t: float, arm: Arm) -> str:
    return e1v2_world.snapshot_text(p, t) if arm.state_snapshot else glossary()
