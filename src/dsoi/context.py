"""Version vector and decision context (Def. 5)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from dsoi.canon import CanonicalState

UNSET = "unset"


@dataclass(frozen=True)
class VersionVector:
    """nu: every artifact whose change can alter a decision (design rule 3).

    `gate` (the L5 rung with its repair mode), `source_priority` (the L3 source
    ranks) and `sop_rules` (L2's threshold rules, approver roles and lift issuers)
    vary per study, so the pipeline fills them in from its components when they are
    left UNSET; a pinned value that disagrees with the component is an error
    (Pipeline.nu).
    """

    formulation: str
    solver: str
    kappa: str
    projection: str
    sop: str
    interpreter: str
    resolution: str = "latest@1"  # L3 conflict policy (S4); it changes decisions
    confidence: str = "none"      # thresholds "lo/hi" for yes/no readings (Def. 5, Sec. 7)
    gate: str = UNSET             # L5 rung and repair mode, e.g. "box_robust/project"
    source_priority: str = UNSET  # L3 source ranks used by the priority and span policies
    sop_rules: str = UNSET        # L2 rules: thresholds with approver roles, lift issuers
    monitors: str = UNSET         # L5 trace monitors that can turn a release into a rejection

    @classmethod
    def load(cls, path: str | Path) -> VersionVector:
        with open(path) as fh:
            return cls(**yaml.safe_load(fh))

    def bump(self, **changes: str) -> VersionVector:
        return replace(self, **changes)

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def id(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Context:
    """c = (kappa_D(S), theta, C, J, nu).

    `theta` is stored as a sorted tuple of items so the context is hashable and
    its serialization is deterministic.
    """

    state: CanonicalState
    theta: tuple[tuple[str, object], ...]
    constraints_id: str
    objective_id: str
    nu: VersionVector

    @staticmethod
    def freeze_theta(theta: dict) -> tuple[tuple[str, object], ...]:
        return tuple(sorted((k, _freeze(v)) for k, v in theta.items()))

    def theta_dict(self) -> dict:
        return {k: v for k, v in self.theta}


def _freeze(v: object) -> object:
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v
