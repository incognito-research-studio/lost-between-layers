"""A synthetic L1 interpreter for capacity notices, with configurable failure modes.

It stands in for a stochastic interpreter so that the gate ladder can be studied
over many draws. The parameters are illustrative, not measured; calibrating them
against a real interpreter is the empirical work the paper leaves open (A2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsoi.layers.l1_interpreter import InterpreterSchema, InterpretResult, parse_output
from dsoi.types import Message

VERSION = "synthetic@1"


@dataclass(frozen=True)
class NoiseModel:
    point_sd: float = 0.03            # sd of the read reduction around the stated one
    bias: float = 0.0                 # systematic offset of the read reduction
    half_width: float = 0.05          # half-width of the reported interval (reduction units)
    scope_misread_prob: float = 0.0   # drops the last affected week from valid time
    abstain_prob: float = 0.0         # reports the declared domain [0, nominal] instead

    @classmethod
    def from_dict(cls, d: dict) -> NoiseModel:
        return cls(**d)


@dataclass(frozen=True)
class CapacityNotice:
    """What the supplier wrote: an approximate reduction over a week range."""

    subject: str
    stated_reduction: float
    first_week: int
    last_week: int
    nominal: float

    def text(self) -> str:
        pct = round(100 * self.stated_reduction)
        return (f"Plant X update: capacity reduced by about {pct}% for weeks "
                f"{self.first_week}-{self.last_week}.")

    def span(self) -> str:
        pct = round(100 * self.stated_reduction)
        return f"capacity reduced by about {pct}% for weeks {self.first_week}-{self.last_week}"


class SyntheticInterpreter:
    """Deterministic given its seed, so a run can be replayed exactly."""

    version = VERSION

    def __init__(self, notice: CapacityNotice, noise: NoiseModel, seed: int):
        self.notice = notice
        self.noise = noise
        self.seed = seed

    def interpret(self, message: Message, schema: InterpreterSchema) -> InterpretResult:
        rng = np.random.default_rng(self.seed)
        nz, nt = self.noise, self.notice
        abstain = rng.random() < nz.abstain_prob
        misread = rng.random() < nz.scope_misread_prob
        r_hat = nt.stated_reduction + nz.bias + nz.point_sd * rng.standard_normal()
        if abstain:
            value, lo, hi = nt.nominal, 0.0, nt.nominal
        else:
            value = float(np.clip(nt.nominal * (1 - r_hat), 0, nt.nominal))
            lo = float(np.clip(nt.nominal * (1 - r_hat - nz.half_width), 0, nt.nominal))
            hi = float(np.clip(nt.nominal * (1 - r_hat + nz.half_width), 0, nt.nominal))
        last = nt.last_week - 1 if misread else nt.last_week
        payload = {"kind": "delta", "ops": [{
            "op": "supersede", "subject": nt.subject, "predicate": "capacity",
            "value": value, "lo": lo, "hi": hi, "unit": "units/week",
            "valid_from": nt.first_week, "valid_to": last + 1, "span_text": nt.span(),
        }]}
        out = parse_output(payload, message, schema, VERSION)
        return InterpretResult(out, nu_i=VERSION,
                               raw={"r_hat": r_hat, "abstain": abstain, "misread": misread})
