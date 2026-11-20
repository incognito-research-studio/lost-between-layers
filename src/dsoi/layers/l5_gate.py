"""L5 Constraint & Assurance: G(x, ctx) -> x^e or reject (Def. 7, Sec. VI).

The ladder of gates (Sec. VI), from most to least conservative:
  BoxRobustGate     K-exec over Gamma(o) by worst-case substitution (Cor. 2),
                    provided the reported intervals cover the truth (GSN goal A2).
  StateVertexGate   exact robust set over the reported state box (Prop. 3(b));
                    at least as permissive as the box gate, and equal to it when
                    each coefficient comes from its own quantity (Cor. 2(b)).
  BudgetedGate      Bertsimas-Sim budget; with one uncertain term per row (S1)
                    it coincides with the box or nominal gate.
  ChanceGate        epsilon-quantile of an assumed error distribution.
  NominalGate       K-prop only; violations it lets through are interpretation
                    or specification errors (Prop. 3(c)).
RecheckGate re-reads constraint-relevant facts at execution time (Prop. 2(a)). It is called
a recheck, not an oracle: it gives K-exec only under A-ORACLE, the assumption that the
system of record is current and read atomically with execution, which S3 and the
execution study break on purpose.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
from scipy.stats import norm

from dsoi.constraints import budgeted_feasible, project_l1, solve_lp, solve_lp_budgeted
from dsoi.context import Context
from dsoi.layers.l4_decision import Formulation, LPInstance


@dataclass(frozen=True, eq=False)
class GateResult:
    rung: str
    outcome: str                 # pass | projected | resolved | rejected | fallback
    released: np.ndarray | None  # None = reject (no action released)
    detail: dict = field(default_factory=dict)


class Gate(Protocol):
    rung: str

    @property
    def label(self) -> str: ...

    def apply(self, x: np.ndarray | None, inst: LPInstance, ctx: Context,
              formulation: Formulation) -> GateResult: ...


def gate_label(gate) -> str:
    """The gate's entry in nu: rung plus every setting that changes what it releases."""
    return getattr(gate, "label", gate.rung)


def _repair(x, inst: LPInstance, robust: bool, mode: str, rung: str) -> GateResult:
    if mode == "reject":
        return GateResult(rung, "rejected", None, {"reason": "candidate outside gate set"})
    if mode == "project":
        y = project_l1(x, inst.system, robust=robust)
        outcome = "projected"
    elif mode == "resolve":
        res = solve_lp(inst.c, inst.system, robust=robust)
        y = res.x if res.status == 0 else None
        outcome = "resolved"
    else:
        raise ValueError(mode)
    if y is None:  # gate set empty: Prop. 3 requires rejection
        return GateResult(rung, "rejected", None, {"reason": "gate set is empty"})
    return GateResult(rung, outcome, np.asarray(y))


class NominalGate:
    rung = "nominal"

    def __init__(self, mode: str = "project"):
        self.mode = mode

    @property
    def label(self) -> str:
        return f"{self.rung}/{self.mode}"

    def apply(self, x, inst, ctx, formulation) -> GateResult:
        if x is None:
            return GateResult(self.rung, "rejected", None, {"reason": "no candidate"})
        if inst.system.nominal_feasible(x):
            return GateResult(self.rung, "pass", x)
        return _repair(x, inst, robust=False, mode=self.mode, rung=self.rung)


class BoxRobustGate:
    """Releases only actions in F_box(o) (Cor. 2). `mode` chooses how to repair."""

    rung = "box_robust"

    def __init__(self, mode: str = "project"):
        self.mode = mode

    @property
    def label(self) -> str:
        return f"{self.rung}/{self.mode}"

    def apply(self, x, inst, ctx, formulation) -> GateResult:
        if x is None:
            return _repair(np.zeros(inst.system.n), inst, robust=True, mode="resolve",
                           rung=self.rung)
        if inst.system.robust_feasible(x):
            return GateResult(self.rung, "pass", x)
        return _repair(x, inst, robust=True, mode=self.mode, rung=self.rung)


class BudgetedGate:
    """Bertsimas-Sim rung: at most `gamma` coefficients per row deviate from nominal.

    K-exec holds only if the truth respects the budget; outside it the guarantee
    becomes a probability bound (Sec. VI). gamma = 0 uses b_lo with nominal
    coefficients; gamma >= n is the box gate. Repairs by re-solving.
    """

    def __init__(self, gamma: int, mode: str = "resolve"):
        if mode not in ("resolve", "reject"):
            raise ValueError("budgeted gate repairs by 'resolve' or 'reject'")
        self.gamma = gamma
        self.mode = mode
        self.rung = f"budgeted:{gamma}"

    @property
    def label(self) -> str:
        return f"{self.rung}/{self.mode}"

    def apply(self, x, inst, ctx, formulation) -> GateResult:
        if x is not None and budgeted_feasible(x, inst.system, self.gamma):
            return GateResult(self.rung, "pass", x)
        if self.mode == "reject":
            return GateResult(self.rung, "rejected", None, {"reason": "outside budgeted set"})
        res = solve_lp_budgeted(inst.c, inst.system, self.gamma)
        if res.status != 0:
            return GateResult(self.rung, "rejected", None, {"reason": "budgeted set is empty"})
        return GateResult(self.rung, "resolved", np.asarray(res.x))


class StateVertexGate:
    """Exact robust rung over the reported state box: F_rob(o) of Prop. 3(b).

    Gamma(o) is taken as the product of the reported intervals. Each
    interval-valued key in R(C) is set to its endpoints (2^k states), the
    formulation is instantiated at every vertex, and the rows are stacked.
    This is exact whenever each constraint is multilinear in the state values,
    which holds for every formulation in this package. Unlike the box gate, it
    keeps the link between coefficients that come from one interpreted quantity
    (Sec. VI, second caveat).
    """

    rung = "state_vertex"

    def __init__(self, mode: str = "resolve", max_keys: int = 10):
        self.mode = mode
        self.max_keys = max_keys

    @property
    def label(self) -> str:
        return f"{self.rung}/{self.mode}"

    def vertex_instance(self, inst: LPInstance, ctx: Context,
                        formulation: Formulation) -> LPInstance:
        keys = {k for r in inst.system.rows for k in r.keys}
        uncertain = [(i, e) for i, e in enumerate(ctx.state)
                     if e[0] in keys and e[2] is not None and e[2] != e[3]]
        if len(uncertain) > self.max_keys:
            raise ValueError(f"{len(uncertain)} interval-valued keys: 2^k vertices is too many")
        rows = []
        theta = ctx.theta_dict()
        for corner in itertools.product(*[(e[2], e[3]) for _, e in uncertain]):
            state = list(ctx.state)
            for (i, (key, _v, _lo, _hi, unit)), value in zip(uncertain, corner, strict=True):
                state[i] = (key, value, None, None, unit)
            rows.extend(formulation.build(tuple(state), theta).system.rows)
        return replace(inst, system=replace(inst.system, rows=tuple(rows)))

    def apply(self, x, inst, ctx, formulation) -> GateResult:
        v = self.vertex_instance(inst, ctx, formulation)
        if x is not None and v.system.nominal_feasible(x):
            return GateResult(self.rung, "pass", x)
        start = x if x is not None else np.zeros(inst.system.n)
        return _repair(start, v, robust=False, mode=self.mode, rung=self.rung)


class ChanceGate:
    """Chance-constrained rung for right-hand-side uncertainty.

    Reads each reported interval [b_lo, b_nom] (lower half) as part of a central
    `interval_level` band of a normal error around the point value, and releases
    only actions feasible at the epsilon-quantile of b. epsilon = 0.5 gives the
    nominal gate; epsilon = (1 - interval_level) / 2 gives the box gate.

    The guarantee is "violation probability <= epsilon if the assumed error
    distribution is right": a statistical claim about the interpreter, not a
    guarantee (Sec. VI).
    """

    def __init__(self, epsilon: float, interval_level: float = 0.95, mode: str = "project"):
        if not 0 < epsilon <= 0.5:
            raise ValueError("epsilon must lie in (0, 0.5]")
        self.epsilon = epsilon
        self.interval_level = interval_level
        self.mode = mode
        self.rung = f"chance:{epsilon:g}"

    @property
    def label(self) -> str:
        return f"{self.rung}@{self.interval_level:g}/{self.mode}"

    def quantile_system(self, inst: LPInstance) -> LPInstance:
        z_int = norm.ppf((1 + self.interval_level) / 2)
        z_eps = norm.ppf(1 - self.epsilon)
        rows = []
        for r in inst.system.rows:
            if np.any(r.a_hat):
                raise NotImplementedError("chance gate supports right-hand-side uncertainty only")
            sigma = (r.b_nom - r.b_lo) / z_int
            b = r.b_nom - z_eps * sigma
            rows.append(replace(r, b_nom=b, b_lo=b))
        return replace(inst, system=replace(inst.system, rows=tuple(rows)))

    def apply(self, x, inst, ctx, formulation) -> GateResult:
        q = self.quantile_system(inst)
        if x is not None and q.system.nominal_feasible(x):
            return GateResult(self.rung, "pass", x)
        start = x if x is not None else np.zeros(inst.system.n)
        return _repair(start, q, robust=False, mode=self.mode, rung=self.rung)


class RecheckGate:
    """Execution-time re-read of constraint-relevant facts from the system of record.

    `read_true(key)` returns the authenticated value for a state key in R(C).
    The re-read values are returned in `detail` so L6 logs them before x^e
    (Table 3, K-exec row). Prop. 2(a) needs A-ORACLE and A-SPEC; a stale record or a
    change between the re-read and execution breaks A-ORACLE, not this code.
    """

    rung = "recheck"

    def __init__(self, read_true: Callable[[tuple], float], fallback: str = "project",
                 decimals: int = 6):
        self.read_true = read_true
        self.fallback = fallback
        self.decimals = decimals  # re-read values enter the canonical state: same normalization

    @property
    def label(self) -> str:
        return f"{self.rung}/{self.fallback}/decimals={self.decimals}"

    def apply(self, x, inst, ctx, formulation) -> GateResult:
        keys = sorted({k for r in inst.system.rows for k in r.keys})
        reread = {k: round(float(self.read_true(k)), self.decimals) for k in keys}
        state = tuple(
            (key, reread[key], None, None, unit) if key in reread else (key, v, lo, hi, unit)
            for key, v, lo, hi, unit in ctx.state
        )
        true_inst = formulation.build(state, ctx.theta_dict())
        detail = {"reread": {"|".join(map(str, k)): v for k, v in reread.items()}}
        if x is not None and true_inst.system.nominal_feasible(x):
            return GateResult(self.rung, "pass", x, detail)
        repaired = _repair(x if x is not None else np.zeros(inst.system.n), true_inst,
                           robust=False, mode=self.fallback, rung=self.rung)
        outcome = "fallback" if repaired.released is not None else "rejected"
        return GateResult(self.rung, outcome, repaired.released, {**detail, **repaired.detail})

