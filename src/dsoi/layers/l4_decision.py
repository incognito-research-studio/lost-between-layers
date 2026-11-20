"""L4 Decision: x = pi_nu(kappa_D(S), theta, C, J).

Contract (Table I):
  Post: x in F(S_hat, C), or an infeasibility certificate; duals / active set attached;
        solver settings in nu.
  Guarantees C1 (Prop. 1), K-prop, and G2 via certificates (Prop. 4(b)).

The policy depends on state only through the canonical state, and the solver
runs with fixed method and options, so it is deterministic in its declared
inputs (A-DET).
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dsoi.canon import CanonicalState, lookup
from dsoi.constraints import LinearSystem, UncertainRow, solve_lp
from dsoi.context import Context

ACTIVE_TOL = 1e-6


@functools.cache
def _encoding_source(cls: type) -> str:
    """The code that encodes C and J: the class and its bases, plus the module-level
    helpers each one calls (e.g. `_usable`). Shared LP machinery (`UncertainRow`,
    `solve_lp`) is covered by nu.solver, not here."""
    parts = []
    for klass in cls.__mro__:
        if klass is object or klass.__module__ in ("typing", "builtins"):
            continue
        src = inspect.getsource(klass)
        module = inspect.getmodule(klass)
        helpers = [inspect.getsource(obj) for name, obj in sorted(vars(module).items())
                   if inspect.isfunction(obj) and obj.__module__ == module.__name__
                   and re.search(rf"\b{re.escape(name)}\(", src)]
        parts.append("\n".join([src, *helpers]))
    return "\n".join(parts)


def encoding(formulation: Formulation) -> dict:
    """The encoded constraints and objective of a formulation: its code and parameters.
    The audit log stores this under the content-hash identifiers (A-LOG, Prop. 1(d))."""
    cls = type(formulation)
    return {"class": f"{cls.__module__}.{cls.__qualname__}", "version": formulation.version,
            "params": dict(sorted(vars(formulation).items())), "source": _encoding_source(cls)}


@functools.cache
def _content_ids(cls: type, version: str, params: str) -> tuple[str, str]:
    spec = {"class": f"{cls.__module__}.{cls.__qualname__}", "version": version,
            "params": json.loads(params), "source": _encoding_source(cls)}
    payload = json.dumps(spec, sort_keys=True, default=str)
    return tuple(f"{role}:sha256:" + hashlib.sha256((role + payload).encode()).hexdigest()
                 for role in ("C", "J"))


def encoding_ids(formulation: Formulation) -> tuple[str, str]:
    """(C-id, J-id) as content hashes (A-LOG) of `encoding(formulation)`: any change to
    the encoding changes them, whatever its version label says. Cached per class, label
    and parameters, since builds are frequent and the source is fixed in a process."""
    params = json.dumps(dict(sorted(vars(formulation).items())), sort_keys=True, default=str)
    return _content_ids(type(formulation), formulation.version, params)


@dataclass(frozen=True, eq=False)
class LPInstance:
    """min J(x) = c^T x + const over the system. L is J's Lipschitz constant in l1."""

    c: np.ndarray
    const: float
    system: LinearSystem
    ids: tuple[str, str]            # (C-id, J-id), content hashes of the encoding

    @property
    def constraints_id(self) -> str:
        return self.ids[0]

    @property
    def objective_id(self) -> str:
        return self.ids[1]

    @property
    def lipschitz_l1(self) -> float:
        return float(np.max(np.abs(self.c))) if self.c.size else 0.0

    def objective(self, x: np.ndarray) -> float:
        return float(self.c @ x + self.const)


class Formulation(Protocol):
    version: str

    def build(self, state: CanonicalState, theta: dict) -> LPInstance: ...


class MissingState(KeyError):
    pass


class CapacityPlanFormulation:
    """S1 model. Variables x_w = production in week w of the horizon.

    Constraints: x_w <= capacity(plant, w) (state, possibly interval-valued),
                 0 <= x_w <= demand_w (theta, exact).
    Objective:   J(x) = sum_w (demand_w - x_w), total shortfall; 1-Lipschitz in l1.
    """

    version = "capacity-plan@1"

    def __init__(self, plant: str, horizon: tuple[float, ...]):
        self.subject = f"plant:{plant}"
        self.horizon = tuple(horizon)

    def build(self, state: CanonicalState, theta: dict) -> LPInstance:
        q = lookup(state)
        n = len(self.horizon)
        demand = np.array([float(dict(theta["demand"])[w]) for w in self.horizon])
        rows = []
        for j, w in enumerate(self.horizon):
            key = (self.subject, "capacity", w)
            if key not in q:
                raise MissingState(key)
            cap = q[key]
            e = np.zeros(n)
            e[j] = 1.0
            rows.append(UncertainRow(a_bar=e, a_hat=np.zeros(n), b_nom=cap.value,
                                     b_lo=cap.interval()[0], label=f"cap[{w:g}]", keys=(key,)))
        system = LinearSystem(tuple(rows), lower=np.zeros(n), upper=demand,
                              names=tuple(f"x[{w:g}]" for w in self.horizon))
        return LPInstance(c=-np.ones(n), const=float(demand.sum()), system=system,
                          ids=encoding_ids(self))


def _usable(q: dict, cap_key: tuple, flag_key: tuple) -> tuple[float, float]:
    """(point, lower bound) of capacity times (1 - flag), for a hold or suspension flag.

    The lower bound takes the flag at the top of its interval: a hedged reading
    reported as [0, 1] may be in force, so the box gate must treat it as in force.
    """
    if cap_key not in q:
        raise MissingState(cap_key)
    if flag_key not in q:
        raise MissingState(flag_key)
    cap, flag = q[cap_key], q[flag_key]
    return cap.value * (1.0 - flag.value), cap.interval()[0] * (1.0 - flag.interval()[1])


class LotAllocationFormulation:
    """S2/S5 model. x_l = quantity allocated from lot l in one period.

    Constraints: x_l <= qty_l * (1 - hold_l) (state); sum_l x_l <= demand (theta).
    Objective:   J(x) = demand - sum_l x_l, unmet demand; 1-Lipschitz in l1.
    """

    version = "lot-allocation@1"

    def __init__(self, lots: tuple[str, ...], period: float):
        self.lots = tuple(lots)
        self.period = float(period)

    def build(self, state: CanonicalState, theta: dict) -> LPInstance:
        q = lookup(state)
        n = len(self.lots)
        demand = float(theta["demand"])
        rows = []
        for j, lot in enumerate(self.lots):
            subject = f"lot:{lot}"
            keys = ((subject, "qty", self.period), (subject, "hold", self.period))
            b_nom, b_lo = _usable(q, *keys)
            e = np.zeros(n)
            e[j] = 1.0
            rows.append(UncertainRow(e, np.zeros(n), b_nom, b_lo, f"lot[{lot}]", keys))
        rows.append(UncertainRow(np.ones(n), np.zeros(n), demand, demand, "demand"))
        system = LinearSystem(tuple(rows), lower=np.zeros(n), upper=np.full(n, np.inf),
                              names=tuple(f"x[{lot}]" for lot in self.lots))
        return LPInstance(c=-np.ones(n), const=demand, system=system,
                          ids=encoding_ids(self))


class SupplierMixFormulation:
    """S3 model. q_s = order quantity from supplier s; u = unmet demand.

    Constraints: q_s <= capacity_s * (1 - suspended_s) (state);
                 sum_s q_s + u >= demand (theta).
    Objective:   J = sum_s cost_s q_s + penalty * u; Lipschitz in l1 = max(cost, penalty).
    """

    version = "supplier-mix@1"

    def __init__(self, suppliers: tuple[str, ...], period: float):
        self.suppliers = tuple(suppliers)
        self.period = float(period)

    def build(self, state: CanonicalState, theta: dict) -> LPInstance:
        q = lookup(state)
        n = len(self.suppliers) + 1  # last variable is u
        cost = dict(theta["cost"])
        rows = []
        for j, s in enumerate(self.suppliers):
            subject = f"supplier:{s}"
            keys = ((subject, "capacity", self.period), (subject, "suspended", self.period))
            b_nom, b_lo = _usable(q, *keys)
            e = np.zeros(n)
            e[j] = 1.0
            rows.append(UncertainRow(e, np.zeros(n), b_nom, b_lo, f"supplier[{s}]", keys))
        demand = float(theta["demand"])
        rows.append(UncertainRow(-np.ones(n), np.zeros(n), -demand, -demand, "coverage"))
        c = np.array([float(cost[s]) for s in self.suppliers] + [float(theta["penalty"])])
        system = LinearSystem(tuple(rows), lower=np.zeros(n), upper=np.full(n, np.inf),
                              names=tuple(f"q[{s}]" for s in self.suppliers) + ("u",))
        return LPInstance(c=c, const=0.0, system=system,
                          ids=encoding_ids(self))


class WarehouseFlowFormulation:
    """S8 model: pallets in (x_in) from the plant and out (x_out) to stores at one DC.

    One interpreted quantity, units per pallet c, enters two rows, and one of
    them twice with opposite signs:
      space     c x_in - c x_out <= free_space   (or == free_space if `exact_fill`)
      stores    c x_out >= store_demand
    Objective: J = in_max - x_in, receipts forgone; 1-Lipschitz in l1.

    A reported interval [lo, hi] for c becomes a_bar = point and a_hat = the larger
    distance from the point to an endpoint, a symmetric box that contains it.
    """

    version = "warehouse-flow@1"

    def __init__(self, dc: str, period: float, exact_fill: bool = False):
        self.subject = f"dc:{dc}"
        self.period = float(period)
        self.exact_fill = exact_fill

    def build(self, state: CanonicalState, theta: dict) -> LPInstance:
        q = lookup(state)
        c_key = (self.subject, "units_per_pallet", self.period)
        f_key = (self.subject, "free_space", self.period)
        for k in (c_key, f_key):
            if k not in q:
                raise MissingState(k)
        c, free = q[c_key], q[f_key].value
        lo, hi = c.interval()
        v, h = c.value, max(c.value - lo, hi - c.value)
        demand = float(theta["store_demand"])
        rows = [
            UncertainRow(np.array([v, -v]), np.array([h, h]), free, free, "space",
                         (c_key, f_key)),
            UncertainRow(np.array([0.0, -v]), np.array([0.0, h]), -demand, -demand, "stores",
                         (c_key,)),
        ]
        if self.exact_fill:  # the reverse inequality makes the space row an equality
            rows.append(UncertainRow(np.array([-v, v]), np.array([h, h]), -free, -free,
                                     "space_fill", (c_key, f_key)))
        system = LinearSystem(tuple(rows), lower=np.zeros(2),
                              upper=np.array([float(theta["in_max"]), float(theta["out_max"])]),
                              names=("x_in", "x_out"))
        return LPInstance(c=np.array([-1.0, 0.0]), const=float(theta["in_max"]), system=system,
                          ids=encoding_ids(self))


@dataclass(frozen=True, eq=False)
class Decision:
    x: np.ndarray | None
    status: str                     # optimal | infeasible | error
    objective: float | None
    certificate: dict = field(default_factory=dict)


class LPPolicy:
    def __init__(self, formulation: Formulation):
        self.formulation = formulation

    def decide(self, ctx: Context) -> tuple[Decision, LPInstance]:
        inst = self.formulation.build(ctx.state, ctx.theta_dict())
        res = solve_lp(inst.c, inst.system, robust=False)
        if res.status == 2:
            cert = {"message": res.message, "iis": irreducible_infeasible_subsystem(inst.system)}
            return Decision(None, "infeasible", None, cert), inst
        if res.status != 0:
            return Decision(None, "error", None, {"message": res.message}), inst
        x = np.asarray(res.x)
        duals = [] if res.ineqlin is None else [float(-m) for m in res.ineqlin.marginals]
        active = [r.label for r, y in zip(inst.system.rows, duals, strict=False)
                  if y > ACTIVE_TOL]
        bounds = [[name, side] for side, marg in (("upper", res.upper.marginals),
                                                  ("lower", res.lower.marginals))
                  for name, m in zip(inst.system.names, marg, strict=True)
                  if abs(m) > ACTIVE_TOL]
        cert = {"duals": duals, "active_rows": active, "active_bounds": bounds,
                "objective": inst.objective(x)}
        return Decision(x, "optimal", inst.objective(x), cert), inst


def irreducible_infeasible_subsystem(system: LinearSystem) -> list[str]:
    """Deletion filter: a minimal set of rows that is infeasible with the variable bounds.

    Rows are dropped one at a time while the rest stays infeasible; what remains is
    an IIS relative to the bounds (which are treated as background constraints).
    """
    def feasible(rows) -> bool:
        trial = LinearSystem(tuple(rows), system.lower, system.upper, system.names)
        return solve_lp(np.zeros(system.n), trial).status == 0

    keep = list(system.rows)
    if feasible(keep):
        return []
    for row in list(keep):
        rest = [r for r in keep if r is not row]
        if not feasible(rest):
            keep = rest
    return [r.label for r in keep]
