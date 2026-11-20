"""Linear constraints with interval-valued data (Def. 6, Cor. 2).

Row i reads  a_i(S)^T x <= b_i(S).  For an interpreted state, each coefficient
lies in [a_bar - a_hat, a_bar + a_hat] and the right-hand side satisfies
b_i(S) >= b_lo. Exact data has a_hat = 0 and b_lo = b_nom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

TOL = 1e-7
HIGHS_OPTIONS = {"presolve": True}  # A-DET: fixed method and options


@dataclass(frozen=True, eq=False)
class UncertainRow:
    a_bar: np.ndarray
    a_hat: np.ndarray
    b_nom: float
    b_lo: float
    label: str
    keys: tuple[tuple, ...] = ()  # state keys whose values occur in this row: part of R(C)

    def nominal_lhs(self, x: np.ndarray) -> float:
        return float(self.a_bar @ x)

    def robust_lhs(self, x: np.ndarray) -> float:
        """Worst-case substitution (Soyster): a_bar x + a_hat |x|."""
        return float(self.a_bar @ x + self.a_hat @ np.abs(x))


@dataclass(frozen=True, eq=False)
class LinearSystem:
    """A constraint set C instantiated on a state, plus variable bounds."""

    rows: tuple[UncertainRow, ...]
    lower: np.ndarray  # variable lower bounds (use -inf for free)
    upper: np.ndarray  # variable upper bounds (use +inf for free)
    names: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.names)

    def _in_bounds(self, x: np.ndarray, tol: float) -> bool:
        return bool(np.all(x >= self.lower - tol) and np.all(x <= self.upper + tol))

    def nominal_feasible(self, x: np.ndarray, tol: float = TOL) -> bool:
        """x in F(S_hat, C), using the interpreter's point values."""
        return self._in_bounds(x, tol) and all(
            r.nominal_lhs(x) <= r.b_nom + tol for r in self.rows
        )

    def robust_feasible(self, x: np.ndarray, tol: float = TOL) -> bool:
        """x in F_box(o) of Cor. 2."""
        return self._in_bounds(x, tol) and all(
            r.robust_lhs(x) <= r.b_lo + tol for r in self.rows
        )

    def nominal_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.array([r.a_bar for r in self.rows]).reshape(len(self.rows), self.n)
        b = np.array([r.b_nom for r in self.rows], dtype=float)
        return a, b

    def violated_rows(self, x: np.ndarray, robust: bool, tol: float = TOL) -> list[str]:
        out = []
        for r in self.rows:
            lhs = r.robust_lhs(x) if robust else r.nominal_lhs(x)
            rhs = r.b_lo if robust else r.b_nom
            if lhs > rhs + tol:
                out.append(r.label)
        return out


def _bounds(system: LinearSystem) -> list[tuple[float | None, float | None]]:
    return [
        (None if np.isneginf(lo) else float(lo), None if np.isposinf(hi) else float(hi))
        for lo, hi in zip(system.lower, system.upper, strict=True)
    ]


def solve_lp(c: np.ndarray, system: LinearSystem, robust: bool = False):
    """min c^T x over F(S_hat, C) (robust=False) or over F_box(o) (robust=True).

    The robust case lifts |x| with t_j >= +-x_j (Cor. 2(c)), so an LP stays an LP.
    Returns the scipy result; for the robust lift, `res.x` is truncated to x.
    """
    n = system.n
    if not robust:
        a, b = system.nominal_matrices()
        return linprog(c, A_ub=a if len(b) else None, b_ub=b if len(b) else None,
                       bounds=_bounds(system), method="highs", options=HIGHS_OPTIONS)
    rows, rhs = [], []
    for r in system.rows:
        rows.append(np.concatenate([r.a_bar, r.a_hat]))
        rhs.append(r.b_lo)
    eye = np.eye(n)
    for j in range(n):  # t_j - x_j >= 0 and t_j + x_j >= 0
        rows.append(np.concatenate([eye[j], -eye[j]]))
        rhs.append(0.0)
        rows.append(np.concatenate([-eye[j], -eye[j]]))
        rhs.append(0.0)
    bounds = _bounds(system) + [(0.0, None)] * n
    res = linprog(np.concatenate([c, np.zeros(n)]), A_ub=np.array(rows), b_ub=np.array(rhs),
                  bounds=bounds, method="highs", options=HIGHS_OPTIONS)
    if res.status == 0:
        res.x = res.x[:n]
    return res


def project_l1(x: np.ndarray, system: LinearSystem, robust: bool):
    """P_F(x): a nearest point of F in the l1 norm, as an LP.

    min sum(u)  s.t.  y in F,  u >= y - x,  u >= x - y.
    The l1 norm matches the Lipschitz norm used for Prop. 6 in this package.
    Tie-breaking is whatever deterministic HiGHS returns (an A-DET assumption).
    """
    n = system.n
    # variables: y (n), u (n); when robust, also t (n) for |y|
    k = 3 * n if robust else 2 * n
    c = np.concatenate([np.zeros(n), np.ones(n), np.zeros(k - 2 * n)])
    a_rows, b_rows = [], []

    def row(y=None, u=None, t=None):
        r = np.zeros(k)
        if y is not None:
            r[:n] = y
        if u is not None:
            r[n:2 * n] = u
        if t is not None:
            r[2 * n:3 * n] = t
        return r

    eye = np.eye(n)
    for j in range(n):
        a_rows.append(row(y=eye[j], u=-eye[j]))   # y - u <= x
        b_rows.append(x[j])
        a_rows.append(row(y=-eye[j], u=-eye[j]))  # -y - u <= -x
        b_rows.append(-x[j])
    for r in system.rows:
        if robust:
            a_rows.append(row(y=r.a_bar, t=r.a_hat))
            b_rows.append(r.b_lo)
        else:
            a_rows.append(row(y=r.a_bar))
            b_rows.append(r.b_nom)
    if robust:
        for j in range(n):
            a_rows.append(row(y=eye[j], t=-eye[j]))
            b_rows.append(0.0)
            a_rows.append(row(y=-eye[j], t=-eye[j]))
            b_rows.append(0.0)
    bounds = _bounds(system) + [(0.0, None)] * (k - n)
    res = linprog(c, A_ub=np.array(a_rows), b_ub=np.array(b_rows), bounds=bounds,
                  method="highs", options=HIGHS_OPTIONS)
    if res.status != 0:
        return None
    return res.x[:n]


# -- budgeted uncertainty (Bertsimas and Sim) --------------------------------------


def budgeted_lhs(row: UncertainRow, x: np.ndarray, gamma: int) -> float:
    """a_bar x plus the gamma largest deviation terms a_hat_j |x_j| (integer budget)."""
    terms = np.sort(row.a_hat * np.abs(x))[::-1]
    return float(row.a_bar @ x + terms[:gamma].sum())


def budgeted_feasible(x: np.ndarray, system: LinearSystem, gamma: int,
                      tol: float = TOL) -> bool:
    """x feasible when at most `gamma` coefficients per row deviate from nominal.

    gamma = 0 is the nominal set with b_lo; gamma >= n is the box set F_box."""
    return system._in_bounds(x, tol) and all(
        budgeted_lhs(r, x, gamma) <= r.b_lo + tol for r in system.rows
    )


def solve_lp_budgeted(c: np.ndarray, system: LinearSystem, gamma: int):
    """min c^T x over the budgeted robust counterpart (Bertsimas and Sim, 2004).

    Variables: x (n), y >= |x| (n), z_i (m), p_ij (m * n). Per row i:
        a_bar_i x + gamma z_i + sum_j p_ij <= b_lo_i
        a_hat_ij y_j - z_i - p_ij <= 0          for every j
    An LP stays an LP. Returns the scipy result with `res.x` truncated to x.
    """
    n, m = system.n, len(system.rows)
    nv = 2 * n + m + m * n
    xs, ys, zs = 0, n, 2 * n
    ps = 2 * n + m
    rows, rhs = [], []
    for i, r in enumerate(system.rows):
        a = np.zeros(nv)
        a[xs:xs + n] = r.a_bar
        a[zs + i] = gamma
        a[ps + i * n:ps + (i + 1) * n] = 1.0
        rows.append(a)
        rhs.append(r.b_lo)
        for j in range(n):
            if r.a_hat[j] == 0:
                continue
            a = np.zeros(nv)
            a[ys + j] = r.a_hat[j]
            a[zs + i] = -1.0
            a[ps + i * n + j] = -1.0
            rows.append(a)
            rhs.append(0.0)
    for j in range(n):  # y_j >= |x_j|
        for sign in (1.0, -1.0):
            a = np.zeros(nv)
            a[xs + j] = sign
            a[ys + j] = -1.0
            rows.append(a)
            rhs.append(0.0)
    bounds = _bounds(system) + [(0.0, None)] * (n + m + m * n)
    cost = np.concatenate([c, np.zeros(nv - n)])
    res = linprog(cost, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds,
                  method="highs", options=HIGHS_OPTIONS)
    if res.status == 0:
        res.x = res.x[:n]
    return res
