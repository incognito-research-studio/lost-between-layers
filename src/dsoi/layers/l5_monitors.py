"""L5 trace monitors (Table 2, L5 (ii)): rules over the decision trace.

Every action a gate releases is checked against each monitor before L2 admits it and
anything executes. A violated rule holds the action back from execution; the record keeps
the release, names the rule and gives the reason (runtime verification over the log, in
the spirit of AgentLTL's temporal rules). The release is not changed, because a rule over
the trace reads inputs outside the logged tuple and would otherwise break C2. Monitors
read only the audit log and the proposed release, and their labels are part of nu.

  CumulativeCap           a rate limit: executed quantity of one variable over any
                          window of given length stays within a cap
  SameInputsSameRelease   C2 at run time: a release that differs from an earlier one
                          with the same logged tuple (hash kappa_D, theta, C-id, J-id,
                          z, nu) has no logged explanation and is held back; z records
                          execution-time observations, empty without a re-read
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dsoi.layers.l6_audit import AuditLog


class Monitor(Protocol):
    @property
    def label(self) -> str: ...

    def check(self, released: Mapping[str, float], t: float, logged: tuple,
              names: tuple[str, ...], trace: AuditLog) -> str | None: ...


@dataclass(frozen=True)
class CumulativeCap:
    variable: str
    cap: float
    window: float
    tol: float = 1e-9

    @property
    def label(self) -> str:
        return f"cap({self.variable}<={self.cap:g}/{self.window:g})"

    def check(self, released, t, logged, names, trace):
        if self.variable not in names:
            return None
        j = names.index(self.variable)
        done = sum(rec.x_exec[j] for rec in trace.decisions()
                   if rec.exec_ack is not None and rec.x_exec is not None
                   and t - self.window < rec.t <= t)
        proposed = released.get(self.variable, 0.0)
        if done + proposed > self.cap + self.tol:
            return (f"{self.variable}: {done:g} executed in the last {self.window:g} plus "
                    f"{proposed:g} exceeds {self.cap:g}")
        return None


@dataclass(frozen=True)
class SameInputsSameRelease:
    tol: float = 1e-9

    @property
    def label(self) -> str:
        return "c2"

    def check(self, released, t, logged, names, trace):
        x = [released.get(n, 0.0) for n in names]
        for rec in trace.decisions():
            if rec.logged_tuple() == logged and rec.x_released is not None and not np.allclose(
                    rec.x_released, x, atol=self.tol, rtol=0):
                return (f"record {rec.id} released another action on the same logged "
                        f"inputs; the change is not attributable to nu")
        return None


def monitors_label(monitors: tuple[Monitor, ...]) -> str:
    """The monitors' entry in nu."""
    return ";".join(m.label for m in monitors) or "none"
