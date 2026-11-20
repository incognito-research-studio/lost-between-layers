"""Sandbox stand-in for dsoi.layers.l5_monitors (fault trial, round 2).

The real module, which holds the L5 trace-monitor rules, is withheld from the
fault-generation sandbox. pipeline.py needs only these two names. With no monitors
configured, which is how every sandbox episode runs, the pipeline behaves exactly as with
the real module: no rule is checked and nu's monitor entry is "none".
"""

from __future__ import annotations

from typing import Protocol


class Monitor(Protocol):
    @property
    def label(self) -> str: ...

    def check(self, released, t, logged, names, trace) -> str | None: ...


def monitors_label(monitors) -> str:
    return ";".join(m.label for m in monitors) or "none"
