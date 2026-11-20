"""Version registry: pins nu; promotion requires passing Inv on the scenario suite (R5).

Passing a finite suite is evidence that Inv holds, not proof (Sec. IV).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dsoi.context import VersionVector

# Inv = {G1, R2, C1, K-prop, audit completeness}
INV = ("G1", "R2", "C1", "K-prop", "audit")


@dataclass(frozen=True)
class Transition:
    old: dict
    new: dict
    promoted: bool
    results: dict


@dataclass
class VersionRegistry:
    current: VersionVector
    history: list[Transition] = field(default_factory=list)

    def promote(self, candidate: VersionVector,
                suite: Callable[[VersionVector], dict[str, bool]]) -> bool:
        """Run the Inv regression suite on `candidate`; activate it only if every check passes.

        A candidate equal to the active vector is refused: R5 attributes every change in
        behaviour to a version bump."""
        if candidate.as_dict() == self.current.as_dict():
            raise ValueError("promotion must change the version vector")
        results = suite(candidate)
        missing = [p for p in INV if p not in results]
        if missing:
            raise ValueError(f"suite did not report {missing}")
        ok = all(results.values())
        self.history.append(Transition(self.current.as_dict(), candidate.as_dict(), ok, results))
        if ok:
            self.current = candidate
        return ok
