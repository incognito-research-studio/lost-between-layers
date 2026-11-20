"""95% Wilson intervals for the rates quoted in Table 6, computed from stored results only.

Within one setting, runs are independent draws, so a per-setting interval is valid.
Gates within a study share draws (common random numbers), and so do settings except in
S4, so those differences are paired; this module reports per-setting intervals and does
not pool settings that share draws. The S1 sweep stores rates with runs = 200; each rate is
k/200, so k = round(rate * 200) is exact.

    uv run dsoi-intervals            # writes results/intervals.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    centre = (k + z * z / 2) / (n + z * z)
    half = z / (n + z * z) * math.sqrt(k * (n - k) / n + z * z / 4)
    return max(0.0, centre - half), min(1.0, centre + half)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _true(v: str) -> bool:
    return v == "True"


def collect(results: Path) -> list[dict]:
    out: list[dict] = []

    def add(study, quantity, setting, k, n, note=""):
        lo, hi = wilson(k, n)
        out.append({"study": study, "quantity": quantity, "setting": setting, "k": k, "n": n,
                    "rate": round(k / n, 4), "wilson_lo": round(lo, 4),
                    "wilson_hi": round(hi, 4), "note": note})

    # S1 sweep: one row per setting and gate; k = round(rate * runs).
    for r in _rows(results / "sweep_s1" / "cells.csv"):
        n = int(r["runs"])
        k = round(float(r["k_exec_violation_rate"]) * n)
        setting = (f"sd={r['point_sd']} hw={r['half_width']} bias={r['bias']} "
                   f"misread={r['scope_misread_prob']} gate={r['gate']}")
        add("S1", "K-exec violation", setting, k, n)

    # S3: violation after a miss; denominator = runs with a miss (the same runs per setting).
    s3 = _rows(results / "sim_s3" / "runs.csv")
    for p in sorted({r["p_stale_given_miss"] for r in s3}, key=float):
        missed = [r for r in s3 if r["gate"] == "recheck" and r["p_stale_given_miss"] == p
                  and _true(r["miss"])]
        add("S3", "recheck violation after a miss", f"p_stale_given_miss={p}",
            sum(not _true(r["k_exec"]) for r in missed), len(missed), "n = runs with a miss")

    # S4: violation among decided runs, per setting and policy (settings use different draws).
    s4 = _rows(results / "sim_s4" / "runs.csv")
    keys = sorted({(r["config"], r["portal_error_rate"], r["planner_sd"]) for r in s4})
    for config, per, sd in keys:
        decided = [r for r in s4 if (r["config"], r["portal_error_rate"], r["planner_sd"]) ==
                   (config, per, sd) and not _true(r["deferred"])]
        if decided:
            add("S4", "K-exec violation among decided runs",
                f"{config} portal_error={per} planner_sd={sd}",
                sum(not _true(r["k_exec"]) for r in decided), len(decided))

    # S8: violation over all runs (rejections count as compliant) and over released runs.
    s8 = _rows(results / "sim_s8" / "runs.csv")
    for sd, hw, gate in sorted({(r["point_sd"], r["half_width"], r["gate"]) for r in s8}):
        rs = [r for r in s8 if (r["point_sd"], r["half_width"], r["gate"]) == (sd, hw, gate)]
        released = [r for r in rs if not _true(r["rejected"])]
        setting = f"sd={sd} hw={hw} gate={gate}"
        add("S8", "K-exec violation, all runs", setting,
            sum(not _true(r["k_exec"]) for r in rs), len(rs), "rejections count as compliant")
        if released:
            add("S8", "K-exec violation, released runs", setting,
                sum(not _true(r["k_exec"]) for r in released), len(released))
        add("S8", "rejection", setting, len(rs) - len(released), len(rs))

    # Execution: violations on TOCTOU runs, per setting (settings share the same drops).
    ex = _rows(results / "sim_exec" / "runs.csv")
    for fault, toc, gate in sorted({(r["p_exec_fault"], r["p_toctou"], r["gate"]) for r in ex}):
        drops = [r for r in ex if (r["p_exec_fault"], r["p_toctou"], r["gate"]) ==
                 (fault, toc, gate) and _true(r["toctou"])]
        if drops:
            add("execution", "violation on a TOCTOU run",
                f"p_exec_fault={fault} p_toctou={toc} gate={gate}",
                sum(not _true(r["k_exec"]) for r in drops), len(drops), "any cause")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsoi-intervals", description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results") / "intervals.csv")
    args = ap.parse_args(argv)
    rows = collect(args.results)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} intervals to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
