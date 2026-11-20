"""dsoi-plot: figures from a sweep (install with the `plots` extra).

Encoding: the four ladder rungs are ordinal (nominal -> chance 0.2 -> chance
0.05 -> box, increasingly conservative), so they share one blue ramp from light
to dark (validated as an ordinal ramp). The recheck is a different lever and takes
orange. Markers, line styles and direct labels carry identity in grayscale print.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
RUNGS = {  # label, color, marker, linestyle
    "nominal": ("nominal", "#6da7ec", "o", "-"),
    "chance:0.2": ("chance, ε=0.2", "#2a78d6", "^", "-."),
    "chance:0.05": ("chance, ε=0.05", "#1c5cab", "v", ":"),
    "box_robust": ("box-robust", "#0d366b", "s", "-"),
    "recheck": ("system-of-record recheck", "#eb6834", "D", "--"),
}
SLICE = {"half_width": "0.05", "bias": "0.0", "scope_misread_prob": "0.0"}
# 13 Nov 2026 (figure audit F2): sideways shift in sigma points per gate in panel (a), so that
# the error bars of neighbouring gates stay apart; stated in the caption.
DODGE = {"nominal": -0.16, "chance:0.2": -0.08, "chance:0.05": 0.0, "box_robust": 0.08,
         "recheck": 0.16}
# Property name in sans serif, as the text sets it (figure audit F3).
KEXEC = r"$\mathsf{K}$-$\mathsf{exec}$"
XMAX_A = 8.3


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2,
        "ytick.color": TEXT_2, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embedded TrueType fonts
        "savefig.facecolor": "white",
    })


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("point_sd", "k_exec_violation_rate", "unsafe_miscoverage_rate",
                  "mean_excess_shortfall"):
            r[k] = float(r[k])
    return rows


def in_slice(r: dict) -> bool:
    return all(r[k] == v for k, v in SLICE.items())


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson interval for k of n, the method of Table 11 (added 13 Nov 2026)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def _count(r: dict) -> tuple[int, int]:
    n = int(r["runs"])
    k = round(r["k_exec_violation_rate"] * n)
    assert abs(k / n - r["k_exec_violation_rate"]) < 1e-9, (r["cell"], r["gate"])
    return k, n


def gate_ladder_figure(rows: list[dict], out: Path, sigma_tradeoff: float = 0.03) -> list[Path]:
    _style()
    # Upright sans for \mathsf (the property name K-exec), as the text sets it.
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.sf": "Arial",
                         "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic"})
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.16, 2.35),
                                           gridspec_kw={"wspace": 0.5})
    # (a) violation rate vs reading error
    # 13 Nov 2026: 95% Wilson intervals over the runs of each setting at the measured
    # sigma values, and the violation rates the chance-constrained gates promise.
    for eps, gate in ((0.2, "chance:0.2"), (0.05, "chance:0.05")):
        ax_a.axhline(100 * eps, color=RUNGS[gate][1], linewidth=0.8, linestyle=(0, (4, 3)),
                     zorder=1)
        # Label outside the right edge, where no data are drawn.
        ax_a.annotate(f"ε={eps:g}", (XMAX_A, 100 * eps), xytext=(3, 0), textcoords="offset points",
                      ha="left", va="center", fontsize=6, color=RUNGS[gate][1],
                      annotation_clip=False)
    handles = []
    for gate, (label, color, marker, ls) in RUNGS.items():
        sel = sorted((r for r in rows if r["gate"] == gate and in_slice(r)),
                     key=lambda r: r["point_sd"])
        xs = [100 * r["point_sd"] + DODGE[gate] for r in sel]
        ys = [100 * r["k_exec_violation_rate"] for r in sel]
        cis = [wilson(*_count(r)) for r in sel]
        lo = [y - 100 * a for y, (a, _) in zip(ys, cis)]
        hi = [100 * b - y for y, (_, b) in zip(ys, cis)]
        ax_a.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=0.7,
                      capsize=1.5, capthick=0.7, zorder=2, clip_on=False)
        (h,) = ax_a.plot(xs, ys, color=color, marker=marker, linestyle=ls, linewidth=1.2,
                         markersize=3.6, markeredgecolor="white", markeredgewidth=0.5,
                         clip_on=False, zorder=3, label=label)
        handles.append(h)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               frameon=False, fontsize=7, handlelength=2.6, columnspacing=1.4)
    ax_a.set_xlabel("interpreter reading error σ (points)")
    ax_a.set_ylabel(f"{KEXEC} violations (%)")
    ax_a.set_xlim(-0.3, XMAX_A)
    ax_a.set_xticks([0, 2, 4, 6, 8])
    ax_a.set_ylim(0, 60)
    ax_a.set_title("(a) violations vs reading error", loc="left", color=TEXT)

    # (b) cost-compliance trade-off at one reading error
    for gate, (label, color, marker, _ls) in RUNGS.items():
        (r,) = [r for r in rows if r["gate"] == gate and in_slice(r)
                and abs(r["point_sd"] - sigma_tradeoff) < 1e-12]
        x, y = r["mean_excess_shortfall"], 100 * r["k_exec_violation_rate"]
        ax_b.plot([x], [y], marker=marker, color=color, markersize=5.5, linestyle="none",
                  markeredgecolor="white", markeredgewidth=0.6, clip_on=False, zorder=3)
        # box-robust label below its point, clear of the chance label (figure audit F4)
        off = (6, 0) if gate == "box_robust" else (5, 3)
        ax_b.annotate(label, (x, y), xytext=off, textcoords="offset points",
                      va="center" if gate == "box_robust" else "baseline",
                      fontsize=6.5, color=TEXT)
    ax_b.set_xlabel("mean shortfall above true optimum (units)")
    ax_b.set_ylabel(f"{KEXEC} violations (%)")
    ax_b.set_xlim(-20, 200)
    ax_b.set_ylim(0, 60)
    ax_b.set_title(f"(b) cost vs compliance at σ = {100 * sigma_tradeoff:g}", loc="left",
                   color=TEXT)

    # (c) box-robust violations equal unsafe-side miscoverage, every cell
    box = [r for r in rows if r["gate"] == "box_robust"]
    xs = [100 * r["unsafe_miscoverage_rate"] for r in box]
    ys = [100 * r["k_exec_violation_rate"] for r in box]
    top = max(max(xs), max(ys)) * 1.05 or 1
    ax_c.plot([0, top], [0, top], color=TEXT_2, linewidth=0.7, linestyle="--", zorder=1)
    ax_c.plot(xs, ys, marker="s", color=RUNGS["box_robust"][1], linestyle="none",
              markersize=3.4, markeredgecolor="white", markeredgewidth=0.4, zorder=3,
              clip_on=False)
    ax_c.annotate("y = x", (top * 0.72, top * 0.52), fontsize=6.5, color=TEXT_2)
    ax_c.set_xlim(0, top)
    ax_c.set_ylim(0, top)
    ax_c.set_xlabel("truth below reported interval (%)")
    ax_c.set_ylabel("box-robust violations (%)")
    ax_c.set_title(f"(c) S1 lower-tail misses ({len(box)} settings)", loc="left",
                   color=TEXT)

    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "fig_gate_ladder.pdf", out / "fig_gate_ladder.png"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsoi-plot", description=__doc__)
    ap.add_argument("cells", type=Path, help="cells.csv from dsoi-sweep")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or args.cells.parent / "figures"
    for p in gate_ladder_figure(load(args.cells), out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
