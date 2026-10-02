"""Pedagogical two-panel figure of a 2x3 contingency-table fibre in 2D.

Requested by M. Hazelton: an ongoing toy example projectable onto two
dimensions so the geometric issues are visible to readers new to discrete
fibre sampling.  Both fibres are ENUMERATED from the real FibreProblem, and the
plane is the free-coordinate (x12, x13) projection; every other cell of the
2x3 table is then fixed by the margins.

  Panel (a): the running 2x3 table, balanced margins rows (4,4), cols (3,3,2),
             fibre size 10.  Points coloured by the bottleneck cell x11 = 4 - x12
             - x13; single-ray hit-and-run moves are axis-aligned.  As the
             Poisson mean of x11 vanishes, mass concentrates on the tilted
             x11 = 0 edge and the moves that raise x11 are proposed rarely
             (the loading obstruction).

  Panel (b): unbalanced margins rows (6,6), cols (1,5,6), fibre size 12.  The
             fibre is a thin, 45-degree-tilted strip (large axis defect).  A
             single-ray move steps one lattice unit; a reflective proposal
             bounces off the tilted walls and traverses the whole fibre.

Run:  python3 experiments/fig_toy_fibre.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

from fibresampler.fibre import FibreProblem
from fibresampler.problems import A_TABLE23, TABLE23_COLS1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def table_with_margins(y):
    """2x3 table FibreProblem with A_TABLE23 and margin vector y=(row1,c1,c2,c3)."""
    return FibreProblem(A_TABLE23, np.asarray(y, dtype=np.int64),
                        cols1=TABLE23_COLS1, name=f"2x3 y={tuple(y)}")


def free_coords(prob):
    """(x12, x13) = columns 1,2 of the state vector (the free A2 block)."""
    S = prob.states
    return S[:, 1], S[:, 2], S[:, 0]     # x12, x13, x11(=bottleneck)


def hull_polygon(xs, ys):
    """Convex hull of the fibre points, as an ordered vertex list (2D)."""
    pts = np.column_stack([xs, ys]).astype(float)
    c = pts.mean(0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    P = pts[order]
    # keep only true extreme points (drop interior/collinear) via a light hull pass
    from math import inf
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    uniq = sorted(set(map(tuple, pts.tolist())))
    uniq = np.array(uniq)
    lower = []
    for p in uniq:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in uniq[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1])


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.4))

    # ---------------- Panel (a): balanced running example ----------------
    pa = table_with_margins((4, 3, 3, 2))          # rows (4,4), cols (3,3,2)
    x12, x13, x11 = free_coords(pa)
    hull = hull_polygon(x12, x13)
    axA.add_patch(MplPolygon(hull, closed=True, facecolor="#eef3f8",
                             edgecolor="#5b7fa6", lw=1.4, zorder=0))
    sc = axA.scatter(x12, x13, c=x11, cmap="viridis", s=170, zorder=3,
                     edgecolor="black", linewidth=0.6)
    for a, b, v in zip(x12, x13, x11):
        axA.text(a, b, str(int(v)), color="white", ha="center", va="center",
                 fontsize=8, fontweight="bold", zorder=4)
    # single-ray moves through the interior point (2,1): axis-aligned rays
    axA.annotate("", xy=(3.15, 1), xytext=(-0.15, 1),
                 arrowprops=dict(arrowstyle="<->", color="#b4341f", lw=1.6), zorder=2)
    axA.annotate("", xy=(2, 2.15), xytext=(2, -0.15),
                 arrowprops=dict(arrowstyle="<->", color="#b4341f", lw=1.6), zorder=2)
    axA.text(3.05, 1.28, "single-ray\nmoves", color="#b4341f", fontsize=8.5, ha="right")
    axA.plot([3, 2], [1, 2], color="#7a3b8f", lw=3.2, zorder=1, alpha=0.9)
    axA.text(2.75, 1.78, r"$x_{11}\!=\!0$ edge", color="#7a3b8f", fontsize=8.5,
             rotation=-45, ha="center", va="bottom")
    axA.text(0.55, 0.30, r"$x_{11}$ large", color="#2a4d2a", fontsize=8.5)
    axA.set_title(r"(a) balanced margins $(4,4)\,|\,(3,3,2)$: the running fibre", fontsize=10)
    cb = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.03)
    cb.set_label(r"bottleneck cell $x_{11}$", fontsize=9)

    # ---------------- Panel (b): unbalanced -> thin tilted fibre ----------------
    pb = table_with_margins((6, 1, 5, 6))          # rows (6,6), cols (1,5,6)
    x12b, x13b, x11b = free_coords(pb)
    hullb = hull_polygon(x12b, x13b)
    axB.add_patch(MplPolygon(hullb, closed=True, facecolor="#f6eef2",
                             edgecolor="#a6688a", lw=1.4, zorder=0))
    axB.scatter(x12b, x13b, s=95, color="#3b3b3b", zorder=3)
    # single-ray move (one lattice step)
    axB.annotate("", xy=(1, 5), xytext=(0, 5),
                 arrowprops=dict(arrowstyle="->", color="#b4341f", lw=2.4), zorder=4)
    axB.text(0.15, 5.35, "single-ray step", color="#b4341f", fontsize=8.5)
    # reflective proposal: a REALISED trajectory produced by the implemented sampler
    # (ReflectiveSampler.bounce_to_wall), not a schematic; bounce vertices are marked.
    from fibresampler.reflective import ReflectiveSampler
    Sb = ReflectiveSampler(pb, bmax=8)
    start_b = np.array([1, 0, 5, 0, 5, 1], dtype=np.int64)     # (x12,x13)=(0,5)
    vB, _, walls = Sb.bounce_to_wall(start_b.copy(), Sb.dirs[0].copy(), 8)  # +e_{x12}
    refl = ([(int(start_b[1]), int(start_b[2]))]
            + [(int(w[1]), int(w[2])) for w in walls]
            + [(int(vB[1]), int(vB[2]))])
    if len(refl) > 1 and refl[-1] == refl[-2]:                 # drop zero-length final leg
        refl = refl[:-1]
    rx, ry = zip(*refl)
    axB.plot(rx, ry, color="#1f6f4a", lw=2.6, zorder=2, solid_capstyle="round")
    axB.scatter(rx, ry, s=34, color="#1f6f4a", zorder=5)       # bounce points marked
    axB.scatter([rx[0]], [ry[0]], s=72, color="#1f6f4a", zorder=6)
    axB.annotate("", xy=(rx[-1], ry[-1]), xytext=(rx[0], ry[0]),
                 arrowprops=dict(arrowstyle="->", color="#9aa0a6", lw=1.1,
                                 ls=(0, (4, 3))), zorder=1)
    axB.text(3.35, 2.7, "net displacement", color="#9aa0a6", fontsize=7.5,
             ha="left", rotation=-38)
    axB.text(2.9, 4.2, "reflective proposal\n(realised legs,\nbounce points marked)",
             color="#1f6f4a", fontsize=8.5, ha="left")
    axB.set_title(r"(b) unbalanced margins $(6,6)\,|\,(1,5,6)$: thin tilted fibre", fontsize=10)

    for ax in (axA, axB):
        ax.set_xlabel(r"$x_{12}$", fontsize=10)
        ax.set_ylabel(r"$x_{13}$", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", alpha=0.4)
        ax.margins(0.12)

    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_toy_fibre.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=130)
    print("fibre sizes:", pa.fibre_size, pb.fibre_size)
    print("wrote", out)


if __name__ == "__main__":
    main()
