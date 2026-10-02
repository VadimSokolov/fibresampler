"""Construct P3: a thin-tilted, moderate-mean fibre where reflection should help.

Prediction 3/4 need a fibre polytope with high axis defect kappa = L/a (diameter /
longest axis-aligned chord) but no loading bottleneck (all coordinate means can be
>= 1).  Even with a 0/1 configuration matrix, the free-frame walls are the rows of
A1^{-1}A2, which for a totally unimodular A carry +-1 entries and hence can be
*difference* walls u - v = const -- a thin diagonal band, oblique to the PLB axes.
Choosing the A1 block is exactly the manuscript's "rotating the route basis".

We search small 0/1 matrices and A1 choices for the largest kappa at an enumerable
fibre, then confirm by exact SLEM that the reflective sampler beats single-ray there.

Run:  python3 experiments/build_p3.py
"""

from __future__ import annotations

import os
import sys
from itertools import combinations, product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.fibre import enumerate_fibre, build_plb, FibreProblem
from fibresampler.spectral import slem
from fibresampler.reflective import reflective_transition_matrix


def kappa_of(free):
    """Axis defect L/a from an array of free coordinates (N x m)."""
    N, m = free.shape
    if N < 2:
        return 0.0, 0.0, 1
    d = free[:, None, :] - free[None, :, :]
    L = float(np.sqrt((d ** 2).sum(2)).max())
    a = 1
    for j in range(m):
        others = [c for c in range(m) if c != j]
        groups = {}
        for row in free:
            groups.setdefault(tuple(row[others]), []).append(int(row[j]))
        for vals in groups.values():
            a = max(a, max(vals) - min(vals))
    return L / a, L, a


def search(n, r, ncols_try=None, fibre_lo=30, fibre_hi=4000, y_scales=(1, 2, 3)):
    """Yield (kappa, A, y, cols1) for high-defect enumerable fibres."""
    best = []
    cols = list(range(r))
    # candidate 0/1 columns: nonzero patterns over n links
    patterns = [p for p in product([0, 1], repeat=n) if any(p)]
    seen = set()
    for combo in combinations(range(len(patterns)), r):
        A = np.array([patterns[c] for c in combo], dtype=np.int64).T  # n x r
        if np.linalg.matrix_rank(A) < n:
            continue
        if (A.sum(0) == 0).any() or (A.sum(1) == 0).any():
            continue
        key = A.tobytes()
        if key in seen:
            continue
        seen.add(key)
        for sc in y_scales:
            # a feasible y: pick an interior x0 (all ones times scale) and set y=A x0
            x0 = np.full(r, sc, dtype=np.int64)
            y = A @ x0
            states = enumerate_fibre(A, y)
            if not (fibre_lo <= len(states) <= fibre_hi):
                continue
            free_all = None
            for c1 in combinations(range(r), n):
                try:
                    U, info = build_plb(A, c1)
                except Exception:
                    continue
                if not info["integral"]:
                    continue
                cols2 = info["cols2"]
                free = states[:, list(cols2)]
                k, L, a = kappa_of(free)
                if k >= 2.0:
                    best.append((k, A.copy(), y.copy(), tuple(c1), len(states), L, a))
    best.sort(key=lambda t: -t[0])
    return best


def evaluate(A, y, cols1, label):
    prob = FibreProblem(A, y, cols1=cols1, name=label)
    # SR (B=0 Metropolis) vs reflective at a couple of bounce budgets, uniform target
    g = {}
    for bm in (0, 2, 4):
        Q, pi = reflective_transition_matrix(prob, "uniform", None, bmax=bm)
        g[bm] = slem(Q, pi)[1]
    return prob, g


def main():
    print("Searching for thin-tilted fibres (kappa >= 2)...")
    found = []
    for (n, r) in [(2, 4), (3, 5), (2, 5), (3, 6)]:
        res = search(n, r)
        if res:
            print(f"  n={n},r={r}: {len(res)} candidates, top kappa={res[0][0]:.2f} "
                  f"(fibre {res[0][4]}, L={res[0][5]:.2f}, a={res[0][6]})")
            found.extend(res[:5])
    if not found:
        print("no tilted fibre found in the searched families")
        return
    found.sort(key=lambda t: -t[0])
    print("\nTop candidates and reflection benefit (exact SLEM gap, uniform target):")
    print(f"  {'kappa':>6} {'fibre':>6} {'gapB0(SR)':>10} {'gapB2':>9} {'gapB4':>9} {'B4/B0':>7}")
    shown = 0
    for k, A, y, c1, nst, L, a in found:
        prob, g = evaluate(A, y, c1, f"P3 kappa={k:.1f}")
        ratio = g[4] / g[0] if g[0] > 0 else float("nan")
        flag = "  <-- reflection helps" if ratio > 1.05 else ""
        print(f"  {k:6.2f} {nst:6d} {g[0]:10.5f} {g[2]:9.5f} {g[4]:9.5f} {ratio:7.2f}{flag}")
        if shown == 0 or ratio > 1.05:
            print(f"        A={A.tolist()} y={y.tolist()} cols1={c1}")
        shown += 1
        if shown >= 8:
            break


if __name__ == "__main__":
    main()
