"""Is the loading obstruction a property of the lattice basis?  Exact test on every multi-lobe fibre.

The companion paper proves that the single-ray gap in the partition lattice basis (PLB) is
Theta(mu^{k*}) as a bottleneck cell mean mu -> 0, under a hypothesis on the MOVE SET (the
dominant stratum x_j* = 0 splits into lobes under the moves that hold the bottleneck cell
fixed).  Here the move set changes: a unimodular recombination V of the PLB columns, chosen
(a) structurally, by Hermite reduction of the bottleneck row of U so that m - 1 moves hold
the bottleneck cell fixed ("hold-fixed"), or (b) statistically, by LLL in the Mahalanobis
metric of the target ("selected").  Same number m of moves, same lattice.

For each fibre: exact single-ray gaps over mu in {1, 0.3, ..., 1e-6}, log-log slope over
mu <= 0.03, and the number of lobes (components of the stratum x_j* = 0 under the moves that
hold j* fixed) for the PLB and for the recombined move set.  Output:
results/refQ1_loading_generality.txt.  Deterministic.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.basis import (RayView, free_points, hold_fixed_basis, int_det,          # noqa: E402
                                is_irreducible, recommend_basis, union_moves, with_basis)
from fibresampler.fibre import FibreProblem                                                # noqa: E402
from fibresampler.problems import A_TABLE23, P1, TABLE23_COLS1                             # noqa: E402
from fibresampler.spectral import normalised_pi, slem, sr_transition_matrix                # noqa: E402

OUT = os.path.join(REPO, "results", "refQ1_loading_generality.txt")
MUS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 1e-4, 1e-6]


def logw(states, jstar, mu):
    X = np.asarray(states, float)
    theta = np.ones(X.shape[1])
    theta[jstar] = mu
    return (X * np.log(theta) - gammaln(X + 1.0)).sum(1)


def gap(view, lw, pi):
    return slem(sr_transition_matrix(view, lw), pi)[1]


def slope(gs, mu_max=0.03):
    """Log-log slope of the gap in mu.  Gaps below 1e-14 are numerically zero (a reducible frame), so a slope fitted
    to them would be noise; fewer than three usable points gives nan."""
    keep = [(m, g) for m, g in zip(MUS, gs) if m <= mu_max and g > 1e-14]
    if len(keep) < 3:
        return float("nan")
    return float(np.polyfit(np.log([m for m, _ in keep]), np.log([g for _, g in keep]), 1)[0])


def n_lobes(states, U_cols, jstar):
    """Components of D = {x_jstar = 0} under moves (columns of U_cols, either sign) that hold
    x_jstar fixed, restricted to the fibre."""
    D = [tuple(int(v) for v in s) for s in states if s[jstar] == 0]
    Dset = set(D)
    parent = {d: d for d in D}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for d in D:
        for k in range(U_cols.shape[1]):
            u = U_cols[:, k]
            if u[jstar] != 0:
                continue
            for sgn in (1, -1):
                e = tuple(int(v) for v in np.array(d) + sgn * u)
                if e in Dset:
                    parent[find(d)] = find(e)
    return len({find(d) for d in D})


def fibres():
    out = [("P1", P1(), 4),
           ("table(4,4|3,3,2)", FibreProblem(A_TABLE23, np.array([4, 3, 3, 2]), cols1=TABLE23_COLS1), 0)]
    canon = (np.array([[1, 1, 1, 1], [1, 2, 0, 1]], int), np.array([4, 5], int), (0, 1), 0)
    seen = {(canon[0].tobytes(), canon[1].tobytes())}
    out.append(("k*=2 canonical 2x4", FibreProblem(*canon[:2], cols1=canon[2], name="kstar2"), canon[3]))
    hits = []
    for fn in sorted(glob.glob(os.path.join(REPO, "results", "kstar2_hits_*.json"))):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        for h in d.get("hits", []):
            key = (np.array(h["A"], int).tobytes(), np.array(h["y"], int).tobytes())
            if key in seen:
                continue
            seen.add(key)
            hits.append((int(h["fibre"]), h))
    hits.sort(key=lambda t: t[0])
    for k, (fib, h) in enumerate(hits[:5]):
        A, y = np.array(h["A"], int), np.array(h["y"], int)
        try:
            out.append((f"k*=2 hit {k + 1} (N={fib})", FibreProblem(A, y, cols1=tuple(h["cols1"])), int(h["jstar"])))
        except Exception:
            continue
    return out


def main():
    W = 124
    out = ["=" * W,
           "Loading obstruction versus the lattice basis: exact single-ray gaps, product-Poisson target (cell j* has mean mu, others 1)",
           "PLB = the paper's partition lattice basis; hold-fixed = Hermite recombination of the bottleneck row (identity of j* only);",
           "selected = Mahalanobis LLL / surrogate climb on the target's exact free-coordinate law; union = PLB and hold-fixed moves together",
           "=" * W]
    summary = []
    for name, prob, jstar in fibres():
        pv = RayView(prob.states, prob.U, index=prob.index, plb_info=prob.plb_info, name=name)
        fr = free_points(prob)
        m = prob.n_moves
        V_hf, n_slice = hold_fixed_basis(prob.U, [jstar])
        assert abs(int_det(V_hf)) == 1
        hf = with_basis(pv, V_hf)
        un = union_moves(pv, V_hf)
        U_hf = prob.U @ V_hf
        lob0 = n_lobes(prob.states, prob.U, jstar)
        lob1 = n_lobes(prob.states, U_hf, jstar)
        out.append("")
        out.append(f"{name}: N={prob.fibre_size}, m={m}, bottleneck cell {jstar}; hold-fixed V columns "
                   f"{[tuple(int(v) for v in V_hf[:, k]) for k in range(m)]}, {n_slice} slice move(s); "
                   f"irreducible: PLB {is_irreducible(pv)}, hold-fixed {is_irreducible(hf)}; lobes: PLB {lob0}, hold-fixed {lob1}")
        rows = {"PLB": [], "hold-fixed": [], "selected": [], "union": []}
        for mu in MUS:
            lw = logw(prob.states, jstar, mu)
            pi = normalised_pi(lw)
            rows["PLB"].append(gap(pv, lw, pi))
            rows["hold-fixed"].append(gap(hf, lw, pi) if is_irreducible(hf) else 0.0)
            V_sel, _ = recommend_basis(fr, pi)
            rows["selected"].append(gap(with_basis(pv, V_sel), lw, pi))
            rows["union"].append(gap(un, lw, pi))
        out.append(f"  {'mu':>8} " + " ".join(f"{k:>12}" for k in rows))
        for i, mu in enumerate(MUS):
            out.append(f"  {mu:>8.0e} " + " ".join(f"{rows[k][i]:>12.6f}" for k in rows))
        sl = {k: slope(v) for k, v in rows.items()}
        out.append("  slope (mu <= 0.03): " + ", ".join(f"{k} {sl[k]:.3f}" for k in rows))
        summary.append((name, lob0, lob1, sl, rows["PLB"][-1], rows["hold-fixed"][-1], rows["selected"][-1]))
    out.append("")
    out.append("SUMMARY (slope of log gap vs log mu; gap at mu = 1e-6)")
    out.append(f"  {'fibre':>26} {'lobes PLB':>9} {'lobes hf':>8} | {'slope PLB':>9} {'slope hf':>9} {'slope sel':>9} | "
               f"{'gap PLB':>10} {'gap hf':>10} {'gap sel':>10}")
    for name, l0, l1, sl, g0, g1, g2 in summary:
        out.append(f"  {name:>26} {l0:>9d} {l1:>8d} | {sl['PLB']:>9.3f} {sl['hold-fixed']:>9.3f} {sl['selected']:>9.3f} | "
                   f"{g0:>10.2e} {g1:>10.2e} {g2:>10.2e}")
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
