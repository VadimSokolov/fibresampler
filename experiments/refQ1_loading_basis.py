"""Does a change of lattice basis touch the LOADING obstruction?

The companion paper proves that single-ray hit-and-run in the partition lattice basis (PLB)
has gap Theta(mu) as a bottleneck coordinate mean mu -> 0 (Prediction 1, Theorem on the
loading obstruction).  A reduced basis changes which rays exist, and a heat-bath ray can jump
over low-weight points, so it is not obvious a priori that the obstruction survives a change
of basis.  This script measures it, exactly:

  fibre P1 (N = 55, m = 4) with the Poisson target whose fifth cell mean is mu (Prediction 1),
  mu in {1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 1e-4};
  bases: PLB (identity), the pilot-free "exact moments" selection (recommend_basis on the
  target's own free-coordinate law), the best of many random unimodular bases, and the
  union kernel PLB + selected.

and reports the log-log slope of each gap against mu over mu <= 0.03.  A slope of 1 for every
basis means the obstruction is a property of the fibre and the target, not of the lattice
basis (conservation of difficulty); a slope below 1 for some basis would be an escape.

Output: results/refQ1_loading_basis.txt.  Deterministic (seeded).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.basis import (RayView, free_points, random_unimodular, recommend_basis,   # noqa: E402
                                union_moves, with_basis)
from fibresampler.problems import P1, p1_poisson_theta                                       # noqa: E402
from fibresampler.spectral import log_weights, normalised_pi, slem, sr_transition_matrix      # noqa: E402

OUT = os.path.join(REPO, "results", "refQ1_loading_basis.txt")
SEED = 20261002
MUS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 1e-4]
N_RANDOM = 1500


def gap(pv, lw, pi):
    return slem(sr_transition_matrix(pv, lw), pi)[1]


def slope(mus, gaps, mu_max=0.03):
    keep = [(m, g) for m, g in zip(mus, gaps) if m <= mu_max]
    x = np.log([m for m, _ in keep])
    y = np.log([g for _, g in keep])
    return float(np.polyfit(x, y, 1)[0])


def main():
    rng = np.random.default_rng(SEED)
    prob = P1()
    pv = RayView(prob.states, prob.U, index=prob.index, plb_info=prob.plb_info, name="P1")
    fr = free_points(prob)
    m = prob.n_moves
    cands = [random_unimodular(m, rng) for _ in range(N_RANDOM)]
    views = [with_basis(pv, V) for V in cands]

    rows = {"PLB": [], "selected": [], "best random": [], "union(PLB,sel)": [], "worst random": []}
    chosen = []
    for mu in MUS:
        lw = log_weights(prob, "poisson", p1_poisson_theta(mu))
        pi = normalised_pi(lw)
        rows["PLB"].append(gap(pv, lw, pi))
        V, info = recommend_basis(fr, pi)
        chosen.append(info["chosen"])
        rows["selected"].append(gap(with_basis(pv, V), lw, pi))
        rows["union(PLB,sel)"].append(gap(union_moves(pv, V), lw, pi))
        g_rand = [gap(v, lw, pi) for v in views]
        rows["best random"].append(max(g_rand))
        rows["worst random"].append(min(g_rand))

    W = 118
    out = ["=" * W,
           "Loading obstruction versus lattice basis: exact single-ray gaps on P1 (N=55), Poisson target, fifth-cell mean mu",
           f"{N_RANDOM} random unimodular 4x4 bases per mu (seeded); 'selected' = recommend_basis on the target's exact free-coordinate law",
           "=" * W,
           f"  {'mu':>8} " + " ".join(f"{k:>16}" for k in rows) + "   selector"]
    for i, mu in enumerate(MUS):
        out.append(f"  {mu:>8.0e} " + " ".join(f"{rows[k][i]:>16.6f}" for k in rows) + f"   {chosen[i]}")
    out.append("")
    out.append("log-log slope of gap vs mu over mu <= 0.03 (1 = the Theta(mu) loading law):")
    for k in rows:
        out.append(f"  {k:>16}: {slope(MUS, rows[k]):.3f}")
    out.append("")
    ratio = [rows["best random"][i] / rows["PLB"][i] for i in range(len(MUS))]
    out.append("best random basis / PLB gap ratio by mu: " + ", ".join(f"{r:.2f}" for r in ratio))
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
