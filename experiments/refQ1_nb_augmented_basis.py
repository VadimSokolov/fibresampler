"""Does a hold-fixed change of basis dissolve the loading obstruction of the GAMMA-AUGMENTED chain?

The companion's Theorem (conserved loading obstruction) proves gap(K_R) = Theta(R mu) for the gamma-augmented
negative-binomial chain under the multi-lobe hypothesis on the PLB moves, and its converse (Proposition converse)
gives a floor when the stratum {x_{j*} = 0} is connected under the moves that hold j* fixed.  Here the SAME P1
fibre, target (NB, alpha = 1.9, bottleneck mean mu, others 1) and heat-bath rays are run in the PLB frame and in
the hold-fixed frame (Hermite reduction of the bottleneck row of U), and the exact one-refresh operator K_1
(quadrature over the gamma latent) is diagonalised.  Output: results/refQ1_nb_augmented_basis.txt.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.augment import nb_augmented_operator                                    # noqa: E402
from fibresampler.basis import RayView, hold_fixed_basis, int_det, with_basis             # noqa: E402
from fibresampler.problems import P1                                                       # noqa: E402
from fibresampler.spectral import log_weights, normalised_pi, slem, sr_transition_matrix  # noqa: E402

OUT = os.path.join(REPO, "results", "refQ1_nb_augmented_basis.txt")
MUS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6]
ALPHA = 1.9
JSTAR = 4


def slope(gs, mu_max=0.03):
    keep = [(m, g) for m, g in zip(MUS, gs) if m <= mu_max and g > 0]
    return float(np.polyfit(np.log([m for m, _ in keep]), np.log([g for _, g in keep]), 1)[0])


def main():
    prob = P1()
    pv = RayView(prob.states, prob.U, index=prob.index, plb_info=prob.plb_info, name="P1")
    V, n_slice = hold_fixed_basis(prob.U, [JSTAR])
    assert abs(int_det(V)) == 1
    hf = with_basis(pv, V)
    out = ["=" * 100,
           "Gamma-augmented (NB, alpha = 1.9) one-refresh operator K_1 on P1, bottleneck cell 5 with mean mu: PLB versus hold-fixed basis",
           f"hold-fixed V columns {[tuple(int(v) for v in V[:, k]) for k in range(V.shape[1])]}, {n_slice} slice moves; exact quadrature, SLEM of K_1",
           "=" * 100,
           f"  {'mu':>8} {'K_1 PLB':>12} {'K_1 hold-fixed':>15} {'Poisson PLB':>13} {'Poisson hf':>12} | ratio"]
    g_plb, g_hf = [], []
    for mu in MUS:
        mvec = np.ones(prob.r)
        mvec[JSTAR] = mu
        row = []
        for view in (pv, hf):
            K, pi = nb_augmented_operator(view, mvec, ALPHA)
            row.append(slem(K, pi)[1])
        g_plb.append(row[0])
        g_hf.append(row[1])
        theta = np.ones(prob.r)
        theta[JSTAR] = mu
        lw = log_weights(prob, "poisson", theta)
        pi = normalised_pi(lw)
        pois = [slem(sr_transition_matrix(v, lw), pi)[1] for v in (pv, hf)]
        out.append(f"  {mu:>8.0e} {row[0]:>12.4e} {row[1]:>15.5f} {pois[0]:>13.4e} {pois[1]:>12.5f} | {row[1] / row[0]:>9.1f}")
    out.append("")
    out.append(f"log-log slope of gap(K_1) vs mu over mu <= 0.03: PLB {slope(g_plb):.3f}, hold-fixed {slope(g_hf):.3f}")
    text = "\n".join(out) + "\n"
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
