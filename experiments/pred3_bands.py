"""Prediction 3 band family: reproducible regeneration of the kappa-sweep table.

The four thin-tilted bands quoted in Section 9.4 (Prediction 3) and in
empirical.md Exp 4b.1 were originally evaluated by an inline construction;
this script regenerates the table deterministically so every quoted ratio
traces to a checked-in run.

For each band: exact Gibbs single-ray hit-and-run gap (the baseline), exact
reflective gaps at bounce budgets B_max in {2, 4} (pi_B uniform on
{0,...,B_max}), and the best-reflective / Gibbs-SR ratio, all under the
uniform target.  Expected (empirical.md): ratios 1.1, 3.8, 7.3, 10.5 at
kappa = 2.15, 4.26, 7.08, 10.6, with best B_max = 2, 2, 4, 4.

Output: results/pred3_bands.txt.  Deterministic (no RNG).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.fibre import ray_endpoints
from fibresampler.reflective import reflective_transition_matrix
from fibresampler.spectral import normalised_pi, slem, sr_transition_matrix
from fibresampler.synthetic import BandProblem

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "pred3_bands.txt")

BANDS = [(2, 12), (2, 24), (2, 40), (1, 30)]


class RayAdapter:
    """Expose FibreProblem's ray() interface on a BandProblem (whose moves are
    the columns of U acting on the full slack-extended state)."""

    def __init__(self, band):
        self._b = band
        self.states = band.states
        self.index = band.index
        self.U = band.U

    @property
    def n_moves(self):
        return self.U.shape[1]

    @property
    def fibre_size(self):
        return len(self.states)

    def ray(self, i, j):
        x = self.states[i]
        u = self.U[:, j]
        b_min, b_max = ray_endpoints(x, u)
        out = []
        for b in range(b_min, b_max + 1):
            k = self.index.get(tuple(x + b * u))
            if k is not None:
                out.append(k)
        return out


def main():
    lines = ["Prediction 3 band family: exact gaps, uniform target",
             "Gibbs-SR baseline vs best reflective over B_max in {2,4}",
             "=" * 72,
             f"{'kappa':>7} {'fibre':>6} {'GibbsSR':>9} {'B2':>9} {'B4':>9} "
             f"{'best':>5} {'ratio':>7}"]
    for (w, C) in BANDS:
        band = BandProblem(w, C)
        kap, _, _ = band.kappa()
        lw = np.zeros(band.fibre_size)
        pi = normalised_pi(lw)
        g_sr = slem(sr_transition_matrix(RayAdapter(band), lw), pi)[1]
        gaps = {}
        for bm in (2, 4):
            Q, _ = reflective_transition_matrix(band, bmax=bm, log_w=lw)
            gaps[bm] = slem(Q, pi)[1]
        best_b = max(gaps, key=gaps.get)
        ratio = gaps[best_b] / g_sr
        lines.append(f"{kap:7.2f} {band.fibre_size:6d} {g_sr:9.4f} "
                     f"{gaps[2]:9.4f} {gaps[4]:9.4f} B{best_b:<4d} {ratio:7.2f}")
    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
