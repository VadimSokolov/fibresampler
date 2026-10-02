"""Task S -- slice-sampling conserved corollary (P1, NB target, alpha=1.9).

Discrete slice sampler: pick a move direction, draw u ~ Uniform(0, pi(x)), update x
by hit-and-run restricted to the slice {x' on the ray : pi(x') >= u}, taking x'
uniform on the slice.  We assemble the EXACT x-marginal transition matrix (the
u-integral is piecewise-constant with breakpoints at the ray's pi-values, so it is
evaluated in closed form) and its exact SLEM over the mu5 sweep.

Prediction P-S1: gap = Theta(mu5), slope ~ 1 -- the loading obstruction is invariant
across auxiliary-variable schemes (the slice height u tracks the vanishing corridor
weight exactly as the gamma latent lambda tracked the vanishing mean).  This is the
corollary that augmentation-by-slicing does not escape either; contrast Task V, where
adding a *uniform* temperature rung (beta=0) does escape.

Run:  python3 experiments/taskS_slice.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.problems import P1, p1_poisson_theta
from fibresampler.spectral import slem, normalised_pi
from fibresampler.augment import nb_log_pmf

PROB = P1()
N = PROB.fibre_size
ALPHA = 1.9
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def slice_transition_matrix(prob, log_w):
    """Exact x-marginal transition matrix of the discrete slice sampler for the
    target pi ∝ exp(log_w).  Direction chosen uniformly among the moves; on the ray
    the slice update T(i->k) = E_{u~U(0,pi_i)}[ 1{pi_k>=u} / |{m: pi_m>=u}| ] is the
    closed-form integral over u (piecewise-constant, breakpoints at the ray pi's)."""
    M = prob.fibre_size
    Nm = prob.n_moves
    w = np.exp(log_w - log_w.max())          # unnormalised pi (positive scale)
    Q = np.zeros((M, M))
    for i in range(M):
        for j in range(Nm):
            ray = prob.ray(i, j)
            if len(ray) == 1:
                Q[i, i] += 1.0 / Nm
                continue
            pr = w[list(ray)]                # pi on the ray
            pi_i = w[i]
            # breakpoint levels in (0, pi_i]: the distinct pi-values <= pi_i, sorted
            levels = np.unique(pr[pr <= pi_i + 1e-300])
            levels = levels[levels <= pi_i]
            prev = 0.0
            # integrate u from 0 to pi_i across intervals (prev, lev]
            for lev in levels:
                width = lev - prev
                if width > 0:
                    inslice = pr >= lev - 1e-300         # states with pi >= lev (=slice for u in (prev,lev])
                    cnt = int(inslice.sum())
                    if cnt > 0:
                        contrib = (width / pi_i) / cnt   # per in-slice state
                        for idx_on_ray, s in enumerate(ray):
                            if inslice[idx_on_ray]:
                                Q[i, s] += (1.0 / Nm) * contrib
                prev = lev
    return Q


def nb_logw(mu5):
    return nb_log_pmf(PROB, p1_poisson_theta(mu5), ALPHA)


def db_violation(Q, pi):
    F = pi[:, None] * Q
    return float(np.abs(F - F.T).max())


def main():
    out = []
    out.append("Task S -- discrete slice sampler on P1 (NB target, alpha=1.9)")
    out.append("=" * 66)
    # correctness gates at mu5=1: row-stochastic, reversible, irreducible
    lw1 = nb_logw(1.0); pi1 = normalised_pi(lw1)
    Q1 = slice_transition_matrix(PROB, lw1)
    rows_ok = np.allclose(Q1.sum(1), 1.0)
    dbe1 = db_violation(Q1, pi1)
    out.append(f"GATE mu5=1: rows sum to 1: {rows_ok};  max DB violation: {dbe1:.2e}  "
               f"{'PASS' if rows_ok and dbe1 < 1e-9 else 'FAIL'}")
    out.append("")
    mus = [1.0, 0.1, 0.01, 1e-3, 1e-4]
    out.append(f"  {'mu5':>8} {'slice gap':>12} {'SR-NB gap':>12} {'maxDB':>9}")
    slice_gaps, srnb_gaps = [], []
    from fibresampler.spectral import sr_transition_matrix
    for mu5 in mus:
        lw = nb_logw(mu5); pi = normalised_pi(lw)
        Qs = slice_transition_matrix(PROB, lw)
        _, gs, _ = slem(Qs, pi)
        _, gnb, _ = slem(sr_transition_matrix(PROB, lw), pi)
        slice_gaps.append(gs); srnb_gaps.append(gnb)
        out.append(f"  {mu5:>8g} {gs:>12.5e} {gnb:>12.5e} {db_violation(Qs, pi):>9.1e}")
    # slope over asymptotic tail
    tail = slice(1, None)
    x = np.log(mus[tail]); ys = np.log(slice_gaps[tail])
    slope_slice = float(np.polyfit(x, ys, 1)[0])
    slope_srnb = float(np.polyfit(x, np.log(srnb_gaps[tail]), 1)[0])
    out.append("")
    out.append(f"LOG-LOG SLOPE (mu5<=0.1):  slice = {slope_slice:+.3f}   SR-NB = {slope_srnb:+.3f}")
    held = abs(slope_slice - 1.0) < 0.15
    out.append(f"P-S1 (gap = Theta(mu5), slope ~ 1): slice slope {slope_slice:+.3f}  "
               f"{'HELD' if held else 'CHECK'} -- obstruction invariant across aux schemes.")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.loglog(mus, srnb_gaps, "s-", color="black", label=fr"SR-NB  ~$\mu_5^{{{slope_srnb:.2f}}}$")
        ax.loglog(mus, slice_gaps, "o-", color="C0", label=fr"slice  ~$\mu_5^{{{slope_slice:.2f}}}$")
        ax.set_xlabel(r"loading mean $\mu_5$"); ax.set_ylabel(r"exact spectral gap $1-\lambda_\star$")
        ax.set_title(r"Task S: slice sampling is conserved too (P1, $\alpha=1.9$)")
        ax.legend(frameon=False); ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        p = os.path.join(RESULTS, "fig_taskS_slice.pdf")
        fig.savefig(p); fig.savefig(p.replace(".pdf", ".png"), dpi=130)
        out.append(f"\nfigure -> {p}")
    except Exception as e:
        out.append(f"[figure skipped: {e}]")

    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "taskS_numbers.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
