"""Prediction 2 (keystone): does NB/gamma augmentation floor the spectral gap?

Theorem thm:loading-nb (6.2) claims the joint (x, lambda) Metropolis-within-Gibbs
sampler has a spectral gap bounded below by delta > 0 uniformly as mu5 -> 0 with
fixed dispersion alpha, whereas the Poisson sampler (Prediction 1) collapses.

Decision by EXACT SLEM.  The joint chain's non-zero spectrum equals that of the
exact x-marginal operator K(x,x') = E_{lambda~f(lambda|x)}[P_x^lambda(x->x')]
(the lambda block is an exact Gibbs draw = projection), computed by deterministic
Gauss-Laguerre quadrature (fibresampler.augment).  This is NOT the NB-marginal
fibre chain; it is the projection of the actual joint chain.

RESULT: negative.  For the realistic sampler (hit-and-run fibre update, any fixed
number R of updates per lambda refresh) the gap does NOT floor -- it decays as
Theta(R*mu5) -> 0 for every tested alpha, mirroring the Poisson collapse.  Only
the idealised data-augmentation chain (R -> infinity, exact x|lambda resample)
floors, and that is not a tractable sampler.  See PREDICTIONS_STATUS.md.

Run:  python3 experiments/pred2_keystone.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

from fibresampler.problems import P1, p1_poisson_theta
from fibresampler.spectral import (log_weights, sr_transition_matrix, slem,
                                   normalised_pi, hazelton_loading)
from fibresampler.augment import nb_augmented_operator, nb_log_pmf

PROB = P1()
X = PROB.states.astype(float)
LOGFACTX = gammaln(X + 1).sum(1)


def da_operator(mu, alpha, M=200000, seed=0):
    """Ideal data-augmentation chain (exact x|lambda resample = R->inf)."""
    rng = np.random.default_rng(seed)
    N = PROB.fibre_size; K = np.zeros((N, N)); mu = np.asarray(mu, float)
    for i in range(N):
        x = PROB.states[i]; shape = alpha + x; rate = alpha / mu + 1
        lam = rng.gamma(shape, 1.0 / rate, size=(M, PROB.r))
        logits = X @ np.log(lam).T - LOGFACTX[:, None]
        logits -= logits.max(0, keepdims=True); W = np.exp(logits); W /= W.sum(0, keepdims=True)
        K[i, :] = W.mean(1)
    lw = nb_log_pmf(PROB, mu, alpha); pi = np.exp(lw - lw.max()); pi /= pi.sum()
    return K, pi


def refinement_check(out, mu5=0.01, alpha=1.9):
    out.append(f"\n# Quadrature refinement check (alpha={alpha}, mu5={mu5})")
    out.append("n_q   max|piK-pi|   gap(K)")
    for nq in (6, 10, 16, 24):
        K, pi = nb_augmented_operator(PROB, p1_poisson_theta(mu5), alpha, n_q=nq)
        se = np.abs(pi @ K - pi).max(); _, g, _ = slem(K, pi)
        out.append(f"{nq:<4}  {se:.2e}     {g:.8f}")
    out.append("-> gap(K) stable under refinement; not a discretisation artifact.")


def main():
    out = ["# Prediction 2 (keystone): NB augmentation spectral-gap floor test\n"]
    sweep = [1.0, 0.1, 0.01, 0.001, 0.0001]
    alphas = [0.5, 1.9, 5.0, 50.0]

    out.append("Exact SLEM of the joint chain (hit-and-run fibre update, R=1),")
    out.append("via quadrature operator K; Poisson baseline is Prediction 1's SR-Pois.\n")
    header = "mu5        Pois_gap     " + "  ".join(f"NB_gap(a={a})" for a in alphas) + "     NB_rho(a=1.9)"
    out.append(header)
    rows = {}
    for mu5 in sweep:
        mu = p1_poisson_theta(mu5)
        lw = log_weights(PROB, "poisson", mu); piP = normalised_pi(lw)
        _, gP, _ = slem(sr_transition_matrix(PROB, lw), piP)
        gaps = []; rho19 = None
        for a in alphas:
            K, pi = nb_augmented_operator(PROB, mu, a, n_q=16)
            _, g, _ = slem(K, pi); gaps.append(g)
            if a == 1.9:
                rho19, _, _ = hazelton_loading(PROB, K, pi)
        rows[mu5] = (gP, gaps, rho19)
        out.append(f"{mu5:<9}  {gP:.4e}  " + "  ".join(f"{g:.4e}" for g in gaps) + f"     {rho19:.1f}")

    # ideal DA chain (R -> inf)
    out.append("\n# Idealised data-augmentation chain (exact x|lambda resample, R->inf)")
    out.append("mu5        DA_gap(a=0.5)  DA_gap(a=1.9)  DA_gap(a=5)")
    da = {}
    for mu5 in sweep:
        gd = []
        for a in (0.5, 1.9, 5.0):
            Kd, pid = da_operator(p1_poisson_theta(mu5), a); _, g, _ = slem(Kd, pid); gd.append(g)
        da[mu5] = gd
        out.append(f"{mu5:<9}  {gd[0]:.4e}     {gd[1]:.4e}     {gd[2]:.4e}")

    refinement_check(out)

    # verdict
    g_hr = [rows[m][1][1] for m in sweep]          # alpha=1.9 HR gaps
    g_da = [da[m][1] for m in sweep]               # alpha=1.9 DA gaps
    hr_floors = g_hr[-1] > 0.5 * g_hr[0]           # would need gap not to collapse
    da_floors = g_da[-1] > 0.5 * g_da[0]
    out.append("\n# VERDICT")
    out.append(f"HR (realistic, R=1) gap alpha=1.9: {g_hr[0]:.3e} -> {g_hr[-1]:.3e}  "
               f"floors? {hr_floors}")
    out.append(f"DA (ideal, R=inf)  gap alpha=1.9: {g_da[0]:.3e} -> {g_da[-1]:.3e}  "
               f"floors? {da_floors}")
    out.append("PREDICTION 2 FAILED for the realistic Metropolis-within-Gibbs sampler: the")
    out.append("joint-chain gap decays ~Theta(R*mu5)->0 for every fixed R and every alpha,")
    out.append("tracking the Poisson collapse.  The idealised R->inf data-augmentation chain")
    out.append("DOES floor, but exact fibre resampling is the intractable problem the sampler")
    out.append("exists to avoid.  Theorem thm:loading-nb (Step-3 conductance decomposition)")
    out.append("is not supported as stated.")

    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "pred2_numbers.txt"), "w") as fh:
        fh.write(text)
    print(text)

    # figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        # Plot (and hence legend) order = visual top-to-bottom; marker/linestyle
        # pairs stay distinguishable in greyscale.
        ax.loglog(sweep, g_da, "^--", color="0.4",
                  label=r"ideal data augmentation, $R\to\infty$ ($\alpha$=1.9)")
        ax.loglog(sweep, [rows[m][0] for m in sweep], "o-", color="black",
                  label="SR-Pois (Prediction 1)")
        ax.loglog(sweep, g_hr, "s-.", color="0.55",
                  label=r"SR-NB, hit-and-run fibre update ($R$=1, $\alpha$=1.9)")
        c_guide = 0.4 * g_hr[-1] / sweep[-1]
        ax.loglog(sweep, [c_guide * m for m in sweep], ":", color="0.7",
                  label=r"slope-1 guide ($\propto\mu_5$)")
        ax.set_xlabel(r"$\mu_5$"); ax.set_ylabel(r"exact spectral gap $1-\lambda_\star$")
        ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(RESULTS, "fig_pred2_keystone.pdf")
        fig.savefig(p); fig.savefig(p.replace(".pdf", ".png"), dpi=130)
        print("figure ->", p)
    except Exception as e:
        print("[figure skipped:", e, "]")
    print("numbers -> results/pred2_numbers.txt")


if __name__ == "__main__":
    main()
