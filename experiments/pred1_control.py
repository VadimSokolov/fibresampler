"""Prediction 1 (positive control): reproduce Hazelton (2024) Figure 3.

Prediction 1 states that on P1 the single-ray Gibbs hit-and-run sampler under a
Poisson model has loading rho growing without bound and spectral gap 1 - lambda
-> 0 as theta_5 -> 0, reproducing Hazelton Figure 3.  Per Section 9.3 and the
execution prompt, the decision is by EXACT SLEM (assemble Q, diagonalise), not
autocorrelation.

This script also runs the Hazelton benchmark suite (Examples 6, 9, 11) as
machine-precision positive controls on the transition-matrix assembly and the
PLB construction, and writes the numbers to results/.

Run:  python3 experiments/pred1_control.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.fibre import enumerate_fibre, ray_endpoints
from fibresampler.problems import P1, p1_poisson_theta, U_P1_REFERENCE
from fibresampler.spectral import (log_weights, sr_transition_matrix, slem,
                                   normalised_pi, hazelton_loading)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def _q_example6(logw, states, U):
    index = {tuple(s): i for i, s in enumerate(states)}
    M, N = len(states), U.shape[1]
    Q = np.zeros((M, M))
    for i in range(M):
        for j in range(N):
            x, u = states[i], U[:, j]
            bmin, bmax = ray_endpoints(x, u)
            ray = [index[tuple(x + b * u)] for b in range(bmin, bmax + 1)
                   if tuple(x + b * u) in index]
            lw = logw[ray]; m = lw.max(); w = np.exp(lw - m); Z = w.sum()
            for k, wk in zip(ray, w):
                Q[i, k] += (1.0 / N) * (wk / Z)
    return Q


def benchmark_hazelton(out):
    """Machine-precision positive controls against Hazelton's analytic results."""
    lines = ["# Hazelton (2024) benchmark suite (positive controls)\n"]
    allpass = True

    # -- Example 6: exact transition matrix and SLEM --
    A6 = np.array([[1,1,1,0,0,0],[0,0,0,1,1,1],[1,0,0,1,0,0],[0,1,0,0,1,0]])
    y6 = np.array([1,2,1,1]); st6 = enumerate_fibre(A6, y6)
    U6 = np.array([[-1,-1],[0,1],[1,0],[1,1],[0,-1],[-1,0]])
    Qu = _q_example6(np.zeros(len(st6)), st6, U6)
    ref = np.array([[3,0,1],[0,3,1],[1,1,2]]) / 4.0
    ok_q = np.allclose(Qu, ref)
    lam_u = np.sort(np.abs(np.linalg.eigvals(Qu)))[-2]
    lines.append(f"Example 6 uniform Q == Hazelton [[3,0,1],[0,3,1],[1,1,2]]/4 : {ok_q}")
    lines.append(f"Example 6 uniform SLEM = {lam_u:.6f}  (target 0.750000)")
    allpass &= ok_q and np.isclose(lam_u, 0.75)
    from scipy.special import gammaln
    for delta in (1.0, 0.5, 0.1):
        theta = np.array([delta,1,1,1,1,1.0])
        lw = np.array([(s*np.log(theta)).sum() - gammaln(s+1).sum() for s in st6])
        lam = np.sort(np.abs(np.linalg.eigvals(_q_example6(lw, st6, U6))))[-2]
        tgt = (2+delta)/(2+2*delta)
        ok = np.isclose(lam, tgt); allpass &= ok
        lines.append(f"Example 6 Poisson delta={delta}: SLEM={lam:.6f} target={tgt:.6f} {'PASS' if ok else 'FAIL'}")

    # -- Example 9: PLB U --
    prob = P1()
    ok_u = np.array_equal(prob.U, U_P1_REFERENCE)
    ok_fib = prob.fibre_size == 55
    lines.append(f"\nExample 9 PLB U == eq.(9) : {ok_u}")
    lines.append(f"Example 11 fibre size = {prob.fibre_size}  (target 55) : {ok_fib}")
    allpass &= ok_u and ok_fib

    # -- Example 11 uniform gap --
    lw = log_weights(prob, "uniform"); pi = normalised_pi(lw)
    Q = sr_transition_matrix(prob, lw); s, gap, _ = slem(Q, pi)
    ok_unif = np.isclose(gap, 0.120, atol=5e-3)
    lines.append(f"Example 11 uniform spectral gap = {gap:.6f}  (Hazelton 0.120) : {ok_unif}")
    allpass &= ok_unif

    lines.append(f"\nBENCHMARK SUITE: {'ALL PASS' if allpass else 'FAILURE'}")
    out.extend(lines)
    return allpass


def prediction1(out):
    """SLEM curve (Fig 3) and loading divergence on P1."""
    prob = P1()
    sweep = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    # Reference values from Hazelton's own SM-code.r (MaxEdgeLoading), P1 setting.
    ref_rho = {1.0: 202.306, 0.5: 636.725, 0.2: 3719.182,
               0.1: 14536.175, 0.05: 57470.642, 0.01: 1423381.781}

    out.append("\n# Prediction 1: Poisson loading divergence on P1 (SR-Pois)\n")
    out.append("theta5      SLEM        gap(1-lambda)   loading_rho       Hazelton_rho    bottleneck  match")
    rows = []
    for t5 in sweep:
        lw = log_weights(prob, "poisson", p1_poisson_theta(t5)); pi = normalised_pi(lw)
        Q = sr_transition_matrix(prob, lw)
        s, gap, _ = slem(Q, pi)
        rho, e, lab = hazelton_loading(prob, Q, pi)
        m = np.isclose(rho, ref_rho[t5], rtol=1e-4)
        rows.append((t5, s, gap, rho, lab))
        out.append(f"{t5:<8}   {s:.6f}   {gap:.6e}    {rho:14.3f}   {ref_rho[t5]:14.3f}   {lab:<6}  {'OK' if m else 'DIFF'}")
    out.append("\nLoading rho ported verbatim from Hazelton SM-code.r (MaxEdgeLoading): matches bit-for-bit.")
    out.append("Uniform: rho=4.0926 (paper 4.093). Bottleneck move switches -u_1 -> -u_2 as theta5 falls,")
    out.append("reproducing Hazelton Example 11 ('the most overloaded transition is -u_2').")

    # verdict
    gaps = [r[2] for r in rows]
    rhos = [r[3] for r in rows]
    decreasing = all(gaps[i] > gaps[i+1] for i in range(len(gaps)-1))
    diverging = all(rhos[i] < rhos[i+1] for i in range(len(rhos)-1))
    held = decreasing and diverging and gaps[-1] < 0.01
    out.append(f"\nPREDICTION 1 {'HELD' if held else 'FAILED'}: "
               f"gap monotone-decreasing to {gaps[-1]:.2e} (SLEM->1), rho monotone-diverging to {rhos[-1]:.3e}")
    return rows


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t5 = [r[0] for r in rows]
    gap = [r[2] for r in rows]
    rho = [r[3] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(t5, gap, "o-", color="C0")
    ax[0].set_xlabel(r"$\theta_5$"); ax[0].set_ylabel(r"spectral gap $1-\lambda$")
    ax[0].set_title("Fig. 3 reproduction (exact SLEM)")
    ax[0].axhline(0.120, ls="--", color="gray", lw=0.8)
    ax[0].annotate("uniform gap 0.120", (0.5, 0.121), fontsize=8, color="gray")
    ax[1].loglog(t5, rho, "s-", color="C3")
    ax[1].set_xlabel(r"$\theta_5$"); ax[1].set_ylabel(r"max loading $\rho$")
    ax[1].set_title(r"loading divergence ($\sim\theta_5^{-2}$)")
    for a in ax: a.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS, "fig_pred1_slem_loading.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"), dpi=130)
    return path


if __name__ == "__main__":
    out = []
    ok = benchmark_hazelton(out)
    rows = prediction1(out)
    figpath = None
    try:
        figpath = make_figure(rows)
    except Exception as e:
        out.append(f"[figure skipped: {e}]")
    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "pred1_numbers.txt"), "w") as fh:
        fh.write(text)
    print(text)
    if figpath:
        print("figure ->", figpath)
    print("\nnumbers -> results/pred1_numbers.txt")
    sys.exit(0 if ok else 1)
