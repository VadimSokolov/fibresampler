"""Prediction 2 generality probe: is the Theta(R*mu) collapse problem-independent?

The Prediction 2 negative (pred2_keystone.py) is exact on P1.  Before reframing
the manuscript we confirm the obstruction is GENERAL, not P1-specific, by
replicating the scaling law on a second, structurally different problem: a 2x3
contingency table (Markov-basis setting) with a Poisson bottleneck at cell x11.

For each problem we fit the two exponents of the conjectured general law
    gap(K_R) ~ c * R^q * mu^p       (obstruction if p ~ 1 and, at fixed mu, q ~ 1)
where K_R is the joint-chain operator with R hit-and-run fibre updates per gamma
refresh.  If both problems give p ~ 1 (SR-NB gap vanishes linearly in mu for any
fixed R) and only the R->inf data-augmentation chain floors, the negative is a
general obstruction: no bounded-R augmented hit-and-run sampler floors.

Run:  python3 experiments/pred2_generality.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

from fibresampler.problems import P1, table_2x3, TABLE23_BOTTLENECK
from fibresampler.spectral import log_weights, sr_transition_matrix, slem, normalised_pi
from fibresampler.augment import nb_augmented_operator, da_operator, kr_operators


def mu_vector(prob, jstar, m):
    mu = np.ones(prob.r); mu[jstar] = m
    return mu


def loglog_slope(xs, ys):
    lx, ly = np.log(np.asarray(xs)), np.log(np.asarray(ys))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return slope


def probe(prob, jstar, alpha, out, mus=(0.1, 0.01, 0.001), Rs=(1, 2, 4, 8)):
    out.append(f"\n## Problem: {prob.name}   (fibre {prob.fibre_size}, shrink coord {jstar}, alpha={alpha})")

    # SR-Pois and exact SR-NB (R=1) gaps over the mu sweep
    pois, nb1 = [], []
    for m in mus:
        mu = mu_vector(prob, jstar, m)
        lw = log_weights(prob, "poisson", mu); piP = normalised_pi(lw)
        pois.append(slem(sr_transition_matrix(prob, lw), piP)[1])
        K, pi = nb_augmented_operator(prob, mu, alpha, n_q=16)
        nb1.append(slem(K, pi)[1])
    p_pois = loglog_slope(mus, pois)
    p_nb = loglog_slope(mus, nb1)
    out.append(f"  mu sweep {list(mus)}")
    out.append(f"  SR-Pois gap:      {['%.3e'%g for g in pois]}   slope p={p_pois:.2f}")
    out.append(f"  SR-NB gap (R=1):  {['%.3e'%g for g in nb1]}   slope p={p_nb:.2f}   (obstruction if p~1)")

    # R-dependence at the smallest mu: gap(K_R) vs R (before saturation)
    m_small = min(mus)
    Kdict, pi = kr_operators(prob, mu_vector(prob, jstar, m_small), alpha, list(Rs), M=800)
    gaps_R = [slem(Kdict[R], pi)[1] for R in Rs]
    q = loglog_slope(Rs, gaps_R)
    out.append(f"  gap(K_R) at mu={m_small}: R={list(Rs)} -> {['%.3e'%g for g in gaps_R]}   slope q={q:.2f}   (linear-in-R if q~1)")

    # DA chain (R->inf) floor across the sweep
    da = [slem(*da_operator(prob, mu_vector(prob, jstar, m), alpha))[1] for m in mus]
    da_floors = da[-1] > 0.5 * da[0]
    out.append(f"  DA (R->inf) gap:  {['%.3e'%g for g in da]}   floors? {da_floors}")
    return dict(pois=pois, nb1=nb1, p_nb=p_nb, q=q, da=da, mus=list(mus))


def main():
    out = ["# Prediction 2 generality probe: Theta(R*mu) obstruction across problems\n"]
    res = {}
    res["P1"] = probe(P1(), 4, 1.9, out)                       # P1: shrink coord 5 (x5, index 4)
    res["2x3"] = probe(table_2x3(), TABLE23_BOTTLENECK, 1.9, out)

    out.append("\n# CONCLUSION")
    p_ok = all(abs(res[k]["p_nb"] - 1.0) < 0.25 for k in res)   # SR-NB gap ~ mu^1
    q_ok = all(abs(res[k]["q"] - 1.0) < 0.35 for k in res)      # gap(K_R) ~ R^1 (pre-saturation)
    da_ok = all(res[k]["da"][-1] > 0.5 * res[k]["da"][0] for k in res)
    for k in res:
        out.append(f"  {k}: SR-NB gap slope p={res[k]['p_nb']:.2f} (mu), K_R slope q={res[k]['q']:.2f} (R), DA floors={res[k]['da'][-1]>0.5*res[k]['da'][0]}")
    out.append(f"\n  p~1 on both problems: {p_ok};  q~1 on both: {q_ok};  DA floors on both: {da_ok}")
    if p_ok and q_ok and da_ok:
        out.append("  => GENERAL OBSTRUCTION confirmed: for any FIXED R, the joint-chain gap of")
        out.append("     an augmented hit-and-run sampler vanishes as Theta(R*mu) -> 0 (the gamma")
        out.append("     latent tracks the shrinking marginal mean); only the intractable R->inf")
        out.append("     exact-resample chain floors.  The Prediction-2 negative is not P1-specific.")
    else:
        out.append("  => law did NOT replicate cleanly; negative may be problem-specific -- inspect.")

    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "pred2_generality.txt"), "w") as fh:
        fh.write(text)
    print(text)
    print("numbers -> results/pred2_generality.txt")


if __name__ == "__main__":
    main()
