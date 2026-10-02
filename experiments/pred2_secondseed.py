"""Second-seed stability of the Monte-Carlo-assembled operators (referee point).

The R>=2 R-sweep operators (kr_operators, M=800) and the exact-resample R->inf
operator (da_operator, M=2e5) are Monte Carlo over the gamma latent, so their
gaps carry a Monte Carlo error the paper quotes as "about fifteen per cent" at
mu=1e-3.  The referee asked whether the reported R-exponent q and the
exact-resample floor are stable to the Monte Carlo seed.  This script recomputes
BOTH quantities at the manuscript settings (alpha=1.9, mu=1e-3, R in {1,2,4,8})
for the reference seeds (kr seed=3, da seed=0) AND a second seed (seed=1), on
both evidence problems (P1 and the 2x3 table), and reports the relative shift.

Run:  python3 experiments/pred2_secondseed.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

from fibresampler.problems import P1, table_2x3, TABLE23_BOTTLENECK
from fibresampler.spectral import slem
from fibresampler.augment import da_operator, kr_operators


def mu_vector(prob, jstar, m):
    mu = np.ones(prob.r); mu[jstar] = m
    return mu


def loglog_slope(xs, ys):
    lx, ly = np.log(np.asarray(xs)), np.log(np.asarray(ys))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return slope


def q_exponent(prob, jstar, alpha, seed, mu=1e-3, Rs=(1, 2, 4, 8), M=800):
    Kdict, pi = kr_operators(prob, mu_vector(prob, jstar, mu), alpha, list(Rs), M=M, seed=seed)
    gaps_R = [slem(Kdict[R], pi)[1] for R in Rs]
    return loglog_slope(Rs, gaps_R), gaps_R


def da_floor(prob, jstar, alpha, seed, mu=1e-3, M=200000):
    K, pi = da_operator(prob, mu_vector(prob, jstar, mu), alpha, M=M, seed=seed)
    return slem(K, pi)[1]


def reldiff(a, b):
    return abs(a - b) / abs(b) if b else float("nan")


def main():
    out = ["# Second-seed stability of MC-assembled operators (alpha=1.9, mu=1e-3)\n"]
    probs = [("P1", P1(), 4), ("2x3", table_2x3(), TABLE23_BOTTLENECK)]
    ref_kr_seed, ref_da_seed, second_seed = 3, 0, 1

    for name, prob, jstar in probs:
        out.append(f"## {name} (fibre {prob.fibre_size})")
        # q-exponent: reference seed vs second seed
        q_ref, gR_ref = q_exponent(prob, jstar, 1.9, ref_kr_seed)
        q_2, gR_2 = q_exponent(prob, jstar, 1.9, second_seed)
        out.append(f"  q-exponent  gap(K_R) ~ R^q, R in [1,2,4,8]:")
        out.append(f"    seed={ref_kr_seed} (ref): q={q_ref:.3f}   gaps={['%.3e'%g for g in gR_ref]}")
        out.append(f"    seed={second_seed} (2nd): q={q_2:.3f}   gaps={['%.3e'%g for g in gR_2]}")
        out.append(f"    |dq| = {abs(q_ref - q_2):.3f}  ({100*reldiff(q_2, q_ref):.1f}% of q)")
        # exact-resample floor: reference seed vs second seed
        f_ref = da_floor(prob, jstar, 1.9, ref_da_seed)
        f_2 = da_floor(prob, jstar, 1.9, second_seed)
        out.append(f"  exact-resample floor 1-slem(K_inf):")
        out.append(f"    seed={ref_da_seed} (ref): floor={f_ref:.4f}")
        out.append(f"    seed={second_seed} (2nd): floor={f_2:.4f}")
        out.append(f"    |dfloor|/floor = {100*reldiff(f_2, f_ref):.1f}%\n")

    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "pred2_secondseed.txt"), "w") as fh:
        fh.write(text)
    print(text)
    print("numbers -> results/pred2_secondseed.txt")


if __name__ == "__main__":
    main()
