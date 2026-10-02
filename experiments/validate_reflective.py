"""Correctness gate for the reflective lattice sampler (Section 5).

Before trusting any Prediction 3/4 number we verify the exact reflective transition
matrix is a valid reversible chain for the target, exactly as exact SLEM was the gate
for Predictions 1-2.  Checks:

  (1) row-stochastic  (every row sums to 1);
  (2) detailed balance pi_i Q_ij == pi_j Q_ji against the product-Poisson target
      (this is the real test: a wrong reverse-path probability breaks it);
  (3) B=0 reduces to a valid single-ray (Metropolis hit-and-run) reversible chain;
  (4) irreducibility (the chain connects the whole fibre) and a sane SLEM.

Run:  python3 experiments/validate_reflective.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.problems import P1, table_2x3, p1_poisson_theta
from fibresampler.spectral import slem
from fibresampler.reflective import reflective_transition_matrix


def check(prob, target, theta, bmax, label):
    Q, pi = reflective_transition_matrix(prob, target=target, theta=theta, bmax=bmax)
    N = prob.fibre_size
    row = np.abs(Q.sum(1) - 1.0).max()
    db = np.abs(pi[:, None] * Q - (pi[:, None] * Q).T).max()      # |pi_i Q_ij - pi_j Q_ji|
    # reachability from state 0 via Q's nonzero pattern
    reach = np.zeros(N, bool); reach[0] = True
    for _ in range(N):
        reach = reach | (Q[reach].sum(0) > 0)
    irred = bool(reach.all())
    sv, gap, _ = slem(Q, pi)
    stat = np.abs(pi @ Q - pi).max()                              # pi stationary
    ok = row < 1e-12 and db < 1e-12 and stat < 1e-12 and irred
    print(f"  [{ 'OK ' if ok else 'FAIL'}] {label}: rowsum_err={row:.1e} "
          f"detbal_err={db:.1e} pi_stat_err={stat:.1e} irred={irred} "
          f"SLEM={sv:.6f} gap={gap:.6f}")
    return ok, gap


def main():
    print("Reflective sampler correctness gate (exact transition matrix):")
    results = []
    # P1: uniform and Poisson, several bounce budgets
    results.append(check(P1(), "uniform", None, 0, "P1 uniform  B=0 (Metropolis SR)")[0])
    results.append(check(P1(), "uniform", None, 1, "P1 uniform  Bmax=1")[0])
    results.append(check(P1(), "uniform", None, 2, "P1 uniform  Bmax=2")[0])
    results.append(check(P1(), "poisson", p1_poisson_theta(1.0), 2, "P1 Poisson th5=1   Bmax=2")[0])
    results.append(check(P1(), "poisson", p1_poisson_theta(0.1), 2, "P1 Poisson th5=0.1 Bmax=2")[0])
    # second problem: contingency table
    results.append(check(table_2x3(), "uniform", None, 2, "2x3 table uniform  Bmax=2")[0])
    print(f"\n{'ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED'} "
          f"({sum(results)}/{len(results)})")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
