"""Verify Assumption A (two-lobe bottleneck) on the evidence problems.

The integrated Theorem (thm:loading-nb / prop:loading-obstruction) is conditional on
Assumption~\\ref{ass:lobes}: with D = {x : x_{j*}=0} the dominant stratum, the graph
on D whose edges are hit-and-run moves using only PLB directions u with u_{j*}=0
must split into >=2 components each carrying stationary mass bounded away from 0 as
mu_{j*}->0.  If this fails on P1 / the 2x3 table, the theorem does not cover the
problems the exact-SLEM evidence is drawn from.

This is a finite computation (delete the u_{j*}!=0 directions, find components of D,
weigh them).  Run:  python3 experiments/verify_assumption_lobes.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.problems import P1, table_2x3, TABLE23_BOTTLENECK
from fibresampler.spectral import log_weights, normalised_pi


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def lobes(prob, jstar, mu_star):
    """Components of D={x_{j*}=0} under moves u with u_{j*}=0, and their masses under
    a product-Poisson target with mean mu_star on j* and 1 elsewhere (small mu_star
    approximates the NB->0 regime; D dominates)."""
    D_idx = [i for i, x in enumerate(prob.states) if x[jstar] == 0]
    Dset = set(D_idx)
    # PLB directions that do not touch coordinate j*
    dirs = [j for j in range(prob.n_moves) if prob.U[jstar, j] == 0]
    pos = {idx: k for k, idx in enumerate(D_idx)}
    uf = UF(len(D_idx))
    for i in D_idx:
        for j in dirs:
            for s in prob.ray(i, j):           # ray-connected states (all have x_{j*}=0)
                if s in Dset:
                    uf.union(pos[i], pos[s])
    comps = {}
    for i in D_idx:
        comps.setdefault(uf.find(pos[i]), []).append(i)
    # masses under product-Poisson with theta[j*]=mu_star
    theta = np.ones(prob.r); theta[jstar] = mu_star
    lw = log_weights(prob, "poisson", theta); pi = normalised_pi(lw)
    massD = pi[D_idx].sum()
    comp_mass = sorted((pi[idxs].sum() for idxs in comps.values()), reverse=True)
    return len(D_idx), len(comps), massD, comp_mass, dirs


def report(prob, jstar, label):
    print(f"\n## {label}   (j* = coordinate index {jstar})")
    print(f"   PLB moves with u_j*=0 (intra-D directions): ", end="")
    for mu in (0.1, 0.01, 0.001):
        nD, ncomp, massD, cmass, dirs = lobes(prob, jstar, mu)
        frac = [f"{m/massD:.3f}" for m in cmass[:4]]
        if mu == 0.1:
            print(dirs)
            print(f"   |D|={nD}  components of D under those moves: {ncomp}")
        two_lobe = ncomp >= 2 and (cmass[1] / massD if ncomp >= 2 else 0) > 0.02
        print(f"   mu*={mu:6}: pi(D)={massD:.4f}  comp mass-fractions (top4): {frac}  "
              f"{'>=2 lobes, both massive' if two_lobe else 'NOT two-lobe'}")
    return ncomp, (cmass[1] / massD if ncomp >= 2 else 0.0)


def main():
    print("Assumption A (two-lobe bottleneck) verification on the evidence problems")
    p1 = P1()
    n1, f1 = report(p1, 4, "P1 (Hazelton Ex.11), bottleneck theta5 = coord index 4")
    tb = table_2x3()
    n2, f2 = report(tb, TABLE23_BOTTLENECK, "2x3 contingency table, bottleneck cell x11 = index 0")
    print("\n# VERDICT")
    ok1 = n1 >= 2 and f1 > 0.02
    ok2 = n2 >= 2 and f2 > 0.02
    print(f"  P1: {'Assumption A HOLDS' if ok1 else 'Assumption A FAILS'} "
          f"({n1} components, 2nd-lobe fraction {f1:.3f})")
    print(f"  2x3 table: {'Assumption A HOLDS' if ok2 else 'Assumption A FAILS'} "
          f"({n2} components, 2nd-lobe fraction {f2:.3f})")
    print(f"\n  => Theorem hypothesis {'covers' if (ok1 and ok2) else 'does NOT fully cover'} "
          f"the exact-SLEM evidence problems.")


if __name__ == "__main__":
    main()
