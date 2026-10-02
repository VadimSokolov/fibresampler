"""FMB reference (Section 9.2): full-Markov-basis hit-and-run via 4ti2.

Computes the minimal Markov basis of A with 4ti2's `markov` executable
(conda-forge 4ti2, installed at ~/miniconda3/bin), then assembles the EXACT
single-ray Gibbs hit-and-run transition matrix over the full basis on the
enumerated fibre and diagonalises, exactly as for the PLB samplers.

Deliverables:
  1. P1 (Hazelton Ex. 2/11): FMB gap vs PLB gap under the uniform target and
     the Poisson theta_5 sweep of Prediction 1.  Expectation: the loading
     collapse is basis-independent (the obstruction lives in the target, not
     the move set), so the FMB gap collapses at the same Theta(theta_5) rate.
  2. 2x3 table: 4ti2 returns the three classical basic moves (the PLB's two
     plus one redundant combination); gap comparison across the mu sweep.

Output: results/fmb_reference.txt.  Deterministic (no RNG).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.fibre import ray_endpoints
from fibresampler.problems import P1, p1_poisson_theta, table_2x3, TABLE23_BOTTLENECK
from fibresampler.spectral import log_weights, sr_transition_matrix, slem, normalised_pi

MARKOV_BIN = os.path.expanduser("~/miniconda3/bin/markov")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fmb_reference.txt")


def markov_basis(A: np.ndarray) -> np.ndarray:
    """Minimal Markov basis of {x >= 0 integer : Ax = y} moves, via 4ti2."""
    with tempfile.TemporaryDirectory() as td:
        stem = os.path.join(td, "conf")
        with open(stem + ".mat", "w") as fh:
            fh.write(f"{A.shape[0]} {A.shape[1]}\n")
            for row in A:
                fh.write(" ".join(str(int(v)) for v in row) + "\n")
        subprocess.run([MARKOV_BIN, stem], check=True, capture_output=True)
        with open(stem + ".mar") as fh:
            header = fh.readline().split()
            k, r = int(header[0]), int(header[1])
            rows = [list(map(int, fh.readline().split())) for _ in range(k)]
    B = np.array(rows, dtype=np.int64)
    assert B.shape == (k, r) and not (A @ B.T).any()
    return B


class MoveSetProblem:
    """Duck-typed FibreProblem over an arbitrary move set on the same fibre."""

    def __init__(self, base, moves_rows: np.ndarray, name: str):
        self.states = base.states
        self.index = base.index
        self.U = moves_rows.T.astype(np.int64)  # moves as columns
        self.name = name

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


def gap(prob, log_w):
    Q = sr_transition_matrix(prob, log_w)
    pi = normalised_pi(log_w)
    _, g, _ = slem(Q, pi)
    return g


def lobe_count(prob, jstar):
    """Connected components of the zero stratum D = {x_{j*}=0} under the
    moves that hold x_{j*} fixed (the Assumption ass:lobes decomposition)."""
    D = [i for i in range(prob.fibre_size) if prob.states[i][jstar] == 0]
    Dset = set(D)
    parent = {i: i for i in D}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    moves0 = [k for k in range(prob.n_moves) if prob.U[jstar, k] == 0]
    for i in D:
        for k in moves0:
            for t in prob.ray(i, k):
                if t in Dset and t != i:
                    parent[find(i)] = find(t)
    return len({find(i) for i in D})


def main():
    lines = ["FMB reference: exact SLEM of full-Markov-basis (4ti2) hit-and-run",
             "=" * 72]

    p1 = P1()
    B1 = markov_basis(p1.A)
    lines.append(f"P1: 4ti2 minimal Markov basis has {B1.shape[0]} moves "
                 f"(PLB has {p1.n_moves}); fibre size {p1.fibre_size}")
    fmb1 = MoveSetProblem(p1, B1, "P1-FMB")

    lw_u = log_weights(p1, "uniform", None)
    g_plb_u, g_fmb_u = gap(p1, lw_u), gap(fmb1, lw_u)
    lines.append(f"  uniform target: gap PLB={g_plb_u:.6f}  FMB={g_fmb_u:.6f}  "
                 f"FMB/PLB={g_fmb_u / g_plb_u:.3f}")
    lines.append(f"  {'theta5':>8} {'gap PLB':>12} {'gap FMB':>12} {'FMB/PLB':>8}")
    sweep = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    fmb_gaps = []
    for th5 in sweep:
        lw = log_weights(p1, "poisson", p1_poisson_theta(th5))
        gp, gf = gap(p1, lw), gap(fmb1, lw)
        fmb_gaps.append(gf)
        lines.append(f"  {th5:8.2f} {gp:12.6f} {gf:12.6f} {gf / gp:8.3f}")
    slope = (np.log(fmb_gaps[-1]) - np.log(fmb_gaps[0])) / (
        np.log(sweep[-1]) - np.log(sweep[0]))
    lines.append(f"  FMB endpoint slope of gap vs theta5 over [0.01,1]: {slope:.3f} "
                 f"(Theta(theta5) collapse, basis-independent)")

    tab = table_2x3()
    Bt = markov_basis(tab.A)
    lines.append(f"table 2x3: 4ti2 minimal Markov basis has {Bt.shape[0]} moves "
                 f"(PLB has {tab.n_moves}); fibre size {tab.fibre_size}")
    fmbt = MoveSetProblem(tab, Bt, "table-FMB")
    for name, prob in [("PLB", tab), ("FMB", fmbt)]:
        lines.append(f"  zero-stratum components under u_(j*)=0 moves, {name}: "
                     f"{lobe_count(prob, TABLE23_BOTTLENECK)}")
    lines.append(f"  {'mu':>8} {'gap PLB':>12} {'gap FMB':>12} {'FMB/PLB':>8}")
    for mu in (1.0, 0.1, 0.01):
        theta = np.full(tab.states.shape[1], 1.0)
        theta[TABLE23_BOTTLENECK] = mu
        lw = log_weights(tab, "poisson", theta)
        gp, gf = gap(tab, lw), gap(fmbt, lw)
        lines.append(f"  {mu:8.2f} {gp:12.6f} {gf:12.6f} {gf / gp:8.3f}")

    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
