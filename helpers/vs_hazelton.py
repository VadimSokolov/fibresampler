"""Where the rim construction sits relative to Hazelton-Polson-Sokolov (2026).

Their Theorem 5.14: tempering to the uniform rung (beta_min = 0) is necessary
AND sufficient to escape the loading obstruction.  Sufficiency is proved under
Assumption 5.1 together with the hypothesis that M_U is an *augmenting* PLB --
i.e. that the move set connects the fibre.

This script asks what happens when that hypothesis fails: the move set spans
ker_Z(A) but leaves F_{A,y} disconnected.  Every scheme on the Proposition 5.19
template then has gap exactly zero at every beta, because flattening reweights
states without adding edges.  The rim extension does add edges.
"""
import sys, numpy as np, networkx as nx
sys.path.insert(0, '.')
from fibre_core import enumerate_fibre, swap_moves, components
from scipy.special import gammaln


def poisson_fibre_logw(X, theta):
    X = np.asarray(X, float)
    return float((X * np.log(theta) - gammaln(X + 1.0)).sum())


def build_kernel(states, moves, logw, lower=0):
    """Gibbs (heat-bath) hit-and-run along each move direction: the ray through
    x in direction u, resampled from the target restricted to the ray."""
    idx = {s.tobytes(): i for i, s in enumerate(states)}
    n = len(states)
    Q = np.zeros((n, n))
    for i, x in enumerate(states):
        for u in moves:
            ray = []
            b = 0
            while True:                       # walk the ray both ways
                y = x + b * u
                if y.min() < lower or y.tobytes() not in idx:
                    break
                ray.append(b); b += 1
            b = -1
            while True:
                y = x + b * u
                if y.min() < lower or y.tobytes() not in idx:
                    break
                ray.append(b); b -= 1
            lw = np.array([logw(x + b * u) for b in ray])
            p = np.exp(lw - lw.max()); p /= p.sum()
            for b, pb in zip(ray, p):
                Q[i, idx[(x + b * u).tobytes()]] += pb / len(moves)
    return Q


def slem(Q, pi):
    """Second-largest eigenvalue modulus via the pi-symmetrised operator."""
    d = np.sqrt(pi)
    S = (Q * d[:, None]) / d[None, :]
    S = (S + S.T) / 2
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    return max(abs(ev[1]), abs(ev[-1]))


def tempering_gap(states, moves, logw, L_betas, lower=0):
    """Idealised simulated tempering: exact pseudo-priors equalise level
    marginals; half the steps are fibre updates, half are level moves."""
    n, K = len(states), len(L_betas)
    blocks, pis = [], []
    for b in L_betas:
        lw = np.array([logw(s) * b for s in states])
        p = np.exp(lw - lw.max()); p /= p.sum()
        pis.append(p)
        blocks.append(build_kernel(states, moves, lambda s, bb=b: bb * logw(s), lower))
    N = n * K
    Q = np.zeros((N, N))
    for k in range(K):                                  # fibre arm
        Q[k*n:(k+1)*n, k*n:(k+1)*n] += 0.5 * blocks[k]
    for k in range(K):                                  # level arm
        for i in range(n):
            for dk in (-1, 1):
                k2 = k + dk
                if not (0 <= k2 < K):
                    Q[k*n+i, k*n+i] += 0.5 * 0.5
                    continue
                a = min(1.0, (pis[k2][i] / pis[k][i]))
                Q[k*n+i, k2*n+i] += 0.5 * 0.5 * a
                Q[k*n+i, k*n+i] += 0.5 * 0.5 * (1 - a)
    pi = np.concatenate([p / K for p in pis])
    return 1 - slem(Q, pi)
