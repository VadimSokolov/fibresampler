"""Core routines: fibre enumeration, move sets, connectivity under
lattice bases, and extended (rim) fibres allowing negative cells.

Two-way tables with fixed margins are used throughout as the running
example.  The design matrix is totally unimodular, so a *Markov* basis
(all 2x2 swaps) connects every fibre; a *lattice* basis (adjacent 2x2
swaps only) spans the kernel but need not connect the fibre once
non-negativity binds.  That gap is exactly what the mixture/extended
fibre construction is meant to repair.
"""
import itertools, numpy as np, networkx as nx
from scipy.special import gammaln


def enumerate_fibre(r, c, lower=0):
    """All integer m x n tables with margins (r,c) and entries >= lower."""
    m, n = len(r), len(c)
    out = []

    def rec(i, rows, colrem):
        if i == m:
            if all(v == 0 for v in colrem):
                out.append(np.array(rows, dtype=int))
            return
        # enumerate row i with sum r[i], entries >= lower, <= colrem[j] - lower*(m-i-1)
        def rowrec(j, acc, rem):
            if j == n - 1:
                v = rem
                hi = colrem[j] - lower * (m - i - 1)
                if lower <= v <= hi:
                    rowrec(j + 1, acc + [v], 0)
                return
            if j == n:
                nc = [colrem[k] - acc[k] for k in range(n)]
                rec(i + 1, rows + [acc], nc)
                return
            hi = min(rem - lower * (n - j - 1), colrem[j] - lower * (m - i - 1))
            v = lower
            while v <= hi:
                rowrec(j + 1, acc + [v], rem - v)
                v += 1
        rowrec(0, [], r[i])

    rec(0, [], list(c))
    return out


def swap_moves(m, n, adjacent_only):
    """2x2 basic swaps.  adjacent_only=True gives a lattice basis."""
    mv = []
    rows = [(i, i + 1) for i in range(m - 1)] if adjacent_only else list(itertools.combinations(range(m), 2))
    cols = [(j, j + 1) for j in range(n - 1)] if adjacent_only else list(itertools.combinations(range(n), 2))
    for (i, ip) in rows:
        for (j, jp) in cols:
            M = np.zeros((m, n), dtype=int)
            M[i, j] = M[ip, jp] = 1
            M[i, jp] = M[ip, j] = -1
            mv.append(M)
    return mv


def components(tables, moves, lower=0):
    """Connected components of the fibre graph induced by +/- moves."""
    idx = {t.tobytes(): k for k, t in enumerate(tables)}
    G = nx.Graph()
    G.add_nodes_from(range(len(tables)))
    for k, t in enumerate(tables):
        for M in moves:
            for s in (1, -1):
                u = t + s * M
                if u.min() >= lower:
                    q = idx.get(u.tobytes())
                    if q is not None:
                        G.add_edge(k, q)
    return G, list(nx.connected_components(G))


def logF(X):
    """Fisher-Yates log-weight, up to the constant term."""
    return -gammaln(np.asarray(X, dtype=float) + 1.0).sum()
