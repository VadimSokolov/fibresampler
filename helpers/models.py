"""General design matrices, integer kernel (lattice) bases, and fibre
enumeration for models beyond the two-way table.

The two-way table is totally unimodular and therefore the easy case.  The
models that motivate this work are not: the no-three-way-interaction model and
logistic regression designs both have non-unimodular A, where Markov bases are
expensive or unavailable and lattice bases are the only practical option.
"""
import itertools, numpy as np
from sympy import Matrix


def design_no3way(d1, d2, d3):
    """A for the no-three-way-interaction model on a d1 x d2 x d3 table:
    all three two-dimensional margins are fixed."""
    cells = list(itertools.product(range(d1), range(d2), range(d3)))
    rows = []
    for (a, b) in itertools.product(range(d1), range(d2)):
        rows.append([1 if (i == a and j == b) else 0 for (i, j, k) in cells])
    for (a, c) in itertools.product(range(d1), range(d3)):
        rows.append([1 if (i == a and k == c) else 0 for (i, j, k) in cells])
    for (b, c) in itertools.product(range(d2), range(d3)):
        rows.append([1 if (j == b and k == c) else 0 for (i, j, k) in cells])
    return np.array(rows, dtype=int), cells


def design_two_way(m, n):
    cells = list(itertools.product(range(m), range(n)))
    rows = [[1 if i == a else 0 for (i, j) in cells] for a in range(m)]
    rows += [[1 if j == b else 0 for (i, j) in cells] for b in range(n)]
    return np.array(rows, dtype=int), cells


def lattice_basis(A):
    """A Z-basis of ker_Z(A), returned as integer vectors."""
    ns = Matrix(A.tolist()).nullspace()
    B = []
    for v in ns:
        den = np.lcm.reduce([int(x.q) for x in v])
        w = np.array([int(x * den) for x in v], dtype=object)
        g = np.gcd.reduce([abs(int(t)) for t in w if t != 0])
        B.append(np.array([int(t) // g for t in w], dtype=int))
    return B


def enumerate_fibre_general(A, y, lower=0, upper=None):
    """Brute-force enumeration of {x : Ax = y, lower <= x <= upper}."""
    k = A.shape[1]
    if upper is None:
        upper = int(y.max())
    out, x = [], np.zeros(k, dtype=int)

    def rec(i):
        if i == k:
            if np.array_equal(A @ x, y):
                out.append(x.copy())
            return
        for v in range(lower, upper + 1):
            x[i] = v
            # prune on rows whose support is fully assigned
            ok = True
            for r in range(A.shape[0]):
                sup = np.nonzero(A[r])[0]
                if sup.max() <= i:
                    if A[r] @ x != y[r]:
                        ok = False
                        break
                else:
                    partial = A[r, :i + 1] @ x[:i + 1]
                    if partial > y[r] - lower * (sup > i).sum():
                        ok = False
                        break
            if ok:
                rec(i + 1)
        x[i] = lower

    rec(0)
    return out


def move_components(states, moves, lower=0):
    import networkx as nx
    idx = {s.tobytes(): t for t, s in enumerate(states)}
    G = nx.Graph(); G.add_nodes_from(range(len(states)))
    for t, s in enumerate(states):
        for b in moves:
            for sg in (1, -1):
                u = s + sg * b
                if u.min() >= lower:
                    q = idx.get(u.tobytes())
                    if q is not None:
                        G.add_edge(t, q)
    return G, list(nx.connected_components(G))
