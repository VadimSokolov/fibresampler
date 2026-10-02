"""Self-reducible estimation of the fibre cardinality |F_y|.

Row-peeling recursion.  For a fixed admissible first row w,

    |F(r ; c)|  =  |F(r_2..m ; c - w)| / P(row_1 = w),

which is exact, and the sub-problem is a table with one fewer row and no
structural zeros.  The conditional probability is estimated from a fibre
sampler; the recursion terminates at m = 1, where the count is 1.  Choosing w
near the modal row keeps every factor well away from zero.

This is Mount's counting-from-generation argument (thesis Sections 3.6-3.9,
3.19) run on the extended-fibre sampler rather than on a Markov basis, so that
it applies when a Markov basis is unavailable.
"""
import numpy as np
from fibre_core import enumerate_fibre, swap_moves
import mixture_sampler as ms
from mixture_sampler import normalisers, run


def modal_row(r, c, i=0):
    """Rounded independence row, repaired to satisfy the constraints."""
    T = sum(c)
    w = np.array([int(round(r[i] * cj / T)) for cj in c])
    w = np.clip(w, 0, np.array(c))
    d = r[i] - w.sum()
    j = 0
    while d != 0:                                   # repair rounding drift
        k = j % len(c)
        if d > 0 and w[k] < c[k]:
            w[k] += 1; d -= 1
        elif d < 0 and w[k] > 0:
            w[k] -= 1; d += 1
        j += 1
        if j > 10000:
            break
    return w


def estimate_count(r, c, n_iter=20000, pi=.5, delta=1., seed=0, exact_small=6):
    """Returns (log-estimate, list of per-stage factors)."""
    ms.UNIFORM = True          # counting requires the uniform target
    r, c = list(r), list(c)
    log_total, stages = 0.0, []
    rng = np.random.default_rng(seed)
    while len(r) > 1:
        F = enumerate_fibre(r, c, 0)
        if len(F) == 0:
            return -np.inf, stages
        Fx = enumerate_fibre(r, c, -1)
        lZf, lZg = normalisers(F, Fx, delta)
        L = swap_moves(len(r), len(c), True)
        d, z = run(F[0], L, n_iter, pi, delta, lZf, lZg, rng, 'h')
        sel = [x for x, zz in zip(d, z) if zz == 1]
        w = modal_row(r, c)
        hits = sum(1 for X in sel if np.array_equal(X[0], w))
        if hits == 0:                                # fall back to the sample mode
            from collections import Counter
            w = np.array(Counter(tuple(X[0]) for X in sel).most_common(1)[0][0])
            hits = sum(1 for X in sel if np.array_equal(X[0], w))
        p = hits / len(sel)
        stages.append(dict(m=len(r), row=w.tolist(), p=p, n=len(sel)))
        log_total += -np.log(p)
        c = [ci - wi for ci, wi in zip(c, w)]
        r = r[1:]
    return log_total, stages


def exact_count(r, c):
    return len(enumerate_fibre(list(r), list(c), 0))
