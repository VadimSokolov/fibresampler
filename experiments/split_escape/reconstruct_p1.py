"""Reconstruct Hazelton et al. (2024) Example 2 from the manuscript's fingerprints:
  - A in {0,1}^{4x8}, full row rank, no zero column, y=(4,4,2,2)
  - |F_{A,y}| = 55, fibre dimension 4
  - a unimodular partition PLB whose uniform-target SLEM gap = 0.119828
  - exactly two of the four PLB moves carry a nonzero in the bottleneck coordinate
  - deleting those two moves splits the x_{j*}=0 stratum into K=3 components
  - Poisson gaps ~6.5e-2 at theta5=1 and ~1.3e-3 at theta5=0.01
"""
import numpy as np
from itertools import combinations, product as iproduct

y = np.array([4, 4, 2, 2])

cols = [np.array(c) for c in iproduct([0, 1], repeat=4) if any(c)]  # 15 candidates

def enumerate_fibre(A):
    """All x >= 0 integer with A x = y (columns processed with residual pruning)."""
    r = A.shape[1]
    out = []
    def rec(j, res, x):
        if j == r:
            if not res.any():
                out.append(tuple(x))
            return
        a = A[:, j]
        ub = min(res[k] // a[k] if a[k] else 10**9 for k in range(4))
        # feasibility prune: remaining columns must be able to zero the residual
        for v in range(ub + 1):
            x.append(v)
            rec(j + 1, res - v * a, x)
            x.pop()
            if len(out) > 200:  # cap
                return
    rec(0, y.copy(), [])
    return out

def ray_gibbs_kernel(states, logw, moves):
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))
    live = np.isfinite(logw)
    for i, s in enumerate(states):
        if not live[i]:
            continue
        sv = np.array(s)
        for u in moves:
            ray = [i]
            for sgn in (1, -1):
                b = sgn
                while True:
                    t = tuple(sv + b * u)
                    if t in idx and live[idx[t]]:
                        ray.append(idx[t]); b += sgn
                    else:
                        break
            lw = logw[ray]; lw = lw - lw.max()
            p = np.exp(lw); p /= p.sum()
            for j, pj in zip(ray, p):
                P[i, j] += pj / len(moves)
    return P

def slem_gap(P, logw):
    lw = logw - logw.max()
    pi = np.exp(lw); pi /= pi.sum()
    s = np.sqrt(pi)
    S = (s[:, None] * P) / s[None, :]
    S = 0.5 * (S + S.T)
    ev = np.sort(np.abs(np.linalg.eigvalsh(S)))[::-1]
    return 1.0 - ev[1]

# ---- stage 1: fibre-size filter ---------------------------------------
survivors = []
for combo in combinations(range(15), 8):
    A = np.column_stack([cols[k] for k in combo])
    if np.linalg.matrix_rank(A) < 4:
        continue
    F = enumerate_fibre(A)
    if len(F) == 55:
        survivors.append((combo, A, F))
print(f"stage 1: {len(survivors)} matrices with |F| = 55")

# ---- stage 2: PLB with uniform gap 0.119828 ---------------------------
TARGET = 0.119828
hits = []
for combo, A, F in survivors:
    logw0 = np.zeros(len(F))
    for S in combinations(range(8), 4):
        A1 = A[:, S]
        d = round(np.linalg.det(A1))
        if abs(d) != 1:
            continue
        A1inv = np.linalg.inv(A1)
        free = [j for j in range(8) if j not in S]
        moves = []
        ok = True
        for j in free:
            m = np.zeros(8)
            adj = -A1inv @ A[:, j]
            for k, sk in enumerate(S):
                m[sk] = adj[k]
            m[j] = 1.0
            mi = np.rint(m)
            if not np.allclose(m, mi, atol=1e-9):
                ok = False; break
            moves.append(mi.astype(int))
        if not ok:
            continue
        P = ray_gibbs_kernel(F, logw0, moves)
        g = slem_gap(P, logw0)
        if abs(g - TARGET) < 5e-6:
            hits.append((combo, S, A, F, moves, g))
print(f"stage 2: {len(hits)} (A, partition) pairs with uniform gap = {TARGET}")

# ---- stage 3: bottleneck structure + Poisson sweep --------------------
def log_pois(k, mu):
    from math import lgamma, log
    return k * log(mu) - mu - lgamma(k + 1)

seen = set()
for combo, S, A, F, moves, g in hits:
    key = tuple(sorted(combo))
    for jstar in range(8):
        touch = [k for k, u in enumerate(moves) if u[jstar] != 0]
        if len(touch) != 2:
            continue
        # lobes of D = {x_jstar = 0} under the other two moves
        D = [s for s in F if s[jstar] == 0]
        if not D:
            continue
        others = [moves[k] for k in range(4) if k not in touch]
        # union-find on D
        idxD = {s: i for i, s in enumerate(D)}
        parent = list(range(len(D)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for s in D:
            for u in others:
                t = tuple(np.array(s) + u)
                if t in idxD:
                    ra, rb = find(idxD[s]), find(idxD[t])
                    parent[ra] = rb
        K = len({find(i) for i in range(len(D))})
        if K != 3:
            continue
        # Poisson gap fingerprints
        def pois_gap(t5):
            mus = np.ones(8); mus[jstar] = t5
            lw = np.array([sum(log_pois(s[c], mus[c]) for c in range(8)) for s in F])
            return slem_gap(ray_gibbs_kernel(F, lw, moves), lw)
        g1, g001 = pois_gap(1.0), pois_gap(0.01)
        tag = (key, tuple(sorted(S)), jstar)
        if tag in seen:
            continue
        seen.add(tag)
        print(f"\nMATCH combo={combo} partition(A1 cols)={S} jstar={jstar}")
        print("A =\n", A)
        print("moves (columns of U in full coords):")
        for u in moves:
            print("  ", u)
        print(f"uniform gap {g:.6f} | Pois gap theta5=1: {g1:.4g} "
              f"(target 6.5e-2) | theta5=0.01: {g001:.4g} (target 1.3e-3) | K={K}")
