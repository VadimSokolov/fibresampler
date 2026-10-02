"""Experiment 4, CANONICAL check: run the split-escape suite on the PAPER's
own A_P1 (fibresampler/problems.py), not Nick's fingerprint reconstruction.

Paper's A_P1 (links x routes), Y_P1=(4,4,2,2), bottleneck theta5 = column 4
(0-indexed; route uses link 3 only), A1 = P1_COLS1 = columns {0,4,5,6}.

If |F|, uniform gap, Poisson sweep, K, tempering floor and the split floors all
match Nick's reconstruction, the two are the same network up to relabelling and
the split-escape floors hold on the Martin-validated matrix.
"""
import numpy as np
from math import lgamma, log
from fibrelib import ladder_kernel, slem_gap

# ---- PAPER's canonical Ex 2 (fibresampler/problems.py) ----------------
A = np.array([
    [1, 1, 1, 0, 0, 0, 1, 0],   # link 1
    [0, 1, 1, 0, 0, 1, 0, 1],   # link 2
    [0, 0, 1, 0, 1, 0, 0, 1],   # link 3
    [0, 0, 0, 1, 0, 0, 1, 0],   # link 4
])
y = np.array([4, 4, 2, 2])
jstar = 4                       # theta5 (0-indexed col 4, route {link 3})
S = (0, 4, 5, 6)                # P1_COLS1, A1 columns

# ---- fibre ------------------------------------------------------------
def enumerate_fibre(A, y):
    out = []
    def rec(j, res, x):
        if j == A.shape[1]:
            if not res.any():
                out.append(tuple(x))
            return
        a = A[:, j]
        ub = min(res[k] // a[k] if a[k] else 10**9 for k in range(len(y)))
        for v in range(ub + 1):
            x.append(v); rec(j + 1, res - v * a, x); x.pop()
    rec(0, y.copy(), [])
    return out

F = enumerate_fibre(A, y)
print(f"|F| = {len(F)}  (gate: 55)")

# ---- PLB moves --------------------------------------------------------
A1 = A[:, list(S)]
d = round(np.linalg.det(A1)); print(f"det(A1) = {d}  (gate: |det|=1)"); assert abs(d) == 1, d
A1inv = np.linalg.inv(A1)
free = [j for j in range(8) if j not in S]
moves = []
for j in free:
    m = np.zeros(8)
    adj = -A1inv @ A[:, j]
    for k, sk in enumerate(S):
        m[sk] = adj[k]
    m[j] = 1.0
    moves.append(np.rint(m).astype(int))
moves = [tuple(m) for m in moves]
touch = [k for k, m in enumerate(moves) if m[jstar] != 0]
print(f"moves touching bottleneck coord {jstar+1}: {[k+1 for k in touch]} (gate: two)")

# ---- kernels ----------------------------------------------------------
def ray_gibbs(states, logw, mvs):
    idx = {s: i for i, s in enumerate(states)}
    n = len(states); P = np.zeros((n, n))
    live = np.isfinite(logw)
    for i, s in enumerate(states):
        if not live[i]:
            P[i, i] = 1.0; continue
        sv = np.array(s)
        for u in mvs:
            ray = [i]
            for sgn in (1, -1):
                b = sgn
                while True:
                    t = tuple(sv + b * np.array(u))
                    if t in idx and live[idx[t]]:
                        ray.append(idx[t]); b += sgn
                    else:
                        break
            lw = logw[ray]; lw = lw - lw.max()
            p = np.exp(lw); p /= p.sum()
            for jj, pj in zip(ray, p):
                P[i, jj] += pj / len(mvs)
    return P

def gap_of(logw):
    P = ray_gibbs(F, logw, moves)
    lw = logw - logw.max()
    pi = np.exp(lw); pi /= pi.sum()
    return slem_gap(P, pi)[0]

def log_pois(k, mu): return k * log(mu) - mu - lgamma(k + 1)
def log_nb(k, mu, a):
    return (lgamma(a + k) - lgamma(a) - lgamma(k + 1)
            + k * (log(mu) - log(mu + a)) + a * (log(a) - log(mu + a)))

gu = gap_of(np.zeros(len(F)))
print(f"\nuniform-target gap = {gu:.6f}  (gate: 0.119828)")
print("Poisson sweep (Prediction 1):")
for t5 in [1, 0.5, 0.2, 0.1, 0.05, 0.01]:
    mus = np.ones(8); mus[jstar] = t5
    lw = np.array([sum(log_pois(s[k], mus[k]) for k in range(8)) for s in F])
    print(f"  theta5={t5:5g}: gap={gap_of(lw):.4g}")

# ---- lobes ------------------------------------------------------------
D = [s for s in F if s[jstar] == 0]
others = [np.array(moves[k]) for k in range(4) if k not in touch]
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
            parent[find(idxD[s])] = find(idxD[t])
comps = {}
for s in D:
    comps.setdefault(find(idxD[s]), []).append(s)
print(f"lobes K = {len(comps)} (gate 3), sizes {sorted(len(v) for v in comps.values())}, |D|={len(D)}")

# ---- ladder suite -----------------------------------------------------
alpha = 1.9
def nb_logw(mu5):
    mus = np.ones(8); mus[jstar] = mu5
    return np.array([sum(log_nb(s[k], mus[k], alpha) for k in range(8)) for s in F])
def unif(mask): return np.where(mask, 0.0, -np.inf)

mus5 = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
rows = {k: [] for k in ["SR-NB", "T3-temper", "S2-split", "S3-split", "S-keep", "S-excl"]}
dbmax = 0.0
for mu5 in mus5:
    lw = nb_logw(mu5); lwn = lw - lw.max()
    rows["SR-NB"].append(gap_of(lw))
    ladders = {
        "T3-temper": [lw, 0.5 * lw, np.zeros_like(lw)],
        "S2-split":  [lw, np.zeros_like(lw)],
        "S3-split":  [lw, unif(lwn >= 0.5 * (lwn.max() + lwn.min())), np.zeros_like(lw)],
        "S-keep":    [lw, unif(lwn >= 1.5 * np.log(mu5))],
        "S-excl":    [lw, unif(lwn >= 0.5 * np.log(mu5))],
    }
    for name, rung in ladders.items():
        ker = [ray_gibbs(F, r, moves) for r in rung]
        P, pi, _ = ladder_kernel(F, rung, ker)
        g, db = slem_gap(P, pi)
        rows[name].append(g); dbmax = max(dbmax, db)

print(f"\nmax detailed-balance violation: {dbmax:.2e}")
print("(P1 canonical) ladder suite, alpha = 1.9:")
print("mu5      " + "".join(f"{k:>11s}" for k in rows))
for t, mu5 in enumerate(mus5):
    print(f"{mu5:8.0e} " + "".join(f"{rows[k][t]:11.4g}" for k in rows))
print("\nGATES vs reconstruction: split S2 floor ~0.0165, S3 floor ~0.0182, "
      "tempering ~0.00943, S-keep ~0.0097, S-excl collapses (+1.000)")
