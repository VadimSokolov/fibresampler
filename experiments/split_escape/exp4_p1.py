"""Experiment 4: (P1) = Hazelton et al. (2024) Ex 2 / Ex 11, reconstructed from
the manuscript's fingerprints (see reconstruct_p1.py) and canonicalised so the
bottleneck route is coordinate 5, means theta = (1,1,1,1,theta5,1,1,1).

Gates (from the manuscript):
  |F| = 55; uniform-target gap 0.119828; Poisson gap 6.5e-2 -> 1.3e-3 over
  theta5 in [0.01, 1]; K = 3 lobes; three-rung tempering floor 0.0094 (alpha=1.9).
New measurements: split-ladder dichotomy and floors on (P1).
"""
import numpy as np
from math import lgamma, log
from itertools import combinations
from fibrelib import ladder_kernel, slem_gap

# ---- canonical Ex 2 ---------------------------------------------------
# links x routes; bottleneck route (uses link 4 only) at index 4
A = np.array([
    # r1 r2 r3 r4 r5* r6 r7 r8
    [0, 0, 0, 1,  0,  1, 1, 1],   # link 1
    [1, 1, 0, 0,  0,  1, 1, 0],   # link 2  (routes reordered below)
]).tolist()
# build explicitly from the reconstruction's column set, canonical order:
c = {'0001':(0,0,0,1),'0010':(0,0,1,0),'0100':(0,1,0,0),'0101':(0,1,0,1),
     '1000':(1,0,0,0),'1010':(1,0,1,0),'1100':(1,1,0,0),'1101':(1,1,0,1)}
order = ['0010','0100','0101','1000','0001','1010','1100','1101']  # bottleneck '0001' at idx 4
A = np.array([c[k] for k in order]).T           # 4 x 8
y = np.array([4, 4, 2, 2])
jstar = 4
S = (0, 1, 3, 4)                                # A1 columns (unimodular)

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
d = round(np.linalg.det(A1)); assert abs(d) == 1, d
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
print("PLB moves:")
for m in moves:
    print("  ", m)
touch = [k for k, m in enumerate(moves) if m[jstar] != 0]
print(f"moves touching bottleneck coordinate {jstar+1}: {[k+1 for k in touch]} (gate: two moves)")

# ---- kernels on full coordinates -------------------------------------
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

# ---- gate: uniform + Poisson sweep -----------------------------------
gu = gap_of(np.zeros(len(F)))
print(f"\nuniform-target gap = {gu:.6f}  (gate: 0.119828)")
print("\nPoisson sweep (Prediction 1 replication):")
for t5 in [1, 0.5, 0.2, 0.1, 0.05, 0.01]:
    mus = np.ones(8); mus[jstar] = t5
    lw = np.array([sum(log_pois(s[k], mus[k]) for k in range(8)) for s in F])
    print(f"  theta5 = {t5:5g}: gap = {gap_of(lw):.4g}")

# ---- gate: lobes ------------------------------------------------------
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
print(f"\nlobes of D under u_(jstar)=0 moves: K = {len(comps)} "
      f"(gate: 3), sizes {sorted(len(v) for v in comps.values())}, |D| = {len(D)}")

# ---- ladders (alpha = 1.9) -------------------------------------------
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
print("\n(P1) ladder suite, alpha = 1.9:")
print("mu5      " + "".join(f"{k:>11s}" for k in rows))
for t, mu5 in enumerate(mus5):
    print(f"{mu5:8.0e} " + "".join(f"{rows[k][t]:11.4g}" for k in rows))
x = np.log(mus5[-3:])
print("\nterminal log-log slopes (last 3 points):")
for k, v in rows.items():
    print(f"  {k:10s}: {np.polyfit(x, np.log(np.array(v[-3:])), 1)[0]:+.3f}")

# corridor content of the thresholded rungs at smallest mu
lwn = nb_logw(mus5[-1]); lwn -= lwn.max()
x5 = np.array([s[jstar] for s in F])
for name, expo in [("S-keep", 1.5), ("S-excl", 0.5)]:
    mask = lwn >= expo * np.log(mus5[-1])
    print(f"{name}: bottom rung keeps x5 values {sorted(set(x5[mask]))} "
          f"({mask.sum()}/{len(F)} states)")
