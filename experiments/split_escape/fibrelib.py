"""Exact-operator prototyping library for the reflective split sampler.

Fibres are enumerated; all kernels are assembled in closed form (or by exact
trajectory enumeration for the reflective proposal); SLEM by symmetric
eigendecomposition. Detailed balance is checked on every assembled kernel.
"""
import numpy as np
from math import lgamma, log
from itertools import product as iproduct

# ---------------------------------------------------------------- weights

def log_nb(k, mu, alpha):
    """log NB pmf with mean mu, dispersion alpha."""
    return (lgamma(alpha + k) - lgamma(alpha) - lgamma(k + 1)
            + k * (log(mu) - log(mu + alpha))
            + alpha * (log(alpha) - log(mu + alpha)))

def log_pois(k, mu):
    return k * log(mu) - mu - lgamma(k + 1)

# ---------------------------------------------------------------- fibres

def table_fibre():
    """2x3 table, margins (4,4)|(3,3,2); free coords (x12,x13); 10 states."""
    states = []
    for x12 in range(4):
        for x13 in range(3):
            if 1 <= x12 + x13 <= 4:
                states.append((x12, x13))
    return states

def table_full(s):
    x12, x13 = s
    return (4 - x12 - x13, x12, x13, x12 + x13 - 1, 3 - x12, 2 - x13)

TABLE_CONSTRAINTS = [  # a . x + c >= 0 in free coords (x12,x13)
    (np.array([1, 0]), 0),    # x12 >= 0
    (np.array([0, 1]), 0),    # x13 >= 0
    (np.array([-1, 0]), 3),   # x12 <= 3   (x22 >= 0)
    (np.array([0, -1]), 2),   # x13 <= 2   (x23 >= 0)
    (np.array([1, 1]), -1),   # x12+x13 >= 1  (x21 >= 0)
    (np.array([-1, -1]), 4),  # x12+x13 <= 4  (x11 >= 0)
]

def table_logw(mu_star, alpha=1.9, mu_other=1.0, model="nb"):
    f = log_nb if model == "nb" else (lambda k, m, a=None: log_pois(k, m))
    out = []
    for s in table_fibre():
        c = table_full(s)
        lw = (f(c[0], mu_star, alpha) if model == "nb" else log_pois(c[0], mu_star))
        for cell in c[1:]:
            lw += (f(cell, mu_other, alpha) if model == "nb" else log_pois(cell, mu_other))
        out.append(lw)
    return np.array(out)

def band_fibre(w, C):
    """Diagonal band {(s,t)>=0 : |s-t|<=w, s+t<=C}."""
    return [(s, t) for s in range(C + 1) for t in range(C + 1)
            if abs(s - t) <= w and s + t <= C]

def band_constraints(w, C):
    return [
        (np.array([1, 0]), 0),
        (np.array([0, 1]), 0),
        (np.array([-1, 1]), w),   # s - t <= w
        (np.array([1, -1]), w),   # t - s <= w
        (np.array([-1, -1]), C),  # s + t <= C
    ]

def axis_defect(states):
    pts = np.array(states, float)
    # diameter
    L = 0.0
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        L = max(L, d.max())
    # longest axis-aligned chord
    a = 1
    for j in range(pts.shape[1]):
        others = [k for k in range(pts.shape[1]) if k != j]
        from collections import defaultdict
        by = defaultdict(list)
        for p in pts:
            by[tuple(p[o] for o in others)].append(p[j])
        for v in by.values():
            a = max(a, int(max(v) - min(v)))
    return L / a

# ---------------------------------------------------------------- kernels

def gibbs_ray_kernel(states, logw, dirs):
    """Single-ray heat-bath hit-and-run for target ∝ exp(logw) (support logw>-inf)."""
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))
    live = np.isfinite(logw)
    for i, s in enumerate(states):
        if not live[i]:
            continue
        for d in dirs:
            # enumerate ray states in support
            ray = [i]
            for sgn in (1, -1):
                b = sgn
                while True:
                    t = tuple(np.array(s) + b * np.array(d))
                    if t in idx and live[idx[t]]:
                        ray.append(idx[t]); b += sgn
                    else:
                        break
            lw = logw[ray]; lw = lw - lw.max()
            p = np.exp(lw); p /= p.sum()
            for j, pj in zip(ray, p):
                P[i, j] += pj / len(dirs)
    return P

def _walldist(x, d, cons):
    b = None
    for a, c in cons:
        ad = int(a @ d)
        if ad < 0:
            m = (int(a @ x) + c) // (-ad)
            b = m if b is None else min(b, m)
    return 10**9 if b is None else b

def _binding(x_end, d, cons):
    for k, (a, c) in enumerate(cons):
        if int(a @ d) < 0 and int(a @ (x_end + d)) + c < 0:
            return k
    return None

def _reflect(d, a):
    aa = int(a @ a)
    num = 2 * int(a @ d)
    if num % aa != 0:
        return None  # non-lattice specular (never for {0,+-1} normals here)
    return d - (num // aa) * a

def _traj(x0, d1, B, cons):
    """Run B bounces from x0 with initial dir d1. Returns (xB, dT, bmaxT, walls) or None."""
    x = np.array(x0); d = np.array(d1); walls = []
    for _ in range(B):
        bm = _walldist(x, d, cons)
        if bm == 0:
            return None
        x = x + bm * d
        k = _binding(x, d, cons)
        if k is None:
            return None
        a = cons[k][0]
        d2 = _reflect(d, a)
        if d2 is None:
            return None
        # invertibility guard
        back = _reflect(-d2, a)
        if back is None or not np.array_equal(back, -d):
            return None
        walls.append((tuple(x), k))
        d = d2
    bT = _walldist(x, d, cons)
    if bT == 0:
        return None
    return x, d, bT, walls

def reflective_kernel(states, logw, cons, Bmax, dim):
    """Reflective lattice proposal (Algorithm 1) Metropolised for target exp(logw).
    Exact assembly by trajectory enumeration, with the reverse-retrace check."""
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))
    live = np.isfinite(logw)
    dirs = []
    for j in range(dim):
        e = np.zeros(dim, int); e[j] = 1
        dirs += [e.copy(), -e.copy()]
    pdir = 1.0 / len(dirs); pB = 1.0 / (Bmax + 1)
    for i, s in enumerate(states):
        if not live[i]:
            P[i, i] = 1.0
            continue
        x0 = np.array(s)
        acc_out = 0.0
        for d1 in dirs:
            for B in range(Bmax + 1):
                fw = _traj(x0, d1, B, cons)
                if fw is None:
                    continue  # abort -> hold (mass stays on diagonal)
                xB, dT, bT, walls = fw
                for b in range(1, bT + 1):
                    cand = xB + b * dT
                    tc = tuple(cand)
                    if tc not in idx:
                        continue  # off-lattice/should not happen; hold
                    j = idx[tc]
                    fwd = pdir * pB * (1.0 / bT)
                    # reverse pass
                    rev = _traj(cand, -dT, B, cons)
                    rp = 0.0
                    if rev is not None:
                        yB, dTr, bTr, wallsr = rev
                        # retrace: reverse bounce points must mirror forward's,
                        # reverse terminal dir must be -d1 and reach x0
                        ok = np.array_equal(dTr, -d1)
                        if ok and B > 0:
                            fpts = [wpt for wpt, _ in walls]
                            rpts = [wpt for wpt, _ in wallsr]
                            ok = (rpts == fpts[::-1])
                        if ok:
                            # distance from yB to x0 along dTr
                            diff = np.array(x0) - yB
                            # must be a positive multiple of dTr within range
                            nz = np.nonzero(dTr)[0]
                            k0 = nz[0]
                            if dTr[k0] != 0 and diff[k0] % dTr[k0] == 0:
                                bb = diff[k0] // dTr[k0]
                                if bb >= 1 and bb <= bTr and np.array_equal(bb * dTr, diff):
                                    rp = pdir * pB * (1.0 / bTr)
                    if rp == 0.0 or not live[j]:
                        a = 0.0
                    else:
                        r = np.exp(min(0.0, logw[j] - logw[i])) if logw[j] - logw[i] < 0 else 1.0
                        # full MH ratio: f(j) rev / f(i) fwd
                        ratio = np.exp(logw[j] - logw[i]) * (rp / fwd)
                        a = min(1.0, ratio)
                    P[i, j] += fwd * a
                    acc_out += fwd * a if j != i else 0.0
        P[i, i] += 1.0 - P[i].sum()
    return P

# ---------------------------------------------------------------- ladders

def ladder_kernel(states, rung_logw, fibre_kernels):
    """Joint chain on (rung, state). Step: w.p. 1/2 fibre move at current rung
    (kernel fibre_kernels[l]), w.p. 1/2 level move l->l+-1 Metropolis with exact
    normalisers (equalised level marginals). Returns (P, pi, labels)."""
    L = len(rung_logw); n = len(states)
    Z = []
    for lw in rung_logw:
        m = np.max(lw[np.isfinite(lw)])
        Z.append(m + log(np.sum(np.exp(lw[np.isfinite(lw)] - m))))
    # joint support
    labels = [(l, i) for l in range(L) for i in range(n) if np.isfinite(rung_logw[l][i])]
    pos = {lab: k for k, lab in enumerate(labels)}
    N = len(labels)
    P = np.zeros((N, N))
    pi = np.zeros(N)
    for k, (l, i) in enumerate(labels):
        pi[k] = np.exp(rung_logw[l][i] - Z[l]) / L
        # fibre arm
        row = fibre_kernels[l][i]
        for j in np.nonzero(row)[0]:
            P[k, pos[(l, j)]] += 0.5 * row[j]
        # level arm
        for dl in (-1, 1):
            l2 = l + dl
            if l2 < 0 or l2 >= L:
                P[k, k] += 0.25
                continue
            if np.isfinite(rung_logw[l2][i]):
                a = min(1.0, np.exp((rung_logw[l2][i] - Z[l2]) - (rung_logw[l][i] - Z[l])))
            else:
                a = 0.0
            if a > 0 and (l2, i) in pos:
                P[k, pos[(l2, i)]] += 0.25 * a
            P[k, k] += 0.25 * (1 - a)
    return P, pi / pi.sum(), labels

# ---------------------------------------------------------------- spectra

def slem_gap(P, pi):
    """1 - SLEM via symmetric similarity; also returns max detailed-balance violation."""
    s = np.sqrt(pi)
    S = (s[:, None] * P) / s[None, :]
    db = np.abs(S - S.T).max()
    S = 0.5 * (S + S.T)
    ev = np.linalg.eigvalsh(S)
    ev = np.sort(np.abs(ev))[::-1]
    return 1.0 - ev[1], db

def fibre_gap(P, logw):
    lw = logw - logw.max()
    pi = np.exp(lw); pi /= pi.sum()
    return slem_gap(P, pi)
