"""How hard is simulated tempering to tune?  (Martin Hazelton's second point.)

Linus Fromm coded the simulated-tempering (ST) construction of the paper and found it has plenty of
tuning parameters that were hard to set.  This script quantifies that with EXACT computation on P1
(fibre size 55): which knobs change the exponent of the loading obstruction, which only change a
constant, and how much each can be mis-set before the efficiency is lost.

Chain (identical to experiments/taskV_tempering.py when q=1/2, exact weights, linear ladder):
  state (x, level l), K = L+1 rungs, beta_0 = 1 > ... > beta_{K-1}, target pi_beta(x) ~ pi_NB(x)^beta;
  one iteration = a fibre update w.p. 1-q (single-ray Gibbs at the current rung) or, w.p. q, a
  nearest-neighbour level proposal accepted by Metropolis with the pseudo-prior weights
  exp(-logZhat_l).  With misspecified logZhat = logZ + eps the joint law is
  pi(x, l) ~ pi_{beta_l}(x) exp(-eps_l): the level marginal moves, x | level is unchanged, and the
  estimator that uses only level-0 visits stays consistent.

Metrics, both exact (symmetrised eigendecomposition of the joint chain):
  gap    1 - SLEM of the joint chain (gap1 = 1 - lambda_2 is also printed where it differs);
  ESS    for the actual level-0 ratio estimator of E[x_j]: h(x,l) = 1{l=0}(x_j - theta_j)/P(l=0),
         sigma^2 = sum_{k>=2} (1+lam_k)/(1-lam_k) <h, v_k>^2, ESS per iteration = Var(x_j)/sigma^2.
Everything is reported as a ratio to single-rung SR-NB (K=1, same mu) per ITERATION (a fibre update
or a level move), and per FIBRE UPDATE (= per iteration / (1-q)).  "x5" is the vanishing-mean cell
(index 4).  "worst8" is the WORST-COORDINATE ESS (min over the 8 route counts) as a ratio to the same
quantity for SR-NB; the binding coordinates are x14 and x24 (indices 2 and 7), NOT x5, whose own
estimator is fast under SR-NB.  "worst7" is the same quantity excluding x5.

Parts (all run by default, about 8 minutes on a laptop):
  gates  five validation gates (taskV agreement, K=1 = SR-NB, reversibility incl. misspecified weights, AV vs exact_iat,
         Monte Carlo ST chain vs the exact asymptotic variance)
  A      number of rungs K (and the paper's scaled rule K* = ceil(log2 1/mu) + 1)
  B      ladder spacing at equal K: linear, geometric, quadratic, equal-acceptance
  C      beta_min (the only knob that moves the exponent in mu): gap and ESS slopes
  D      weight misspecification: iid errors, systematic drift, and the single weight of the two-rung ladder
  E      level-move probability q under three cost models
  F      weights from a pilot of iid exact draws (best case); F2 weights from an MCMC pilot (single-ray Gibbs per rung)
  G      parallel tempering: exact for K=2,3 (sparse, 55^K states) and Monte Carlo for K up to 24, against ideal ST
  H      Poisson (not NB) check of the headline claims
  R      recipe grid (K x spacing x q x cost model): fraction of the best configuration reached by candidate defaults
  I      does the overlap rule for K transfer to a more concentrated target?
  S      headline ratios derived from A and B (K=2 against K=3 and the scaled rule, efficiency kept as K grows, spacing gains)
Run:  python3 experiments/refQ2_tempering_tuning.py [--parts gates,A,...] [--out PATH]
Output: results/refQ2_tempering_tuning.txt   (deterministic, seeded with SEED)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import taskV_tempering as tv                                                  # noqa: E402
from fibresampler.problems import p1_poisson_theta                            # noqa: E402
from fibresampler.spectral import (log_weights, normalised_pi,                # noqa: E402
                                   sr_transition_matrix)
from pred5_ess import exact_iat, iat_geyer                                    # noqa: E402

SEED = 20261002
PROB = tv.PROB
N = PROB.fibre_size
M = PROB.n_moves
X = PROB.states.astype(float)          # (N, 8) route counts
RARE = 4                               # index of the vanishing-mean cell x5 (theta_5 = mu)
RAYS = [[np.array(PROB.ray(i, j)) for j in range(M)] for i in range(N)]
RMAX = max(len(r) for row in RAYS for r in row)
OUT_DEFAULT = os.path.join(REPO, "results", "refQ2_tempering_tuning.txt")
NONRARE = [j for j in range(8) if j != RARE]


# =============================================================================================
# targets, chains, exact analysis
# =============================================================================================
def sr_block(lwb):
    """Single-ray Gibbs matrix for log-weights lwb (same as spectral.sr_transition_matrix)."""
    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(M):
            r = RAYS[i][j]
            w = np.exp(lwb[r] - lwb[r].max())
            Q[i, r] += w / (w.sum() * M)
    return Q


class Target:
    """Base target pi (beta = 1) on the enumerated fibre: NB (alpha=1.9) or Poisson, theta_5 = mu."""

    def __init__(self, kind, mu, theta=None):
        self.kind, self.mu = kind, float(mu)
        if theta is not None:
            th = np.array(theta, float)
            th[RARE] = mu
            self.lw = log_weights(PROB, "poisson", th)
        else:
            self.lw = tv.nb_logw(mu) if kind == "nb" else \
                log_weights(PROB, "poisson", p1_poisson_theta(mu))
        self.pi = normalised_pi(self.lw)
        self.mean = self.pi @ X
        self.var = self.pi @ (X - self.mean) ** 2
        self._blk = {}
        self._sr = None

    def block(self, beta):
        b = float(beta)
        if b not in self._blk:
            self._blk[b] = sr_block(b * self.lw)
        return self._blk[b]

    def logZ(self, betas):
        return np.array([logsumexp(b * self.lw) for b in betas])

    def joint(self, betas, q=0.5, eps=None):
        """Dense joint transition matrix on (level, x), index = level * N + x, and its stationary law."""
        betas = np.asarray(betas, float)
        K = len(betas)
        lzu = self.logZ(betas)
        if eps is not None:
            lzu = lzu + np.asarray(eps, float)
        logpi = betas[:, None] * self.lw[None, :] - lzu[:, None]
        logpi -= logsumexp(logpi)
        pi = np.exp(logpi).ravel()
        if K == 1:
            return self.block(betas[0]), pi, lzu
        P = np.zeros((K * N, K * N))
        ar = np.arange(N)
        for e in range(K):
            sl = slice(e * N, (e + 1) * N)
            P[sl, sl] += (1.0 - q) * self.block(betas[e])
            for ep in (e - 1, e + 1):
                if 0 <= ep < K:
                    dl = (betas[ep] - betas[e]) * self.lw - (lzu[ep] - lzu[e])
                    A = np.exp(np.minimum(0.0, dl))
                    P[e * N + ar, ep * N + ar] += 0.5 * q * A
                    P[e * N + ar, e * N + ar] += 0.5 * q * (1.0 - A)
                else:
                    P[sl, sl] += 0.5 * q * np.eye(N)
        return P, pi, lzu

    def sr(self):
        """Single-rung SR-NB baseline analysis (cached)."""
        if self._sr is None:
            P, pi, _ = self.joint(np.array([1.0]))
            H, _ = h_matrix(self, pi, 1)
            gap, gap1, sig2, _ = analyse(P, pi, H)
            self._sr = dict(gap=gap, gap1=gap1, sig2=sig2, ess=self.var / sig2)
        return self._sr


def h_matrix(tgt, pi, K):
    """Influence functions of the level-0 ratio estimator for the 8 route counts."""
    p0 = float(pi[:N].sum())
    H = np.zeros((K * N, 8))
    H[:N, :] = (X - tgt.mean) / p0
    return H, p0


def analyse(P, pi, H):
    """gap, one-sided gap and exact asymptotic variances of the mean-zero columns of H."""
    d = np.sqrt(pi)
    S = (d[:, None] * P) / d[None, :]
    S = 0.5 * (S + S.T)
    lam, V = np.linalg.eigh(S)
    coef = V[:, :-1].T @ (d[:, None] * H)
    w = (1.0 + lam[:-1]) / (1.0 - lam[:-1])
    sig2 = w @ coef ** 2
    gap = 1.0 - max(abs(lam[-2]), abs(lam[0]))
    gap1 = 1.0 - lam[-2]
    return gap, gap1, sig2, lam


def metrics(tgt, betas, q=0.5, eps=None):
    betas = np.asarray(betas, float)
    P, pi, _ = tgt.joint(betas, q, eps)
    K = len(betas)
    H, p0 = h_matrix(tgt, pi, K)
    gap, gap1, sig2, _ = analyse(P, pi, H)
    sr = tgt.sr()
    ess = tgt.var / sig2
    R = ess / sr["ess"]                                      # per-coordinate ratio to SR-NB, per iteration
    fu = 1.0 / (1.0 - q) if K > 1 else 1.0
    # Rw: ratio of the WORST-coordinate ESS (the binding coordinate, not the worst ratio)
    return dict(K=K, q=q, gap=gap, gap1=gap1, p0=p0, ess=ess, R=R, R5=R[RARE],
                Rw=ess.min() / sr["ess"].min(), jw=int(ess.argmin()), fu=fu, essmin=ess.min())


def ladder_linear(K, bmin=0.0):
    if K == 1:
        return np.array([1.0])
    return 1.0 - (1.0 - bmin) * np.arange(K) / (K - 1)


def ladder_geom(K):
    """Geometric in beta from 1 down to the linear ladder's smallest positive rung 1/(K-1), then 0."""
    if K <= 2:
        return ladder_linear(K)
    return np.concatenate([(K - 1.0) ** (-np.arange(K - 1) / (K - 2.0)), [0.0]])


def ladder_quad(K):
    return ladder_linear(K) ** 2


def overlap(tgt, b1, b2):
    """sum_x min(pi_b1, pi_b2): the mean Metropolis acceptance of the level move between b1 and b2."""
    return float(np.minimum(normalised_pi(b1 * tgt.lw), normalised_pi(b2 * tgt.lw)).sum())


def _next_rung(tgt, b, a):
    if overlap(tgt, b, 0.0) >= a:
        return 0.0
    lo, hi = 0.0, b
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if overlap(tgt, b, mid) >= a:
            hi = mid
        else:
            lo = mid
    return hi


def ladder_eqacc(tgt, K):
    """Greedy equal-acceptance ladder: K rungs from 1 to 0 with all adjacent overlaps (nearly) equal."""
    if K <= 2:
        return ladder_linear(K)

    def build(a):
        b = [1.0]
        for _ in range(K - 2):
            b.append(_next_rung(tgt, b[-1], a))
        return b

    lo, hi = 0.0, 1.0 - 1e-9
    for _ in range(40):
        a = 0.5 * (lo + hi)
        b = build(a)
        g = 1.0 if b[-1] == 0.0 else overlap(tgt, b[-1], 0.0)
        if a - g > 0:
            hi = a
        else:
            lo = a
    b = build(0.5 * (lo + hi))
    out = np.array(b + [0.0])
    assert np.all(np.diff(out) < 0), "degenerate equal-acceptance ladder"
    return out


def scaledK(mu):
    """K = L + 1 with the paper's scaled rule L = ceil(log2(1/mu))."""
    return 1 + max(1, int(np.ceil(np.log2(1.0 / mu)))) if mu < 1 else 2


def fmt(x, w=9, p=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return f"{'n/a':>{w}}"
    return f"{x:>{w}.{p}g}"


def slope(mus, ys):
    return float(np.polyfit(np.log(mus), np.log(ys), 1)[0])


def lslope(mus, ys):
    return float((np.log(ys[-1]) - np.log(ys[-2])) / (np.log(mus[-1]) - np.log(mus[-2])))


# =============================================================================================
# numba samplers (validation of the exact ST analysis, and parallel tempering)
# =============================================================================================
def build_ray_tables(tgt, betas):
    """Per-rung, per-state, per-direction cumulative heat-bath probabilities, padded to RMAX."""
    K = len(betas)
    ridx = -np.ones((N, M, RMAX), np.int64)
    rlen = np.zeros((N, M), np.int64)
    for i in range(N):
        for j in range(M):
            r = RAYS[i][j]
            ridx[i, j, :len(r)] = r
            rlen[i, j] = len(r)
    cum = np.ones((K, N, M, RMAX))
    for e, b in enumerate(betas):
        lwb = b * tgt.lw
        for i in range(N):
            for j in range(M):
                r = RAYS[i][j]
                w = np.exp(lwb[r] - lwb[r].max())
                cum[e, i, j, :len(r)] = np.cumsum(w) / w.sum()
                cum[e, i, j, len(r) - 1] = 1.0
    return ridx, rlen, cum


@njit(cache=True)
def _st_sim(T, burn, K, betas, lw, logZ, q, rlen, ridx, cum, x0, seed):
    np.random.seed(seed)
    x = x0
    lev = 0
    xs = np.empty(T, np.uint8)
    ls = np.empty(T, np.uint8)
    m = ridx.shape[1]
    for t in range(T + burn):
        if np.random.random() < 1.0 - q:
            j = np.random.randint(0, m)
            v = np.random.random()
            k = 0
            while k < rlen[x, j] - 1 and cum[lev, x, j, k] < v:
                k += 1
            x = ridx[x, j, k]
        else:
            lp = lev + 1 if np.random.random() < 0.5 else lev - 1
            if lp >= 0 and lp < K:
                d = (betas[lp] - betas[lev]) * lw[x] - (logZ[lp] - logZ[lev])
                if d >= 0.0 or np.random.random() < np.exp(d):
                    lev = lp
        if t >= burn:
            xs[t - burn] = x
            ls[t - burn] = lev
    return xs, ls


@njit(cache=True)
def _pt_sim(T, burn, K, betas, lw, p_fibre, rlen, ridx, cum, x0, seed):
    np.random.seed(seed)
    x = np.full(K, x0, np.int64)
    tr = np.empty(T, np.uint8)
    m = ridx.shape[1]
    for t in range(T + burn):
        if np.random.random() < p_fibre:
            e = np.random.randint(0, K)
            j = np.random.randint(0, m)
            i = x[e]
            v = np.random.random()
            k = 0
            while k < rlen[i, j] - 1 and cum[e, i, j, k] < v:
                k += 1
            x[e] = ridx[i, j, k]
        else:
            e = np.random.randint(0, K - 1)
            d = (betas[e] - betas[e + 1]) * (lw[x[e + 1]] - lw[x[e]])
            if d >= 0.0 or np.random.random() < np.exp(d):
                tmp = x[e]
                x[e] = x[e + 1]
                x[e + 1] = tmp
        if t >= burn:
            tr[t - burn] = x[0]
    return tr


def batch_means_av(h, nb=500):
    n = (len(h) // nb) * nb
    bm = h[:n].reshape(nb, -1).mean(1)
    return (n // nb) * bm.var(ddof=1)


# =============================================================================================
# parts
# =============================================================================================
def part_gates(out):
    out.append("=" * 118)
    out.append("GATES (all must pass before any number below is used)")
    out.append("=" * 118)
    ok = True
    # G1: bit-level agreement with taskV at the scaled ladder, mu=1e-3, L=10, q=1/2, exact weights
    tg = Target("nb", 1e-3)
    g_new = metrics(tg, ladder_linear(11), 0.5)["gap"]
    g_old = float(tv.gap_of_tempered(1e-3, 10)[0])
    d = abs(g_new - g_old)
    out.append(f"G1  generalised builder vs taskV gap_of_tempered(1e-3, L=10): {g_new:.12f} vs {g_old:.12f}  "
               f"|diff|={d:.2e}  {'PASS' if d < 1e-10 else 'FAIL'}")
    ok &= d < 1e-10
    # G2: K=1 equals SR-NB, and the fast block builder equals spectral.sr_transition_matrix
    for mu in (1e-2, 1e-4):
        t = Target("nb", mu)
        dq = np.abs(t.block(1.0) - sr_transition_matrix(PROB, t.lw)).max()
        d2 = abs(t.sr()["gap"] - tv.sr_nb_gap(mu))
        out.append(f"G2  mu={mu:g}: max|block - sr_transition_matrix|={dq:.1e}, |gap(K=1) - SR-NB gap|={d2:.1e}  "
                   f"{'PASS' if dq < 1e-13 and d2 < 1e-10 else 'FAIL'}")
        ok &= dq < 1e-13 and d2 < 1e-10
    # G3: exact joint reversibility, exact and misspecified weights, several q
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for mu, K, q, sig in [(1e-3, 11, 0.5, 0.0), (1e-4, 15, 0.5, 1.0), (1e-4, 15, 0.1, 2.0),
                          (1e-6, 8, 0.9, 0.5), (1e-2, 4, 0.25, 1.0)]:
        t = Target("nb", mu)
        eps = sig * rng.standard_normal(K)
        P, pi, _ = t.joint(ladder_linear(K), q, eps)
        F = pi[:, None] * P
        db = np.abs(F - F.T).max() / F.max()
        lvl = pi.reshape(K, N).sum(1)
        pred = np.exp(-eps) / np.exp(-eps).sum()
        dl = np.abs(lvl - pred).max()
        worst = max(worst, db)
        out.append(f"G3  reversibility mu={mu:g} K={K} q={q} sigma={sig}: max relative DB violation={db:.1e}; "
                   f"level marginal vs exp(-eps)/sum: {dl:.1e}  {'PASS' if db < 1e-9 and dl < 1e-9 else 'FAIL'}")
        ok &= db < 1e-9 and dl < 1e-9
    # G4: exact AV vs pred5_ess.exact_iat for the single-rung chain
    t = Target("nb", 1e-2)
    P, pi, _ = t.joint(np.array([1.0]))
    iat_ref = np.array([exact_iat(P, pi, X[:, j]) for j in range(8)])
    iat_me = t.sr()["sig2"] / t.var
    d4 = np.abs(iat_me / iat_ref - 1).max()
    out.append(f"G4  SR-NB asymptotic variance vs pred5_ess.exact_iat (8 coordinates, mu=1e-2): "
               f"max relative difference {d4:.1e}  {'PASS' if d4 < 1e-8 else 'FAIL'}")
    ok &= d4 < 1e-8
    # G5: Monte Carlo ST chain vs the exact sigma^2 of the level-0 estimator (mu=1e-2, K=8, 3 seeds)
    mu, K, q = 1e-2, 8, 0.5
    t = Target("nb", mu)
    bet = ladder_linear(K)
    ridx, rlen, cum = build_ray_tables(t, bet)
    lz = t.logZ(bet)
    P, pi, _ = t.joint(bet, q)
    H, p0 = h_matrix(t, pi, K)
    _, _, sig2, _ = analyse(P, pi, H)
    av = []
    for s in range(3):
        xs, ls = _st_sim(8_000_000, 200_000, K, bet, t.lw, lz, q, rlen, ridx, cum,
                         int(np.argmax(t.lw)), SEED + s)
        h = np.where(ls == 0, X[xs.astype(np.int64), RARE] - t.mean[RARE], 0.0) / p0
        av.append(batch_means_av(h))
    av = np.array(av)
    ratio = av.mean() / sig2[RARE]
    out.append(f"G5  Monte Carlo ST (mu=1e-2, K=8, 3 x 8e6 its) batch-means AV of the x5 estimator "
               f"{av.mean():.4f} (range {av.min():.4f}-{av.max():.4f}) vs exact {sig2[RARE]:.4f}  "
               f"ratio {ratio:.3f}  {'PASS' if abs(ratio - 1) < 0.1 else 'FAIL'}")
    ok &= abs(ratio - 1) < 0.1
    out.append(f"GATES {'ALL PASS' if ok else 'FAILED'}")
    out.append("")
    return ok


def part_A(out):
    out.append("=" * 118)
    out.append("A. NUMBER OF RUNGS K (linear ladder 1 -> 0, q=1/2, exact weights); ratios to SR-NB, per iteration (x2 per fibre update)")
    out.append("   R5 = ESS(x5) ratio; Rw = worst-coordinate ESS ratio (jw = binding coordinate of the tempered chain); "
               "best = max over the listed K; ov = smallest adjacent overlap of the ladder (mean level-move acceptance)")
    out.append("=" * 118)
    Ks = [2, 3, 4, 6, 8, 12, 16, 24]
    for mu in (1.0, 1e-2, 1e-4, 1e-6, 1e-8):
        t = Target("nb", mu)
        sr = t.sr()
        Kl = sorted(set(Ks + [scaledK(mu)]))
        res = {K: metrics(t, ladder_linear(K)) for K in Kl}
        b5 = max(res[K]["R5"] for K in Ks)
        bw = max(res[K]["Rw"] for K in Ks)
        out.append("")
        out.append(f"mu={mu:g}  (paper's scaled rule K*={scaledK(mu)}, marked *)  SR-NB: gap={sr['gap']:.4g}, "
                   f"ESS/iter x5={sr['ess'][RARE]:.4g}, worst8={sr['ess'].min():.4g} (coordinate {int(sr['ess'].argmin())}); "
                   f"overlap(target, uniform)={overlap(t, 1.0, 0.0):.4f}")
        out.append(f"  {'K':>4} {'gap':>10} {'R5':>10} {'R5/best':>8} {'Rw':>10} {'jw':>3} {'Rw/best':>8} {'ov':>6}")
        for K in Kl:
            r = res[K]
            b = ladder_linear(K)
            ov = min(overlap(t, b[i], b[i + 1]) for i in range(K - 1))
            mark = "*" if K == scaledK(mu) else " "
            out.append(f"  {K:>3}{mark} {r['gap']:>10.4g} {r['R5']:>10.4g} {r['R5'] / b5:>8.3f} {r['Rw']:>10.4g} "
                       f"{r['jw']:>3} {r['Rw'] / bw:>8.3f} {ov:>6.3f}")
    t = Target("nb", 1e-4)
    out.append("")
    out.append("PER-COORDINATE ESS per iteration at mu=1e-4 (index: route count; 4 = x34 = the vanishing cell x5; the cells x14 and x24 "
               "(indices 2 and 7) share link 3 with it and are the binding coordinates)")
    out.append(f"  {'':>12} " + " ".join(f"{f'idx {j}':>10}" for j in range(8)))
    rows = [("SR-NB", t.sr()["ess"])] + [(f"ST K={K}", metrics(t, ladder_linear(K))["ess"]) for K in (2, 3, scaledK(1e-4))]
    for name, e in rows:
        out.append(f"  {name:>12} " + " ".join(f"{v:>10.3e}" for v in e))
    out.append("")


def part_B(out):
    out.append("=" * 118)
    out.append("B. LADDER SPACING at equal K (all end at beta=0; q=1/2, exact weights); ratios to SR-NB per iteration")
    out.append("   linear: 1-l/(K-1); geom: geometric from 1 to 1/(K-1) then 0; quad: (1-l/(K-1))^2; "
               "eqacc: greedy equal mean level-move acceptance (exact overlaps)")
    out.append("=" * 118)
    for mu in (1e-2, 1e-4, 1e-6):
        t = Target("nb", mu)
        for K in (4, 8, 15):
            lads = {"linear": ladder_linear(K), "geom": ladder_geom(K), "quad": ladder_quad(K),
                    "eqacc": ladder_eqacc(t, K)}
            res = {k: metrics(t, b) for k, b in lads.items()}
            b5 = max(r["R5"] for r in res.values())
            bw = max(r["Rw"] for r in res.values())
            out.append("")
            out.append(f"mu={mu:g} K={K}")
            out.append(f"  {'spacing':>8} {'gap':>10} {'R5':>10} {'R5/best':>8} {'Rw':>10} {'Rw/best':>8}  "
                       f"min adjacent overlap  rungs")
            for k, r in res.items():
                b = lads[k]
                ov = min(overlap(t, b[i], b[i + 1]) for i in range(K - 1))
                out.append(f"  {k:>8} {r['gap']:>10.4g} {r['R5']:>10.4g} {r['R5'] / b5:>8.3f} {r['Rw']:>10.4g} "
                           f"{r['Rw'] / bw:>8.3f}  {ov:>19.3f}  "
                           + (" ".join(f"{v:.3g}" for v in b) if k == "eqacc" else ""))
    out.append("")


def part_C(out):
    out.append("=" * 118)
    out.append("C. beta_min (the ladder 1 -> beta_min, linear, K fixed; q=1/2, exact weights) vs the exponent in mu")
    out.append("   slopes are d log(y) / d log(mu) over mu in {1e-2,...,1e-6} (all five points) and over the last pair "
               "(1e-5 -> 1e-6)")
    out.append("   E5 = ESS/iter of the x5 estimator, Ew = min over the 8 coordinates of ESS/iter, "
               "SR-NB slopes are the reference")
    out.append("=" * 118)
    mus = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    tg = {mu: Target("nb", mu) for mu in mus}
    sr = {mu: tg[mu].sr() for mu in mus}
    out.append("")
    out.append(f"SR-NB (K=1): gap " + " ".join(f"{sr[m]['gap']:.3e}" for m in mus)
               + f"  slope {slope(mus, [sr[m]['gap'] for m in mus]):+.3f}  last {lslope(mus, [sr[m]['gap'] for m in mus]):+.3f}")
    e5 = [sr[m]["ess"][RARE] for m in mus]
    ew = [sr[m]["ess"].min() for m in mus]
    out.append(f"             E5  slope {slope(mus, e5):+.3f}  last {lslope(mus, e5):+.3f};   "
               f"Ew slope {slope(mus, ew):+.3f}  last {lslope(mus, ew):+.3f}")
    for K in (12, 24):
        out.append("")
        out.append(f"K = {K}")
        out.append(f"  {'beta_min':>8} | {'gap at mu=1e-2..1e-6':>56} | {'gap slope':>9} {'last':>7} | "
                   f"{'E5 slope':>8} {'last':>7} | {'Ew slope':>8} {'last':>7}")
        for bm in (0.0, 0.05, 0.1, 0.25, 0.5):
            rs = [metrics(tg[mu], ladder_linear(K, bm)) for mu in mus]
            g = [r["gap"] for r in rs]
            e5 = [r["ess"][RARE] for r in rs]
            ew = [r["essmin"] for r in rs]
            out.append(f"  {bm:>8.2f} | " + " ".join(f"{v:>10.3e}" for v in g[:5]) + f" | {slope(mus, g):>+9.3f} "
                       f"{lslope(mus, g):>+7.3f} | {slope(mus, e5):>+8.3f} {lslope(mus, e5):>+7.3f} | "
                       f"{slope(mus, ew):>+8.3f} {lslope(mus, ew):>+7.3f}")
    out.append("")


def _draw_stats(t, bet, q, eps_list, ref):
    """efficiency ratios (vs exact weights on the same ladder) for a list of eps vectors."""
    r5, rw, occ, gp = [], [], [], []
    for eps in eps_list:
        r = metrics(t, bet, q, eps)
        r5.append(r["R5"] / ref["R5"])
        rw.append(r["Rw"] / ref["Rw"])
        occ.append(r["p0"])
        gp.append(r["gap"] / ref["gap"])
    return np.array(r5), np.array(rw), np.array(occ), np.array(gp)


def part_D(out):
    out.append("=" * 118)
    out.append("D. WEIGHT MISSPECIFICATION on the scaled ladder (linear, q=1/2): log-weight error eps_l iid N(0, sigma^2) "
               "(40 draws each, seed fixed)")
    out.append("   ratios are to the SAME chain with exact weights (so 1.0 = no loss); occ0 = stationary P(level 0) "
               "(exact weights: 1/K)")
    out.append("=" * 118)
    rng = np.random.default_rng(SEED + 4)
    for mu in (1e-4, 1e-6):
        t = Target("nb", mu)
        K = scaledK(mu)
        bet = ladder_linear(K)
        ref = metrics(t, bet, 0.5)
        out.append("")
        out.append(f"mu={mu:g}, K={K}: exact-weights reference gap={ref['gap']:.4g}, R5={ref['R5']:.4g}, "
                   f"Rw={ref['Rw']:.4g}, occ0={ref['p0']:.4f}")
        out.append(f"  {'sigma':>6} | {'R5 ratio: median':>16} {'p10':>8} {'min':>8} | {'worst8 ratio: median':>20} "
                   f"{'p10':>8} {'min':>8} | {'gap ratio med':>13} | {'occ0 median':>11} {'occ0 p10':>9}")
        tol = None
        for sig in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            eps_list = [sig * rng.standard_normal(K) for _ in range(40 if sig > 0 else 1)]
            r5, rw, occ, gp = _draw_stats(t, bet, 0.5, eps_list, ref)
            out.append(f"  {sig:>6.2f} | {np.median(r5):>16.4f} {np.percentile(r5, 10):>8.4f} {r5.min():>8.4f} | "
                       f"{np.median(rw):>20.4f} {np.percentile(rw, 10):>8.4f} {rw.min():>8.4f} | "
                       f"{np.median(gp):>13.4f} | {np.median(occ):>11.4f} {np.percentile(occ, 10):>9.4f}")
            if min(np.median(r5), np.median(rw)) >= 0.8:
                tol = sig
        out.append(f"  tolerance: largest sigma with median efficiency >= 80% of exact weights (both x5 and worst8): "
                   f"sigma = {tol}")
        out.append(f"  systematic drift eps_l = s * l/L (level marginal ~ exp(-s l/L)):")
        out.append(f"  {'s':>6} | {'R5 ratio':>9} {'worst8 ratio':>13} {'gap ratio':>10} {'occ0':>8}")
        for s in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
            eps = s * np.arange(K) / (K - 1)
            r5, rw, occ, gp = _draw_stats(t, bet, 0.5, [eps], ref)
            out.append(f"  {s:>+6.1f} | {r5[0]:>9.4f} {rw[0]:>13.4f} {gp[0]:>10.4f} {occ[0]:>8.4f}")
        # the minimal ladder K=2 has ONE free weight, delta = eps_1 - eps_0 (level-1 occupancy ~ exp(-delta))
        b2 = ladder_linear(2)
        ref2 = metrics(t, b2, 0.5)
        out.append(f"  K=2 ladder {{1,0}}: ONE weight, delta = log Zhat_uniform - log Z_uniform relative to the target rung "
                   f"(exact-weights Rw={ref2['Rw']:.4g}, gap={ref2['gap']:.4g})")
        out.append(f"  {'delta':>6} | {'worst8 ratio':>12} {'R5 ratio':>9} {'gap ratio':>10} {'occ0':>8}")
        for dlt in (-3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
            r = metrics(t, b2, 0.5, np.array([0.0, dlt]))
            out.append(f"  {dlt:>+6.2f} | {r['Rw'] / ref2['Rw']:>12.4f} {r['R5'] / ref2['R5']:>9.4f} "
                       f"{r['gap'] / ref2['gap']:>10.4f} {r['p0']:>8.4f}")
        grid = np.arange(-6.0, 6.0001, 0.05)
        rat = np.array([metrics(t, b2, 0.5, np.array([0.0, d_]))["Rw"] / ref2["Rw"] for d_ in grid])
        ok = grid[rat >= 0.8]
        out.append(f"  K=2 weight: worst8 ratio >= 0.8 for delta in [{ok.min():+.2f}, {ok.max():+.2f}]; maximised at delta={grid[rat.argmax()]:+.2f} "
                   f"with ratio {rat.max():.4f}")
    out.append("")


def part_E(out):
    out.append("=" * 118)
    out.append("E. LEVEL-MOVE PROBABILITY q (scaled ladder, linear, exact weights); worst-coordinate ESS ratio Rw to SR-NB under three "
               "cost models")
    out.append("   cost per iteration = (1-q) + q*rho with a fibre update costing 1 and a level move costing rho: rho=1 is per iteration, "
               "rho=0.1 a cheap level move, rho=0 is per fibre update (= per iteration / (1-q))")
    out.append("   gap = 1 - SLEM, gap1 = 1 - lambda_2 (they differ when a negative eigenvalue dominates)")
    out.append("=" * 118)
    qs = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    for mu, K in ((1e-4, scaledK(1e-4)), (1e-6, scaledK(1e-6)), (1e-4, 2), (1e-6, 2)):
        t = Target("nb", mu)
        bet = ladder_linear(K)
        res = {q: metrics(t, bet, q) for q in qs}
        eff = {rho: {q: res[q]["Rw"] / ((1 - q) + q * rho) for q in qs} for rho in (1.0, 0.1, 0.0)}
        e5 = {q: res[q]["R5"] for q in qs}
        out.append("")
        out.append(f"mu={mu:g}, K={K}")
        out.append(f"  {'q':>5} {'gap':>9} {'gap1':>9} | {'Rw rho=1':>10} {'/best':>6} | {'Rw rho=0.1':>11} {'/best':>6} | "
                   f"{'Rw rho=0':>10} {'/best':>6} | {'R5 rho=1':>9} {'/best':>6}")
        for q in qs:
            r = res[q]
            out.append(f"  {q:>5.2f} {r['gap']:>9.4g} {r['gap1']:>9.4g} | {eff[1.0][q]:>10.4g} "
                       f"{eff[1.0][q] / max(eff[1.0].values()):>6.3f} | {eff[0.1][q]:>11.4g} "
                       f"{eff[0.1][q] / max(eff[0.1].values()):>6.3f} | {eff[0.0][q]:>10.4g} "
                       f"{eff[0.0][q] / max(eff[0.0].values()):>6.3f} | {e5[q]:>9.4g} {e5[q] / max(e5.values()):>6.3f}")
    out.append("")


def stepping_counts(t, bet, counts, reverse):
    """Cumulative log Zhat_l - log Zhat_0 from per-rung sample counts over the 55 fibre states."""
    K = len(bet)
    n = counts.sum(1)
    est = np.zeros(K - 1)
    for l in range(K - 1):
        if not reverse:
            est[l] = np.log(counts[l] @ np.exp((bet[l + 1] - bet[l]) * t.lw) / n[l])
        else:
            est[l] = -np.log(counts[l + 1] @ np.exp((bet[l] - bet[l + 1]) * t.lw) / n[l + 1])
    return np.concatenate([[0.0], np.cumsum(est)])


def iid_counts(t, bet, n, rng):
    return np.array([rng.multinomial(n, normalised_pi(b * t.lw)) for b in bet], float)


def _pilot_row(out, label, t, bet, lz, ref, eps_all, r5, rw):
    eps_all, r5, rw = np.array(eps_all), np.array(r5), np.array(rw)
    out.append(f"  {label:>22} {eps_all[:, -1].mean():>+11.4f} {eps_all[:, -1].std(ddof=1):>11.4f} "
               f"{np.abs(eps_all).max(1).mean():>11.4f} | {np.median(r5):>9.4f} {np.percentile(r5, 10):>7.4f} | "
               f"{np.median(rw):>10.4f} {np.percentile(rw, 10):>7.4f} {(rw >= 0.8).mean():>9.2f}")


def part_F(out):
    out.append("=" * 118)
    out.append("F. WEIGHTS FROM A PILOT, BEST CASE: stepping-stone estimate of log Z_l from n iid EXACT draws per rung, 40 replicates")
    out.append("   forward: E_{pi_l}[exp((b_{l+1}-b_l) lw)] from draws of the colder rung (for the two-rung ladder this is the harmonic-mean "
               "estimator); reverse: the same ratio from draws of the hotter rung (bounded summands)")
    out.append("   eps_l = log Zhat_l - log Z_l (cumulative, eps_0 = 0); ratios are to exact weights on the same ladder (1.0 = no loss)")
    out.append("=" * 118)
    rng = np.random.default_rng(SEED + 6)
    for mu in (1e-2, 1e-4, 1e-6):
        t = Target("nb", mu)
        out.append("")
        out.append(f"mu={mu:g}")
        for K in sorted({2, 4, scaledK(mu)}):
            bet = ladder_linear(K)
            lz = t.logZ(bet)
            ref = metrics(t, bet, 0.5)
            out.append(f"  K={K} (exact-weights R5={ref['R5']:.4g}, Rw={ref['Rw']:.4g})")
            out.append(f"  {'direction, n/rung':>22} {'mean eps_L':>11} {'sd eps_L':>11} {'mean max|eps|':>11} | {'R5 med':>9} {'p10':>7} | "
                       f"{'worst8 med':>10} {'p10':>7} {'frac>=0.8':>9}")
            for rev in (False, True):
                for n in (100, 1000, 10000, 100000):
                    eps_all, r5, rw = [], [], []
                    for _ in range(40):
                        eps = stepping_counts(t, bet, iid_counts(t, bet, n, rng), rev) - (lz - lz[0])
                        r = metrics(t, bet, 0.5, eps)
                        eps_all.append(eps)
                        r5.append(r["R5"] / ref["R5"])
                        rw.append(r["Rw"] / ref["Rw"])
                    _pilot_row(out, f"{'reverse' if rev else 'forward'} n={n}", t, bet, lz, ref, eps_all, r5, rw)
    out.append("")


def part_F2(out):
    out.append("=" * 118)
    out.append("F2. WEIGHTS FROM AN MCMC PILOT (what Linus actually has): single-ray Gibbs chain at each rung, n iterations after n/10 burn-in, "
               "started from ONE common state, 40 replicates")
    out.append("   start 'mode' = the mode of pi_NB (inside one of the three x5=0 components of the fibre); 'rand' = a uniformly random fibre "
               "state per replicate (common to all rungs)")
    out.append("   the pilot at rung beta needs about mu^(-beta) iterations to cross between the three components, so the top rungs are stuck "
               "when n is small")
    out.append("   eps_l = log Zhat_l - log Z_l; ratios are to exact weights on the same ladder (1.0 = no loss); fwd/rev as in F")
    out.append("=" * 118)
    rng = np.random.default_rng(SEED + 7)
    for mu in (1e-4, 1e-6):
        t = Target("nb", mu)
        out.append("")
        out.append(f"mu={mu:g}")
        for K in [2, 4, scaledK(mu)]:
            bet = ladder_linear(K)
            lz = t.logZ(bet)
            ref = metrics(t, bet, 0.5)
            ridx, rlen, cum = build_ray_tables(t, bet)
            out.append(f"  K={K} (exact-weights Rw={ref['Rw']:.4g}, R5={ref['R5']:.4g})")
            out.append(f"  {'start, dir, n/rung':>22} {'mean eps_L':>11} {'sd eps_L':>11} {'mean max|eps|':>11} | {'R5 med':>9} {'p10':>7} | "
                       f"{'worst8 med':>10} {'p10':>7} {'frac>=0.8':>9}")
            for start in ("mode", "rand"):
                for n in (100, 1000, 10000, 100000):
                    acc = {True: ([], [], []), False: ([], [], [])}
                    for rep in range(40):
                        x0 = int(np.argmax(t.lw)) if start == "mode" else int(rng.integers(N))
                        counts = np.zeros((K, N))
                        for e in range(K):
                            tr = _pt_sim(n, max(n // 10, 10), 1, bet[e:e + 1], t.lw, 1.0, rlen, ridx, cum[e:e + 1], x0,
                                         SEED + 1000 * rep + e)
                            counts[e] = np.bincount(tr, minlength=N)
                        for rev in (True, False):
                            eps = stepping_counts(t, bet, counts, rev) - (lz - lz[0])
                            r = metrics(t, bet, 0.5, eps)
                            acc[rev][0].append(eps)
                            acc[rev][1].append(r["R5"] / ref["R5"])
                            acc[rev][2].append(r["Rw"] / ref["Rw"])
                    for rev in (True, False):
                        _pilot_row(out, f"{start} {'rev' if rev else 'fwd'} n={n}", t, bet, lz, ref, *acc[rev])
    out.append("")


def pt_exact(t, bet, coords, maxit=8000):
    """Exact sparse parallel tempering with K = 2 or 3 rungs on N^K states (random scan: a rung update w.p. 1/2,
    rung uniform; else a swap of a uniformly chosen adjacent pair).  Returns the gap and ESS/iter (CG solves)."""
    K = len(bet)
    n = N ** K
    idx = np.arange(n)
    dig = [(idx // N ** (K - 1 - l)) % N for l in range(K)]
    I = sp.identity(N, format="csr")
    Qs = [sp.csr_matrix(t.block(b)) for b in bet]
    F = None
    for l in range(K):
        term = None
        for m_ in range(K):
            mat = Qs[l] if m_ == l else I
            term = mat if term is None else sp.kron(term, mat, format="csr")
        F = term if F is None else F + term
    Sw = None
    for l in range(K - 1):
        dd = (bet[l] - bet[l + 1]) * (t.lw[dig[l + 1]] - t.lw[dig[l]])
        a = np.exp(np.minimum(0.0, dd))
        sidx = idx + (dig[l + 1] - dig[l]) * N ** (K - 1 - l) + (dig[l] - dig[l + 1]) * N ** (K - 2 - l)
        Ml = sp.coo_matrix((a, (idx, sidx)), shape=(n, n)).tocsr() + sp.diags(1.0 - a)
        Sw = Ml if Sw is None else Sw + Ml
    P = 0.5 * F / K + 0.5 * Sw / (K - 1)
    pr = [normalised_pi(b * t.lw) for b in bet]
    pi = pr[0]
    for q_ in pr[1:]:
        pi = np.multiply.outer(pi, q_)
    pi = pi.ravel()
    d = np.sqrt(pi)
    S = sp.diags(d) @ P @ sp.diags(1.0 / d)
    asym = abs(S - S.T).max()
    S = 0.5 * (S + S.T)
    defl = spla.LinearOperator((n, n), matvec=lambda v: S @ v - d * (d @ v), dtype=float)
    lam2 = spla.eigsh(defl, k=1, which="LA", tol=1e-10, ncv=40, maxiter=200000, return_eigenvectors=False)[0]
    lamn = spla.eigsh(S, k=1, which="SA", tol=1e-10, ncv=40, maxiter=200000, return_eigenvectors=False)[0]
    res = dict(gap=1.0 - max(abs(lam2), abs(lamn)), gap1=1.0 - lam2, lamn=lamn, asym=asym, ess={}, cgit={}, n=n)
    op = spla.LinearOperator((n, n), matvec=lambda v: v - S @ v + d * (d @ v), dtype=float)
    for j in coords:
        ht = d * (X[dig[0], j] - t.mean[j])
        cnt = [0]

        def cb(_):
            cnt[0] += 1

        g, info = spla.cg(op, ht, rtol=1e-10, atol=0.0, maxiter=maxit, callback=cb)
        sig2 = 2.0 * float(ht @ g) - float(ht @ ht)
        res["ess"][j] = t.var[j] / sig2 if info == 0 else float("nan")
        res["cgit"][j] = cnt[0]
    return res


def part_G(out, T=2_000_000, burn=200_000):
    out.append("=" * 118)
    out.append("G. WEIGHT-FREE ALTERNATIVE: parallel tempering (PT) on the same linear ladder 1 -> 0; one iteration = a rung "
               "update w.p. 1/2 (rung uniform) or a random adjacent swap")
    out.append("   ESS/iter = 1/IAT of the rung-0 trace (Geyer initial-sequence IAT, 3 seeds x 2e6 iterations after 2e5 burn-in); "
               "ST ideal = exact weights, same ladder, q=1/2")
    out.append("   x5 is rare for mu<=1e-4 (the trace is constant), so the comparison metric is worst7 = worst of the 7 other "
               "coordinates; x5 shown only where the trace has >= 500 nonzero entries")
    out.append("=" * 118)
    Ks = [2, 3, 4, 6, 8, 12, 16, 24]
    for mu in (1e-2, 1e-4, 1e-6):
        t = Target("nb", mu)
        sr = t.sr()
        ess_sr7 = sr["ess"][NONRARE].min()
        out.append("")
        out.append(f"mu={mu:g}   SR-NB exact: ESS/iter x5={sr['ess'][RARE]:.4g}, worst7={ess_sr7:.4g} "
                   f"(coordinate {NONRARE[int(sr['ess'][NONRARE].argmin())]}), gap={sr['gap']:.4g}")
        out.append(f"  {'K':>3} | {'ST ideal: worst7':>16} {'x5':>10} | {'PT MC: worst7':>13} {'seeds range':>17} "
                   f"{'x5':>10} {'nnz x5':>7} | {'PT/ST worst7':>12} | {'PT/SR worst7':>12}")
        x0 = int(np.argmax(t.lw))
        for K in Ks:
            bet = ladder_linear(K)
            st = metrics(t, bet, 0.5)
            ridx, rlen, cum = build_ray_tables(t, bet)
            e7_runs, e5_runs, nnz = [], [], []
            for s_ in range(3):
                tr = _pt_sim(T, burn, K, bet, t.lw, 0.5, rlen, ridx, cum, x0, SEED + 100 * K + s_)
                tr = tr.astype(np.int64)
                iats = np.array([iat_geyer(X[tr, j]) for j in range(8)])
                e7_runs.append(np.nanmin(1.0 / iats[NONRARE]))
                nnz.append(int((X[tr, RARE] > 0).sum()))
                e5_runs.append(1.0 / iats[RARE] if nnz[-1] >= 500 else float("nan"))
            e7 = np.mean(e7_runs)
            e5 = np.mean(e5_runs) if np.all(np.isfinite(e5_runs)) else float("nan")
            st7 = st["ess"][NONRARE].min()
            out.append(f"  {K:>3} | {st7:>16.4g} {st['ess'][RARE]:>10.3g} | {e7:>13.4g} "
                       f"{min(e7_runs):>8.3g}-{max(e7_runs):<8.3g} {fmt(e5, 10, 3)} {int(np.mean(nnz)):>7d} | "
                       f"{e7 / st7:>12.3f} | {e7 / ess_sr7:>12.3f}")
        for Kx in (2, 3):
            t0 = time.time()
            bx = ladder_linear(Kx)
            ex = pt_exact(t, bx, coords=[RARE] + NONRARE)
            stx = metrics(t, bx, 0.5)
            e7x = min(ex["ess"][j] for j in NONRARE)
            out.append(f"  exact PT, K={Kx} ({ex['n']} joint states, symmetry defect {ex['asym']:.1e}, CG iterations max "
                       f"{max(ex['cgit'].values())}, {time.time() - t0:.0f}s): gap={ex['gap']:.4g} "
                       f"(gap1={ex['gap1']:.4g}, lambda_min={ex['lamn']:.3g}), ESS/iter worst7={e7x:.4g}, x5={ex['ess'][RARE]:.4g};  "
                       f"ST ideal K={Kx}: gap={stx['gap']:.4g}, worst7={stx['ess'][NONRARE].min():.4g}, x5={stx['ess'][RARE]:.4g}")
    out.append("")


def part_H(out):
    out.append("=" * 118)
    out.append("H. POISSON (not NB) target: the same headline claims (q=1/2, exact weights); ratios to SR-Pois per iteration")
    out.append("=" * 118)
    for mu in (1e-2, 1e-4):
        t = Target("pois", mu)
        sr = t.sr()
        out.append("")
        out.append(f"mu={mu:g}   SR-Pois: gap={sr['gap']:.4g}, ESS/iter x5={sr['ess'][RARE]:.4g}, "
                   f"worst8={sr['ess'].min():.4g}")
        out.append(f"  {'ladder':>34} {'K':>3} {'gap':>10} {'R5':>10} {'Rw':>10} {'jw':>3}")
        K0 = scaledK(mu)
        cfg = [("fixed K=3 {1,1/2,0}", ladder_linear(3)),
               (f"scaled K={K0} linear to 0", ladder_linear(K0)),
               ("K=3 {1, 3/4, 1/2} (beta_min=1/2)", ladder_linear(3, 0.5)),
               (f"K={K0} linear to beta_min=1/2", ladder_linear(K0, 0.5)),
               (f"K={K0} linear to beta_min=1/4", ladder_linear(K0, 0.25))]
        for name, b in cfg:
            r = metrics(t, b)
            out.append(f"  {name:>34} {len(b):>3} {r['gap']:>10.4g} {r['R5']:>10.4g} {r['Rw']:>10.4g} {r['jw']:>3}")
    # slopes with mu for the Poisson targets, fixed K=3 vs K=12 to 0 vs beta_min=1/2
    mus = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    tg = {mu: Target("pois", mu) for mu in mus}
    out.append("")
    out.append("Poisson gap slopes over mu in {1e-2,...,1e-6}: all five points / last pair (1e-5 -> 1e-6)")
    g = [tg[mu].sr()["gap"] for mu in mus]
    out.append(f"  SR-Pois (K=1): gap {' '.join(f'{v:.3e}' for v in g)}  slope {slope(mus, g):+.3f}  last {lslope(mus, g):+.3f}")
    for name, mk in [("K=3 {1,1/2,0}", lambda mu: ladder_linear(3)),
                     ("K=12 linear to 0", lambda mu: ladder_linear(12)),
                     ("K=12 linear to beta_min=1/2", lambda mu: ladder_linear(12, 0.5))]:
        g = [metrics(tg[mu], mk(mu))["gap"] for mu in mus]
        out.append(f"  {name:>27}: gap {' '.join(f'{v:.3e}' for v in g)}  slope {slope(mus, g):+.3f}  "
                   f"last {lslope(mus, g):+.3f}")
    out.append("")


def part_R(out):
    out.append("=" * 118)
    out.append("R. RECIPE GRID: worst-coordinate ESS ratio Rw (to SR-NB) over K x spacing x q (K in 2,3,4,6,8,12,K*; q in .25,.5,.75,.9,.95), "
               "per unit cost; cost per iteration = (1-q) + q*rho (rho = cost of a level move relative to a fibre update)")
    out.append("   'frac of best' = efficiency of a candidate default divided by the best configuration in the whole grid at the same mu and rho")
    out.append("=" * 118)
    qs = (0.25, 0.5, 0.75, 0.9, 0.95)
    rhos = (1.0, 0.1, 0.01)
    mus = (1e-2, 1e-4, 1e-6)
    defaults = {"K=2,q=.5": (2, "linear", 0.5), "K=2,q=.75": (2, "linear", 0.75), "K=3 eqacc,q=.5": (3, "eqacc", 0.5),
                "K=3 eqacc,q=.75": (3, "eqacc", 0.75), "K=4 eqacc,q=.75": (4, "eqacc", 0.75),
                "K=4 linear,q=.5": (4, "linear", 0.5), "scaled linear,q=.5": ("sc", "linear", 0.5),
                "scaled eqacc,q=.5": ("sc", "eqacc", 0.5), "K=8 linear,q=.5": (8, "linear", 0.5)}
    table = {}
    for mu in mus:
        t = Target("nb", mu)
        Kl = [2, 3, 4, 6, 8, 12, scaledK(mu)]
        res = {}
        for K in Kl:
            for spc in ("linear", "geom", "quad", "eqacc"):
                if K == 2 and spc != "linear":
                    continue
                bet = {"linear": ladder_linear, "geom": ladder_geom, "quad": ladder_quad}[spc](K) if spc != "eqacc" \
                    else ladder_eqacc(t, K)
                for q in qs:
                    res[(K, spc, q)] = metrics(t, bet, q)
        for rho in rhos:
            score = {k: r["Rw"] / ((1 - k[2]) + k[2] * rho) for k, r in res.items()}
            kb = max(score, key=score.get)
            table[(mu, rho)] = {name: score[(scaledK(mu) if cfg[0] == "sc" else cfg[0], cfg[1], cfg[2])] / score[kb]
                                for name, cfg in defaults.items()}
            out.append("")
            out.append(f"mu={mu:g}, rho={rho}: best configuration K={kb[0]}, spacing={kb[1]}, q={kb[2]} with Rw per unit cost "
                       f"{score[kb]:.4g} (gap {res[kb]['gap']:.4g}); scaled K*={scaledK(mu)}")
            top = sorted(score, key=score.get, reverse=True)[:5]
            out.append("   top five: " + "; ".join(f"K={k[0]} {k[1]} q={k[2]} ({score[k] / score[kb]:.3f})" for k in top))
    out.append("")
    out.append("FRACTION OF BEST for candidate defaults (rows) at each mu, rho (columns)")
    cols = [(mu, rho) for mu in mus for rho in rhos]
    out.append(f"  {'default':>20} | " + " ".join(f"{f'{mu:g}/{rho}':>11}" for mu, rho in cols) + " |    worst")
    for name in defaults:
        vals = [table[c][name] for c in cols]
        out.append(f"  {name:>20} | " + " ".join(f"{v:>11.3f}" for v in vals) + f" | {min(vals):>8.3f}")
    out.append("")
    out.append("PARAMETERS A PRACTITIONER MUST CHOOSE")
    out.append("  ST, K rungs: K; the K-2 interior rungs (or ONE target overlap for the equal-acceptance rule); beta_min (not free: it must be 0); q; "
               "the K-1 log-weights, plus the pilot length that estimates them")
    out.append("  ST, K=2:     q; ONE log-weight (log of the target to uniform normaliser ratio) and its pilot length")
    out.append("  PT, K rungs: K; the K-2 interior rungs (or the one overlap target); the update/swap mix; no weights, no q")
    out.append("  numbers to set beyond K itself (beta_min = 0 is fixed; the equal-acceptance rule replaces the K-2 interior rungs by one overlap target, "
               "and K=2 has no interior rung):")
    out.append(f"  {'K':>4} | {'ST equal-acceptance':>19} {'ST free rungs':>14} | {'PT equal-acceptance':>19} {'PT free rungs':>14}")
    for K in (2, 3, scaledK(1e-4)):
        inner_eq, inner_free = (1, K - 2) if K > 2 else (0, 0)
        out.append(f"  {K:>4} | {inner_eq + 1 + (K - 1) + 1:>19} {inner_free + 1 + (K - 1) + 1:>14} | {inner_eq + 1:>19} {inner_free + 1:>14}")
    out.append("")


def part_I(out):
    out.append("=" * 118)
    out.append("I. DOES THE OVERLAP RULE TRANSFER?  A concentrated Poisson target on the same fibre: theta_j = exp(s u_j) (u = +1,-1,+1,-1,0,-1,+1,-1), "
               "theta_5 = mu = 1e-4")
    out.append("   s = 0 is the paper's P1 target; larger s makes the target more concentrated, so its overlap with uniform (the mean acceptance of a "
               "one-step jump to beta=0) falls")
    out.append("   Rw = worst-coordinate ESS ratio to SR-Pois at the same theta, q=1/2, exact weights; ov = smallest adjacent overlap of the ladder")
    out.append("=" * 118)
    u = np.array([1, -1, 1, -1, 0, -1, 1, -1], float)
    for sc in (0.0, 1.0, 2.0, 3.0, 4.0):
        t = Target("pois", 1e-4, theta=np.exp(sc * u))
        sr = t.sr()
        out.append("")
        out.append(f"s={sc:g}: overlap(target, uniform)={overlap(t, 1.0, 0.0):.4f}; SR gap={sr['gap']:.4g}, SR worst8 ESS/iter={sr['ess'].min():.4g}")
        out.append(f"  {'K':>3} {'linear Rw':>11} {'ov':>6} {'eqacc Rw':>11} {'ov':>6}")
        best = (0.0, None)
        for K in (2, 3, 4, 6, 8, 12):
            row = f"  {K:>3}"
            for spc in ("linear", "eqacc"):
                b = ladder_linear(K) if spc == "linear" else ladder_eqacc(t, K)
                r = metrics(t, b, 0.5)
                ov = min(overlap(t, b[i], b[i + 1]) for i in range(K - 1))
                row += f" {r['Rw']:>11.4g} {ov:>6.3f}"
                if r["Rw"] > best[0]:
                    best = (r["Rw"], (K, spc, ov))
            out.append(row)
        out.append(f"  best: K={best[1][0]} {best[1][1]} (min adjacent overlap {best[1][2]:.3f}), Rw={best[0]:.4g}")
        out.append(f"  K=2 as a fraction of that best: {metrics(t, ladder_linear(2), 0.5)['Rw'] / best[0]:.3f}")
    out.append("")


def part_S(out):
    out.append("=" * 118)
    out.append("S. HEADLINE RATIOS (derived here from the same exact quantities as A and B, so that every ratio quoted in the summary of the study is in this file)")
    out.append("   Rw = worst-coordinate ESS ratio to SR-NB per iteration, q=1/2, exact weights, linear ladder 1 -> 0 unless stated")
    out.append("=" * 118)
    out.append("")
    out.append("S1. two rungs {1,0} against the fixed three-rung ladder {1,1/2,0} and against the scaled rule K* = ceil(log2 1/mu) + 1")
    out.append(f"  {'mu':>7} {'Rw K=2':>10} {'Rw K=3':>10} {'K*':>4} {'Rw K*':>10} {'K=2/K=3':>8} {'K=2/K*':>8}")
    for mu in (1e-2, 1e-4, 1e-6, 1e-8):
        t = Target("nb", mu)
        r2, r3 = (metrics(t, ladder_linear(K))["Rw"] for K in (2, 3))
        ks = scaledK(mu)
        rs = metrics(t, ladder_linear(ks))["Rw"]
        out.append(f"  {mu:>7g} {r2:>10.4g} {r3:>10.4g} {ks:>4d} {rs:>10.4g} {r2 / r3:>8.3f} {r2 / rs:>8.2f}")
    out.append("")
    out.append("S2. efficiency kept as K grows: K * Rw(K) / (2 * Rw(2)), which is 1 when Rw falls exactly like 1/K")
    Ks = (2, 3, 4, 6, 8, 12, 16, 24)
    out.append(f"  {'mu':>7} " + " ".join(f"{f'K={K}':>7}" for K in Ks))
    for mu in (1e-2, 1e-4, 1e-6):
        t = Target("nb", mu)
        r2 = metrics(t, ladder_linear(2))["Rw"]
        out.append(f"  {mu:>7g} " + " ".join(f"{K * metrics(t, ladder_linear(K))['Rw'] / (2 * r2):>7.3f}" for K in Ks))
    out.append("")
    out.append("S3. spacing at equal K: Rw(equal-acceptance) / Rw(linear), and Rw(equal-acceptance) / Rw(geometric)")
    out.append(f"  {'mu':>7} " + " ".join(f"{f'K={K} lin':>9} {f'K={K} geo':>9}" for K in (4, 8, 15)))
    for mu in (1e-2, 1e-4, 1e-6):
        t = Target("nb", mu)
        row = f"  {mu:>7g} "
        for K in (4, 8, 15):
            re = metrics(t, ladder_eqacc(t, K))["Rw"]
            row += f"{re / metrics(t, ladder_linear(K))['Rw']:>9.3f} {re / metrics(t, ladder_geom(K))['Rw']:>9.3f} "
        out.append(row.rstrip())
    out.append("")


PARTS = {"gates": part_gates, "A": part_A, "B": part_B, "C": part_C, "D": part_D, "E": part_E,
         "F": part_F, "F2": part_F2, "G": part_G, "H": part_H, "R": part_R, "I": part_I, "S": part_S}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="gates,A,B,C,D,E,F,F2,G,H,R,I,S")
    ap.add_argument("--out", default=OUT_DEFAULT)
    a = ap.parse_args()
    out = ["Tempering tuning study on P1 (fibre size 55), NB target alpha=1.9, theta_5 = mu, exact computation",
           f"seed {SEED}; q = level-move probability; K = number of rungs; conventions in the module docstring", ""]
    for name in a.parts.split(","):
        t0 = time.time()
        res = PARTS[name](out)
        out.append(f"[part {name} done in {time.time() - t0:.0f}s]")
        out.append("")
        print(f"part {name} done in {time.time() - t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write("\n".join(out) + "\n")
        if name == "gates" and res is False:
            print("GATES FAILED; stopping")
            break
    print("\n".join(out))


if __name__ == "__main__":
    main()
