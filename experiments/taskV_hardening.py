"""Task V hardening -- four confirmations feeding the tempered-sampler theorem.

  1. Exponent law: gap = Theta(mu^{beta_min}) on P1, at beta_min in {0,.25,.5,.75,1}.
     Five points on the line "exponent = beta_min" turns the pattern into a law.
  2. Generality: the constant floor (3-rung ladder to beta=0) replicates on the 2x3
     table, with a different constant.
  3. Corridor height k on P1: the provable law is gap = Theta(mu^{k*beta_min}); the
     observed exponent equals beta_min iff k=1 (cheapest inter-lobe crossing raises
     x_{j*} exactly once).  Confirm k=1.
  4. Level overhead in the escape regime: gap vs L at fixed mu -- is it 1/L or 1/L^2?
     (The mechanism behind the polylog remark.)

All exact SLEM, prob-parameterized so P1 and the table share one tempering routine.
Run:  python3 experiments/taskV_hardening.py
"""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.problems import (P1, p1_poisson_theta, table_2x3,
                                   TABLE23_BOTTLENECK)
from fibresampler.spectral import sr_transition_matrix, slem, normalised_pi
from fibresampler.augment import nb_log_pmf

ALPHA = 1.9
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# general prob-parameterized idealized simulated tempering
# ---------------------------------------------------------------------------
def tempered_gap(prob, lw, betas):
    """Exact joint (x, level) spectral gap and max DB violation for the random-scan
    reversible simulated-tempering chain on target pi ∝ exp(lw), ladder `betas`."""
    N = prob.fibre_size
    betas = np.asarray(betas, float)
    Lp1 = len(betas)
    blocks = [sr_transition_matrix(prob, b * lw) for b in betas]
    if Lp1 == 1:
        pi = normalised_pi(betas[0] * lw)
        F = pi[:, None] * blocks[0]
        return slem(blocks[0], pi)[1], float(np.abs(F - F.T).max())
    logZ = np.array([logsumexp(b * lw) for b in betas])
    logpi = np.array([betas[e] * lw - logZ[e] - np.log(Lp1) for e in range(Lp1)])
    pi = np.exp(logpi).ravel(); pi /= pi.sum()
    Pf = np.zeros((Lp1 * N, Lp1 * N))
    for e in range(Lp1):
        Pf[e * N:(e + 1) * N, e * N:(e + 1) * N] = blocks[e]
    Pl = np.zeros((Lp1 * N, Lp1 * N))
    for e in range(Lp1):
        for ep in (e - 1, e + 1):
            if 0 <= ep < Lp1:
                dlog = (betas[ep] - betas[e]) * lw - (logZ[ep] - logZ[e])
                A = np.minimum(1.0, np.exp(dlog))
                for i in range(N):
                    Pl[e * N + i, ep * N + i] += 0.5 * A[i]
                    Pl[e * N + i, e * N + i] += 0.5 * (1.0 - A[i])
            else:
                Pl[e * N:(e + 1) * N, e * N:(e + 1) * N] += 0.5 * np.eye(N)
    P = 0.5 * Pf + 0.5 * Pl
    F = pi[:, None] * P
    return slem(P, pi)[1], float(np.abs(F - F.T).max())


def slope(mus, gaps):
    return float(np.polyfit(np.log(mus), np.log(gaps), 1)[0])


def floor_ladder(beta_min, dbeta=0.25):
    """Ladder 1 -> beta_min in steps of dbeta (floor = beta_min, rounded exact)."""
    b = np.round(np.arange(1.0, beta_min - 1e-9, -dbeta), 10)
    if b[-1] != round(beta_min, 10):
        b = np.append(b, round(beta_min, 10))
    return b


def p1_nb_logw(mu5):
    return nb_log_pmf(P1(), p1_poisson_theta(mu5), ALPHA)


# ---------------------------------------------------------------------------
# Run 1 -- exponent law on P1
# ---------------------------------------------------------------------------
def run1_exponent_law(out):
    out.append("=" * 74)
    out.append("RUN 1 -- exponent law: gap = Theta(mu5^{beta_min}) on P1 (alpha=1.9)")
    out.append("=" * 74)
    prob = P1()
    mus = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    beta_mins = [0.0, 0.25, 0.5, 0.75, 1.0]
    out.append(f"  ladders in steps of dbeta=0.25 down to beta_min; slope fit over mu5<=0.01")
    out.append(f"  {'beta_min':>9} {'ladder':>22} {'measured slope':>15} {'predicted':>10} {'maxDB':>9}")
    measured = []
    for bm in beta_mins:
        betas = floor_ladder(bm)
        gaps, dbes = [], []
        for mu5 in mus:
            g, dbe = tempered_gap(prob, p1_nb_logw(mu5), betas)
            gaps.append(g); dbes.append(dbe)
        s = slope(mus, gaps); measured.append(s)
        lad = "[" + ",".join(f"{b:g}" for b in betas) + "]"
        out.append(f"  {bm:>9.2f} {lad:>22} {s:>15.3f} {bm:>10.2f} {max(dbes):>9.1e}")
    law_ok = all(abs(m - b) < 0.06 for m, b in zip(measured, beta_mins))
    out.append(f"  => exponent = beta_min to <0.06 on all 5 points: {'LAW CONFIRMED' if law_ok else 'CHECK'}")
    out.append("")
    return beta_mins, measured


# ---------------------------------------------------------------------------
# Run 2 -- generality of the constant floor on the 2x3 table
# ---------------------------------------------------------------------------
def run2_table_floor(out):
    out.append("=" * 74)
    out.append("RUN 2 -- constant floor replicates on the 2x3 table (3-rung [1,0.5,0])")
    out.append("=" * 74)
    tab = table_2x3()
    r = tab.states.shape[1]

    def tab_logw(mu):
        muv = np.ones(r); muv[TABLE23_BOTTLENECK] = mu
        return nb_log_pmf(tab, muv, ALPHA)

    betas = np.array([1.0, 0.5, 0.0])
    out.append(f"  table N={tab.fibre_size}")
    out.append(f"  {'mu':>8} {'SR-NB gap':>11} {'temp[1,.5,0] gap':>17} {'local slope':>12} {'maxDB':>9}")
    prev = None
    gaps = []
    for mu in [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-12, 1e-16]:
        lw = tab_logw(mu)
        g, dbe = tempered_gap(tab, lw, betas)
        gnb = slem(sr_transition_matrix(tab, lw), normalised_pi(lw))[1]
        ls = "" if prev is None else f"{(np.log(g)-np.log(prev[1]))/(np.log(mu)-np.log(prev[0])):+.4f}"
        out.append(f"  {mu:>8g} {gnb:>11.3e} {g:>17.6f} {ls:>12} {dbe:>9.1e}")
        prev = (mu, g); gaps.append(g)
    flat = abs(gaps[-1] - gaps[-2]) / gaps[-2] < 0.01
    out.append(f"  => tempered gap floors at ~{gaps[-1]:.5f} (flat: {'yes' if flat else 'NO'}); "
               f"SR-NB collapses. Escape is problem-independent.")
    out.append("")


# ---------------------------------------------------------------------------
# Run 3 -- corridor height k on P1
# ---------------------------------------------------------------------------
def run3_corridor_height(out):
    out.append("=" * 74)
    out.append("RUN 3 -- corridor height k on P1 (cheapest inter-lobe crossing)")
    out.append("=" * 74)
    prob = P1()
    jstar = 4                       # bottleneck coordinate (theta_5 -> index 4)
    states = prob.states
    N = prob.fibre_size
    # full move graph: edge i-k if k on some ray through i
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(prob.n_moves):
            for k in prob.ray(i, j):
                if k != i:
                    adj[i].add(k)
    # lobes: components of D={x_{j*}=0} under moves with u_{j*}=0
    D = [i for i in range(N) if states[i][jstar] == 0]
    Dset = set(D)
    moves0 = [j for j in range(prob.n_moves) if prob.U[jstar, j] == 0]
    adjD = {i: set() for i in D}
    for i in D:
        for j in moves0:
            for k in prob.ray(i, j):
                if k in Dset and k != i:
                    adjD[i].add(k)
    # union-find lobes
    lobe = {i: i for i in D}
    def find(a):
        while lobe[a] != a:
            lobe[a] = lobe[lobe[a]]; a = lobe[a]
        return a
    for i in D:
        for k in adjD[i]:
            lobe[find(i)] = find(k)
    comps = {}
    for i in D:
        comps.setdefault(find(i), []).append(i)
    lobe_id = {i: c for c, mem in enumerate(comps.values()) for i in mem}
    nlobes = len(comps)
    out.append(f"  |D|={len(D)}, lobes={nlobes} (Assumption A)")
    # k = min over inter-lobe paths of max x_{j*} along the path.  BFS minimizing the
    # path-maximum of x_{j*} (a min-max / widest-path search from lobe 0).
    INF = 10**9
    src_lobe = 0
    best = {i: INF for i in range(N)}
    # start: all lobe-0 states have path-max = their x_{j*}=0
    import heapq
    pq = []
    for i in D:
        if lobe_id[i] == src_lobe:
            best[i] = 0; heapq.heappush(pq, (0, i))
    reached_other = INF
    while pq:
        h, i = heapq.heappop(pq)
        if h > best[i]:
            continue
        if i in Dset and lobe_id.get(i, src_lobe) != src_lobe:
            reached_other = min(reached_other, h); break
        for k in adj[i]:
            nh = max(h, int(states[k][jstar]))
            if nh < best[k]:
                best[k] = nh; heapq.heappush(pq, (nh, k))
    out.append(f"  cheapest crossing raises x_(j*) to max height k = {reached_other}")
    out.append(f"  => provable law gap=Theta(mu^(k*beta_min)); observed exponent=beta_min "
               f"consistent with k={reached_other}. {'k=1 CONFIRMED' if reached_other==1 else 'k!=1 -- adjust theorem'}")
    out.append("")


# ---------------------------------------------------------------------------
# Run 4 -- level overhead: gap vs L at fixed mu in the escape regime
# ---------------------------------------------------------------------------
def run4_level_overhead(out):
    out.append("=" * 74)
    out.append("RUN 4 -- level overhead: gap vs L at fixed mu (escape regime, ladder ->0)")
    out.append("=" * 74)
    prob = P1()
    mu5 = 1e-8
    lw = p1_nb_logw(mu5)
    Ls = [2, 4, 8, 16, 32, 64]
    out.append(f"  mu5={mu5:g}; linear ladder [1..0] with L+1 rungs")
    out.append(f"  {'L':>4} {'gap':>12} {'gap*L':>12} {'gap*L^2':>12}")
    gaps = []
    for L in Ls:
        betas = np.round(np.linspace(1.0, 0.0, L + 1), 12)
        g, _ = tempered_gap(prob, lw, betas)
        gaps.append(g)
        out.append(f"  {L:>4} {g:>12.5e} {g*L:>12.5e} {g*L*L:>12.5e}")
    p = -slope(Ls, gaps)          # gap ~ L^{-p}
    out.append(f"  => gap ~ 1/L^{p:.2f}  (p~1 => level path-length bottleneck; "
               f"p~2 => level diffusion bottleneck)")
    out.append("")


def make_law_figure(beta_mins, measured):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="exponent = $\\beta_{\\min}$")
    ax.plot(beta_mins, measured, "o", color="C3", ms=8)
    ax.set_xlabel(r"ladder floor $\beta_{\min}$")
    ax.set_ylabel(r"measured log-log slope of gap vs $\mu_5$")
    ax.set_title("Task V exponent law (P1)")
    ax.legend(frameon=False); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    p = os.path.join(RESULTS, "fig_taskV_exponent_law.pdf")
    fig.savefig(p); fig.savefig(p.replace(".pdf", ".png"), dpi=130)
    return p


def main():
    out = []
    bmins, measured = run1_exponent_law(out)
    run2_table_floor(out)
    run3_corridor_height(out)
    run4_level_overhead(out)
    try:
        p = make_law_figure(bmins, measured)
        out.append(f"figure -> {p}")
    except Exception as e:
        out.append(f"[figure skipped: {e}]")
    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "taskV_hardening_numbers.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
