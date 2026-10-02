"""Tier C -- counting-based and rare-event estimators for the fibre normaliser.

Ports Nick Polson's companion bundle (nested_sampling.py, rubinstein_methods.py,
split_sampling.py) into our harness and reproduces every headline number against
EXACT enumeration, on two contingency-table fibres:

  * the balanced 2x3 table (4,4)|(3,3,2), |F|=10 (our ``table_2x3``), the running
    example whose two wall states (2,2),(3,1) are the K=2, k*=1 lobes;
  * the larger (12,12)|(8,8,8) table, |F|=61, five wall states.

Four estimators, each of the fibre count |F| or the normaliser C = sum_x L(x)
(equivalently the evidence Z = C/|F| under the uniform reference pi_0):

  (CE)  cross-entropy / adaptive SIS counting over the self-reducibility tree: the
        subtree-leaf-count proposal is the zero-variance importance density;
  (SE)  stochastic enumeration (Rubinstein-Vaisman-Dolgin; B=1 is Knuth's estimator);
  (NS)  nested sampling with the UNIFORM fibre measure as reference (a flattening
        escape; returns the normaliser and freezes the lobe census at the pinch);
  (SP)  multilevel splitting (Birge-Chang-Polson): the vertical telescoping product,
        the rare-event twin of the tempering ladder, and two-directional.

Everything is checked against exact enumeration (|F|, C, Z all closed-form here), so
the estimators are validated, not merely reported.  Numbers -> results/counting_numbers.txt.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.special import gammaln, logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.fibre import enumerate_fibre

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# 2x3 table fibre: A rows = row-1 total + three column totals; cells
# (x11,x12,x13,x21,x22,x23).  x11 is the loaded bottleneck cell (mean mu_star).
# ---------------------------------------------------------------------------
def table_fibre(r1, cols):
    c1, c2, c3 = cols
    A = np.array([[1, 1, 1, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 1]], dtype=np.int64)
    y = np.array([r1, c1, c2, c3], dtype=np.int64)
    states = enumerate_fibre(A, y)          # rows (x11,x12,x13,x21,x22,x23)
    return states


def log_like(states, mu_star):
    """Fibre-conditional Poisson log-weight; only x11 carries mean mu_star."""
    theta = np.array([mu_star, 1, 1, 1, 1, 1], dtype=float)
    X = states.astype(float)
    return X @ np.log(theta) - gammaln(X + 1).sum(1)


# ---------------------------------------------------------------------------
# self-reducibility tree over (x11, x12): a 2x3 table has dof 2, so fixing
# (x11,x12) determines the row.  Leaves = fibre states.
# ---------------------------------------------------------------------------
def build_tree(states):
    """Return level-1 children x11 with subtree leaf-counts, and, per x11, the
    level-2 children x12 (each a single leaf)."""
    x11 = states[:, 0]
    x12 = states[:, 1]
    roots = sorted(set(int(v) for v in x11))
    subtree = {k: int((x11 == k).sum()) for k in roots}          # leaves under x11=k
    children2 = {k: sorted(int(states[i, 1]) for i in range(len(states)) if states[i, 0] == k)
                 for k in roots}
    return roots, subtree, children2


def ce_count(states, mode, n_samp, seed):
    """|F| = E_q[1/q(path)] over root-to-leaf paths.  mode='uniform' or 'ce'
    (subtree-count-proportional = zero-variance)."""
    rng = np.random.default_rng(seed)
    roots, subtree, children2 = build_tree(states)
    roots = np.array(roots)
    est = np.empty(n_samp)
    for s in range(n_samp):
        if mode == "uniform":
            k = int(rng.choice(roots)); q1 = 1.0 / len(roots)
            kids = children2[k]; q2 = 1.0 / len(kids)
        else:                                                    # CE-optimal tilt
            w = np.array([subtree[int(k)] for k in roots], float)
            k = int(rng.choice(roots, p=w / w.sum())); q1 = subtree[k] / w.sum()
            kids = children2[k]; q2 = 1.0 / len(kids)            # each child a single leaf
        est[s] = 1.0 / (q1 * q2)
    return est.mean(), est.std()


def stochastic_enumeration(states, B, reps, seed):
    """B-cursor tree-size estimator (Knuth for B=1): est = prod_t N_t/b_{t-1}, where
    b_{t-1} is the number of cursors retained at the previous level (b_0=1 root) and
    N_t the total children of those cursors.  Keeps at most B nodes per level,
    subsampling without replacement; exact once B reaches the tree width."""
    rng = np.random.default_rng(seed)
    roots, subtree, children2 = build_tree(states)
    roots = list(roots)
    ests = np.empty(reps)
    for rep in range(reps):
        # level 1: single root (b0=1) -> N1 = len(roots) x11-children
        N1 = len(roots)
        b1 = min(N1, B)
        retained = roots if N1 <= B else list(rng.choice(roots, b1, replace=False))
        w = N1 / 1.0                                    # N1 / b0
        # level 2: children of the retained x11-nodes are leaves
        N2 = sum(len(children2[int(k)]) for k in retained)
        w *= N2 / b1                                    # N2 / b1
        ests[rep] = w
    return ests.mean(), ests.std()


# ---------------------------------------------------------------------------
# nested sampling with uniform reference (flattening); exact + live-point
# ---------------------------------------------------------------------------
def ns_exact(states, mu_star):
    """Exact evidence Z = (1/|F|) sum L and the corridor pinch levels."""
    logL = log_like(states, mu_star)
    n = len(states)
    logZ = logsumexp(logL) - np.log(n)
    Z = float(np.exp(logZ)); C = float(np.exp(logsumexp(logL)))
    # corridor = states with x11 >= 1; wall/lobe states have x11 == 0
    wall = states[:, 0] == 0
    corridor = ~wall
    Lmax = float(np.exp(logL.max()))
    Lpinch = float(np.exp(logL[corridor].max() - logL.max()))     # relative to Lmax
    # once L* passes the top corridor weight, only the wall (lobe) states remain
    Xpinch = float((logL > logL[corridor].max()).sum()) / n
    return dict(Z=Z, C=C, logZ=logZ, Lpinch=Lpinch, Xpinch=Xpinch,
                n_wall=int(wall.sum()), n_corr=int(corridor.sum()))


def ns_livepoint(states, mu_star, N, seed, max_iter=6000):
    """Standard nested sampling: N uniform live points, delete the lowest-L, shrink
    prior mass by exp(-1/N), replace by a uniform draw above threshold.  Returns the
    log-evidence estimate and the iteration at which the corridor pinches (the live
    set stops bridging the lobes)."""
    rng = np.random.default_rng(seed)
    logL = log_like(states, mu_star)
    n = len(states)
    corridor_top = logL[states[:, 0] >= 1].max()
    live = rng.integers(0, n, size=N)                    # uniform live points (indices)
    shells = []
    i_pinch = None
    for it in range(max_iter):
        j = int(np.argmin(logL[live]))
        Lstar = logL[live[j]]
        logX = -it / N
        shell = np.exp(logX) - np.exp(-(it + 1) / N)
        shells.append(Lstar + np.log(max(shell, 1e-300)))
        # corridor pinch: no live point is a corridor (x11>=1) state above Lstar
        above = live[logL[live] > Lstar]
        if i_pinch is None and (len(above) == 0 or (states[above, 0] >= 1).sum() == 0):
            i_pinch = it
        # replace worst by a uniform draw with L > Lstar
        pool = np.nonzero(logL > Lstar)[0]
        if len(pool) == 0:
            break
        live[j] = int(rng.choice(pool))
    logZ_est = logsumexp(shells) if shells else -np.inf
    return logZ_est, (i_pinch if i_pinch is not None else max_iter)


# ---------------------------------------------------------------------------
# multilevel splitting: C(mu*) = C(1) * prod_k E_{pi_{mu_k}}[(mu_{k+1}/mu_k)^{x11}]
# each factor computed EXACTLY (enumerable) and by sampling; two-directionality.
# ---------------------------------------------------------------------------
def splitting_exact(states, mu_star, K):
    """The telescoping product with each level expectation computed exactly."""
    levels = np.geomspace(1.0, mu_star, K + 1)
    x11 = states[:, 0].astype(float)
    C = float(np.exp(logsumexp(log_like(states, 1.0))))          # C(1)
    for k in range(K):
        lw = log_like(states, levels[k])
        p = np.exp(lw - logsumexp(lw))                           # pi_{mu_k}
        ratio = float((p * (levels[k + 1] / levels[k]) ** x11).sum())
        C *= ratio
    return C, levels


def splitting_sampled(states, mu_star, K, M, seed):
    """Same telescoping, each level expectation estimated by M draws from pi_{mu_k}
    (exact multinomial over the enumerated fibre), plus up/down level-move rates for
    the two-directionality diagnostic (contrast: nested sampling has down-rate 0)."""
    rng = np.random.default_rng(seed)
    levels = np.geomspace(1.0, mu_star, K + 1)
    x11 = states[:, 0].astype(float)
    C = float(np.exp(logsumexp(log_like(states, 1.0))))
    for k in range(K):
        lw = log_like(states, levels[k])
        p = np.exp(lw - logsumexp(lw))
        draws = rng.choice(len(states), size=M, p=p)
        C *= float(((levels[k + 1] / levels[k]) ** x11[draws]).mean())
    # two-directionality: a level-index random walk with acceptance to adjacent level
    a = 0; ups = downs = 0
    lws = [log_like(states, lv) for lv in levels]
    # sample a representative state at each level once, walk the ladder
    xi = int(rng.choice(len(states), p=np.exp(lws[0] - logsumexp(lws[0]))))
    for _ in range(4000):
        step = 1 if rng.random() < 0.5 else -1
        b = a + step
        if 0 <= b <= K:
            acc = min(1.0, np.exp(lws[b][xi] - lws[a][xi]))
            if rng.random() < acc:
                a = b; ups += (step > 0); downs += (step < 0)
        # refresh state at current level
        pa = np.exp(lws[a] - logsumexp(lws[a])); xi = int(rng.choice(len(states), p=pa))
    tot = max(ups + downs, 1)
    return C, ups / tot, downs / tot


def main():
    out = []
    W = 80
    out.append("=" * W)
    out.append("Tier C -- counting / rare-event estimators of the fibre normaliser")
    out.append("all estimators cross-checked against EXACT enumeration")
    out.append("=" * W)

    # ---- balanced fibre |F|=10
    S10 = table_fibre(4, (3, 3, 2))
    out.append(f"\nbalanced fibre (4,4)|(3,3,2): |F| = {len(S10)} (exact enumeration)")
    roots, subtree, _ = build_tree(S10)
    out.append(f"  self-reducibility tree: x11 children {roots}, leaf counts "
               f"{ {k: subtree[k] for k in roots} }, total {sum(subtree.values())}")

    out.append("\n[CE] cross-entropy counting of |F| (E_q[1/q(path)]):")
    for M in (200, 2000):
        mu_m, mu_s = ce_count(S10, "uniform", M, seed=1)
        ce_m, ce_s = ce_count(S10, "ce", M, seed=1)
        out.append(f"  M={M:>5}: uniform |F|={mu_m:6.3f} (sd/mean={mu_s/mu_m:.3f})   "
                   f"CE-optimal |F|={ce_m:6.3f} (sd/mean={ce_s/max(ce_m,1e-9):.1e})")

    out.append("\n[SE] stochastic enumeration (B=1 is Knuth), reps=4000:")
    for B in (1, 2, 3, 4):
        m, s = stochastic_enumeration(S10, B, reps=4000, seed=0)
        tag = "Knuth" if B == 1 else "SE   "
        out.append(f"  B={B} ({tag}): mean={m:6.3f}  sd={s:5.3f}")

    out.append("\n[NS] nested sampling, uniform reference (returns C = Z|F|):")
    for mu in (0.05, 0.01):
        e = ns_exact(S10, mu)
        out.append(f"  mu*={mu:g}: exact Z={e['Z']:.6e}  C=Z|F|={e['C']:.4f}  "
                   f"L*_pinch={e['Lpinch']:.3e} (~mu*)  X_pinch={e['Xpinch']:.3f} (fixed)  "
                   f"[{e['n_wall']} wall / {e['n_corr']} corridor states]")

    out.append("\n[SP] multilevel splitting, C = C(1) prod_k E[(mu_{k+1}/mu_k)^x11]:")
    for mu in (1e-2, 1e-3):
        Cx, _ = splitting_exact(S10, mu, K=6)
        Cs, up, dn = splitting_sampled(S10, mu, K=6, M=4000, seed=0)
        Cexact = float(np.exp(logsumexp(log_like(S10, mu))))
        out.append(f"  mu*={mu:g}: exact C={Cexact:.5f}  splitting(exact telescope)={Cx:.5f}  "
                   f"splitting(sampled)={Cs:.5f} ({abs(Cs-Cexact)/Cexact*100:.1f}% err)")
    out.append(f"       level-index moves: up={up:.2f}, down={dn:.2f}  "
               f"(nested sampling is one-directional: down=0)")

    # ---- larger fibre |F|=61
    S61 = table_fibre(12, (8, 8, 8))
    out.append(f"\nlarger fibre (12,12)|(8,8,8): |F| = {len(S61)} (exact enumeration)")
    e61 = ns_exact(S61, 0.15)
    out.append(f"  [NS] exact log Z = {e61['logZ']:.4f}  (mu*=0.15); "
               f"{e61['n_wall']} wall states")
    out.append("  [NS] live-point evidence error vs N (mean |logZ_NS - logZ_exact|, 40 reps):")
    for N in (5, 10, 20, 40, 80, 160):
        errs = []
        for rep in range(40):
            lz, _ = ns_livepoint(S61, 0.15, N, seed=1000 * N + rep)
            errs.append(abs(lz - e61["logZ"]))
        out.append(f"       N={N:>4}: {np.mean(errs):.4f}")

    txt = "\n".join(out)
    print(txt)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "counting_numbers.txt"), "w") as f:
        f.write(txt + "\n")
    print("\nnumbers written: results/counting_numbers.txt")


if __name__ == "__main__":
    main()
