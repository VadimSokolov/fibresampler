"""Exact spectral analysis of fibre samplers on small fibres.

This module assembles the one-step transition matrix ``Q`` of a fibre sampler
explicitly and diagonalises it, giving the exact second-largest eigenvalue
modulus (SLEM) and spectral gap.  Predictions 1 and 2 are decided here, not by
autocorrelation estimates (per Section 9.3 and the protocol).

It also computes Sinclair's maximum edge loading ``rho`` for the augmenting
up-and-down canonical paths of Hazelton (2024), the quantity whose divergence is
the mechanism behind the Poisson loading obstruction (Theorem 5.1(ii)).

Currently the single-ray Gibbs hit-and-run sampler (SR) is implemented; the
reflective proposal transition matrix is added with the full experiment grid.
"""

from __future__ import annotations

import numpy as np

__all__ = ["log_weights", "sr_transition_matrix", "slem",
           "updown_paths", "sinclair_loading"]


# ---------------------------------------------------------------------------
# Target weights
# ---------------------------------------------------------------------------
def log_weights(prob, model: str = "poisson", theta=None) -> np.ndarray:
    """Unnormalised log target log f(x) for every fibre state.

    model='uniform' -> f == 1.  model='poisson' -> product Poisson with mean
    vector ``theta`` (normalising constants cancel in every ray/ratio, so the
    factorials are kept only for numerical faithfulness).
    """
    states = prob.states
    if model == "uniform":
        return np.zeros(len(states))
    if model == "poisson":
        theta = np.asarray(theta, dtype=float)
        with np.errstate(divide="ignore"):
            logt = np.log(theta)
        from scipy.special import gammaln
        # x_i log theta_i - log(x_i!) ; handle theta_i=0 (=> x_i must be 0)
        lw = np.zeros(len(states))
        for i, x in enumerate(states):
            term = 0.0
            for xi, lt, th in zip(x, logt, theta):
                if th == 0.0:
                    if xi > 0:
                        term = -np.inf
                        break
                else:
                    term += xi * lt
            lw[i] = term
        lw -= np.array([gammaln(x + 1).sum() for x in states])
        return lw
    raise ValueError(f"unknown model {model!r}")


def normalised_pi(log_w: np.ndarray) -> np.ndarray:
    m = np.max(log_w[np.isfinite(log_w)])
    w = np.where(np.isfinite(log_w), np.exp(log_w - m), 0.0)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Single-ray Gibbs hit-and-run transition matrix
# ---------------------------------------------------------------------------
def sr_transition_matrix(prob, log_w: np.ndarray) -> np.ndarray:
    """Transition matrix Q of the Gibbs (heat-bath) single-ray hit-and-run
    sampler H(M_U): pick a move direction uniformly, then resample the whole
    ray from the conditional target.  Reversible w.r.t. pi ∝ exp(log_w).
    """
    M = prob.fibre_size
    N = prob.n_moves
    Q = np.zeros((M, M))
    # precompute per-state, per-direction rays once
    for i in range(M):
        for j in range(N):
            ray = prob.ray(i, j)
            lw = log_w[list(ray)]
            m = np.max(lw[np.isfinite(lw)])
            w = np.where(np.isfinite(lw), np.exp(lw - m), 0.0)
            Z = w.sum()
            if Z <= 0:
                Q[i, i] += 1.0 / N   # degenerate ray: hold
                continue
            for k, wk in zip(ray, w):
                Q[i, k] += (1.0 / N) * (wk / Z)
    return Q


def slem(Q: np.ndarray, pi: np.ndarray):
    """Exact SLEM and spectral gap of a pi-reversible stochastic matrix.

    Symmetrise S = D^{1/2} Q D^{-1/2} (D=diag pi) so eigenvalues are real, then
    SLEM = max modulus among all but the Perron eigenvalue 1.
    """
    d = np.sqrt(pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = (d[:, None] * Q) / d[None, :]
    S = 0.5 * (S + S.T)                    # kill round-off asymmetry
    evals = np.linalg.eigvalsh(S)
    evals = np.sort(evals)                 # ascending
    lam1 = evals[-1]                       # ~ 1
    rest = np.abs(evals[:-1])
    slem_val = rest.max()
    return slem_val, 1.0 - slem_val, evals


# ---------------------------------------------------------------------------
# Sinclair loading via augmenting up-and-down canonical paths
# ---------------------------------------------------------------------------
def _updown_order(prob, participating, diff):
    """Order the participating moves so that, in every coordinate (row) of x,
    all positive changes precede negative ones (Hazelton 2024, eq. 11 / proof of
    Thm A.4).  The signed move for j is v_j = sign(diff_j) * U[:,j]; we require
    p before q whenever some row has v_p>0 and v_q<0.  Kahn topological sort with
    smallest-index tie-break; falls back to a feasibility search on the (rare)
    event of a cycle (cannot occur for a genuinely augmenting basis).
    """
    U = prob.U
    signed = {j: np.sign(diff[j]) * U[:, j] for j in participating}
    # required precedence edges p -> q
    succ = {j: set() for j in participating}
    indeg = {j: 0 for j in participating}
    for p in participating:
        for q in participating:
            if p == q:
                continue
            if ((signed[p] > 0) & (signed[q] < 0)).any():
                if q not in succ[p]:
                    succ[p].add(q)
                    indeg[q] += 1
    order, avail = [], sorted(j for j in participating if indeg[j] == 0)
    while avail:
        j = avail.pop(0)
        order.append(j)
        for q in sorted(succ[j]):
            indeg[q] -= 1
            if indeg[q] == 0:
                avail.append(q)
        avail.sort()
    if len(order) == len(participating):
        return order
    return None   # cycle: caller falls back to feasibility search


def updown_paths(prob):
    """Augmenting up-and-down canonical path for every ordered pair of states.

    Each participating move is applied once as a single hit-and-run edge (an
    axis jump in free coordinates), in the up-and-down order of Hazelton (2024).
    Returns {(i,k): [ (u,v), ... ]} where each (u,v) is a directed fibre edge.
    """
    from itertools import permutations
    states = prob.states
    free = np.array([prob.free_coords(x) for x in states])
    free_lookup = {tuple(int(t) for t in f): i for i, f in enumerate(free)}

    def state_of(free_vec):
        return free_lookup.get(tuple(int(t) for t in free_vec))

    def build(a, i, order, b):
        cur, cur_idx, edges = a.copy(), i, []
        for j in order:
            nxt = cur.copy(); nxt[j] = b[j]
            nxt_idx = state_of(nxt)
            if nxt_idx is None:
                return None            # infeasible corner
            edges.append((cur_idx, nxt_idx))
            cur, cur_idx = nxt, nxt_idx
        return edges

    paths = {}
    for i in range(len(states)):
        a = free[i]
        for k in range(len(states)):
            if i == k:
                continue
            b = free[k]
            diff = b - a
            participating = [j for j in range(prob.n_moves) if diff[j] != 0]
            order = _updown_order(prob, participating, diff)
            edges = build(a, i, order, b) if order is not None else None
            if edges is None:                       # fallback: any feasible order
                for perm in permutations(participating):
                    edges = build(a, i, list(perm), b)
                    if edges is not None:
                        break
            if edges is None:
                raise RuntimeError(f"no augmenting path for pair ({i},{k})")
            paths[(i, k)] = edges
    return paths


def sinclair_loading(prob, Q: np.ndarray, pi: np.ndarray, paths=None):
    """Sinclair maximum edge loading rho for the given canonical paths.

    rho = max_e  (1 / (pi_u Q_uv))  * sum_{(x,x'): path uses e} pi_x pi_x' ,
    over directed edges e=(u,v) with pi_u Q_uv > 0.  Also returns the argmax
    edge and its move label, to compare with Hazelton's "most overloaded
    transition" statements.
    """
    if paths is None:
        paths = updown_paths(prob)
    flow = {}
    for (i, k), edges in paths.items():
        w = pi[i] * pi[k]
        for e in edges:
            flow[e] = flow.get(e, 0.0) + w
    best_rho, best_e = 0.0, None
    for e, fl in flow.items():
        u, v = e
        cap = pi[u] * Q[u, v]
        if cap <= 0:
            continue
        load = fl / cap
        if load > best_rho:
            best_rho, best_e = load, e
    move_label = None
    if best_e is not None:
        move_label = _edge_move_label(prob, best_e)
    return best_rho, best_e, move_label


def hazelton_loading(prob, Q: np.ndarray, pi: np.ndarray):
    """Maximum edge loading rho, ported verbatim from Hazelton's MaxEdgeLoading
    (SM-code.r, lines 140-203), so it reproduces his published values bit-for-bit
    (P1: 4.093 / 202.3 / 14536 / 1423382).

    For each ORDERED pair (a,b) the canonical path is characterised by its
    free-coordinate displacement z2 = free(b) - free(a); an ordered pair is
    charged to a directed edge e (which moves the free coords by ``edge2``, a
    single-axis multiple b*e_j) iff z2 == edge2 on the support of edge2.  The
    loading of e is  sum_{charged pairs} pi(a)pi(b) / P(e), with denominator the
    transition PROBABILITY P(from->to) (NOT the capacity pi*P).
    """
    N = prob.fibre_size
    free = np.array([prob.free_coords(x) for x in prob.states])   # (N, d) int
    D = free[None, :, :] - free[:, None, :]        # D[a,b] = free[b]-free[a]
    piouter = np.outer(pi, pi)                     # prob.product over ordered pairs
    rho, best_edge = 0.0, None
    for frm in range(N):
        for to in range(N):
            if frm == to or Q[frm, to] <= 0:
                continue
            edge2 = free[to] - free[frm]
            supp = edge2 != 0
            if not supp.any():
                continue
            # ordered pairs whose free-displacement matches edge2 on its support
            match = (D[:, :, supp] == edge2[supp]).all(axis=2)
            numer = piouter[match].sum()
            load = numer / Q[frm, to]
            if load > rho:
                rho, best_edge = load, (frm, to)
    label = _edge_move_label(prob, best_edge) if best_edge is not None else None
    return rho, best_edge, label


def _edge_move_label(prob, edge):
    """Identify which +-u_j realises a directed edge (u,v), for reporting."""
    u, v = edge
    dx = prob.states[v] - prob.states[u]
    for j in range(prob.n_moves):
        col = prob.U[:, j]
        # dx must be a positive or negative integer multiple of column j
        nz = col != 0
        if not nz.any():
            continue
        ratios = dx[nz] / col[nz]
        if np.allclose(ratios, ratios[0]) and np.allclose(dx[~nz], 0):
            b = ratios[0]
            sign = "+" if b > 0 else "-"
            return f"{sign}u_{j+1}"
    return "?"
