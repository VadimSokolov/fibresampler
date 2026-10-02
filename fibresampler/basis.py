"""Change of lattice basis as an alternative to reflection (Section 6 of Paper I).

A fibre in its free-coordinate frame is the set of integer points of a convex polytope,
and single-ray hit-and-run moves along the coordinate axes.  Reflection repairs a bad
axis defect *trajectory by trajectory*.  A change of basis repairs it *once*: replace
the free-frame basis ``I`` by a unimodular ``V`` (an integer matrix with determinant
+-1) whose columns are short in the Mahalanobis metric of the fibre, so that the polytope
looks round in the new coordinates.  The move set ``U V`` spans the same lattice
``ker_Z(A)``; only the directions change.

The reduction is Lenstra-Lenstra-Lovasz (LLL) in the quadratic form ``Sigma^{-1}``, where
``Sigma`` is the covariance of the free coordinates under the target.  ``Sigma`` is exact
on an enumerated fibre and estimated from a pilot run otherwise; the basis is then frozen,
so the main chain is an ordinary fixed-kernel reversible Gibbs sampler (no adaptation).

Two safeguards live here because a reduced basis is *not* automatically augmenting:

  * ``union_moves`` keeps the certified PLB moves and appends the reduced vectors, so
    irreducibility is inherited from the PLB and the mixed kernel never does worse than
    the PLB sampler by more than the mixing weight (the no-hurt floor of Prop. nohurt);
  * ``is_irreducible`` checks connectivity of an enumerated fibre under a move set.

Everything is exact integer arithmetic on the lattice side; floating point enters only in
the Gram-Schmidt step of LLL, on matrices of order r-n.
"""

from __future__ import annotations

import copy
from itertools import product

import numpy as np

__all__ = [
    "int_det", "lll_reduce", "free_covariance", "reduce_basis", "with_basis",
    "union_moves", "weighted_union_matrix", "surrogate_gap", "hill_climb_basis",
    "recommend_basis", "hold_fixed_basis", "RayView", "free_points", "is_irreducible", "is_augmenting",
    "to_frame", "enumerate_unimodular_2d", "random_unimodular",
]


# ---------------------------------------------------------------------------
# exact integer determinant (Bareiss), used to certify unimodularity
# ---------------------------------------------------------------------------
def int_det(M) -> int:
    """Exact determinant of an integer matrix by fraction-free (Bareiss) elimination."""
    A = [[int(v) for v in row] for row in np.asarray(M)]
    n = len(A)
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if swap is None:
                return 0
            A[k], A[swap] = A[swap], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


# ---------------------------------------------------------------------------
# LLL in a quadratic form
# ---------------------------------------------------------------------------
def lll_reduce(M: np.ndarray, delta: float = 0.99, max_iter: int = 100000) -> np.ndarray:
    """LLL-reduce the standard basis of Z^m in the inner product <x,y> = x^T M y.

    Returns an integer unimodular matrix ``V`` whose COLUMNS are the reduced basis
    vectors (written in the original free coordinates).  ``M`` must be symmetric
    positive definite; ``delta`` in (1/4, 1) is the Lovasz constant.

    Gram-Schmidt data are read off a Cholesky factor of the current Gram matrix
    ``G = V^T M V``: with ``G = L L^T``, the squared GS norm of b_i is ``L_ii^2`` and
    the GS coefficient ``mu_ij = L_ij / L_jj``.  Only integer column operations are
    applied to V, so unimodularity is preserved exactly.
    """
    M = np.asarray(M, dtype=float)
    m = M.shape[0]
    V = np.eye(m, dtype=np.int64)
    if m == 1:
        return V
    k, it = 1, 0

    def chol(Vc):
        G = Vc.T @ M @ Vc
        G = 0.5 * (G + G.T)
        jitter = 0.0
        scale = max(float(np.trace(G)) / m, 1e-300)
        for _ in range(8):
            try:
                return np.linalg.cholesky(G + jitter * scale * np.eye(m))
            except np.linalg.LinAlgError:
                jitter = 1e-14 if jitter == 0.0 else jitter * 100.0
        raise np.linalg.LinAlgError("Gram matrix not positive definite even with jitter")

    L = chol(V)
    while k < m:
        it += 1
        if it > max_iter:
            raise RuntimeError("LLL did not converge")
        # size-reduce b_k against b_{k-1},...,b_0
        for j in range(k - 1, -1, -1):
            q = int(np.rint(L[k, j] / L[j, j]))
            if q != 0:
                V[:, k] -= q * V[:, j]
                L = chol(V)
        mu = L[k, k - 1] / L[k - 1, k - 1]
        if L[k, k] ** 2 >= (delta - mu ** 2) * L[k - 1, k - 1] ** 2:   # Lovasz
            k += 1
        else:
            V[:, [k - 1, k]] = V[:, [k, k - 1]]
            L = chol(V)
            k = max(k - 1, 1)
    return V


def free_covariance(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Covariance of free-coordinate points (rows), optionally pi-weighted."""
    X = np.asarray(points, dtype=float)
    if weights is None:
        return np.atleast_2d(np.cov(X.T, bias=True))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mu = w @ X
    Xc = X - mu
    return (Xc * w[:, None]).T @ Xc


def precision_floor(S: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    """Inverse of the covariance ``S`` with its eigenvalues floored at ``ridge`` times the
    largest, symmetrised and rescaled to unit mean diagonal.  A frozen coordinate (a pilot
    that never moves it) makes ``S`` singular; the floor caps the penalty at a large finite
    value so the Mahalanobis metric stays positive definite in floating point."""
    S = 0.5 * (np.atleast_2d(S) + np.atleast_2d(S).T)
    lam, Q = np.linalg.eigh(S)
    floor = max(float(lam.max()) * ridge, 1e-12)
    M = (Q / np.maximum(lam, floor)) @ Q.T
    M = 0.5 * (M + M.T)
    return M / max(float(np.trace(M)) / M.shape[0], 1e-300)


def reduce_basis(points: np.ndarray, weights: np.ndarray | None = None,
                 delta: float = 0.99, ridge: float = 1e-9) -> np.ndarray:
    """Unimodular V reducing the free frame in the Mahalanobis metric of ``points``.

    A frozen coordinate (zero variance) would make the metric singular; a ridge
    proportional to the mean variance keeps it positive definite without changing the
    ordering of the informative directions.
    """
    S = free_covariance(points, weights)
    M = precision_floor(S, ridge)
    V = lll_reduce(M, delta=delta)
    if abs(int_det(V)) != 1:
        raise AssertionError("reduced basis is not unimodular")
    return V


# ---------------------------------------------------------------------------
# problem views
# ---------------------------------------------------------------------------
class RayView:
    """Duck-typed problem exposing the ray interface of ``FibreProblem`` for any
    state array and integer move matrix, including a duck-typed band."""

    def __init__(self, states, U, index=None, plb_info=None, name=""):
        self.states = np.asarray(states)
        self.U = np.asarray(U, dtype=np.int64)
        self.index = index if index is not None else {tuple(s): i for i, s in enumerate(self.states)}
        self.plb_info = plb_info
        self.name = name

    @property
    def n_moves(self):
        return self.U.shape[1]

    @property
    def fibre_size(self):
        return len(self.states)

    @property
    def r(self):
        return self.states.shape[1]

    def ray(self, i, j):
        from .fibre import ray_endpoints
        x = self.states[i]
        u = self.U[:, j]
        b_min, b_max = ray_endpoints(x, u)
        out = []
        for b in range(b_min, b_max + 1):
            k = self.index.get(tuple(x + b * u))
            if k is not None:
                out.append(k)
        return out


def with_basis(prob, V: np.ndarray) -> RayView:
    """The same fibre sampled along the columns of ``U V`` (full-space moves)."""
    V = np.asarray(V, dtype=np.int64)
    return RayView(prob.states, prob.U @ V, index=prob.index,
                   plb_info=getattr(prob, "plb_info", None),
                   name=f"{getattr(prob, 'name', '')} [reduced]")


def union_moves(prob, V: np.ndarray) -> RayView:
    """Certified PLB moves followed by the reduced-basis moves (2m directions)."""
    V = np.asarray(V, dtype=np.int64)
    U2 = np.concatenate([prob.U, prob.U @ V], axis=1)
    return RayView(prob.states, U2, index=prob.index,
                   plb_info=getattr(prob, "plb_info", None),
                   name=f"{getattr(prob, 'name', '')} [PLB + reduced]")


def weighted_union_matrix(prob, V: np.ndarray, log_w: np.ndarray, plb_weight: float = 0.1):
    """Transition matrix of the weighted union kernel: with probability ``plb_weight`` the
    direction is drawn from the certified PLB moves, otherwise from the reduced moves; the
    whole ray is then resampled from the target.  Irreducible whenever the PLB is, and its
    gap is at least ``plb_weight`` times the PLB gap (the no-hurt floor)."""
    V = np.asarray(V, dtype=np.int64)
    m = prob.n_moves
    red = with_basis(prob, V)
    M = prob.fibre_size
    Q = np.zeros((M, M))
    for view_, wgt in ((prob, plb_weight / m), (red, (1.0 - plb_weight) / m)):
        for i in range(M):
            for j in range(m):
                ray = view_.ray(i, j)
                lw = log_w[list(ray)]
                mx = np.max(lw[np.isfinite(lw)])
                p = np.where(np.isfinite(lw), np.exp(lw - mx), 0.0)
                p = p / p.sum()
                for k, pk in zip(ray, p):
                    Q[i, k] += wgt * pk
    return Q


# ---------------------------------------------------------------------------
# Gaussian surrogate and basis selection
# ---------------------------------------------------------------------------
def surrogate_gap(Sigma: np.ndarray, V: np.ndarray, Q: np.ndarray | None = None) -> float:
    """Gap of random-scan Gibbs along the columns of V on N(0, Sigma): lambda_min of the
    unit-diagonal precision, divided by the number of directions (Roberts and Sahu 1997).  ``Q`` is the
    (floored) precision; pass it to avoid recomputing it inside a search."""
    if Q is None:
        Q = precision_floor(Sigma)
    Qv = np.asarray(V, dtype=float).T @ Q @ np.asarray(V, dtype=float)
    d = np.sqrt(np.diag(Qv))
    return float(np.linalg.eigvalsh(Qv / np.outer(d, d))[0] / Q.shape[0])


def hill_climb_basis(Sigma: np.ndarray, start: np.ndarray | None = None, bound: int = 3,
                     max_sweeps: int = 200) -> tuple:
    """Greedy ascent of the surrogate gap over unimodular column shears V <- V (I +- E_ij),
    entries capped at ``bound``.  Returns (V, surrogate_gap)."""
    m = Sigma.shape[0]
    Q = precision_floor(Sigma)
    V = np.eye(m, dtype=np.int64) if start is None else np.array(start, dtype=np.int64)
    f = surrogate_gap(Sigma, V, Q)
    for _ in range(max_sweeps):
        best_f, best_W = f, None
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                for s in (-1, 1):
                    W = V.copy()
                    W[:, i] += s * V[:, j]
                    if np.abs(W).max() > bound:
                        continue
                    g = surrogate_gap(Sigma, W, Q)
                    if g > best_f + 1e-12:
                        best_f, best_W = g, W
        if best_W is None:
            break
        f, V = best_f, best_W
    return V, f


def recommend_basis(points: np.ndarray, weights: np.ndarray | None = None,
                    bound: int = 3, min_gain: float = 1.10, delta: float = 0.99):
    """Pilot-based basis selection: keep the PLB unless a candidate improves the Gaussian
    surrogate gap by at least ``min_gain``.  Candidates are the identity, the LLL basis
    and the hill-climbed basis started from each.  Returns (V, info)."""
    S = free_covariance(points, weights)
    m = S.shape[0]
    Qp = precision_floor(S)
    I = np.eye(m, dtype=np.int64)
    V_lll = reduce_basis(points, weights, delta=delta)
    cands = {"identity": I, "lll": V_lll}
    cands["climb(I)"] = hill_climb_basis(S, I, bound)[0]
    cands["climb(lll)"] = hill_climb_basis(S, V_lll, bound)[0]
    scores = {k: surrogate_gap(S, V, Qp) for k, V in cands.items()}
    best = max(scores, key=scores.get)
    chosen = best if scores[best] >= min_gain * scores["identity"] else "identity"
    info = {"chosen": chosen, "surrogate": scores,
            "gain": scores[chosen] / scores["identity"]}
    return cands[chosen], info


def hold_fixed_basis(U: np.ndarray, rows) -> tuple[np.ndarray, int]:
    """Bottleneck-fixing recombination: a unimodular V with ``U[rows] @ V = [0 | H]``.

    The first ``n_slice`` columns of ``U V`` hold the coordinates ``rows`` fixed exactly and
    span the sublattice ``ker_Z(A) cap {z_rows = 0}``; the remaining columns are the only
    ones that move them.  Column-style Hermite reduction of the integer submatrix
    ``U[rows]`` (Euclid on each row), so it needs the IDENTITY of the bottleneck cells and
    nothing about the target.  Returns ``(V, n_slice)``."""
    B = np.asarray(U)[list(rows)].astype(np.int64).copy()
    r, m = B.shape
    V = np.eye(m, dtype=np.int64)
    active = m
    for i in range(r):
        while True:
            nz = [j for j in range(active) if B[i, j] != 0]
            if len(nz) <= 1:
                break
            jmin = min(nz, key=lambda j: abs(int(B[i, j])))
            for j in nz:
                if j != jmin:
                    q = int(B[i, j]) // int(B[i, jmin])
                    B[:, j] -= q * B[:, jmin]
                    V[:, j] -= q * V[:, jmin]
        nz = [j for j in range(active) if B[i, j] != 0]
        if nz:
            j, last = nz[0], active - 1
            if j != last:
                B[:, [j, last]] = B[:, [last, j]]
                V[:, [j, last]] = V[:, [last, j]]
            active -= 1
    return V, active


def free_points(prob) -> np.ndarray:
    """Free-coordinate points (rows) of an enumerated fibre in its PLB frame."""
    cols2 = list(prob.plb_info["cols2"])
    return np.asarray(prob.states)[:, cols2].astype(np.int64)


def to_frame(points: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Coordinates c with z = V c for free points z (rows); exact since V is unimodular."""
    Vinv = np.rint(np.linalg.inv(np.asarray(V, dtype=float))).astype(np.int64)
    if not np.array_equal(Vinv @ np.asarray(V, dtype=np.int64), np.eye(V.shape[0], dtype=np.int64)):
        raise AssertionError("V^{-1} is not integral")
    return np.asarray(points, dtype=np.int64) @ Vinv.T


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------
def is_irreducible(prob) -> bool:
    """True iff every state of the enumerated fibre is reachable under the ray moves."""
    n = prob.fibre_size
    seen = np.zeros(n, dtype=bool)
    seen[0] = True
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(prob.n_moves):
            for k in prob.ray(i, j):
                if not seen[k]:
                    seen[k] = True
                    stack.append(k)
    return bool(seen.all())


def is_augmenting(prob) -> bool:
    """Augmenting property of Hazelton et al. (2024): any two states are joined by a path
    that applies each move AT MOST ONCE (each application a full ray jump), every partial
    sum feasible.  Exact search over (state, used-move mask); intended for small fibres."""
    n, m = prob.fibre_size, prob.n_moves
    rays = [[prob.ray(i, j) for j in range(m)] for i in range(n)]
    for s in range(n):
        seen = {(s, 0)}
        stack = [(s, 0)]
        reach = {s}
        while stack:
            i, mask = stack.pop()
            for j in range(m):
                if (mask >> j) & 1:
                    continue
                for k in rays[i][j]:
                    node = (k, mask | (1 << j))
                    if node not in seen:
                        seen.add(node)
                        stack.append(node)
                        reach.add(k)
        if len(reach) < n:
            return False
    return True


# ---------------------------------------------------------------------------
# candidate bases for the oracle comparison
# ---------------------------------------------------------------------------
def enumerate_unimodular_2d(bound: int = 4):
    """Every 2x2 integer matrix with |entries| <= bound and determinant +-1."""
    out = []
    rng = range(-bound, bound + 1)
    for a, b, c, d in product(rng, rng, rng, rng):
        if abs(a * d - b * c) == 1:
            out.append(np.array([[a, b], [c, d]], dtype=np.int64))
    return out


def random_unimodular(m: int, rng: np.random.Generator, steps: int = 12) -> np.ndarray:
    """Random unimodular matrix: a product of elementary column additions and swaps."""
    V = np.eye(m, dtype=np.int64)
    for _ in range(steps):
        i, j = rng.choice(m, size=2, replace=False)
        if rng.random() < 0.8:
            V[:, i] += int(rng.choice([-1, 1])) * V[:, j]
        else:
            V[:, [i, j]] = V[:, [j, i]]
    return V
