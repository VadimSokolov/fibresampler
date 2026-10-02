"""Negative-binomial (Poisson-gamma) augmentation and the exact joint-chain
operator for the keystone test (Prediction 2 / Theorem thm:loading-nb).

The augmented model (eq. nb-mixture / nb-joint) is, per coordinate j,
    x_j | lambda_j ~ Pois(lambda_j),   lambda_j ~ Gamma(alpha_j, alpha_j/mu_j),
so marginally x_j ~ NB(mean mu_j, dispersion alpha_j).  The sampler
(Algorithm main, blocks a+b) is a two-block Metropolis-within-Gibbs chain on
(x, lambda): a fibre update at fixed lambda (single-ray Gibbs hit-and-run,
Poisson mean lambda) and an EXACT gamma refresh lambda_j | x_j.

**Exact joint-chain SLEM without discretising 8-d lambda.**  Because the gamma
block is an exact conditional draw, its kernel is an orthogonal projection Pi
onto functions of x.  The joint operator P = P_x Pi therefore shares its non-zero
spectrum with the x-marginal operator

    K(x, x') = E_{lambda ~ f(lambda|x)} [ P_x^lambda(x -> x') ] ,

which is NOT the NB-marginal fibre chain (that would be a different sampler);
it is the exact projection of the joint chain the theorem is about, and it is
reversible w.r.t. the NB marginal pi_NB on the fibre.  The lambda-expectation is
evaluated by deterministic generalised Gauss-Laguerre quadrature per coordinate
(the "lambda-grid"); the quadrature order n_q is the refinement knob.
"""

from __future__ import annotations

import numpy as np
from scipy.special import roots_genlaguerre, gammaln

__all__ = ["nb_log_pmf", "gamma_quadrature", "nb_augmented_operator",
           "da_operator", "kr_operators"]


def nb_log_pmf(prob, mu, alpha) -> np.ndarray:
    """Unnormalised log NB-marginal weight log pi_NB(x) for every fibre state.

    NB(x_j | size=alpha_j, mean=mu_j):
      lgamma(x+a) - lgamma(a) - lgamma(x+1) + a log(a/(a+mu)) + x log(mu/(a+mu)).
    """
    mu = np.broadcast_to(np.asarray(mu, float), (prob.r,))
    alpha = np.broadcast_to(np.asarray(alpha, float), (prob.r,))
    lw = np.zeros(prob.fibre_size)
    for idx, x in enumerate(prob.states):
        term = (gammaln(x + alpha) - gammaln(alpha) - gammaln(x + 1)
                + alpha * np.log(alpha / (alpha + mu))
                + x * np.log(mu / (alpha + mu)))
        lw[idx] = term.sum()
    return lw


def gamma_quadrature(shape, rate, n_q):
    """Nodes/weights approximating E[g(lambda)] for lambda ~ Gamma(shape, rate)
    (rate parametrisation) by n_q-point generalised Gauss-Laguerre:
        E[g] ~ sum_k w_k g(nodes_k),  sum_k w_k = 1.
    """
    t, w = roots_genlaguerre(n_q, shape - 1.0)   # weight t^{shape-1} e^{-t}
    wsum = w.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        # very concentrated gamma (large shape) underflows the quadrature weights;
        # the distribution is then effectively a point mass at its mean.
        return np.array([shape / rate]), np.array([1.0])
    nodes = t / rate
    weights = w / wsum
    return nodes, weights


def nb_augmented_operator(prob, mu, alpha, n_q=10):
    """Exact (quadrature) joint-chain x-operator K and its NB stationary pi.

    Returns (K, pi_NB).  K[i, :] is the expected single-ray Gibbs hit-and-run
    transition row from state i, averaged over lambda ~ f(lambda | x_i) with
    f(lambda_j | x_j) = Gamma(alpha_j + x_j, alpha_j/mu_j + 1).
    """
    mu = np.broadcast_to(np.asarray(mu, float), (prob.r,)).copy()
    alpha = np.broadcast_to(np.asarray(alpha, float), (prob.r,)).copy()
    N = prob.fibre_size
    Nm = prob.n_moves
    K = np.zeros((N, N))

    # precompute support (changing coords) of each move
    supp = [np.nonzero(prob.U[:, j])[0] for j in range(Nm)]

    for i in range(N):
        x = prob.states[i]
        shape = alpha + x                     # per-coordinate gamma shape
        rate = alpha / mu + 1.0               # per-coordinate gamma rate
        for j in range(Nm):
            ray = prob.ray(i, j)              # state indices on the ray (incl. i)
            if len(ray) == 1:
                K[i, i] += 1.0 / Nm           # no feasible move: hold
                continue
            S = supp[j]
            zS = prob.states[ray][:, S]       # (ray_len, |S|) changing coords
            logfact = gammaln(zS + 1).sum(axis=1)          # (ray_len,)
            # tensor quadrature grid over the |S| changing coordinates
            per = [gamma_quadrature(shape[c], rate[c], n_q) for c in S]
            node_lists = [p[0] for p in per]
            wt_lists = [p[1] for p in per]
            grids = np.meshgrid(*node_lists, indexing="ij")
            wgrids = np.meshgrid(*wt_lists, indexing="ij")
            logLam = np.log(np.stack([g.ravel() for g in grids], axis=1))   # (G,|S|)
            gw = np.prod(np.stack([w.ravel() for w in wgrids], axis=1), axis=1)  # (G,)
            # softmax over ray points for every grid node, then average
            logits = zS @ logLam.T - logfact[:, None]      # (ray_len, G)
            logits -= logits.max(axis=0, keepdims=True)
            W = np.exp(logits); W /= W.sum(axis=0, keepdims=True)
            expected_landing = W @ gw                       # (ray_len,)
            for k, s in enumerate(ray):
                K[i, s] += (1.0 / Nm) * expected_landing[k]

    lw = nb_log_pmf(prob, mu, alpha)
    m = lw.max()
    pi = np.exp(lw - m); pi /= pi.sum()
    return K, pi


# ---------------------------------------------------------------------------
# Monte-Carlo joint-chain operators for the R-dependence study (Prediction 2).
# These share the projection reduction with nb_augmented_operator but keep a
# COMMON lambda across all fibre states so that (P^lambda)^R is well defined;
# hence Monte Carlo over lambda rather than per-row quadrature.
# ---------------------------------------------------------------------------
def _precompute_rays(prob):
    return [[prob.ray(i, j) for j in range(prob.n_moves)]
            for i in range(prob.fibre_size)]


def _build_P(prob, log_w, rays):
    N, Nm = prob.fibre_size, prob.n_moves
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(Nm):
            r = rays[i][j]
            if len(r) == 1:
                P[i, i] += 1.0 / Nm; continue
            lw = log_w[r]; lw = lw - lw.max(); w = np.exp(lw); w /= w.sum()
            for k, s in enumerate(r):
                P[i, s] += w[k] / Nm
    return P


def da_operator(prob, mu, alpha, M=200000, seed=0):
    """Ideal data-augmentation chain (exact x|lambda resample, R->inf):
    K_DA(x,.) = E_{lambda~f(lambda|x)}[ p(.|lambda) ].  Returns (K, pi_NB)."""
    from scipy.special import gammaln
    mu = np.broadcast_to(np.asarray(mu, float), (prob.r,))
    alpha = np.broadcast_to(np.asarray(alpha, float), (prob.r,))
    X = prob.states.astype(float); logfactX = gammaln(X + 1).sum(1)
    rng = np.random.default_rng(seed); N = prob.fibre_size; K = np.zeros((N, N))
    for i in range(N):
        x = prob.states[i]
        lam = rng.gamma(alpha + x, 1.0 / (alpha / mu + 1), size=(M, prob.r))
        logits = X @ np.log(lam).T - logfactX[:, None]
        logits -= logits.max(0, keepdims=True); W = np.exp(logits); W /= W.sum(0, keepdims=True)
        K[i, :] = W.mean(1)
    lw = nb_log_pmf(prob, mu, alpha); pi = np.exp(lw - lw.max()); pi /= pi.sum()
    return K, pi


def kr_operators(prob, mu, alpha, Rlist, M=600, seed=3):
    """Projected x-chain with R hit-and-run fibre updates per lambda refresh,
    K_R(x,.) = E_{lambda~f(lambda|x)}[ (P^lambda)^R (x,.) ], for each R in Rlist.
    Returns (dict R->K_R, pi_NB).  Monte Carlo over lambda."""
    from scipy.special import gammaln
    mu = np.broadcast_to(np.asarray(mu, float), (prob.r,))
    alpha = np.broadcast_to(np.asarray(alpha, float), (prob.r,))
    X = prob.states.astype(float); logfactX = gammaln(X + 1).sum(1)
    rays = _precompute_rays(prob); rng = np.random.default_rng(seed)
    N = prob.fibre_size; Rmax = max(Rlist)
    acc = {R: np.zeros((N, N)) for R in Rlist}
    for i in range(N):
        x = prob.states[i]
        lam = rng.gamma(alpha + x, 1.0 / (alpha / mu + 1), size=(M, prob.r))
        for m in range(M):
            P = _build_P(prob, X @ np.log(lam[m]) - logfactX, rays)
            v = np.zeros(N); v[i] = 1.0
            for R in range(1, Rmax + 1):
                v = v @ P
                if R in acc:
                    acc[R][i, :] += v
    lw = nb_log_pmf(prob, mu, alpha); pi = np.exp(lw - lw.max()); pi /= pi.sum()
    return {R: acc[R] / M for R in Rlist}, pi
