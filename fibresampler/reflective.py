"""Reflective lattice sampler (Section 5 of the manuscript, Algorithm 1).

The billiard/reflective proposal chains several axis-aligned lattice segments into
one Metropolis proposal, reflecting off the two face types of the fibre polytope in
the free-coordinate frame:

  * coordinate walls {x_{2,j}=0}: specular reflection is an exact sign flip
    d -> d - 2 d_j e_j  (eq. coord-reflect);
  * image walls {[A1^{-1}(y - A2 x2)]_i = 0}: the "lattice reflection" of
    Definition 5.3, the l1-nearest integer direction to the specular one that
    points back into the polytope and has no larger l1-norm.

Construction used here (a clean, reversible reading of Algorithm 1): the first B
segments run to the wall (deterministic length = feasible max, then reflect), and
the terminal segment B+1 draws a uniform length on its feasible range.  B=0 is
exactly a Metropolis single-ray hit-and-run.  The Metropolis ratio (eq. 15) uses the
probability of the *exact reverse trajectory*; because on the realised path the
reflections are deterministic, the reverse is unique and its probability is a product
of the same factor types (Remark 5.8).

We assemble the exact transition matrix by enumerating every trajectory
(B, initial direction, terminal length) from each fibre state.  This is finite and
gives the reflective sampler's exact SLEM, the same rigour used for Predictions 1-2.
Correctness is checked by detailed balance against the product-Poisson target.
"""

from __future__ import annotations

import numpy as np

from .spectral import log_weights, normalised_pi

__all__ = ["ReflectiveSampler", "reflective_transition_matrix"]


class ReflectiveSampler:
    """Exact reflective-proposal machinery on an enumerated fibre.

    Parameters
    ----------
    prob : FibreProblem
    bmax : int
        Maximum number of bounces B_max.
    pi_B : sequence or None
        Distribution over B in {0,...,bmax}.  Defaults to uniform.  A point mass
        at 0 (``[1.0]``) recovers Metropolis single-ray hit-and-run.
    """

    def __init__(self, prob, bmax=2, pi_B=None):
        self.prob = prob
        self.m = prob.n_moves                       # r - n free coordinates
        self.bmax = int(bmax)
        if pi_B is None:
            pi_B = np.ones(self.bmax + 1) / (self.bmax + 1)
        pi_B = np.asarray(pi_B, float)
        if len(pi_B) < self.bmax + 1:               # pad point-mass / short vectors
            pi_B = np.concatenate([pi_B, np.zeros(self.bmax + 1 - len(pi_B))])
        self.pi_B = pi_B / pi_B.sum()
        self.U = prob.U                              # (r, m) full-x moves
        self.cols1 = list(prob.plb_info["cols1"])
        self.cols2 = list(prob.plb_info["cols2"])
        self.col_is_free = {c: k for k, c in enumerate(self.cols2)}   # full coord -> free index
        # unit directions +/- e_k in the free frame, as full-x moves U (+/- column)
        self.dirs = []                               # list of free-frame unit dirs
        for k in range(self.m):
            e = np.zeros(self.m, dtype=np.int64); e[k] = 1
            self.dirs.append(e.copy()); self.dirs.append(-e.copy())
        self._Ucache = {}

    # -- geometry ---------------------------------------------------------------
    def full_move(self, d):
        """Full-x displacement U d for a free-frame direction d."""
        key = tuple(int(v) for v in d)
        M = self._Ucache.get(key)
        if M is None:
            M = self.U @ np.asarray(d, dtype=np.int64)
            self._Ucache[key] = M
        return M

    def cast(self, x_full, d):
        """Feasible max steps b>=0 with x_full + b (U d) >= 0, and the binding
        full-coordinate that first hits zero.  Returns (bmax, binding_coord)."""
        Ud = self.full_move(d)
        neg = Ud < 0
        if not neg.any():
            return None, None                        # unbounded (should not occur)
        ratios = x_full[neg] // (-Ud[neg])
        bmax = int(ratios.min())
        # binding coordinate(s): those attaining the min at b = bmax (hit 0 next step)
        neg_idx = np.nonzero(neg)[0]
        binding = neg_idx[ratios == bmax]
        # deterministic tie-break: smallest full-coordinate index
        return bmax, int(binding.min())

    def reflect(self, d, binding_coord):
        """Reflect free-frame direction d off the face whose binding full-coord is
        `binding_coord`.  Coordinate wall -> sign flip; image wall -> lattice
        reflection (Def. 5.3).  Returns the reflected free-frame direction, or None
        if no legal reflection exists or the reflection is not invertible on the
        realised path.

        Invertibility (the reverse trajectory retraces this wall) is required for the
        reverse-path Metropolis ratio (eq. 15) to give detailed balance.  Coordinate
        walls are sign flips and always invert; a lattice reflection inverts only when
        reflecting the reversed outgoing direction reproduces the reversed incoming
        one.  Non-invertible bounces are blocked, so forward and reverse agree on
        which trajectories exist and the chain is exactly reversible."""
        if binding_coord in self.col_is_free:        # coordinate wall {x_{2,j}=0}
            j = self.col_is_free[binding_coord]
            dr = d.copy(); dr[j] = -dr[j]
            return dr
        # image wall: normal = row of U at the determined coordinate (= -A1inv A2 row)
        n = self.U[binding_coord, :].astype(np.int64)
        dr = self._lattice_reflect(d, n)
        if dr is None:
            return None
        back = self._lattice_reflect(-dr, n)         # reverse reflects -dr off n
        if back is None or not np.array_equal(back, -d):
            return None                              # non-invertible: block trajectory
        return dr

    def _lattice_reflect(self, d, n):
        """Definition 5.3: l1-nearest integer d' with d'.n > 0 and ||d'||_1 <= ||d||_1,
        closest in l1 to the specular direction d - 2 (d.n) n / ||n||^2.  Ties broken
        lexicographically.  For unit d and {0,+-1} normals this yields a unit dir."""
        n = np.asarray(n, float)
        nn = float(n @ n)
        if nn == 0:
            return None
        d_spec = d - 2.0 * (d @ n) / nn * n
        radius = int(np.abs(d).sum())                 # ||d||_1
        best, best_key = None, None
        for cand in self._l1_ball(radius):
            if cand @ n <= 0:                         # must point back into polytope
                continue
            dist = float(np.abs(cand - d_spec).sum())
            key = (dist, tuple(int(v) for v in cand))  # lexicographic tie-break
            if best_key is None or key < best_key:
                best_key, best = key, cand
        return None if best is None else best.astype(np.int64)

    def _l1_ball(self, radius):
        """All integer vectors in Z^m with ||.||_1 <= radius (radius small)."""
        key = radius
        cache = getattr(self, "_ball_cache", {})
        if key in cache:
            return cache[key]
        pts = []
        def rec(k, rem, cur):
            if k == self.m:
                pts.append(np.array(cur, dtype=np.int64)); return
            for v in range(-rem, rem + 1):
                rec(k + 1, rem - abs(v), cur + [v])
        rec(0, radius, [])
        cache[key] = pts
        self._ball_cache = cache
        return pts

    # -- trajectory -------------------------------------------------------------
    def bounce_to_wall(self, x_full, d, B):
        """Run B wall-bounces from x_full in initial direction d.
        Returns (vB_full, d_terminal, wall_points) where wall_points = [v1,...,vB]
        are the successive wall vertices, or None if the trajectory cannot complete
        B bounces (stuck against a wall, or a non-invertible reflection)."""
        x = x_full.copy(); dcur = d.copy(); walls = []
        for _ in range(B):
            bmax, binding = self.cast(x, dcur)
            if bmax is None or bmax == 0:
                return None                           # stuck against a wall
            x = x + bmax * self.full_move(dcur)       # advance to wall
            dnext = self.reflect(dcur, binding)
            if dnext is None:
                return None
            walls.append(x.copy())
            dcur = dnext
        return x, dcur, walls

    def reverse_prob_factor(self, x_dagger_full, d_terminal, B, x_full, fwd_walls):
        """Probability factor (excluding pi_B and 1/(2m), which cancel) that the
        algorithm started at x_dagger realises the EXACT reverse of the forward
        trajectory whose wall vertices are `fwd_walls` = [v1,...,vB] and which lands
        on x_full.  Returns 1/Lbar only if the reverse deterministically retraces the
        walls in reverse order and its terminal ray reaches x_full; else 0.0."""
        rev = self.bounce_to_wall(x_dagger_full, -d_terminal, B)
        if rev is None:
            return 0.0
        v1_full, d_end, rev_walls = rev               # after B reverse bounces
        # the reverse must retrace the forward walls in reverse order
        if len(rev_walls) != B:
            return 0.0
        for a, b in zip(rev_walls, list(reversed(fwd_walls))):
            if not np.array_equal(a, b):
                return 0.0
        # terminal reverse ray from v1 in direction d_end must reach x_full
        Lbar, _ = self.cast(v1_full, d_end)
        if Lbar is None or Lbar == 0:
            return 0.0
        Ud = self.full_move(d_end)
        diff = x_full - v1_full
        nz = Ud != 0
        if not nz.any():
            return 0.0
        q = diff[nz] // Ud[nz]
        if not np.all(Ud[nz] * q[0] == diff[nz]):     # colinear & integer multiple
            return 0.0
        bbar = int(q[0])
        if bbar < 1 or bbar > Lbar:
            return 0.0
        if not np.array_equal(v1_full + bbar * Ud, x_full):
            return 0.0
        return 1.0 / Lbar


def reflective_transition_matrix(prob, target="poisson", theta=None, bmax=2, pi_B=None,
                                 log_w=None):
    """Exact transition matrix Q of the reflective sampler on prob's fibre.

    Enumerates every (B, initial direction, terminal length) trajectory from each
    state, applies the reverse-path Metropolis rule (eq. 15), and returns the
    row-stochastic Q.  Target is product-Poisson (theta) or uniform.

    ``log_w`` optionally supplies an unnormalised log-target vector directly (one
    entry per fibre state), bypassing ``log_weights``; used when the target is a
    product-Poisson over a coordinate subset (e.g. the real free coordinates of a
    synthetic band, with auxiliary slack coordinates left flat).
    """
    S = ReflectiveSampler(prob, bmax=bmax, pi_B=pi_B)
    N = prob.fibre_size
    lw = log_weights(prob, target, theta) if log_w is None else np.asarray(log_w, float)
    twom = 2.0 * S.m
    Q = np.zeros((N, N))

    for i in range(N):
        x = prob.states[i].astype(np.int64)
        acc = 0.0
        for B in range(S.bmax + 1):
            pB = S.pi_B[B]
            if pB == 0:
                continue
            for d1 in S.dirs:
                # run B wall bounces to reach vB and terminal direction
                if B == 0:
                    vB, d_term, fwd_walls = x, d1.copy(), []
                else:
                    res = S.bounce_to_wall(x, d1, B)
                    if res is None:
                        continue
                    vB, d_term, fwd_walls = res
                L, _ = S.cast(vB, d_term)             # terminal ray length
                if L is None or L == 0:
                    continue
                Ud = S.full_move(d_term)
                qf_base = pB / twom * (1.0 / L)        # forward path prob
                for b in range(1, L + 1):
                    xd = vB + b * Ud
                    k = prob.index.get(tuple(xd))
                    if k is None or k == i:
                        continue
                    qrev_factor = S.reverse_prob_factor(xd, d_term, B, x, fwd_walls)
                    if qrev_factor == 0.0:
                        alpha = 0.0
                    else:
                        qr = pB / twom * qrev_factor
                        ratio = np.exp(lw[k] - lw[i]) * qr / qf_base
                        alpha = min(1.0, ratio)
                    if alpha > 0.0:
                        Q[i, k] += qf_base * alpha
                        acc += qf_base * alpha
        Q[i, i] += 1.0 - acc                          # rejection + holding
    pi = normalised_pi(lw)
    return Q, pi
