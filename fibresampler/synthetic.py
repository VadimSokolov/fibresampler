"""Synthetic fibres for the geometry/loading decoupling of Prediction 4.

Two builders, both producing lightweight duck-typed "problems" that expose exactly
the attributes the reflective machinery reads (``states``, ``index``, ``U``,
``plb_info``, ``n_moves``, ``fibre_size``):

  * ``BandProblem``  -- a thin diagonal band in a two-dimensional free frame
    ``(s,t)``, embedded with three slack coordinates that realise the oblique
    image walls ``|s-t|<=w`` and the long-axis cap ``s+t<=C``.  The axis defect is
    kappa ~ C/(2 sqrt(2) w); this is the exact family behind the Prediction-3
    kappa-sweep (w=2, C in {12,24,40,60} -> kappa ~ {2.15,4.26,7.08,10.6}).  The
    two free coordinates s,t are the *real* count coordinates that carry the
    product-Poisson target; the slacks are auxiliary and left flat.

  * ``product_problem(A, B)`` -- the disjoint (block-diagonal) product of two
    problems.  The fibre is the Cartesian product, the PLB is block-diagonal, and
    a reflective move lives entirely in one block.  This isolates a geometry block
    (a Band) from a loading block (e.g. the 2x3 table) so the two mechanisms of
    Section 5 -- realised path length vs loading -- can be dialled independently.

Every construction is validated by the same exact detailed-balance gate as the
real fibres before its spectral gap is trusted (see experiments/pred4_crossover).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = ["BandProblem", "SlopedBand", "product_problem", "poisson_free_logw"]


class BandProblem:
    """Thin diagonal band {(s,t) : s,t>=0, |s-t|<=w, s+t<=C} in a 2-D free frame.

    Full state x = [s, t, w+s-t, w-s+t, C-s-t] (r=5); free coords cols2=(0,1) carry
    the target, slacks cols1=(2,3,4) realise the walls.  The billiard reflects an
    s-step into a t-step off a difference wall (specular direction of e_s off
    normal [1,-1] is e_t), travelling *along* the long diagonal in one proposal.
    """

    def __init__(self, w: int, C: int, name: str = ""):
        self.w = int(w)
        self.C = int(C)
        self.name = name or f"band(w={w},C={C})"
        # U columns: how full x moves when s (resp. t) increments by 1.
        self.U = np.array([[1, 0],
                           [0, 1],
                           [1, -1],
                           [-1, 1],
                           [-1, -1]], dtype=np.int64)
        self.plb_info = {"cols1": (2, 3, 4), "cols2": (0, 1)}
        states = []
        for s in range(0, C + 1):
            for t in range(0, C + 1):
                if abs(s - t) <= w and s + t <= C:
                    states.append([s, t, w + s - t, w - s + t, C - s - t])
        self.states = np.array(states, dtype=np.int64)
        self.index = {tuple(x): i for i, x in enumerate(self.states)}

    # -- interface expected by the reflective machinery ------------------------
    @property
    def n_moves(self) -> int:
        return self.U.shape[1]

    @property
    def fibre_size(self) -> int:
        return len(self.states)

    @property
    def r(self) -> int:
        return self.U.shape[0]

    def free_coords(self, x):
        return np.asarray(x, dtype=np.int64)[list(self.plb_info["cols2"])]

    # -- geometry --------------------------------------------------------------
    def kappa(self):
        """Axis defect L/a: diameter over longest axis-aligned chord, in (s,t)."""
        free = self.states[:, [0, 1]].astype(float)
        d = free[:, None, :] - free[None, :, :]
        L = float(np.sqrt((d ** 2).sum(2)).max())
        a = 1
        for j in (0, 1):
            other = 1 - j
            groups = {}
            for row in free:
                groups.setdefault(int(row[other]), []).append(int(row[j]))
            for vals in groups.values():
                a = max(a, max(vals) - min(vals))
        return L / a, L, a


class SlopedBand(BandProblem):
    """Thin band around the lattice line q*s = p*t (ridge direction (p, q)):
    {(s,t) : s,t>=0, |q s - p t| <= W, s+t<=C}, in the same slack-extended frame as
    ``BandProblem`` (which is the case p = q = 1, W = w).  The image-wall normals are
    (q,-p), so a non-unit slope (p or q >= 2) puts entries of magnitude >= 2 in the walls
    and exercises the non-invertible-bounce guard of the reflective sampler; a change of
    basis only needs the ridge vector (p,q) and a completing vector.
    """

    def __init__(self, p: int, q: int, W: int, C: int, name: str = ""):
        self.p, self.q, self.W = int(p), int(q), int(W)
        self.w = self.W
        self.C = int(C)
        self.name = name or f"sloped(p={p},q={q},W={W},C={C})"
        self.U = np.array([[1, 0],
                           [0, 1],
                           [q, -p],
                           [-q, p],
                           [-1, -1]], dtype=np.int64)
        self.plb_info = {"cols1": (2, 3, 4), "cols2": (0, 1)}
        states = []
        for s in range(0, C + 1):
            for t in range(0, C + 1):
                if abs(q * s - p * t) <= W and s + t <= C:
                    states.append([s, t, W + q * s - p * t, W - q * s + p * t, C - s - t])
        self.states = np.array(states, dtype=np.int64)
        self.index = {tuple(x): i for i, x in enumerate(self.states)}


def poisson_free_logw(prob, theta_free, free_cols=None):
    """Unnormalised product-Poisson log-target on a *subset* of coordinates.

    ``theta_free`` are the Poisson means for the coordinates in ``free_cols``
    (default: prob.plb_info['cols2']); all other coordinates are left flat.  This
    is the natural target for a synthetic band whose slack coordinates are not
    real counts.
    """
    if free_cols is None:
        free_cols = list(prob.plb_info["cols2"])
    free_cols = list(free_cols)
    theta = np.asarray(theta_free, dtype=float)
    logt = np.log(theta)
    xs = prob.states[:, free_cols].astype(float)
    lw = xs @ logt - gammaln(xs + 1).sum(axis=1)
    return lw


class _ProductProblem:
    """Block-diagonal product of two duck-typed problems (see product_problem)."""

    def __init__(self, A, B, name=""):
        self.A_prob, self.B_prob = A, B
        self.name = name or f"{getattr(A,'name','A')} x {getattr(B,'name','B')}"
        rA, rB = A.states.shape[1], B.states.shape[1]
        mA, mB = A.n_moves, B.n_moves
        # Cartesian product of states, concatenated.
        sa, sb = A.states, B.states
        NA, NB = len(sa), len(sb)
        prod = np.empty((NA * NB, rA + rB), dtype=np.int64)
        self._ij = np.empty((NA * NB, 2), dtype=np.int64)
        k = 0
        for i in range(NA):
            for j in range(NB):
                prod[k, :rA] = sa[i]
                prod[k, rA:] = sb[j]
                self._ij[k] = (i, j)
                k += 1
        self.states = prod
        self.index = {tuple(x): i for i, x in enumerate(self.states)}
        # Block-diagonal U.
        U = np.zeros((rA + rB, mA + mB), dtype=np.int64)
        U[:rA, :mA] = A.U
        U[rA:, mA:] = B.U
        self.U = U
        c1 = list(A.plb_info["cols1"]) + [c + rA for c in B.plb_info["cols1"]]
        c2 = list(A.plb_info["cols2"]) + [c + rA for c in B.plb_info["cols2"]]
        self.plb_info = {"cols1": tuple(c1), "cols2": tuple(c2)}
        self._rA = rA

    @property
    def n_moves(self):
        return self.U.shape[1]

    @property
    def fibre_size(self):
        return len(self.states)

    def logw_from_blocks(self, lwA, lwB):
        """Outer-sum the two blocks' log-targets: lw[i,j] = lwA[i] + lwB[j]."""
        out = np.empty(len(self.states))
        for k, (i, j) in enumerate(self._ij):
            out[k] = lwA[i] + lwB[j]
        return out


def product_problem(A, B, name=""):
    return _ProductProblem(A, B, name=name)
