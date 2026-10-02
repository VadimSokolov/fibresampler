"""Core fibre machinery for count-data linear inverse problems.

Implements the objects of Sections 2 and 5 of the manuscript:

  * the configuration matrix ``A`` and the y-fibre ``F_{A,y}`` (eq. 2--3);
  * exact fibre enumeration for small problems (feasible for |F| <~ 1e4);
  * the partition lattice basis (PLB)  U = [-A1^{-1} A2 ; I]  (eq. 4);
  * the hit-and-run ray endpoints  b_min, b_max  (eq. 6).

Everything here is exact integer arithmetic; no sampling is involved.  The
module is deliberately self-contained (numpy only) so that the spectral
computations that decide Predictions 1 and 2 do not depend on any Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product

import numpy as np

__all__ = ["FibreProblem", "enumerate_fibre", "build_plb", "ray_endpoints"]


# ---------------------------------------------------------------------------
# Fibre enumeration
# ---------------------------------------------------------------------------
def enumerate_fibre(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return every non-negative integer x with A x = y, as rows of an array.

    Exact depth-first enumeration with residual-bound pruning.  Column j is
    assigned a value between 0 and the tightest residual on the links it uses;
    the recursion backtracks whenever a link residual is exceeded.  For the
    small fibres used in the exact-SLEM experiments (P1, P4) this is instant.
    """
    A = np.asarray(A, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    n, r = A.shape
    if y.shape != (n,):
        raise ValueError(f"y has shape {y.shape}, expected {(n,)}")

    solutions: list[np.ndarray] = []
    x = np.zeros(r, dtype=np.int64)

    def recurse(j: int, residual: np.ndarray) -> None:
        if (residual < 0).any():
            return
        if j == r:
            if (residual == 0).all():
                solutions.append(x.copy())
            return
        col = A[:, j]
        # Largest value column j can take without over-shooting any link.
        support = col > 0
        if support.any():
            hi = int((residual[support] // col[support]).min())
        else:
            hi = 0  # a zero column can only take value 0 (assumed absent)
        # Which links can still be reduced by some later column (j+1..r-1)?
        reachable = ((A[:, j + 1:] > 0).any(axis=1) if j + 1 < r
                     else np.zeros(n, dtype=bool))
        for v in range(hi + 1):
            new_res = residual - v * col
            # prune: a positive residual on a link no remaining column touches
            # can never be cleared (values are unbounded, so no other bound holds)
            if ((new_res > 0) & ~reachable).any():
                continue
            x[j] = v
            recurse(j + 1, new_res)
        x[j] = 0

    recurse(0, y.copy())
    if not solutions:
        return np.zeros((0, r), dtype=np.int64)
    return np.array(sorted((tuple(s) for s in solutions)), dtype=np.int64)


# ---------------------------------------------------------------------------
# Partition lattice basis  (eq. 4)
# ---------------------------------------------------------------------------
def _exact_inverse(M: np.ndarray) -> np.ndarray:
    """Exact rational inverse of an integer square matrix via Fraction Gauss."""
    n = M.shape[0]
    aug = [[Fraction(int(M[i, j])) for j in range(n)] +
           [Fraction(1 if i == k else 0) for k in range(n)] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if piv is None:
            raise np.linalg.LinAlgError("A1 is singular")
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return np.array([[aug[i][n + j] for j in range(n)] for i in range(n)], dtype=object)


def build_plb(A: np.ndarray, cols1, free_order=None):
    """Construct the PLB U = [-A1^{-1} A2 ; I] of eq. (4).

    Parameters
    ----------
    A : (n, r) 0/1 configuration matrix.
    cols1 : indices (length n) of columns forming the invertible block A1.
    free_order : optional ordering of the remaining columns; the j-th free
        column becomes basis move u_j.  Defaults to ascending index order.
        Supplying Hazelton's ordering reproduces his printed U exactly.

    Returns
    -------
    U : (r, r-n) integer numpy array whose columns are the PLB moves, in the
        ORIGINAL coordinate ordering of x (row i of U corresponds to x_i).
    info : dict with A1inv (exact), cols1, cols2 (=free_order), integral flag.
    """
    A = np.asarray(A, dtype=np.int64)
    n, r = A.shape
    cols1 = list(cols1)
    if len(cols1) != n:
        raise ValueError("cols1 must have length n")
    cols2_default = [j for j in range(r) if j not in cols1]
    cols2 = list(free_order) if free_order is not None else cols2_default
    if sorted(cols2) != sorted(cols2_default):
        raise ValueError("free_order must be a permutation of the non-A1 columns")

    A1 = A[:, cols1]
    A2 = A[:, cols2]
    A1inv = _exact_inverse(A1)                      # exact rational (object array)
    block = -(A1inv @ A2)                           # (n, r-n), object/Fraction

    # integrality check
    integral = all(Fraction(v).denominator == 1 for v in block.ravel())
    block_int = np.array([[int(Fraction(v)) for v in row] for row in block], dtype=np.int64)

    U = np.zeros((r, r - n), dtype=np.int64)
    for a, row in enumerate(cols1):                 # top block -> A1 rows
        U[row, :] = block_int[a, :]
    for b, row in enumerate(cols2):                 # identity block -> A2 rows
        U[row, b] = 1

    A1inv_int = None
    if integral:
        A1inv_int = np.array([[int(Fraction(v)) for v in r_] for r_ in A1inv], dtype=np.int64)
    info = {"A1inv": A1inv, "A1inv_int": A1inv_int, "cols1": cols1,
            "cols2": cols2, "integral": integral}
    return U, info


# ---------------------------------------------------------------------------
# Hit-and-run ray endpoints  (eq. 6)
# ---------------------------------------------------------------------------
def ray_endpoints(x: np.ndarray, u: np.ndarray):
    """Feasible integer range [b_min, b_max] of x + b u >= 0  (eq. 6).

    b_max = floor(min_{i: u_i<0} x_i / |u_i|),
    b_min = -floor(min_{i: u_i>0} x_i / |u_i|).
    """
    x = np.asarray(x, dtype=np.int64)
    u = np.asarray(u, dtype=np.int64)
    pos = u > 0
    neg = u < 0
    b_min = -int(np.floor((x[pos] / np.abs(u[pos])).min())) if pos.any() else -10**9
    b_max = int(np.floor((x[neg] / np.abs(u[neg])).min())) if neg.any() else 10**9
    return b_min, b_max


# ---------------------------------------------------------------------------
# Problem container
# ---------------------------------------------------------------------------
@dataclass
class FibreProblem:
    """A configuration matrix, an observation y, and a chosen PLB.

    Enumerates the fibre and builds the PLB on construction; caches the ray
    structure needed for transition-matrix assembly.
    """
    A: np.ndarray
    y: np.ndarray
    cols1: tuple
    free_order: tuple | None = None
    name: str = ""
    states: np.ndarray = field(init=False)
    index: dict = field(init=False)
    U: np.ndarray = field(init=False)
    plb_info: dict = field(init=False)

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=np.int64)
        self.y = np.asarray(self.y, dtype=np.int64)
        self.states = enumerate_fibre(self.A, self.y)
        self.index = {tuple(s): i for i, s in enumerate(self.states)}
        self.U, self.plb_info = build_plb(self.A, self.cols1, self.free_order)

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def r(self) -> int:
        return self.A.shape[1]

    @property
    def n_moves(self) -> int:
        return self.U.shape[1]

    @property
    def fibre_size(self) -> int:
        return len(self.states)

    def free_coords(self, x: np.ndarray) -> np.ndarray:
        """The free-coordinate vector x_2 for a state x (the A2 columns)."""
        return np.asarray(x, dtype=np.int64)[list(self.plb_info["cols2"])]

    def ray(self, i: int, j: int):
        """Sorted list of state-indices on the ray through state i in move u_j.

        Includes i itself.  These are exactly the lattice points x + b u_j that
        remain in the fibre, which (since u_j in ker A) is equivalent to
        x + b u_j >= 0.
        """
        x = self.states[i]
        u = self.U[:, j]
        b_min, b_max = ray_endpoints(x, u)
        out = []
        for b in range(b_min, b_max + 1):
            key = tuple(x + b * u)
            k = self.index.get(key)
            if k is not None:
                out.append(k)
        return out
