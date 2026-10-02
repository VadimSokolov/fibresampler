"""Exact Eulerian-submatrix, balancedness and total-unimodularity checks.

Implements the algebraic certificates of Hazelton, McVeagh, Tuffley & van
Brunt (2024, Bernoulli 30(4)) used by Predictions 6 and 7:

  * c0-Eulerian submatrix (Definition 4.2): a non-null submatrix with every
    column sum zero and every row sum even.  Absence in sgn(U) certifies that
    the PLB M_U is a Markov basis and a c-minimal augmenting Markov sub-basis
    for every full-dimensional fibre (Theorem 4.4).
  * cr0-Eulerian submatrix (Definition 4.3): additionally every row sum zero.
    Presence in some column signing of U, with no zero column, certifies that
    M_U fails to be augmenting for at least one fibre (Theorem 4.6).
  * Balanced matrix (Berge): every square submatrix with exactly two nonzeros
    per row and per column has entry sum divisible by 4.  For 0/1 matrices
    this reduces to the absence of odd-order 2-regular submatrices.
  * Total unimodularity: every square minor lies in {-1, 0, 1}.

All searches are exhaustive and exact (integer arithmetic; GF(2) linear
algebra for the even-row-sum condition), sized for the small configuration
matrices of the experimental protocol (top blocks up to ~10 x 16).

A note on squareness: Definition 4.2 states c0-Eulerian for square
submatrices but permits zero rows (a zero row has even sum).  Any non-square
witness with zero-sum columns and even rows pads to a square one with zero
rows and vice versa, so presence is decided by the canonical form searched
here: no zero rows or columns, column sums zero, row sums even.  This matches
the definition used in Hazelton's follow-up (Statistics & Probability
Letters, 2024, Definition 1), which drops squareness.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = [
    "sgn",
    "c0_eulerian_witness",
    "cr0_eulerian_witness",
    "signing_with_cr0",
    "is_balanced",
    "is_totally_unimodular",
]


def sgn(M: np.ndarray) -> np.ndarray:
    """Entrywise sign matrix, in {0, +1, -1} (Hazelton Sec. 4)."""
    return np.sign(np.asarray(M, dtype=np.int64)).astype(np.int64)


# ---------------------------------------------------------------------------
# c0-Eulerian search
# ---------------------------------------------------------------------------
def _candidate_rows(S: np.ndarray) -> list[int]:
    """Rows that can appear non-trivially in a canonical witness.

    A canonical witness row must contain at least two nonzeros (an even,
    positive count of +-1 entries), so rows with fewer than two nonzeros in
    the whole matrix can never participate.  For a PLB U = [T; I] this
    automatically restricts the search to the top block, as observed by
    Hazelton (2024, Sec. 4): identity-block rows carry a single nonzero.
    """
    counts = (S != 0).sum(axis=1)
    return [i for i in range(S.shape[0]) if counts[i] >= 2]


def c0_eulerian_witness(M: np.ndarray):
    """Search sgn(M) for a c0-Eulerian submatrix.

    Returns (rows, cols) of a canonical witness (no zero rows or columns,
    every column sum zero, every row sum even) or None if absent.

    Method: for every subset I of candidate rows, the admissible columns are
    those with a nonzero, zero-sum restriction to I.  A witness on I is a
    non-empty subset J of admissible columns whose support indicators on I
    sum to zero over GF(2) (entries are +-1, so a row's entry sum is even iff
    its nonzero count is even).  Such a J exists iff the support vectors are
    GF(2)-linearly dependent; a dependency is extracted from the elimination.
    """
    S = sgn(M)
    rows_all = _candidate_rows(S)
    ncols = S.shape[1]

    for k in range(2, len(rows_all) + 1):
        for I in combinations(rows_all, k):
            Iarr = list(I)
            sub = S[Iarr, :]
            colsum = sub.sum(axis=0)
            nnz = (sub != 0).sum(axis=0)
            admissible = [j for j in range(ncols) if nnz[j] >= 2 and colsum[j] == 0]
            if len(admissible) < 2:
                continue
            # GF(2) elimination over support-indicator vectors (bitmask on I).
            masks = []
            for j in admissible:
                m = 0
                for a, i in enumerate(Iarr):
                    if sub[a, j] != 0:
                        m |= 1 << a
                masks.append(m)
            # basis[b] = (mask, combo) with combo a bitmask over admissible idx
            basis: dict[int, tuple[int, int]] = {}
            for idx, m in enumerate(masks):
                combo = 1 << idx
                while m:
                    top = m.bit_length() - 1
                    if top in basis:
                        bm, bc = basis[top]
                        m ^= bm
                        combo ^= bc
                    else:
                        basis[top] = (m, combo)
                        break
                else:
                    # m reduced to zero: 'combo' indexes a dependent subset
                    J = [admissible[t] for t in range(len(admissible))
                         if (combo >> t) & 1]
                    used = S[np.ix_(Iarr, J)]
                    keep = [i for a, i in enumerate(Iarr) if (used[a] != 0).any()]
                    return keep, J
    return None


# ---------------------------------------------------------------------------
# cr0-Eulerian search (for Theorem 4.6 failure certificates)
# ---------------------------------------------------------------------------
def cr0_eulerian_witness(M: np.ndarray):
    """Search sgn(M) for a cr0-Eulerian submatrix with no zero column.

    Returns (rows, cols) or None.  Brute force: for each row subset the
    admissible columns are the nonzero zero-sum ones; then search column
    subsets whose row sums over the subset all vanish.  Exponential but the
    inputs here have at most ~12 admissible columns.
    """
    S = sgn(M)
    rows_all = _candidate_rows(S)
    ncols = S.shape[1]

    for k in range(2, len(rows_all) + 1):
        for I in combinations(rows_all, k):
            Iarr = list(I)
            sub = S[Iarr, :]
            colsum = sub.sum(axis=0)
            nnz = (sub != 0).sum(axis=0)
            admissible = [j for j in range(ncols) if nnz[j] >= 2 and colsum[j] == 0]
            if len(admissible) < 2:
                continue
            for m in range(2, len(admissible) + 1):
                for J in combinations(admissible, m):
                    block = sub[:, list(J)]
                    if (block.sum(axis=1) == 0).all():
                        keep = [i for a, i in enumerate(Iarr)
                                if (block[a] != 0).any()]
                        return keep, list(J)
    return None


def signing_with_cr0(M: np.ndarray):
    """Search all column signings U^sigma for a cr0-Eulerian submatrix.

    Implements the hypothesis of Hazelton (2024) Theorem 4.6.  Returns
    (sigma, rows, cols) for the first signing whose signed matrix contains a
    cr0-Eulerian submatrix with no zero column, or None.
    """
    S = sgn(M)
    ncols = S.shape[1]
    for bits in range(1 << ncols):
        sigma = np.array([1 - 2 * ((bits >> j) & 1) for j in range(ncols)],
                         dtype=np.int64)
        wit = cr0_eulerian_witness(S * sigma[None, :])
        if wit is not None:
            return sigma, wit[0], wit[1]
    return None


# ---------------------------------------------------------------------------
# Balancedness (Berge 1970 / Conforti-Cornuejols 2006)
# ---------------------------------------------------------------------------
def is_balanced(M: np.ndarray):
    """Exact balancedness of a {0, +-1} matrix.

    Balanced iff every square submatrix with exactly two nonzeros per row and
    per column has entry sum divisible by 4.  For a 0/1 matrix such a
    submatrix of order q has sum 2q, so the condition is the absence of
    odd-order 2-regular submatrices (odd holes), Berge's definition.

    Rows or columns with fewer than two nonzeros can never participate in a
    2-regular submatrix, so they are peeled off (iteratively) before the
    exhaustive search; this makes configurations padded with unit columns
    (single-link routes, identity blocks) cheap to test.

    Returns (True, None) or (False, (rows, cols)) with a violating submatrix,
    indices referring to the input matrix.
    """
    S = np.asarray(M, dtype=np.int64)
    rows = list(range(S.shape[0]))
    cols = list(range(S.shape[1]))
    while True:
        sub = S[np.ix_(rows, cols)]
        rkeep = [a for a in range(len(rows)) if (sub[a] != 0).sum() >= 2]
        ckeep = [b for b in range(len(cols)) if (sub[:, b] != 0).sum() >= 2]
        if len(rkeep) == len(rows) and len(ckeep) == len(cols):
            break
        rows = [rows[a] for a in rkeep]
        cols = [cols[b] for b in ckeep]
        if not rows or not cols:
            return True, None
    n, m = len(rows), len(cols)
    for q in range(2, min(n, m) + 1):
        for I in combinations(range(n), q):
            sub = S[np.ix_([rows[a] for a in I], cols)]
            # columns with exactly two nonzeros on I
            nnz = (sub != 0).sum(axis=0)
            cand = [b for b in range(m) if nnz[b] == 2]
            if len(cand) < q:
                continue
            for J in combinations(cand, q):
                block = sub[:, list(J)]
                if ((block != 0).sum(axis=1) == 2).all():
                    if int(block.sum()) % 4 != 0:
                        return False, ([rows[a] for a in I],
                                       [cols[b] for b in J])
    return True, None


# ---------------------------------------------------------------------------
# Total unimodularity (exhaustive minors, exact integer determinants)
# ---------------------------------------------------------------------------
def _int_det(M: np.ndarray) -> int:
    """Exact determinant of a small integer matrix (fraction-free Bareiss)."""
    q = M.shape[0]
    a = [[int(M[i, j]) for j in range(q)] for i in range(q)]
    sign = 1
    prev = 1
    for c in range(q - 1):
        piv = next((r for r in range(c, q) if a[r][c] != 0), None)
        if piv is None:
            return 0
        if piv != c:
            a[c], a[piv] = a[piv], a[c]
            sign = -sign
        for r in range(c + 1, q):
            for s in range(c + 1, q):
                a[r][s] = (a[r][s] * a[c][c] - a[r][c] * a[c][s]) // prev
            a[r][c] = 0
        prev = a[c][c]
    return sign * a[q - 1][q - 1]


def is_totally_unimodular(M: np.ndarray):
    """Exhaustive TU check.  Returns (True, None) or (False, (rows, cols, det))."""
    S = np.asarray(M, dtype=np.int64)
    n, m = S.shape
    if not np.isin(S, (-1, 0, 1)).all():
        return False, (None, None, None)
    for q in range(2, min(n, m) + 1):
        for I in combinations(range(n), q):
            for J in combinations(range(m), q):
                d = _int_det(S[np.ix_(list(I), list(J))])
                if d not in (-1, 0, 1):
                    return False, (list(I), list(J), d)
    return True, None
