"""Test problems from Section 9.1 of the manuscript.

Currently implemented:

  * P1  -- Hazelton (2024) Example 2 / Example 11 network.  Eight routes, four
           links, y=(4,4,2,2), fibre size 55.  The exact A and PLB are taken
           verbatim from Hazelton, McVeagh, Tuffley & van Brunt (2024),
           Bernoulli 30(4), Examples 2 and 9.
  * P2  -- Auckland SH16 highway network (21 OD pairs, 7 links; A is 7x21),
           Example (10)
           / Section 6 of the same paper.

The x-ordering and the A1 column choice are chosen to reproduce Hazelton's
printed U (eq. 9) exactly, so that move labels u_1..u_4 match the paper and the
bottleneck-move identification (-u_2 for small theta_5) is directly comparable.
"""

from __future__ import annotations

import numpy as np

from .fibre import FibreProblem

__all__ = ["P1", "P2", "p1_poisson_theta"]


# ---------------------------------------------------------------------------
# P1  --  Hazelton Example 2 / Example 11
# ---------------------------------------------------------------------------
# x ordering: (x_{12}, x_{13}, x_{14}, x_{25}, x_{34}, x_{23}, x_{15}, x_{24})
#             index:  0      1      2      3      4      5      6      7
A_P1 = np.array([
    [1, 1, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 0],
], dtype=np.int64)

Y_P1 = np.array([4, 4, 2, 2], dtype=np.int64)

# Example 9: A1 = columns 1,5,6,7 (1-indexed) -> 0-indexed {0,4,5,6}.
# Free-coordinate order chosen so U columns u_1..u_4 match eq. (9):
#   u_1 <- orig col 2 (0-idx 1), u_2 <- col 3 (0-idx 2),
#   u_3 <- col 8 (0-idx 7),      u_4 <- col 4 (0-idx 3).
P1_COLS1 = (0, 4, 5, 6)
P1_FREE_ORDER = (1, 2, 7, 3)


def P1() -> FibreProblem:
    return FibreProblem(A_P1, Y_P1, cols1=P1_COLS1, free_order=P1_FREE_ORDER,
                        name="P1 (Hazelton Ex.2/Ex.11)")


def p1_poisson_theta(theta5: float) -> np.ndarray:
    """Poisson mean vector theta = (1,1,1,1,theta5,1,1,1) of Example 11."""
    theta = np.ones(8, dtype=float)
    theta[4] = theta5
    return theta


# Hazelton's printed U (eq. 9), for validation.
U_P1_REFERENCE = np.array([
    [-1, -1,  0,  1],
    [ 1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1],
    [ 0, -1, -1,  0],
    [-1, -1, -1,  0],
    [ 0,  0,  0, -1],
    [ 0,  0,  1,  0],
], dtype=np.int64)


# ---------------------------------------------------------------------------
# P2  --  Auckland State Highway 16  (Hazelton 2024, eq. 10 / Section 6)
#
# Linear 8-node highway; link k joins node k to node k+1 (k=1..7).  Drivers
# join at nodes 1-4 (on-ramps) and leave at nodes 3-8 (off-ramps); the trip
# (i, j) uses links i..j-1.  Lexicographic OD order puts (2,3) seventh,
# matching theta_7 = 0.5 in the source.
# ---------------------------------------------------------------------------
P2_OD_PAIRS = [(i, j) for i in (1, 2, 3, 4) for j in range(max(i + 1, 3), 9)]

A_P2 = np.array(
    [[1 if i <= k < j else 0 for (i, j) in P2_OD_PAIRS] for k in range(1, 8)],
    dtype=np.int64)

Y_P2 = np.array([2991, 3352, 3977, 4576, 3849, 2458, 978], dtype=np.int64)

# Short-path partition (Corollary 4.5): columns 1,7,13,18,19,20,21 (1-indexed).
P2_COLS1_PLB1 = (0, 6, 12, 17, 18, 19, 20)
# Low-mean-out-of-A1 partition PLB2: columns 1,8,13,19,20,21 ... (Section 6);
# A1 must have 7 columns.  Hazelton: "A1 comprises columns 1,8,13,19,20,21" is
# six -- the seventh is fixed by invertibility; recovered in problem P2 setup
# when that experiment is built out.  (P2 not needed for Prediction 1.)


def P2(cols1=P2_COLS1_PLB1) -> FibreProblem:
    # Fibre is astronomically large; do NOT enumerate.  Returned for A/PLB only.
    raise NotImplementedError(
        "P2 fibre is too large to enumerate; use the MCMC path (built with the "
        "full grid, not needed for the Prediction 1 control).")


# ---------------------------------------------------------------------------
# P6  --  2x3 contingency table (independent second problem for the generality
#         check of the Prediction 2 negative).  A different configuration class
#         (Markov-basis contingency table, cf. Hazelton Example 6) from the P1
#         road network, with a genuine Poisson loading bottleneck at cell x11.
# ---------------------------------------------------------------------------
# x = (x11, x12, x13, x21, x22, x23); constraints: row-1 sum + three col sums
# (the row-2 sum is redundant and dropped).
A_TABLE23 = np.array([
    [1, 1, 1, 0, 0, 0],   # row 1 total
    [1, 0, 0, 1, 0, 0],   # col 1 total
    [0, 1, 0, 0, 1, 0],   # col 2 total
    [0, 0, 1, 0, 0, 1],   # col 3 total
], dtype=np.int64)
Y_TABLE23 = np.array([4, 3, 3, 2], dtype=np.int64)   # row sums (4,4); col sums (3,3,2)
TABLE23_COLS1 = (0, 3, 4, 5)      # A1 block; x11 (index 0) is the bottleneck cell
TABLE23_BOTTLENECK = 0            # coordinate whose shrinking mean collapses the gap


def table_2x3() -> FibreProblem:
    """2x3 contingency-table fibre (size 10) used to confirm the Prediction 2
    obstruction is problem-independent, not specific to the P1 network."""
    return FibreProblem(A_TABLE23, Y_TABLE23, cols1=TABLE23_COLS1,
                        name="2x3 table (generality check)")
