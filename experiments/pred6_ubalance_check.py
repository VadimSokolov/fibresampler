"""Post-hoc hypothesis check for the Prediction 6 necessity claim.

Theorem thm:balanced (via cor:balanced-iff) requires TWO per-instance
hypotheses of a failing PLB: (a) sgn(U) = U in {0,+-1} is BALANCED, and
(b) the c0-Eulerian witness has at most two nonzero entries per row.
The Hopper run (pred6_balanced.py stage 4) verified (b) for every failing
PLB but recorded (a) only for the featured passing PLB and for A itself;
A-balanced does not imply U-balanced.  This script closes the gap: it
re-runs the identical deterministic PLB census on P4 and checks
is_balanced(U) for every {0,+-1} PLB, pass and fail.

Output: results/pred6_ubalance.txt (counts + any unbalanced offenders).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibresampler.eulerian import c0_eulerian_witness, is_balanced, sgn
from fibresampler.fibre import build_plb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pred6_balanced import A_P4, N4, R4  # noqa: E402  (census constants)

RESULTS = Path(__file__).resolve().parent.parent / "results"

LINES: list[str] = []


def say(msg: str) -> None:
    print(msg)
    LINES.append(msg)


def deg2_witnesses(U: np.ndarray):
    """Exhaustive list of degree-two c0-Eulerian witnesses of sgn(U).

    A degree-two witness is a submatrix (rows I, cols J) with no zero
    column, every column sum zero, and every row having exactly two
    nonzeros (<=2 per the theorem, even and positive for a canonical
    witness, hence exactly 2).  U here is small (11 x 4), so full
    enumeration over row and column subsets is exact.
    """
    S = sgn(U)
    m, n = S.shape
    out = []
    for jmask in range(3, 1 << n):  # at least two columns
        J = [j for j in range(n) if jmask >> j & 1]
        sub_all = S[:, J]
        # rows with exactly two nonzeros within J and even (zero-mod-2) sum
        rows_ok = [i for i in range(m) if (sub_all[i] != 0).sum() == 2]
        for imask in range(1, 1 << len(rows_ok)):
            I = [rows_ok[k] for k in range(len(rows_ok)) if imask >> k & 1]
            if len(I) < 2:
                continue
            sub = S[np.ix_(I, J)]
            if ((sub != 0).sum(axis=0) == 0).any():
                continue  # zero column
            if (sub.sum(axis=0) != 0).any():
                continue  # column sums must vanish
            out.append((tuple(I), tuple(J), sub))
    return out


def main() -> None:
    say("P4 U-balancedness check (hypothesis (a) of thm:balanced, per PLB)")
    say(f"A_P4: {N4} x {R4}, census identical to pred6_balanced.stage1")
    counts = {("pass", True): 0, ("pass", False): 0,
              ("fail", True): 0, ("fail", False): 0}
    offenders = []
    for cols1 in combinations(range(R4), N4):
        if abs(np.linalg.det(A_P4[:, cols1])) < 0.5:
            continue
        U, info = build_plb(A_P4, cols1)
        if not info["integral"] or not np.isin(U, (-1, 0, 1)).all():
            continue
        arm = "pass" if c0_eulerian_witness(U) is None else "fail"
        bal, _ = is_balanced(U)
        counts[(arm, bool(bal))] += 1
        if not bal:
            offenders.append((arm, cols1))
    n_pass = counts[("pass", True)] + counts[("pass", False)]
    n_fail = counts[("fail", True)] + counts[("fail", False)]
    say(f"  c0-PASS PLBs: {n_pass}, of which U balanced: "
        f"{counts[('pass', True)]}")
    say(f"  c0-FAIL PLBs: {n_fail}, of which U balanced: "
        f"{counts[('fail', True)]}")
    gate = (n_pass == 12 and n_fail == 102)
    say(f"  census gate (12 pass / 102 fail as ledgered): "
        f"{'PASS' if gate else 'FAIL'}")
    say(f"  full-U balancedness (hypothesis of thm:balanced as printed): "
        f"holds for {counts[('fail', True)]}/{n_fail} failing PLBs")

    # Witness-level check: the proof of thm:balanced uses only the witness
    # submatrix's balancedness (heredity).  Does every failing PLB carry a
    # degree-two c0-Eulerian witness, and is that witness itself balanced?
    say("")
    say("  witness-level check (exhaustive over submatrices of sgn(U)):")
    n_have_deg2 = 0
    n_all_bal = 0
    n_some_bal = 0
    no_deg2, none_bal = [], []
    for cols1 in combinations(range(R4), N4):
        if abs(np.linalg.det(A_P4[:, cols1])) < 0.5:
            continue
        U, info = build_plb(A_P4, cols1)
        if not info["integral"] or not np.isin(U, (-1, 0, 1)).all():
            continue
        if c0_eulerian_witness(U) is None:
            continue  # passing PLB: no witness to examine
        wits = deg2_witnesses(U)
        if not wits:
            no_deg2.append(cols1)
            continue
        n_have_deg2 += 1
        bal_flags = [bool(is_balanced(sub)[0]) for _, _, sub in wits]
        if all(bal_flags):
            n_all_bal += 1
        if any(bal_flags):
            n_some_bal += 1
        else:
            none_bal.append(cols1)
    say(f"    failing PLBs with a degree-two c0-Eulerian witness: "
        f"{n_have_deg2}/{n_fail}"
        + (f" (missing: {no_deg2})" if no_deg2 else ""))
    say(f"    ... with EVERY degree-two witness balanced: {n_all_bal}")
    say(f"    ... with AT LEAST ONE balanced degree-two witness: {n_some_bal}"
        + (f" (none balanced: {none_bal})" if none_bal else ""))
    (RESULTS / "pred6_ubalance.txt").write_text("\n".join(LINES) + "\n")
    say(f"  wrote {RESULTS / 'pred6_ubalance.txt'}")
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
