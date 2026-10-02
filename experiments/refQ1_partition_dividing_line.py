"""Is the loading obstruction a property of the PARTITION?  Every integral PLB of P1 and of the 2x3 table.

Hazelton et al. (2024) advise putting high-mean coordinates in the invertible block A_1, that is, low-mean
coordinates in the free block.  For a free coordinate j* the identity block of U makes exactly one PLB column
move x_{j*} and the other m - 1 hold it fixed, so the dominant stratum {x_{j*} = 0} is the fibre of A with
column j* deleted, moved by m - 1 PLB columns.  The heuristic is therefore the hold-fixed condition of the
companion's converse (Proposition converse).  This script tests the dividing line EXACTLY: for every column
subset S of size n with |det A_S| = 1 (so U is integral), build the PLB, and report

  - whether j* is a pivot (in S) or a free coordinate,
  - how many PLB columns hold x_{j*} fixed, the number of lobes of the stratum {x_{j*}=0} under those moves,
  - whether the PLB is irreducible and exactly augmenting,
  - the exact single-ray gap at mu in {1, ..., 1e-6} (Poisson target, cell j* with mean mu, others 1) and the
    log-log slope over mu <= 0.03 (1 = the Theta(mu) law; 0 = a floor).

Output: results/refQ1_partition_dividing_line.txt.  Deterministic.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.basis import RayView, int_det, is_augmenting, is_irreducible           # noqa: E402
from fibresampler.fibre import FibreProblem                                               # noqa: E402
from fibresampler.problems import A_P1, A_TABLE23, Y_P1, Y_TABLE23                        # noqa: E402
from fibresampler.spectral import normalised_pi, slem, sr_transition_matrix                # noqa: E402
from refQ1_loading_generality import MUS, logw, n_lobes, slope                             # noqa: E402
from verify_assumptions import border_drains                                              # noqa: E402

OUT = os.path.join(REPO, "results", "refQ1_partition_dividing_line.txt")


def study(name, A, y, jstar):
    n = A.shape[0]
    rows = []
    for S in itertools.combinations(range(A.shape[1]), n):
        if abs(int_det(A[:, S])) != 1:
            continue
        try:
            prob = FibreProblem(A, y, cols1=S, name=name)
        except Exception:
            continue
        pv = RayView(prob.states, prob.U, index=prob.index, plb_info=prob.plb_info, name=name)
        hold = int((prob.U[jstar] == 0).sum())
        lobes = n_lobes(prob.states, prob.U, jstar)
        irr = is_irreducible(pv)
        aug = is_augmenting(prob)
        U = np.asarray(prob.U)
        moves = [U[:, b].astype(np.int64) for b in range(U.shape[1])]
        nd, nb, _ = border_drains(prob, jstar, moves + [-u for u in moves])
        gaps = []
        for mu in MUS:
            lw = logw(prob.states, jstar, mu)
            pi = normalised_pi(lw)
            gaps.append(slem(sr_transition_matrix(pv, lw), pi)[1] if irr else 0.0)
        rows.append(dict(S=S, free=jstar not in S, hold=hold, m=prob.n_moves, lobes=lobes, irr=irr, aug=aug, drain=(nd == nb),
                         gap_small=gaps[-1], slope=slope(gaps) if irr else float("nan")))
    return rows


def main():
    W = 112
    out = ["=" * W,
           "Loading obstruction versus the PARTITION: every integral PLB (|det A_1| = 1), exact single-ray gaps",
           "j* = the cell whose Poisson mean mu shrinks (others 1); free = j* in the free block; hold = PLB columns with u_{j*} = 0",
           "=" * W]
    all_rows = []
    for name, A, y, jstar in (("P1 (8 routes, 4 links; cell 5)", A_P1, Y_P1, 4),
                              ("2x3 table (4,4|3,3,2); cell x11", A_TABLE23, Y_TABLE23, 0)):
        rows = study(name, A, y, jstar)
        out.append("")
        out.append(f"{name}: {len(rows)} integral PLBs")
        out.append(f"  {'A_1 columns':>14} {'j*':>5} {'hold/m':>7} {'lobes':>5} {'irr':>4} {'aug':>4} {'drain':>5} {'gap(mu=1e-6)':>13} {'slope':>7}")
        for r in sorted(rows, key=lambda r: (not r["free"], r["S"])):
            out.append(f"  {str(tuple(c + 1 for c in r['S'])):>14} {'free' if r['free'] else 'pivot':>5} "
                       f"{r['hold']:>3}/{r['m']:<3} {r['lobes']:>5d} {'y' if r['irr'] else 'n':>4} {'y' if r['aug'] else 'n':>4} "
                       f"{'y' if r['drain'] else 'n':>5} {r['gap_small']:>13.3e} {r['slope']:>7.3f}")
        piv = [r for r in rows if not r["free"] and r["irr"]]
        fre = [r for r in rows if r["free"] and r["irr"]]
        out.append(f"  pivot (irreducible): n={len(piv)}, slope {np.nanmin([r['slope'] for r in piv]):.3f} to "
                   f"{np.nanmax([r['slope'] for r in piv]):.3f}, lobes {min(r['lobes'] for r in piv)} to {max(r['lobes'] for r in piv)}; "
                   f"free (irreducible): n={len(fre)}, slope {np.nanmin([r['slope'] for r in fre]):.3f} to "
                   f"{np.nanmax([r['slope'] for r in fre]):.3f}, lobes {min(r['lobes'] for r in fre)} to {max(r['lobes'] for r in fre)}")
        n_aug_free = sum(1 for r in fre if r["aug"])
        out.append(f"  free partitions that are exactly augmenting: {n_aug_free} of {len(fre)}")
        irr_rows = [r for r in rows if r["irr"]]
        for label, sel in (("K = 1 ", [r for r in irr_rows if r["lobes"] == 1]), ("K >= 2", [r for r in irr_rows if r["lobes"] >= 2])):
            if sel:
                out.append(f"  {label}: n={len(sel)}, slope {min(r['slope'] for r in sel):.3f} to {max(r['slope'] for r in sel):.3f}, "
                           f"gap(1e-6) {min(r['gap_small'] for r in sel):.2e} to {max(r['gap_small'] for r in sel):.2e}, "
                           f"border drained on all: {all(r['drain'] for r in sel)}")
        n_red = sum(1 for r in rows if not r["irr"])
        out.append(f"  reducible PLBs (gap exactly 0 at every mu, not admissible samplers): {n_red}")
        all_rows.extend(rows)
    irr_all = [r for r in all_rows if r["irr"]]
    k1 = [r for r in irr_all if r["lobes"] == 1]
    k2 = [r for r in irr_all if r["lobes"] >= 2]
    out.append("")
    out.append(f"BOTH FIBRES: {len(all_rows)} integral PLBs, {len(irr_all)} irreducible; dividing line: "
               f"K = 1 on {len(k1)} (slope {min(r['slope'] for r in k1):.3f} to {max(r['slope'] for r in k1):.3f}), "
               f"K >= 2 on {len(k2)} (slope {min(r['slope'] for r in k2):.3f} to {max(r['slope'] for r in k2):.3f}); "
               f"every irreducible K = 1 partition drains: {all(r['drain'] for r in k1)}")
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
