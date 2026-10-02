"""Verify the converse's two hypotheses for the hold-fixed basis on every multi-lobe fibre of the loading study.

Proposition converse (companion, Section conservation) gives gap(K_R) >= gamma_0 for every R and every small
bottleneck mean when (a) the dominant stratum D = {x_{j*} = 0} is connected under the moves with u_{j*} = 0 and
(b) every border state drains to D by a never-raising path.  Both are finite computations.  For each fibre of
refQ1_loading_generality.py this script runs them for the PLB moves and for the hold-fixed moves U V (V from
Hermite reduction of the bottleneck row of U), reporting the number of lobes K, the drained fraction of the
border and the longest drain path, and the corridor height k* where K >= 2.  It also records the largest entry and
the mean number of nonzeros per move of U and of U V, since a recombined move set is only useful if its moves stay short.
Output: results/refQ1_hold_fixed_assumptions.txt.  Deterministic.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.basis import hold_fixed_basis, int_det                                  # noqa: E402
from refQ1_loading_generality import fibres                                                # noqa: E402
from verify_assumptions import border_drains, corridor_height, lobe_components             # noqa: E402

OUT = os.path.join(REPO, "results", "refQ1_hold_fixed_assumptions.txt")


def signed_moves(U):
    cols = [np.asarray(U)[:, b].astype(np.int64) for b in range(np.asarray(U).shape[1])]
    return cols + [-u for u in cols]


def report(prob, jstar, U):
    moves = signed_moves(U)
    K, D, lobe_of = lobe_components(prob, jstar, moves)
    nd, nb, mlen = border_drains(prob, jstar, moves)
    kstar = corridor_height(prob, jstar, moves, lobe_of) if K >= 2 else None
    return K, len(D), nd, nb, mlen, kstar


def main():
    W = 120
    out = ["=" * W,
           "Hypotheses of the converse on the PLB and on the hold-fixed basis: lobes K of D under the u_{j*}=0 moves, border drain, corridor height k*",
           "=" * W,
           f"  {'fibre':>26} {'N':>5} {'|D|':>4} {'|B|':>4} | {'PLB: K':>7} {'drained':>9} {'k*':>3} | {'hold-fixed: K':>13} {'drained':>9} {'max len':>8} {'slice moves':>11} | {'max|U|':>6} {'max|UV|':>7} {'nnz U':>6} {'nnz UV':>6}"]
    ok_all = True
    for name, prob, jstar in fibres():
        V, n_slice = hold_fixed_basis(prob.U, [jstar])
        assert abs(int_det(V)) == 1
        U_hf = prob.U @ V
        K0, nD, nd0, nb0, ml0, ks0 = report(prob, jstar, prob.U)
        K1, _, nd1, nb1, ml1, _ = report(prob, jstar, U_hf)
        ok_all &= (K1 == 1 and nd1 == nb1)
        out.append(f"  {name:>26} {prob.fibre_size:>5d} {nD:>4d} {nb0:>4d} | {K0:>7d} {f'{nd0}/{nb0}':>9} {str(ks0):>3} | "
                   f"{K1:>13d} {f'{nd1}/{nb1}':>9} {ml1:>8d} {n_slice:>11d} | "
                   f"{int(np.abs(prob.U).max()):>6d} {int(np.abs(U_hf).max()):>7d} "
                   f"{float((prob.U != 0).sum(0).mean()):>6.3f} {float((U_hf != 0).sum(0).mean()):>6.3f}")
    out.append("")
    out.append("hypotheses (a) K = 1 and (b) full border drain hold for the hold-fixed basis on every fibre listed: "
               + ("YES" if ok_all else "NO"))
    text = "\n".join(out) + "\n"
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
