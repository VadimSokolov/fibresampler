"""Is there a niche where reflection beats a change of basis?  Exact test on near-golden-slope bands.

On a thin band around the lattice line q s = p t the PLB axes are tilted and the ridge vector
(p, q) is itself a lattice vector, so a unimodular basis containing it aligns exactly (the
`sloped` fibres of refQ1_basis_change.py).  The adversarial case for a change of basis is a
ridge whose slope is badly approximated by short lattice vectors: consecutive Fibonacci ratios
p/q = F_{k+1}/F_k.  The aligned vector (q, p) is then LONG (norm F_{k+1}-ish), the band width W
is held small relative to it, and the shortest lattice vectors that stay inside the band are
the Fibonacci convergents.  Reflection does not need the ridge to be a short lattice vector,
only the image walls (normals (q,-p)), but the lattice reflection rule is not exactly specular
there.  Both are measured by the exact SLEM, uniform target, SR / SR-LLL / selected / oracle /
Ref-B2 / Ref-B4, same operators as refQ1_basis_change.py.

Output: results/refQ1_fibonacci_slope.txt.  Deterministic.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.basis import (RayView, enumerate_unimodular_2d, free_points,            # noqa: E402
                                recommend_basis, to_frame, with_basis)
from fibresampler.reflective import reflective_transition_matrix                          # noqa: E402
from fibresampler.spectral import normalised_pi, slem, sr_transition_matrix                # noqa: E402
from fibresampler.synthetic import SlopedBand                                              # noqa: E402
from refK_obliqueness import geom_and_oblq                                                 # noqa: E402

OUT = os.path.join(REPO, "results", "refQ1_fibonacci_slope.txt")
# (p, q, W, C): ridge q s = p t with p/q a Fibonacci ratio; W is the half-width in the units of q s - p t
CASES = [(2, 1, 2, 30), (3, 2, 3, 40), (5, 3, 4, 60), (8, 5, 6, 80), (13, 8, 8, 120)]


def view(prob):
    return RayView(prob.states, prob.U, index=prob.index, plb_info=prob.plb_info, name=prob.name)


def gap_sr(pv, lw, pi):
    return slem(sr_transition_matrix(pv, lw), pi)[1]


def main():
    W = 132
    out = ["=" * W,
           "Near-golden-slope bands: exact single-ray gaps in the PLB, LLL-selected and oracle bases, versus the reflective sampler",
           "ridge q s = p t with p/q a Fibonacci ratio; uniform target; oracle = best unimodular V with entries |.|<=4 (exhaustive, m=2)",
           "=" * W,
           f"  {'(p,q;W,C)':>14} {'N':>6} {'kappa':>6} {'kap_sel':>7} | {'SR':>9} {'sel':>9} {'orc|.|<=4':>9} {'Ref-B2':>9} {'Ref-B4':>9} | "
           f"{'sel/SR':>7} {'orc/SR':>7} {'sel/Ref':>8} {'sel/orc':>8}  V_sel"]
    for (p, q, Wd, C) in CASES:
        prob = SlopedBand(p, q, Wd, C)
        pv = view(prob)
        fr = free_points(prob)
        lw = np.zeros(prob.fibre_size)
        pi = normalised_pi(lw)
        g_sr = gap_sr(pv, lw, pi)
        V_sel, info = recommend_basis(fr, pi)
        g_sel = gap_sr(with_basis(pv, V_sel), lw, pi)
        best_g, best_V = 0.0, None
        for V in enumerate_unimodular_2d(4):
            g = gap_sr(with_basis(pv, V), lw, pi)
            if g > best_g:
                best_g, best_V = g, V
        gr = {}
        for b in (2, 4):
            Q, _ = reflective_transition_matrix(prob, bmax=b, log_w=lw)
            gr[b] = slem(Q, pi)[1]
        kap = geom_and_oblq(fr)[0]
        kap_sel = geom_and_oblq(to_frame(fr, V_sel))[0]
        vs = [tuple(int(v) for v in V_sel[:, k]) for k in range(V_sel.shape[1])]
        g_ref = max(gr[2], gr[4])
        ratio_ref = f"{g_sel / g_ref:>8.2f}" if g_ref > 1e-6 else f"{'(Ref~0)':>8}"
        ratio_sr = f"{g_sel / g_sr:>7.1f}" if g_sr > 1e-9 else f"{'(SR~0)':>7}"
        ratio_orc_sr = f"{best_g / g_sr:>7.1f}" if g_sr > 1e-9 else f"{'(SR~0)':>7}"
        out.append(f"  {f'({p},{q};{Wd},{C})':>14} {prob.fibre_size:>6d} {kap:>6.2f} {kap_sel:>7.2f} | {g_sr:>9.3g} {g_sel:>9.4f} {best_g:>9.4f} "
                   f"{gr[2]:>9.3g} {gr[4]:>9.3g} | {ratio_sr} {ratio_orc_sr} {ratio_ref} {g_sel / best_g:>8.2f}  {vs}")
    out.append("")
    out.append("reading: sel/Ref > 1 means the selected basis beats the better of Ref-B2 / Ref-B4 per iteration (before the 2-8x step cost).")
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
