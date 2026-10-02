"""Closed-form check of the band proposition behind the change-of-basis lever.

On the unit band F_{w,C} = {s,t >= 0: |s-t| <= w, s+t <= C} with the uniform law:

  (1) PLB frame, single-ray hit-and-run: gap <= 2 w^2 / Var_pi(s+t) = O(kappa^-2).  The test
      function f = s+t changes by at most 2w per heat-bath step, so the Dirichlet form is at
      most 2 w^2 and the variational characterisation of the gap gives the bound.
  (2) Reduced basis V = [(1,1),(1,0)], coordinates (a,b) = (t, s-t): the single-ray sampler is
      the average of the two conditional-expectation projections, so its gap is EXACTLY
      (1-rho)/2 with rho the Gebelein maximal correlation of (a,b), and rho <= phi, where
      phi^2 = sum_{a,b} pi^2/(pi_a pi_b) - 1 = sum_{(a,b) in R} 1/(c(a) r(b)) - 1 is the
      mean-square contingency (c(a), r(b) are the column and row counts of the region R).
  (3) Explicit bound: phi^2 <= y := (2w+1)/(C-w+1) for C >= w (each row count r(b) is at least
      (C-w+1)/2 and the column counts c(a) cancel against the number of columns), hence
      gap >= (1 - sqrt(y))/2.
  (4) The ratio of the reduced to the PLB-frame gap grows like kappa^2.

Everything is exact (SVD of the joint table, dense eigenvalues of the SR operator where the
fibre is small).  Output: results/refQ1_band_bound.txt.  Deterministic.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.basis import RayView, with_basis                                # noqa: E402
from fibresampler.spectral import normalised_pi, slem, sr_transition_matrix        # noqa: E402
from fibresampler.synthetic import BandProblem                                     # noqa: E402

OUT = os.path.join(REPO, "results", "refQ1_band_bound.txt")


def region(w, C):
    """Lattice points of the band in the reduced coordinates (a, b) = (t, s-t)."""
    return [(a, b) for b in range(-w, w + 1) for a in range(max(0, -b), (C - b) // 2 + 1)]


def contingency(w, C):
    """(N, rho, phi^2, Var(s+t)) of the uniform law on the band, from the joint table."""
    pts = region(w, C)
    a_vals = sorted({p[0] for p in pts})
    b_vals = sorted({p[1] for p in pts})
    ai = {v: i for i, v in enumerate(a_vals)}
    bi = {v: i for i, v in enumerate(b_vals)}
    J = np.zeros((len(a_vals), len(b_vals)))
    for a, b in pts:
        J[ai[a], bi[b]] += 1
    N = J.sum()
    J /= N
    T = J / np.sqrt(np.outer(J.sum(1), J.sum(0)))
    sv = np.linalg.svd(T, compute_uv=False)
    sigma = np.array([2 * a + b for a, b in pts], float)          # s + t = 2a + b
    return int(N), float(sv[1]), float((sv[1:] ** 2).sum()), float(sigma.var())


def exact_gaps(w, C):
    """Exact single-ray gaps from the transition matrices in the PLB and reduced frames."""
    prob = BandProblem(w, C)
    pv = RayView(prob.states, prob.U, index=prob.index, plb_info=prob.plb_info, name="band")
    lw = np.zeros(prob.fibre_size)
    pi = normalised_pi(lw)
    g_plb = slem(sr_transition_matrix(pv, lw), pi)[1]
    V = np.array([[1, 1], [1, 0]], dtype=np.int64)                # columns (1,1) and (1,0)
    g_red = slem(sr_transition_matrix(with_basis(pv, V), lw), pi)[1]
    return prob.fibre_size, g_plb, g_red


def main():
    W = 112
    out = ["=" * W,
           "Band proposition: PLB-frame gap <= 2w^2/Var(s+t); reduced-basis gap = (1-rho)/2 >= (1 - sqrt(y))/2, y = (2w+1)/(C-w+1)",
           "kappa = C/(2 sqrt2 w); y = (2w+1)/(C-w+1); the bound phi^2 <= y holds for C >= w (grid below: C >= 4w)",
           "=" * W,
           f"  {'w':>2} {'C':>5} {'N':>6} {'kappa':>7} | {'gap PLB':>9} {'2w^2/Var':>9} {'ok1':>3} | "
           f"{'gap red':>9} {'(1-rho)/2':>9} {'id':>3} {'phi^2':>8} {'y':>8} {'ok3':>3} {'lower':>7} {'ok2':>3} | "
           f"{'ratio':>8} {'ratio/k^2':>9}"]
    n_fail = 0
    worst_phi_ratio = 0.0
    for w in (1, 2, 3, 5, 8):
        for C in sorted({4 * w, 4 * w + 1, 12, 13, 24, 25, 40, 41, 80, 160, 400}):
            if C < 4 * w:
                continue
            N, rho, phi2, var_sigma = contingency(w, C)
            kappa = C / (2 * np.sqrt(2) * w)
            y = (2 * w + 1) / (C - w + 1)
            ok3 = phi2 <= y + 1e-12
            lower = 0.5 * (1 - np.sqrt(y))                # gap >= (1 - sqrt(y))/2, from rho <= phi <= sqrt(y)
            if N <= 700:
                _, g_plb, g_red = exact_gaps(w, C)
                bound1 = 2 * w * w / var_sigma
                ok1 = g_plb <= bound1 + 1e-12
                idn = abs(g_red - (1 - rho) / 2) < 1e-8
                ok2 = g_red >= lower - 1e-12
                ratio = g_red / g_plb
                row = (f"  {w:>2d} {C:>5d} {N:>6d} {kappa:>7.2f} | {g_plb:>9.5f} {bound1:>9.5f} {'y' if ok1 else 'N':>3} | "
                       f"{g_red:>9.5f} {(1 - rho) / 2:>9.5f} {'y' if idn else 'N':>3} {phi2:>8.5f} {y:>8.5f} "
                       f"{'y' if ok3 else 'N':>3} {lower:>7.4f} {'y' if ok2 else 'N':>3} | {ratio:>8.2f} {ratio / kappa ** 2:>9.3f}")
                n_fail += (not ok1) + (not idn) + (not ok2) + (not ok3)
            else:
                g_red = (1 - rho) / 2
                ok2 = g_red >= lower - 1e-12
                row = (f"  {w:>2d} {C:>5d} {N:>6d} {kappa:>7.2f} | {'-':>9} {2 * w * w / var_sigma:>9.5f} {'-':>3} | "
                       f"{'-':>9} {g_red:>9.5f} {'-':>3} {phi2:>8.5f} {y:>8.5f} {'y' if ok3 else 'N':>3} "
                       f"{lower:>7.4f} {'y' if ok2 else 'N':>3} | {'-':>8} {'-':>9}")
                n_fail += (not ok3) + (not ok2)
            worst_phi_ratio = max(worst_phi_ratio, phi2 / y)
            out.append(row)
    out.append("")
    out.append(f"checks failed: {n_fail}; largest phi^2/y over the grid: {worst_phi_ratio:.3f}")
    out.append("columns: ok1 = PLB gap <= 2w^2/Var(s+t); id = exact SR gap equals (1-rho)/2; ok3 = phi^2 <= y;")
    out.append("         lower = (1 - sqrt(y))/2; ok2 = reduced gap >= lower; ratio = reduced gap / PLB gap.")

    # leading-order constants of rho^2 and phi^2 as C grows (w fixed): rho^2 C/w and phi^2 C/w
    out.append("")
    out.append("Leading order as C grows (w fixed): rho^2 * C / w and phi^2 * C / w (even C, then odd C)")
    out.append(f"  {'w':>2} {'C':>6} {'rho^2 C/w':>10} {'phi^2 C/w':>10} | {'C+1':>6} {'rho^2 C/w':>10} {'phi^2 C/w':>10}")
    for w in (1, 2, 3, 5):
        for C in (200, 1000, 2000):
            _, r0, p0, _ = contingency(w, C)
            _, r1, p1, _ = contingency(w, C + 1)
            out.append(f"  {w:>2d} {C:>6d} {r0 ** 2 * C / w:>10.4f} {p0 * C / w:>10.4f} | {C + 1:>6d} "
                       f"{r1 ** 2 * (C + 1) / w:>10.4f} {p1 * (C + 1) / w:>10.4f}")
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
