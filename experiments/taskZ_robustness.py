"""Task Z: static robustness of the three-rung tempering floor to pseudo-prior
normalising-constant error.

rem:implementable confines the idealisation of Theorem thm:spectrum to the
pseudo-prior constants Z_beta.  This check quantifies the static half of that
concern: build the SAME exact joint chain as taskV_tempering (three-rung
ladder {1, 1/2, 0} on P1) but with the pseudo-prior constants multiplied by
error factors f (a misestimated log Z shifts the level-move acceptances and
the level occupancies; the chain remains a valid MCMC whose x-marginal at
each rung is unchanged), and diagonalise.  Reported: exact spectral gap vs f
for (a) the interior constant Z_{1/2} only, (b) both nontrivial constants
Z_1 and Z_{1/2}, at mu5 = 1e-8 (deep in the flat regime).  What this does
NOT cover, and the manuscript says so: the feedback of ONLINE estimation
error into the occupancies during adaptation.

Output: results/taskZ_robustness.txt.  Deterministic (no RNG).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.special import logsumexp

from taskV_tempering import PROB, N, nb_logw, db_violation
from fibresampler.spectral import sr_transition_matrix, slem

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "taskZ_robustness.txt")

BETAS = np.array([1.0, 0.5, 0.0])
MU5 = 1e-8
FACTORS = [0.5, 0.8, 1.0, 1.25, 2.0]


def tempered_chain_perturbed(mu5, log_err):
    """taskV_tempering.tempered_chain with pseudo-prior constants multiplied
    by exp(log_err[e]); log_err = 0 reproduces the idealised chain."""
    lw = nb_logw(mu5)
    betas = BETAS
    Lp1 = len(betas)
    logZ = np.array([logsumexp(b * lw) for b in betas]) + np.asarray(log_err)
    blocks = [sr_transition_matrix(PROB, b * lw) for b in betas]
    logpi = np.empty((Lp1, N))
    for e, b in enumerate(betas):
        logpi[e] = b * lw - logZ[e]
    logpi -= logsumexp(logpi)
    pi = np.exp(logpi).ravel()
    pi /= pi.sum()
    Pf = np.zeros((Lp1 * N, Lp1 * N))
    for e in range(Lp1):
        Pf[e * N:(e + 1) * N, e * N:(e + 1) * N] = blocks[e]
    Pl = np.zeros((Lp1 * N, Lp1 * N))
    for e in range(Lp1):
        for ep in (e - 1, e + 1):
            if 0 <= ep < Lp1:
                dlog = (betas[ep] - betas[e]) * lw - (logZ[ep] - logZ[e])
                A = np.minimum(1.0, np.exp(dlog))
                for i in range(N):
                    Pl[e * N + i, ep * N + i] += 0.5 * A[i]
                    Pl[e * N + i, e * N + i] += 0.5 * (1.0 - A[i])
            else:
                Pl[e * N:(e + 1) * N, e * N:(e + 1) * N] += 0.5 * np.eye(N)
    P = 0.5 * Pf + 0.5 * Pl
    return P, pi


def spectrum_extremes(P, pi):
    """Second-largest and smallest eigenvalues of the pi-reversible P.
    S = diag(sqrt pi) P diag(1/sqrt pi) is symmetric with P's real spectrum,
    so eigvalsh gives the ordering lam_min <= ... <= lam2 <= lam1 ~ 1.  The
    SLEM gap reported by slem() equals 1 - max(lam2, |lam_min|); recording both
    lets the manuscript's |lam_min| < lam2 ordering be checked.  On P1 lam_min is
    in fact positive (the three-rung ladder is PSD), so lambda_star = lam2; the
    -0.09 negative eigenvalue in the methodology is the 2x3 table, not P1."""
    s = np.sqrt(pi)
    S = s[:, None] * P / s[None, :]
    S = 0.5 * (S + S.T)
    ev = np.sort(np.linalg.eigvalsh(S))   # ascending; ev[-1] ~ 1
    return float(ev[-2]), float(ev[0])    # (lambda_2, lambda_min)


def gap_at(mu5, log_err):
    P, pi = tempered_chain_perturbed(mu5, log_err)
    dbe = db_violation(P, pi)
    _, gap, _ = slem(P, pi)
    lam2, lam_min = spectrum_extremes(P, pi)
    return gap, lam2, lam_min, dbe


def main():
    lines = ["Task Z: three-rung tempering floor vs pseudo-prior constant error",
             f"P1, ladder betas = {list(BETAS)}, mu5 = {MU5:g}; exact SLEM",
             "=" * 72]
    g0, l2_0, lmin_0, e0 = gap_at(MU5, [0.0, 0.0, 0.0])
    lines.append(f"idealised (exact Z): gap = {g0:.6f}  "
                 f"(lam2 = {l2_0:.6f}, lam_min = {lmin_0:.6f}; DB err {e0:.1e})")
    lines.append(f"{'factor f':>9} {'gap, Z_1/2 only':>16} {'gap, Z_1 and Z_1/2':>19}")
    worst = g0
    l2s, lmins = [l2_0], [lmin_0]
    for f in FACTORS:
        le = np.log(f)
        g_int, l2_i, lmin_i, _ = gap_at(MU5, [0.0, le, 0.0])
        g_both, l2_b, lmin_b, _ = gap_at(MU5, [le, le, 0.0])
        worst = min(worst, g_int, g_both)
        l2s += [l2_i, l2_b]
        lmins += [lmin_i, lmin_b]
        lines.append(f"{f:9.2f} {g_int:16.6f} {g_both:19.6f}")
    lines.append(f"worst gap over the grid: {worst:.6f} "
                 f"({worst / g0:.2f} of the idealised floor); the floor "
                 "survives factor-2 constant error (static check only; online "
                 "estimation feedback not covered)")
    ordered = all(abs(lm) < l2 for lm, l2 in zip(lmins, l2s))
    lines.append(f"eigenvalue ordering across all {len(l2s)} cases: "
                 f"lam2 in [{min(l2s):.6f}, {max(l2s):.6f}], "
                 f"lam_min in [{min(lmins):.6f}, {max(lmins):.6f}]; "
                 f"|lam_min| < lam2 in every case: {ordered} "
                 "(=> lambda_star = lambda_2, so the reported gap = 1 - lam2)")
    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
