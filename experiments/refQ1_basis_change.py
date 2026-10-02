"""Martin Hazelton's question (and referee M11): why reflect, instead of changing basis?

Reflection needs local geometry at every bounce (wall normals, the lattice reflection rule,
an involution guard, and a reverse-path retrace).  If that geometry is going to be
characterised anyway, one can spend it ONCE on a better lattice basis: replace the free
frame I by a unimodular V whose columns are short in the Mahalanobis metric of the fibre
(LLL in Sigma^{-1}), and run the ordinary single-ray Gibbs sampler along the columns of U V.

This script compares, with EXACT second eigenvalues (no Monte Carlo), on every enumerable
fibre of the paper:

  SR       single-ray Gibbs in the PLB frame (the paper's baseline)
  SR-LLL   single-ray Gibbs in the LLL-reduced frame (basis frozen after the pilot)
  SR-U     the certified PLB moves PLUS the reduced moves (2m directions); irreducible by
           construction, so the reduced basis needs no augmentation certificate of its own
  SR-orc   best unimodular basis found by search (exhaustive |entry|<=3 for m=2, random
           search plus LLL for m=4): the headroom no change of basis can exceed
  Ref-B2/4 the reflective sampler of Section 3, B_max in {2,4}, same exact operator as
           Prediction 3

and reports, for each fibre, the axis defect kappa in the original and in the reduced frame,
the mixing efficiency eta = m * gap (eta = 1 iff the target is a product in the basis
coordinates, since random-scan Gibbs over m directions has gap <= 1/m), the per-step cost of
each kernel measured on the same implementation, and the cost-adjusted gap (gap divided by
measured time per step, relative to SR).

Two-dimensional identity used throughout: the single-ray Gibbs sampler on a 2-D fibre is
the average of the two conditional-expectation projections, so its gap is EXACTLY
(1 - rho)/2 with rho the maximal (Gebelein) correlation of the two basis coordinates.  The
script checks the identity against the exact SLEM on every 2-D fibre.

Output: results/refQ1_basis_change.txt.  Deterministic (seeded).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.basis import (RayView, enumerate_unimodular_2d, free_points,           # noqa: E402
                                int_det, is_augmenting, is_irreducible, random_unimodular,
                                recommend_basis, reduce_basis, to_frame, union_moves,
                                weighted_union_matrix, with_basis)
from fibresampler.fibre import FibreProblem                                               # noqa: E402
from fibresampler.problems import (A_TABLE23, P1, TABLE23_COLS1)                          # noqa: E402
from fibresampler.reflective import reflective_transition_matrix                          # noqa: E402
from fibresampler.spectral import log_weights, normalised_pi, slem, sr_transition_matrix  # noqa: E402
from fibresampler.synthetic import BandProblem, SlopedBand, poisson_free_logw             # noqa: E402
from refK_obliqueness import geom_and_oblq                                                # noqa: E402
from pred5_ess import run_ref_chain, run_sr_chain                                         # noqa: E402

OUT = os.path.join(REPO, "results", "refQ1_basis_change.txt")
SEED = 20261002
BANDS = [(2, 12), (2, 24), (2, 40), (1, 30)]          # kappa 2.15, 4.26, 7.08, 10.61
SLOPED = [(3, 2, 4, 30), (3, 2, 3, 30), (5, 3, 6, 40)]  # ridge (p,q): non-unit tilt, wall normal (q,-p)


# ---------------------------------------------------------------------------
# fibres
# ---------------------------------------------------------------------------
def p4_problem(scale=1):
    from pred6_balanced import A_P4, Y0_P4
    return FibreProblem(A_P4, Y0_P4 * scale, cols1=tuple(range(7)), name=f"P4(x{scale})")


def make_fibres():
    fibres = []
    for (w, C) in BANDS:
        fibres.append((f"band(w{w},C{C})", BandProblem(w, C), "band"))
    for (p, q, W, C) in SLOPED:
        fibres.append((f"sloped({p},{q};W{W},C{C})", SlopedBand(p, q, W, C), "band"))
    fibres.append(("P1", P1(), "net"))
    fibres.append(("table(4,4|3,3,2)", FibreProblem(A_TABLE23, np.array([4, 3, 3, 2]),
                                                    cols1=TABLE23_COLS1), "table"))
    fibres.append(("table(6,6|1,5,6)", FibreProblem(A_TABLE23, np.array([6, 1, 5, 6]),
                                                    cols1=TABLE23_COLS1), "table"))
    fibres.append(("P4(x1)", p4_problem(1), "net"))
    fibres.append(("P4(x2)", p4_problem(2), "net"))
    return fibres


def target_logw(prob, kind, target):
    """Unnormalised log-target over the enumerated fibre."""
    if target == "uniform":
        return np.zeros(prob.fibre_size)
    if kind == "band":                        # product Poisson(1) on the two real counts
        return poisson_free_logw(prob, [1.0, 1.0])
    theta = np.ones(prob.states.shape[1])     # product Poisson(1) on every cell
    return log_weights(prob, "poisson", theta)


# ---------------------------------------------------------------------------
# exact arms
# ---------------------------------------------------------------------------
def view(prob):
    return RayView(prob.states, prob.U, index=prob.index, plb_info=prob.plb_info,
                   name=getattr(prob, "name", ""))


def gap_sr(pv, lw, pi):
    Q = sr_transition_matrix(pv, lw)
    return slem(Q, pi)[1]


def gap_ref(prob, lw, pi, bmax):
    Q, _ = reflective_transition_matrix(prob, bmax=bmax, log_w=lw)
    return slem(Q, pi)[1]


def maximal_correlation(c, pi):
    """Gebelein maximal correlation of the two coordinates of c (rows) under pi."""
    a_vals = sorted(set(int(v) for v in c[:, 0]))
    b_vals = sorted(set(int(v) for v in c[:, 1]))
    ai = {v: i for i, v in enumerate(a_vals)}
    bi = {v: i for i, v in enumerate(b_vals)}
    J = np.zeros((len(a_vals), len(b_vals)))
    for (a, b), p in zip(c, pi):
        J[ai[int(a)], bi[int(b)]] += p
    T = J / np.sqrt(np.outer(J.sum(1), J.sum(0)))
    return float(np.linalg.svd(T, compute_uv=False)[1])


def oracle_gap(pv, fr, lw, pi, m, rng, n_random):
    """Best single-ray Gibbs gap over a search of unimodular bases."""
    best_g, best_V = -1.0, None
    if m == 2:
        cands = enumerate_unimodular_2d(3)
    else:
        cands = [np.eye(m, dtype=np.int64)] + [random_unimodular(m, rng) for _ in range(n_random)]
    for V in cands:
        g = gap_sr(with_basis(pv, V), lw, pi)
        if g > best_g:
            best_g, best_V = g, V
    return best_g, best_V


def mean_ray(pv):
    """Mean number of lattice points on a ray (chord length in steps), averaged over
    states and directions."""
    tot, cnt = 0, 0
    for i in range(pv.fibre_size):
        for j in range(pv.n_moves):
            tot += len(pv.ray(i, j))
            cnt += 1
    return tot / cnt


def density(U):
    """Mean number of nonzero entries per move (full space), and max |entry|."""
    return float((U != 0).sum(0).mean()), int(np.abs(U).max())


# ---------------------------------------------------------------------------
# measured per-step cost, identical implementations (pred5_ess chains)
# ---------------------------------------------------------------------------
def step_cost(prob, pv_sr, bmax=None, n_iter=6000, burn=500, reps=3):
    """Microseconds per step of the SR chain on pv_sr, or of the reflective chain on
    prob when bmax is given; uniform target, best of `reps`."""
    zero = lambda X: np.zeros(np.atleast_2d(X).shape[0])                       # noqa: E731
    x0 = np.asarray(prob.states[0], dtype=np.int64)
    best = np.inf
    for r in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence((SEED, r)))
        if bmax is None:
            _, secs, _ = run_sr_chain(pv_sr, zero, x0, n_iter, burn, rng, [0], False)
        else:
            _, secs, _ = run_ref_chain(prob, zero, x0, n_iter, burn, rng, [0], False, bmax=bmax)
        best = min(best, secs)
    return 1e6 * best / (n_iter + burn)


# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    W = 118
    out = ["=" * W,
           "Martin Hazelton / referee M11: change of basis (LLL) versus reflection, EXACT gaps",
           "gap = 1 - SLEM of the exact transition matrix; eta = m * gap (1 = product target);",
           "cost = measured us/step on the same implementation; eff = (gap/cost) relative to SR",
           "=" * W]

    for target in ("uniform", "pois1"):
        out.append("")
        out.append(f"TARGET = {target}" + ("  (product Poisson, every cell mean 1)" if target == "pois1" else ""))
        out.append(f"  {'fibre':>22} {'N':>4} {'m':>2} {'kappa':>6} {'kap_sel':>7} | {'SR':>7} {'SR-LLL':>7} "
                   f"{'SR-sel':>7} {'SR-U.5':>7} {'SR-U.1':>7} {'SR-orc':>7} {'Ref-B2':>7} {'Ref-B4':>7} | "
                   f"{'sel/SR':>6} {'sel/Ref':>7} {'eta_sel':>7} | {'decision':>10} irr augP augS floor")
        for name, prob, kind in make_fibres():
            if target == "pois1" and kind == "band" and name != "band(w2,C24)":
                continue
            N = prob.fibre_size
            m = prob.n_moves
            lw = target_logw(prob, kind, target)
            pi = normalised_pi(lw)
            pv = view(prob)
            fr = free_points(prob)
            g_sr = gap_sr(pv, lw, pi)
            V_lll = reduce_basis(fr, pi)
            assert abs(int_det(V_lll)) == 1
            g_lll = gap_sr(with_basis(pv, V_lll), lw, pi)
            V_sel, info = recommend_basis(fr, pi)
            assert abs(int_det(V_sel)) == 1
            sel = with_basis(pv, V_sel)
            g_sel = gap_sr(sel, lw, pi)
            g_u5 = gap_sr(union_moves(pv, V_sel), lw, pi)
            g_u1 = slem(weighted_union_matrix(pv, V_sel, lw, 0.1), pi)[1]
            irr = is_irreducible(sel)
            aug = (is_augmenting(sel) if N <= 70 else None)
            aug0 = (is_augmenting(pv) if N <= 70 else None)
            floor_ok = g_u1 >= max(0.1 * g_sr, 0.9 * g_sel) - 1e-9     # PSD mixture: gap >= max(lam*gap_PLB, (1-lam)*gap_red)
            assert floor_ok, (name, g_u1, g_sr, g_sel)
            n_rand = 1500 if N <= 70 else (150 if N <= 500 else 0)
            if m == 2 or n_rand > 0:
                g_orc = max(oracle_gap(pv, fr, lw, pi, m, rng, n_rand)[0], g_lll, g_sr, g_sel)
            else:
                g_orc = float("nan")
            g_b2 = gap_ref(prob, lw, pi, 2) if N <= 120 else float("nan")
            g_b4 = gap_ref(prob, lw, pi, 4) if N <= 120 else float("nan")
            kap0 = geom_and_oblq(fr)[0]
            kap1 = geom_and_oblq(to_frame(fr, V_sel))[0]
            ref_best = np.nanmax([g_b2, g_b4]) if N <= 120 else float("nan")
            sel_ref = g_sel / ref_best if np.isfinite(ref_best) else float("nan")
            out.append(f"  {name:>22} {N:>4d} {m:>2d} {kap0:>6.2f} {kap1:>7.2f} | {g_sr:>7.4f} {g_lll:>7.4f} "
                       f"{g_sel:>7.4f} {g_u5:>7.4f} {g_u1:>7.4f} {g_orc:>7.4f} {g_b2:>7.4f} {g_b4:>7.4f} | "
                       f"{g_sel/g_sr:>6.2f} {sel_ref:>7.2f} {m*g_sel:>7.3f} | {info['chosen']:>10} "
                       f"{'yes' if irr else 'NO':>3} {'-' if aug0 is None else ('yes' if aug0 else 'NO'):>4} "
                       f"{'-' if aug is None else ('yes' if aug else 'NO'):>4} {'ok' if floor_ok else 'FAIL':>5}")
            if m == 2 and target == "uniform":
                r0 = maximal_correlation(fr, pi)
                r1 = maximal_correlation(to_frame(fr, V_sel), pi)
                out.append(f"  {'':>22} identity (1-rho)/2: SR {(1-r0)/2:.4f} vs exact {g_sr:.4f}; "
                           f"selected {(1-r1)/2:.4f} vs exact {g_sel:.4f}; rho {r0:.4f} -> {r1:.4f}; V columns "
                           f"{[tuple(int(v) for v in V_sel[:, k]) for k in range(m)]}")
                assert abs((1 - r0) / 2 - g_sr) < 1e-8 and abs((1 - r1) / 2 - g_sel) < 1e-8

    # ------------------------------------------------- pilot-estimated basis
    out.append("")
    out.append("PILOT ROBUSTNESS: basis from the covariance of a SHORT single-ray pilot chain (uniform target),")
    out.append("not from exact fibre moments; exact gap of the resulting reduced basis, 20 pilot seeds each")
    out.append(f"  {'fibre':>22} {'pilot':>6} | {'exact-sel':>9} {'mean':>8} {'min':>8} {'frac>=90%':>9}  (gap of SR with the pilot-selected basis)")
    zero = lambda X: np.zeros(np.atleast_2d(X).shape[0])                       # noqa: E731
    for name, prob, kind in make_fibres():
        if name not in ("band(w2,C24)", "band(w1,C30)", "sloped(3,2;W4,C30)", "table(6,6|1,5,6)",
                        "P1", "P4(x1)"):
            continue
        pv = view(prob)
        fr = free_points(prob)
        lw = np.zeros(prob.fibre_size)
        pi = normalised_pi(lw)
        g_exact = gap_sr(with_basis(pv, recommend_basis(fr, pi)[0]), lw, pi)
        cols2 = list(prob.plb_info["cols2"])
        x0 = np.asarray(prob.states[0], dtype=np.int64)
        for n_pilot in (300, 1000, 5000):
            gaps = []
            for s in range(20):
                prng = np.random.default_rng(np.random.SeedSequence((SEED, 99, n_pilot, s)))
                tr, _, _ = run_sr_chain(pv, zero, x0, n_pilot, 100, prng, cols2, False)
                try:
                    Vp = recommend_basis(tr.astype(float))[0]
                except Exception:
                    Vp = np.eye(len(cols2), dtype=np.int64)
                gaps.append(gap_sr(with_basis(pv, Vp), lw, pi))
            gaps = np.array(gaps)
            out.append(f"  {name:>22} {n_pilot:>6d} | {g_exact:>9.4f} {gaps.mean():>8.4f} {gaps.min():>8.4f} "
                       f"{(gaps >= 0.9 * g_exact).mean():>9.2f}")

    # ------------------------------------------------------------------ cost
    out.append("")
    out.append("MEASURED COST PER STEP (us), uniform target, same implementation (pred5_ess chains)")
    out.append(f"  {'fibre':>17} {'SR':>7} {'SR-LLL':>7} {'SR-U':>7} {'Ref-B2':>7} {'Ref-B4':>7} | "
               f"{'LLL/SR':>6} {'B2/SR':>6} {'B4/SR':>6} | mean ray (steps): SR  LLL | nnz/move SR LLL | max|u| SR LLL")
    for name, prob, kind in make_fibres():
        if prob.fibre_size > 120 and name != "P4(x2)":
            continue
        pv = view(prob)
        fr = free_points(prob)
        pi = normalised_pi(np.zeros(prob.fibre_size))
        V = reduce_basis(fr, pi)
        red, uni = with_basis(pv, V), union_moves(pv, V)
        c_sr = step_cost(prob, pv)
        c_lll = step_cost(prob, red)
        c_u = step_cost(prob, uni)
        if prob.fibre_size <= 120:
            c_b2 = step_cost(prob, pv, bmax=2)
            c_b4 = step_cost(prob, pv, bmax=4)
        else:
            c_b2 = c_b4 = float("nan")
        d0, mx0 = density(pv.U)
        d1, mx1 = density(red.U)
        out.append(f"  {name:>17} {c_sr:>7.1f} {c_lll:>7.1f} {c_u:>7.1f} {c_b2:>7.1f} {c_b4:>7.1f} | "
                   f"{c_lll/c_sr:>6.2f} {c_b2/c_sr:>6.2f} {c_b4/c_sr:>6.2f} | "
                   f"{mean_ray(pv):>14.2f} {mean_ray(red):>5.2f} | {d0:>10.1f} {d1:>4.1f} | {mx0:>6d} {mx1:>3d}")

    out.append("")
    out.append("READING: see the paper's Section 6 (basis reduction).  eta = m*gap measures how close the")
    out.append("frame is to decoupling the target; LLL pushes eta toward 1 where the fibre is a thin tilted")
    out.append("body aligned with a short lattice vector, and leaves a near-isotropic fibre alone.")
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
