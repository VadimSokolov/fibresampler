"""M4 -- real-data-scale ESS/second benchmark (referee point: publication viability).

Martin Hazelton's review asks for ESS-per-second on a realistically sized problem,
not only the small enumerable fibres that decide the exact-SLEM predictions.  This
script runs the four-sampler ablation on

  * P2 = the Auckland State Highway 16 corridor of Hazelton, McVeagh, Tuffley & van
    Brunt (2024, Bernoulli 30(4), Section 6): 21 origin-destination routes on 7
    directed links, real observed link loads y=(2991,3352,3977,4576,3849,2458,978).
    The fibre {x>=0 integer : A x = y} has r-n = 14 free dimensions and is far too
    large to enumerate, so exact SLEM is unavailable and ESS is estimated from long
    chains -- exactly the regime a practitioner faces.

Non-enumerating path.  FibreProblem enumerates on construction; here we build only
the PLB U = [-A1^{-1}A2 ; I] and a feasible start x0 = [A1^{-1}y ; 0] (A1 is
unimodular for the short-path partition, so x0 is a non-negative integer point).
The SR and Ref chains are imported verbatim from experiments/pred5_ess.py -- the
identical kernels whose reversibility and estimator accuracy are gate-checked
exactly on the enumerable P1/product fibres there (that is the correctness anchor
this scale-up inherits).  Every proposal is a lattice move in ker A, so A x = y is
preserved bit-for-bit at every step (checked at the end, gate G0).

Targets.  Two, both reproducible:
  * uniform on the fibre (the beta=0 reference rung the augmenting-basis theory
    certifies as fast-mixing; parameter-free), SR vs Ref -- isolates GEOMETRIC
    mixing at real scale;
  * a max-entropy Poisson route-flow model theta* (the standard network-tomography
    "entropy/gravity" prior: theta* = argmax entropy s.t. A theta*=y, found by
    Newton on the 7 duals) and its negative-binomial counterpart (alpha=1.9), giving
    the full {SR,Ref}x{Pois,NB} grid on a genuine posterior.

Headline metric: worst-coordinate ESS per wall-clock second, with the reflective/
single-ray RATIO the implementation-independent quantity.  The headline cell is
replicated across seeds.  Numbers -> results/m4_numbers.txt.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.problems import A_P2, Y_P2, P2_COLS1_PLB1
from fibresampler.fibre import build_plb, ray_endpoints
from experiments.pred5_ess import run_sr_chain, run_ref_chain, iat_geyer

ALPHA = 1.9
SEED = 20260709
BMAX = 2
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# lightweight, NON-enumerating problem: PLB + feasible start only
# ---------------------------------------------------------------------------
class LiteProblem:
    """A configuration matrix and a chosen PLB, WITHOUT fibre enumeration.

    Exposes exactly the interface the SR/Ref chains and ReflectiveSampler use:
    .U, .plb_info, .n_moves, .n, .r.  No .states / .index (there is no enumeration).
    """

    def __init__(self, A, y, cols1, free_order=None, name=""):
        self.A = np.asarray(A, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)
        self.cols1 = tuple(cols1)
        self.U, self.plb_info = build_plb(self.A, cols1, free_order)
        self.name = name

    @property
    def n(self):
        return self.A.shape[0]

    @property
    def r(self):
        return self.A.shape[1]

    @property
    def n_moves(self):
        return self.U.shape[1]

    def feasible_start(self):
        """x0 = [A1^{-1} y ; 0]; requires A1 unimodular (integral PLB)."""
        A1inv = self.plb_info["A1inv_int"]
        if A1inv is None:
            raise RuntimeError("A1 not unimodular; supply an explicit feasible x0")
        x = np.zeros(self.r, dtype=np.int64)
        x[list(self.plb_info["cols1"])] = A1inv @ self.y
        if not (np.array_equal(self.A @ x, self.y) and (x >= 0).all()):
            raise RuntimeError("computed x0 is not a feasible non-negative point")
        return x


# ---------------------------------------------------------------------------
# max-entropy Poisson route-flow model  theta* : A theta* = y, theta* > 0
# ---------------------------------------------------------------------------
def maxent_theta(A, y, iters=100, tol=1e-8):
    """theta_i = exp((A^T lambda)_i) with lambda solving A exp(A^T lambda) = y.

    The maximum-entropy (equivalently Poisson-conditional-mode / gravity) route
    flow matching the observed link loads.  This is the minimiser of the smooth
    convex dual  phi(lambda) = sum_i exp((A^T lambda)_i) - y . lambda, solved by
    damped Newton with Armijo backtracking (globally convergent; the backtrack
    rejects any step that overflows exp, so lambda never diverges).  Returns a
    strictly positive theta with A theta = y to `tol`.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    n, r = A.shape

    def phi(lam):
        with np.errstate(over="ignore"):
            e = np.exp(A.T @ lam)
        s = e.sum()
        return (np.inf if not np.isfinite(s) else s - y @ lam), e

    lam = np.zeros(n)
    val, e = phi(lam)
    for _ in range(iters):
        g = A @ e - y                                  # gradient of phi
        if np.max(np.abs(g)) < tol * max(1.0, float(y.max())):
            break
        H = (A * e) @ A.T + 1e-9 * np.eye(n)           # A diag(e) A^T, SPD
        d = -np.linalg.solve(H, g)                     # Newton descent direction
        slope = float(g @ d)                           # < 0
        t = 1.0
        for _bt in range(80):
            nv, ne = phi(lam + t * d)
            if nv <= val + 1e-4 * t * slope:
                break
            t *= 0.5
        lam = lam + t * d
        val, e = phi(lam)
    return e


# ---------------------------------------------------------------------------
# vectorised log-targets on raw x rows
# ---------------------------------------------------------------------------
def logw_uniform():
    def f(X):
        X = np.atleast_2d(np.asarray(X))
        return np.zeros(X.shape[0])
    return f


def logw_poisson(theta):
    lt = np.log(theta)

    def f(X):
        X = np.atleast_2d(np.asarray(X, dtype=np.int64)).astype(float)
        return X @ lt - gammaln(X + 1).sum(1)
    return f


def logw_nb(theta, alpha):
    a = float(alpha)
    th = np.asarray(theta, float)
    const = (a * np.log(a / (a + th))).sum()
    p = np.log(th / (a + th))

    def f(X):
        X = np.atleast_2d(np.asarray(X, dtype=np.int64)).astype(float)
        out = (gammaln(X + a) - gammaln(X + 1)).sum(1) - X.shape[1] * gammaln(a)
        return out + const + X @ p
    return f


# ---------------------------------------------------------------------------
# one chain -> ESS summary
# ---------------------------------------------------------------------------
def run_cell(prob, logw_fn, x0, sampler, n_iter, burn, seed, sid, bmax=BMAX):
    rng = np.random.default_rng(np.random.SeedSequence((seed, sid)))
    rec_cols = list(range(prob.r))
    if sampler == "SR":
        trace, secs, extras = run_sr_chain(prob, logw_fn, x0, n_iter, burn, rng,
                                           rec_cols, False)
    else:
        trace, secs, extras = run_ref_chain(prob, logw_fn, x0, n_iter, burn, rng,
                                            rec_cols, False, bmax=bmax)
    # G0: invariant preserved (reconstruct full x from recorded coords == all coords)
    xlast = trace[-1].astype(np.int64)
    g0 = bool(np.array_equal(prob.A @ xlast, prob.y))
    ess = []
    for k in range(trace.shape[1]):
        col = trace[:, k].astype(np.float64)
        if col.std() < 1e-9:                 # a frozen coordinate: chain stuck
            continue
        t = iat_geyer(col)
        if np.isfinite(t):
            ess.append(n_iter / t)
    if not ess:
        return dict(sampler=sampler, bmax=(0 if sampler == "SR" else bmax),
                secs=secs, us_per_step=1e6 * secs / (n_iter + burn),
                    acc=extras["acc"], g0=g0, ess_min=float("nan"), worst_col=-1,
                    ess_min_ps=float("nan"), ess_med=float("nan"), ess_med_ps=float("nan"))
    ess = np.array(ess)
    worst = int(np.argmin(ess))
    return dict(sampler=sampler, bmax=(0 if sampler == "SR" else bmax),
                secs=secs, us_per_step=1e6 * secs / (n_iter + burn),
                acc=extras["acc"], g0=g0, ess_min=float(ess[worst]), worst_col=worst,
                ess_min_ps=float(ess[worst] / secs), ess_med=float(np.median(ess)),
                ess_med_ps=float(np.median(ess) / secs))


def anisotropy_proxy(prob, x0, n_iter, burn, seed):
    """Empirical axis-defect proxy at scale: sqrt(lambda_max/lambda_min) of the
    free-coordinate sample covariance under the UNIFORM fibre law.  ~1 means a
    near-isotropic fibre (reflection has nothing to gain, Prop effective-step);
    large means a tilted fibre (reflection's regime).  A run-time analogue of the
    exact axis defect kappa the enumerable experiments compute."""
    rng = np.random.default_rng(np.random.SeedSequence((seed, 7777)))
    cols2 = list(prob.plb_info["cols2"])
    trace, _, _ = run_sr_chain(prob, logw_uniform(), x0, n_iter, burn, rng, cols2, False)
    X = trace.astype(np.float64)
    C = np.cov(X.T)
    ev = np.linalg.eigvalsh(C); ev = ev[ev > 1e-9]
    cov_spread = float(np.sqrt(ev.max() / ev.min()))       # includes route-SCALE spread
    # correlation matrix: standardises away marginal scale, isolating OBLIQUENESS
    # (the axis-defect kappa that reflection targets; scale heterogeneity is not it)
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    er = np.linalg.eigvalsh(R); er = er[er > 1e-9]
    corr_spread = float(np.sqrt(er.max() / er.min()))
    return cov_spread, corr_spread


def main(n_iter=300_000, burn=30_000, n_seed=3, quick=False):
    if quick:
        n_iter, burn, n_seed = 40_000, 4_000, 1

    FLOOR = 25.0                               # rate floor for the well-conditioned model
    prob = LiteProblem(A_P2, Y_P2, P2_COLS1_PLB1, name="P2 Auckland SH16")
    x0 = prob.feasible_start()
    theta = maxent_theta(prob.A, prob.y)       # raw max-entropy ML fit (has near-empty routes)
    theta_wc = np.maximum(theta, FLOOR)        # floored: a well-conditioned route posterior
    n_empty = int((theta < FLOOR).sum())
    aerr = float(np.max(np.abs(prob.A @ theta - prob.y)))
    cov_spread, corr_spread = anisotropy_proxy(prob, x0, min(n_iter, 100_000), burn, SEED)

    BSWEEP = (2, 4, 8) if quick else (2, 4, 8, 16)

    out = []
    W = 86
    out.append("=" * W)
    out.append("M4 -- real-data-scale ESS/second: P2 Auckland SH16 (Hazelton 2024, Sec 6)")
    out.append(f"r={prob.r} routes, n={prob.n} links, free dim r-n={prob.n_moves}; "
               f"y={list(prob.y)}")
    out.append(f"fibre NOT enumerated; {n_iter:,} iters + {burn:,} burn, {n_seed} seed(s), "
               f"seed base {SEED}")
    out.append(f"max-entropy theta* fit: max|A theta*-y| = {aerr:.2e} "
               f"(theta* range {theta.min():.1f}..{theta.max():.1f}, {n_empty} route(s) below "
               f"floor {FLOOR:.0f} => loading routes); alpha_NB={ALPHA}")
    out.append(f"well-conditioned model = theta floored at {FLOOR:.0f} "
               f"(range {theta_wc.min():.0f}..{theta_wc.max():.0f})")
    out.append(f"anisotropy (uniform fibre): covariance sqrt(lam_max/lam_min)={cov_spread:.1f} "
               f"(route-SCALE spread); correlation sqrt(lam_max/lam_min)={corr_spread:.2f} "
               f"(OBLIQUENESS = the axis defect reflection targets)")
    out.append("Ref = reflective uniform pi_B on {0..B_max}; SR = single-ray Gibbs. "
               "Kernels = pred5_ess.py")
    out.append("=" * W)
    out.append("")
    hdr = (f"  {'target':>8} {'smp':>7} {'us/step':>8} {'acc':>6} "
           f"{'ESSmin':>9} {'ESSmin/s':>9} {'ESSmed':>9} {'ESSmed/s':>9} {'G0':>4}")

    grid = {}
    sid = 0

    def emit(r):
        lab = "SR" if r["sampler"] == "SR" else f"Ref/B{r['bmax']}"
        out.append(f"  {r['_t']:>8} {lab:>7} {r['us_per_step']:>8.1f} {r['acc']:>6.3f} "
                   f"{r['ess_min']:>9.0f} {r['ess_min_ps']:>9.1f} "
                   f"{r['ess_med']:>9.0f} {r['ess_med_ps']:>9.1f} "
                   f"{'ok' if r['g0'] else 'BAD':>4}")

    # ---- uniform target: SR baseline + reflective B_max sweep (geometry at scale)
    out.append("target = uniform  (geometry only; the beta=0 certified reference rung)")
    out.append(hdr)
    r_sr = run_cell(prob, logw_uniform(), x0, "SR", n_iter, burn, SEED, sid); sid += 1
    r_sr["_t"] = "uniform"; grid[("uniform", "SR", 0)] = r_sr; emit(r_sr)
    for B in BSWEEP:
        r = run_cell(prob, logw_uniform(), x0, "Ref", n_iter, burn, SEED, sid, bmax=B); sid += 1
        r["_t"] = "uniform"; grid[("uniform", "Ref", B)] = r; emit(r)
    out.append("")

    # best reflective B_max on the uniform target, by ESS/sec
    ref_cells = [grid[("uniform", "Ref", B)] for B in BSWEEP]
    Bbest = max(BSWEEP, key=lambda B: grid[("uniform", "Ref", B)]["ess_min_ps"])
    out.append(f"  best reflective B_max on uniform (by ESS/sec): B={Bbest}")
    out.append("")

    # ---- well-conditioned model posterior (the "is it practical" benchmark)
    for tname, fn in (("pois-wc", logw_poisson(theta_wc)),
                      ("nb-wc", logw_nb(theta_wc, ALPHA))):
        out.append(f"target = {tname}  (well-conditioned floored route model; "
                   f"NB alpha={ALPHA})")
        out.append(hdr)
        r = run_cell(prob, fn, x0, "SR", n_iter, burn, SEED, sid); sid += 1
        r["_t"] = tname; grid[(tname, "SR", 0)] = r; emit(r)
        r = run_cell(prob, fn, x0, "Ref", n_iter, burn, SEED, sid, bmax=Bbest); sid += 1
        r["_t"] = tname; grid[(tname, "Ref", Bbest)] = r; emit(r)
        out.append("")

    # ---- raw max-entropy posterior: the loading obstruction ON REAL DATA
    out.append(f"target = pois-maxent  (raw ML fit; {n_empty} near-empty route(s) => "
               f"a real-data loading bottleneck)")
    out.append(hdr)
    r = run_cell(prob, logw_poisson(theta), x0, "SR", n_iter, burn, SEED, sid); sid += 1
    r["_t"] = "pois-me"; grid[("pois-me", "SR", 0)] = r; emit(r)
    r = run_cell(prob, logw_poisson(theta), x0, "Ref", n_iter, burn, SEED, sid, bmax=Bbest); sid += 1
    r["_t"] = "pois-me"; grid[("pois-me", "Ref", Bbest)] = r; emit(r)
    out.append("")

    # ---- ratios
    out.append("REFLECTIVE(best) / SINGLE-RAY ratios")
    for tname, B in (("uniform", Bbest), ("pois-wc", Bbest), ("nb-wc", Bbest),
                     ("pois-me", Bbest)):
        rc = grid[(tname, "Ref", B)]; sc = grid[(tname, "SR", 0)]
        out.append(f"  {tname:>9} (B={B}): worst-coord ESS/iter Ref/SR = "
                   f"{rc['ess_min']/sc['ess_min']:.2f}   "
                   f"ESS/sec Ref/SR = {rc['ess_min_ps']/sc['ess_min_ps']:.2f}")
    out.append("")

    # ---- headline seed replication (uniform, SR and best-B reflective)
    if n_seed > 1:
        out.append(f"SEED REPLICATION (uniform, {n_seed} seeds): worst-coord ESS/sec")
        for smp, B in (("SR", 0), ("Ref", Bbest)):
            vals = []
            for s in range(n_seed):
                r = run_cell(prob, logw_uniform(), x0, smp, n_iter, burn,
                             SEED + 1 + s, 900 + s, bmax=(B or BMAX))
                vals.append(r["ess_min_ps"])
            vals = np.array(vals)
            lab = "SR" if smp == "SR" else f"Ref/B{B}"
            out.append(f"  {lab:>7}: {vals.mean():.1f} +/- {vals.std(ddof=1):.1f}  "
                       f"(seeds: {', '.join(f'{v:.1f}' for v in vals)})")
        out.append("")

    txt = "\n".join(out)
    print(txt)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "m4_numbers.txt"), "w") as f:
        f.write(txt + "\n")
    print("\nnumbers written: results/m4_numbers.txt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=300_000)
    ap.add_argument("--burn", type=int, default=30_000)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(a.iters, a.burn, a.seeds, quick=a.quick)
