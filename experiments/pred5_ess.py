"""Prediction 5 -- the ESS/time ablation grid (the paper's headline metric).

Registered claim (Sec. 9.4, Prediction 5): on a problem with both a tilted fibre and
a low-mean coordinate, Ref-NB attains the best ESS/time among the four ablations
{SR,Ref} x {Pois,NB} in the geometry-limited regime; by the loading-obstruction
theorem its gap still collapses in the deep-bottleneck limit (no escape), and past
the crossover blind reflection costs a measured factor (~0.82 in gap).  The
ass:lobes sub-task of Prediction 5 was verified separately
(experiments/verify_assumption_lobes.py: three lobes on P1, two on the table).

Design.  Problem with both mechanisms = the Prediction-4 decoupled product
band(w=2, C=24, kappa=4.26) x 2x3 table, bottleneck mean mu* in {1.0, 0.1, 0.01}
(geometry-limited -> crossover -> loading-limited); control = P1 (kappa ~ 1.2,
nothing for reflection to gain) at theta5 in {1.0, 0.01}.  Samplers are honest
iterative chains that redo their geometric work every step (no fibre enumeration or
precomputed rays at run time):
  SR  = single-ray Gibbs hit-and-run (Hazelton; heat-bath on the ray),
  Ref = reflective proposal, uniform pi_B on {0,1,2} (B=0 component is MH-on-ray),
        per-trajectory reverse-path Metropolis rule identical to
        fibresampler.reflective.reflective_transition_matrix.
Targets: Pois = product Poisson; NB = negative binomial, alpha = 1.9 (the correctly
specified model).  On the product, the band block keeps its Poisson ridge target
(theta = C/4 on s,t; slacks flat) in both arms; the table block switches.

Metrics per (problem, mu, target, sampler): integrated autocorrelation time (Geyer
initial positive monotone sequence) and ESS per coordinate functional (+ the band
long axis s+t), wall-clock seconds and microseconds/step, ESS/sec (headline), and
the exact SLEM of the identical operator.

Validation gates, all exact where the fibre is enumerable:
  G1 target consistency: the chain's closed-form log-target matches the enumerated
     log-weight vector (centred) to 1e-9 on every setting.
  G2 marginal correctness: TV(empirical state distribution, exact pi) small.
  G3 estimator correctness: Geyer IAT matches the EXACT operator IAT
     tau_f = sum_i c_i (1+lam_i)/(1-lam_i) for every functional (MC error only).
  G4 the generic SR operator builder reproduces fibresampler.spectral's on P1.

Honest-reporting notes: ESS/iter is implementation-independent; ESS/sec depends on
this (pure numpy) implementation's per-step constants, but all four samplers share
the same infrastructure, so the RATIOS are the meaningful quantity.  Both are
reported; the verdict is stated under both normalisations.

Outputs: results/pred5_numbers.txt, results/fig_pred5_ess.{pdf,png}.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.special import gammaln, logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.problems import P1, p1_poisson_theta, table_2x3, TABLE23_BOTTLENECK
from fibresampler.spectral import sr_transition_matrix, slem, normalised_pi, log_weights
from fibresampler.fibre import ray_endpoints
from fibresampler.reflective import ReflectiveSampler, reflective_transition_matrix
from fibresampler.augment import nb_log_pmf
from fibresampler.synthetic import BandProblem, poisson_free_logw, product_problem

ALPHA = 1.9
SEED = 20260701
N_ITER = 1_000_000
BURN = 50_000
BAND_W, BAND_C = 2, 24
RIDGE = BAND_C / 4.0
BMAX = 2                                   # uniform pi_B on {0,1,2}, as in Pred 3/4
MUS_PROD = (1.0, 0.1, 0.01)
THETAS_P1 = (1.0, 0.01)
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# closed-form log-targets evaluated on raw x rows (the chains' honest view)
# ---------------------------------------------------------------------------
def logw_fn_product(mu_star, target):
    """Vectorised log-target on product states X (k, 11): band Poisson ridge on
    (s,t) = cols 0,1 (slacks 2-4 flat) + table target on cols 5..10."""
    th_tab = np.array([mu_star, 1, 1, 1, 1, 1], dtype=float)

    def f(X):
        X = np.atleast_2d(np.asarray(X, dtype=np.int64))
        st = X[:, :2].astype(float)
        out = (st * np.log(RIDGE)).sum(1) - gammaln(st + 1).sum(1)
        T = X[:, 5:].astype(float)
        if target == "pois":
            out += (T * np.log(th_tab)).sum(1) - gammaln(T + 1).sum(1)
        else:
            a = ALPHA
            out += (gammaln(T + a) - gammaln(T + 1)).sum(1) - T.shape[1] * gammaln(a)
            out += (a * np.log(a / (a + th_tab))).sum()
            out += (T * np.log(th_tab / (a + th_tab))).sum(1)
        return out

    return f


def logw_fn_p1(theta5, target):
    th = p1_poisson_theta(theta5)

    def f(X):
        X = np.atleast_2d(np.asarray(X, dtype=np.int64)).astype(float)
        if target == "pois":
            return (X * np.log(th)).sum(1) - gammaln(X + 1).sum(1)
        a = ALPHA
        out = (gammaln(X + a) - gammaln(X + 1)).sum(1) - X.shape[1] * gammaln(a)
        out += (a * np.log(a / (a + th))).sum()
        out += (X * np.log(th / (a + th))).sum(1)
        return out

    return f


# ---------------------------------------------------------------------------
# chains (no enumeration at run time)
# ---------------------------------------------------------------------------
def run_sr_chain(prob, logw_fn, x0, n_iter, burn, rng, rec_cols, extra_longaxis):
    """Single-ray Gibbs hit-and-run; returns (trace int16, seconds, extras)."""
    U = prob.U
    m = U.shape[1]
    x = x0.copy()
    nrec = len(rec_cols) + (1 if extra_longaxis else 0)
    trace = np.empty((n_iter, nrec), dtype=np.int16)
    t0 = time.perf_counter()
    for it in range(-burn, n_iter):
        j = int(rng.integers(m))
        u = U[:, j]
        b_min, b_max = ray_endpoints(x, u)
        bs = np.arange(b_min, b_max + 1, dtype=np.int64)
        Xray = x[None, :] + bs[:, None] * u[None, :]
        lw = logw_fn(Xray)
        w = np.exp(lw - lw.max())
        c = np.cumsum(w)
        k = int(np.searchsorted(c, rng.random() * c[-1], side="right"))
        x = Xray[min(k, len(bs) - 1)]
        if it >= 0:
            trace[it, : len(rec_cols)] = x[rec_cols]
            if extra_longaxis:
                trace[it, -1] = x[0] + x[1]
    secs = time.perf_counter() - t0
    return trace, secs, {"acc": 1.0}


def run_ref_chain(prob, logw_fn, x0, n_iter, burn, rng, rec_cols, extra_longaxis,
                  bmax=BMAX):
    """Reflective chain, uniform pi_B on {0..bmax}; per-trajectory reverse-path MH
    identical to reflective_transition_matrix."""
    S = ReflectiveSampler(prob, bmax=bmax)
    twom = 2 * S.m
    x = x0.copy()
    lw_cur = float(logw_fn(x[None, :])[0])
    nrec = len(rec_cols) + (1 if extra_longaxis else 0)
    trace = np.empty((n_iter, nrec), dtype=np.int16)
    n_acc = 0
    t0 = time.perf_counter()
    for it in range(-burn, n_iter):
        B = int(rng.integers(bmax + 1))
        d1 = S.dirs[int(rng.integers(twom))]
        ok = False
        if B == 0:
            vB, d_term, walls = x, d1, []
            ok = True
        else:
            res = S.bounce_to_wall(x, d1, B)
            if res is not None:
                vB, d_term, walls = res
                ok = True
        if ok:
            L, _ = S.cast(vB, d_term)
            if L is not None and L > 0:
                b = int(rng.integers(1, L + 1))
                xd = vB + b * S.full_move(d_term)
                if not np.array_equal(xd, x):
                    qrev = S.reverse_prob_factor(xd, d_term, B, x, walls)
                    if qrev > 0.0:
                        lw_new = float(logw_fn(xd[None, :])[0])
                        alpha = min(1.0, np.exp(lw_new - lw_cur) * qrev * L)
                        if rng.random() < alpha:
                            x = xd
                            lw_cur = lw_new
                            if it >= 0:
                                n_acc += 1
        if it >= 0:
            trace[it, : len(rec_cols)] = x[rec_cols]
            if extra_longaxis:
                trace[it, -1] = x[0] + x[1]
    secs = time.perf_counter() - t0
    return trace, secs, {"acc": n_acc / n_iter}


# ---------------------------------------------------------------------------
# IAT: Geyer initial positive monotone sequence, and the exact operator value
# ---------------------------------------------------------------------------
def iat_geyer(z):
    z = np.asarray(z, float)
    z = z - z.mean()
    n = len(z)
    v0 = float(z @ z) / n
    if v0 <= 1e-12:
        return np.nan
    nf = 1 << int(2 * n - 1).bit_length()
    F = np.fft.rfft(z, nf)
    ac = np.fft.irfft(F * np.conj(F), nf)[:n].real / (v0 * n)
    K = (n - 1) // 2
    G = ac[0 : 2 * K : 2] + ac[1 : 2 * K + 1 : 2]
    nonpos = np.nonzero(G <= 0)[0]
    if len(nonpos):
        G = G[: nonpos[0]]
    if len(G) == 0:
        return 1.0
    G = np.minimum.accumulate(G)
    return max(1.0, 2.0 * float(G.sum()) - 1.0)


def exact_iat(Q, pi, f):
    """tau_f of a reversible kernel, exactly, via symmetrised diagonalisation."""
    d = np.sqrt(pi)
    Ssym = (d[:, None] * Q) / d[None, :]
    Ssym = 0.5 * (Ssym + Ssym.T)
    lam, V = np.linalg.eigh(Ssym)
    fb = f - float(pi @ f)
    varf = float(pi @ fb ** 2)
    if varf <= 1e-14:
        return np.nan
    g = V.T @ (d * fb)
    keep = lam < 1.0 - 1e-12
    return float(((1.0 + lam[keep]) / (1.0 - lam[keep])) @ (g[keep] ** 2) / varf)


def sr_matrix_generic(prob, lw):
    """SR Gibbs operator for duck-typed problems (no .ray method needed)."""
    M = len(prob.states)
    m = prob.U.shape[1]
    Q = np.zeros((M, M))
    for i in range(M):
        x = prob.states[i]
        for j in range(m):
            u = prob.U[:, j]
            b_min, b_max = ray_endpoints(x, u)
            idxs = []
            for b in range(b_min, b_max + 1):
                k = prob.index.get(tuple(x + b * u))
                if k is not None:
                    idxs.append(k)
            w = np.exp(lw[idxs] - np.max(lw[idxs]))
            w /= w.sum()
            for k, wk in zip(idxs, w):
                Q[i, k] += wk / m
    return Q


# ---------------------------------------------------------------------------
# one grid cell
# ---------------------------------------------------------------------------
def build_problem(spec):
    if spec["prob"] == "product":
        band = BandProblem(BAND_W, BAND_C)
        tab = table_2x3()
        prod = product_problem(band, tab)
        lw_band = poisson_free_logw(band, [RIDGE, RIDGE])
        if spec["target"] == "pois":
            th = np.full(6, 1.0)
            th[TABLE23_BOTTLENECK] = spec["mu"]
            lw_tab = log_weights(tab, "poisson", th)
        else:
            th = np.full(6, 1.0)
            th[TABLE23_BOTTLENECK] = spec["mu"]
            lw_tab = nb_log_pmf(tab, th, ALPHA)
        lw_vec = prod.logw_from_blocks(lw_band, lw_tab)
        fn = logw_fn_product(spec["mu"], spec["target"])
        rec_cols = list(range(11))
        names = [f"x{c}" for c in rec_cols] + ["s+t"]
        return prod, lw_vec, fn, rec_cols, True, names, 5 + TABLE23_BOTTLENECK
    p1 = P1()
    th = p1_poisson_theta(spec["mu"])
    lw_vec = (log_weights(p1, "poisson", th) if spec["target"] == "pois"
              else nb_log_pmf(p1, th, ALPHA))
    fn = logw_fn_p1(spec["mu"], spec["target"])
    rec_cols = list(range(8))
    names = [f"x{c}" for c in rec_cols]
    return p1, lw_vec, fn, rec_cols, False, names, 4


def run_setting(spec):
    prob, lw_vec, fn, rec_cols, longax, names, bcol = build_problem(spec)
    # G1: chain target == enumerated target (centred)
    lw_chain = fn(prob.states)
    dev = lw_chain - lw_vec
    g1 = float(np.abs(dev - dev.mean()).max())
    pi = normalised_pi(lw_vec)
    # exact operator for THIS sampler
    if spec["sampler"] == "SR":
        Q = sr_matrix_generic(prob, lw_vec)
    else:
        Q, _ = reflective_transition_matrix(prob, bmax=BMAX, log_w=lw_vec)
    gap = float(slem(Q, pi)[1])
    F = pi[:, None] * Q
    dbe = float(np.abs(F - F.T).max())
    # functionals evaluated on states, for exact IAT
    feats = [prob.states[:, c].astype(float) for c in rec_cols]
    if longax:
        feats.append(prob.states[:, 0].astype(float) + prob.states[:, 1].astype(float))
    tau_exact = [exact_iat(Q, pi, f) for f in feats]
    # the chain
    rng = np.random.default_rng(np.random.SeedSequence((SEED, spec["sid"])))
    x0 = prob.states[int(np.argmax(lw_vec))].astype(np.int64)
    runner = run_sr_chain if spec["sampler"] == "SR" else run_ref_chain
    trace, secs, extras = runner(prob, fn, x0, N_ITER, BURN, rng, rec_cols, longax)
    # G2: marginal TV (state identity recovered from recorded coords is overkill;
    # use the bottleneck-column marginal instead, exact vs empirical)
    bidx = rec_cols.index(bcol)
    vals, cnt = np.unique(trace[:, bidx], return_counts=True)
    emp = np.zeros(int(prob.states[:, bcol].max()) + 1)
    for v, c in zip(vals, cnt):
        emp[int(v)] = c / len(trace)
    exa = np.zeros_like(emp)
    for s, p in zip(prob.states[:, bcol], pi):
        exa[int(s)] += p
    tv = 0.5 * float(np.abs(emp - exa).sum())
    # IATs
    tau_hat, ess, keepnames = [], [], []
    for k in range(trace.shape[1]):
        t = iat_geyer(trace[:, k].astype(np.float64))
        if np.isnan(t):
            continue
        tau_hat.append((k, t))
        ess.append(N_ITER / t)
        keepnames.append(names[k])
    ess = np.array(ess)
    worst = int(np.argmin(ess))
    # G3: estimator vs exact
    g3 = 0.0
    for k, t in tau_hat:
        te = tau_exact[k]
        if te and np.isfinite(te) and te > 1.5:
            g3 = max(g3, abs(t - te) / te)
    return dict(spec=spec, g1=g1, tv=tv, g3=g3, gap=gap, dbe=dbe, secs=secs,
                us_per_step=1e6 * secs / (N_ITER + BURN), acc=extras["acc"],
                ess_min=float(ess[worst]), worst_fn=keepnames[worst],
                ess_min_ps=float(ess[worst] / secs),
                ess_b=float(ess[keepnames.index(f"x{bcol}")]),
                tau_b_hat=float(N_ITER / ess[keepnames.index(f"x{bcol}")]),
                tau_b_exact=float(tau_exact[rec_cols.index(bcol)]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    # G4 first (cheap, in-process): generic SR builder == canonical on P1
    p1 = P1()
    lwp = nb_log_pmf(p1, p1_poisson_theta(0.1), ALPHA)
    dQ = float(np.abs(sr_matrix_generic(p1, lwp) - sr_transition_matrix(p1, lwp)).max())

    specs = []
    sid = 0
    for mu in MUS_PROD:
        for target in ("pois", "nb"):
            for sampler in ("SR", "Ref"):
                specs.append(dict(prob="product", mu=mu, target=target,
                                  sampler=sampler, sid=sid)); sid += 1
    for th in THETAS_P1:
        for target in ("pois", "nb"):
            for sampler in ("SR", "Ref"):
                specs.append(dict(prob="P1", mu=th, target=target,
                                  sampler=sampler, sid=sid)); sid += 1

    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        results = list(ex.map(run_setting, specs))

    out = []
    out.append("=" * 78)
    out.append("PREDICTION 5 -- ESS/time ablation grid ({SR,Ref} x {Pois,NB})")
    out.append(f"N = {N_ITER:,} iterations + {BURN:,} burn-in per chain, seed {SEED};")
    out.append(f"product = band(w={BAND_W},C={BAND_C}, kappa=4.26, Poisson ridge "
               f"theta={RIDGE:.0f}) x 2x3 table (bottleneck mean mu*); alpha_NB = {ALPHA}")
    out.append(f"Ref = reflective, uniform pi_B on {{0..{BMAX}}}; SR = single-ray Gibbs")
    out.append("ass:lobes sub-task: verified separately (three lobes on P1, two on the")
    out.append("table; experiments/verify_assumption_lobes.py, ledgered).")
    out.append("=" * 78)
    out.append("")
    out.append("GATES")
    out.append(f"  G4 generic SR operator == fibresampler.spectral on P1: max dev {dQ:.2e}  "
               f"{'PASS' if dQ < 1e-12 else 'FAIL'}")
    g1w = max(r["g1"] for r in results)
    tvw = max(r["tv"] for r in results)
    g3w = max(r["g3"] for r in results)
    dbw = max(r["dbe"] for r in results)
    out.append(f"  G1 chain target == enumerated target (centred), worst: {g1w:.2e}  "
               f"{'PASS' if g1w < 1e-9 else 'FAIL'}")
    out.append(f"  G2 bottleneck-marginal TV(chain, exact pi), worst: {tvw:.4f}  "
               f"{'PASS' if tvw < 0.02 else 'CHECK'}")
    out.append(f"  G3 Geyer IAT vs exact operator IAT, worst rel dev: {g3w:.3f}  "
               f"{'PASS' if g3w < 0.25 else 'CHECK'}")
    out.append(f"  (exact operator DB violation, worst: {dbw:.1e})")
    out.append("")

    def block(title, rows):
        out.append(title)
        out.append(f"  {'mu':>6} {'target':>6} {'smp':>4} {'gap(exact)':>11} "
                   f"{'tau_b(hat/ex)':>14} {'ESSmin':>9} {'worst':>6} {'us/step':>8} "
                   f"{'ESSmin/s':>9} {'acc':>6}")
        for r in rows:
            s = r["spec"]
            out.append(f"  {s['mu']:>6g} {s['target']:>6} {s['sampler']:>4} "
                       f"{r['gap']:>11.5f} "
                       f"{r['tau_b_hat']:>7.1f}/{r['tau_b_exact']:<6.1f} "
                       f"{r['ess_min']:>9.0f} {r['worst_fn']:>6} "
                       f"{r['us_per_step']:>8.1f} {r['ess_min_ps']:>9.1f} "
                       f"{r['acc']:>6.3f}")
        out.append("")

    prod_rows = [r for r in results if r["spec"]["prob"] == "product"]
    p1_rows = [r for r in results if r["spec"]["prob"] == "P1"]
    block("PRODUCT (both mechanisms): band(geometry) x table(loading)", prod_rows)
    block("P1 CONTROL (kappa ~ 1.2: reflection has nothing to gain)", p1_rows)

    # verdicts
    out.append("=" * 78)
    out.append("VERDICTS (Prediction 5)")
    out.append("=" * 78)

    def cell(mu, target, sampler):
        return next(r for r in prod_rows if r["spec"]["mu"] == mu
                    and r["spec"]["target"] == target and r["spec"]["sampler"] == sampler)

    for mu, tag in ((1.0, "geometry-limited"), (0.1, "crossover"), (0.01, "loading-limited")):
        rank_ps = sorted(((cell(mu, t, s)["ess_min_ps"], f"{s}-{t}")
                          for t in ("pois", "nb") for s in ("SR", "Ref")), reverse=True)
        rank_pi = sorted(((cell(mu, t, s)["ess_min"], f"{s}-{t}")
                          for t in ("pois", "nb") for s in ("SR", "Ref")), reverse=True)
        out.append(f"  mu*={mu:g} ({tag}):")
        out.append("    ESS/sec ranking : " + "  >  ".join(
            f"{n} ({v:.1f})" for v, n in rank_ps))
        out.append("    ESS/iter ranking: " + "  >  ".join(
            f"{n} ({v:.0f})" for v, n in rank_pi))
    g_best = cell(1.0, "nb", "Ref")
    top_ps = max(prod_rows, key=lambda r: r["ess_min_ps"] if r["spec"]["mu"] == 1.0 else -1)
    v1_ps = (g_best["ess_min_ps"] >= max(cell(1.0, t, s)["ess_min_ps"]
             for t in ("pois", "nb") for s in ("SR", "Ref")) - 1e-9)
    v1_pi = (g_best["ess_min"] >= max(cell(1.0, t, s)["ess_min"]
             for t in ("pois", "nb") for s in ("SR", "Ref")) - 1e-9)
    out.append(f"  V1 geometry regime: Ref-NB best?  ESS/sec: {v1_ps}   ESS/iter: {v1_pi}")
    coll = [cell(0.01, t, s)["ess_min"] / cell(1.0, t, s)["ess_min"]
            for t in ("pois", "nb") for s in ("SR", "Ref")]
    out.append(f"  V2 loading regime: ESS/iter collapse factors mu*=1 -> 0.01 "
               f"(SR-p, Ref-p, SR-nb, Ref-nb): "
               + ", ".join(f"{c:.3g}" for c in [coll[0], coll[2], coll[1], coll[3]]))
    rr = cell(0.01, "nb", "Ref")["gap"] / cell(0.01, "nb", "SR")["gap"]
    out.append(f"  V2b past crossover, Ref/SR exact-gap ratio (NB): {rr:.3f} "
               f"(pred4 measured ~0.82 vs its MH-on-ray baseline)")
    def p1cell(th, target, sampler):
        return next(r for r in p1_rows if r["spec"]["mu"] == th
                    and r["spec"]["target"] == target and r["spec"]["sampler"] == sampler)
    for th in THETAS_P1:
        rp = p1cell(th, "pois", "Ref")["ess_min"] / p1cell(th, "pois", "SR")["ess_min"]
        rn = p1cell(th, "nb", "Ref")["ess_min"] / p1cell(th, "nb", "SR")["ess_min"]
        out.append(f"  V3 P1 control, theta5={th:g}: Ref/SR ESS/iter ratio "
                   f"Pois {rp:.2f}, NB {rn:.2f} (expected ~<=1: nothing to gain)")

    txt = "\n".join(out)
    print(txt)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "pred5_numbers.txt"), "w") as f:
        f.write(txt + "\n")
    print("numbers written: results/pred5_numbers.txt")

    try:
        make_figure(prod_rows)
        print("figure written: results/fig_pred5_ess.pdf/.png")
    except Exception as e:                                        # noqa: BLE001
        print(f"figure skipped: {e}")


def make_figure(prod_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combos = [("SR", "pois"), ("Ref", "pois"), ("SR", "nb"), ("Ref", "nb")]
    labels = ["SR-Pois", "Ref-Pois", "SR-NB", "Ref-NB"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    width = 0.2
    xs = np.arange(len(MUS_PROD))
    for ax, key, ttl in ((axes[0], "ess_min_ps", "ESS/second (this implementation)"),
                         (axes[1], "ess_min", "ESS per 1e6 iterations")):
        for k, ((s, t), lab) in enumerate(zip(combos, labels)):
            vals = []
            for mu in MUS_PROD:
                r = next(r for r in prod_rows if r["spec"]["mu"] == mu
                         and r["spec"]["target"] == t and r["spec"]["sampler"] == s)
                vals.append(r[key])
            ax.bar(xs + (k - 1.5) * width, vals, width, label=lab)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"$\\mu_*={m:g}$" for m in MUS_PROD])
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("min-coordinate ESS")
    axes[0].legend(fontsize=7)
    fig.suptitle("Prediction 5: band(geometry) x table(loading), worst-functional ESS",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_pred5_ess.{ext}"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
