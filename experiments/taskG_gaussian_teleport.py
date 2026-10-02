"""Task G -- Gaussian-relaxation independence sampler as a candidate teleporter.

Construction (per problem, P1 and the 2x3 table, alpha = 1.9):
  1. Moment-match a Gaussian to the NB target: m_j = mu_j, V = diag(mu_j + mu_j^2/alpha).
  2. Condition onto the affine ridge {x : Ax = y} by kriging:
         m_c = m - V A'(A V A')^{-1}(A m - y),   V_c = V - V A'(A V A')^{-1} A V,
     restrict to the PLB free coordinates -> nondegenerate (r-n)-dim normal (m2, V2).
  3. Proposal: draw x2 ~ N(m2, V2), round componentwise, lift x1 = A1^{-1}(y - A2 x2),
     reject-and-redraw if infeasible.  The proposal mass of a fibre state is the exact
     Gaussian measure of the unit cell centred at its x2, renormalised over the fibre
     (the redraw).  Metropolised independence sampler (IMH); exact transition matrix;
     exact SLEM.

Predictions under test:
  P-G1  gap-vs-mu slope ~ 0 (|slope| <= 0.1) at small mu on both problems.
  P-G2  Liu (1996) identity: gap = w* = min_x q(x)/pi(x), to numerical precision.
        Primary oracle for the operator assembly.
  P-G3  mismatch degradation: alpha decreasing -> smaller w*; locate the argmin state.
  P-G4  floor comparison: Gaussian-IMH vs 3-rung tempering vs DA(R->inf), with
        work-per-sweep notes.

Validation gates (all must pass before any sweep is interpreted):
  G0  log-space Genz SOV cell masses agree with scipy's mvn CDF inclusion-exclusion
      in the benign regime (relative ~1e-3, QMC-limited).
  G1  exact detailed balance of the assembled IMH kernel.
  G2  q sums to 1 over the fibre after renormalisation (pre-renorm feasible mass is
      the reported feasibility-rejection rate).
  G3  Liu identity wherever the eigensolver can resolve the gap (w* >= 1e-9).
  G4  consistency limit: alpha -> 1e6 and all means = 5 -> no collapse (gap = O(1)).

Numerics: cell masses are computed by a separation-of-variables (Genz 1992) QMC
integrator carried entirely in log space (log_ndtr / ndtri_exp), so masses far below
the float underflow threshold have exact log values.  The conditioned covariance
develops a thin direction of variance ~mu5 as the bottleneck mean shrinks, which is
real physics (the ridge pins the bottleneck coordinate), not a numerical artefact.
Below mu ~ 1e-14 the thin eigenvalue reaches the float roundoff of the conditioning
algebra; such rows are flagged and their log10 w* is an upper bound.

Outputs: results/taskG_numbers.txt, results/fig_taskG_teleport.{pdf,png}.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp
from scipy.stats import multivariate_normal, qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.problems import (P1, p1_poisson_theta, table_2x3,
                                   TABLE23_BOTTLENECK)
from fibresampler.spectral import log_weights, sr_transition_matrix, slem, normalised_pi
from fibresampler.augment import nb_log_pmf
from experiments.taskV_hardening import tempered_gap
from experiments.taskS_slice import slice_transition_matrix

ALPHA = 1.9
SEED = 20260701
MUS = [1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]     # Task V grid (overlay is apples-to-apples)
MUS_DEEP = [1e-12, 1e-16]                          # floor probe, log-space only
BETAS3 = np.array([1.0, 0.5, 0.0])                 # the 3-rung ladder of Task V
# DA(R->inf) floors measured in Prediction 2 (results/pred2_numbers.txt line 31,
# results/pred2_generality.txt line 16, alpha=1.9, deepest mu):
DA_FLOOR = {"P1": 0.6116, "table": 0.7016}
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# log-space Gaussian cell masses (Genz SOV with scrambled-Sobol QMC)
# ---------------------------------------------------------------------------
def _log1mexp(d):
    """log(1 - e^d) elementwise for d <= 0, stable near both ends."""
    d = np.asarray(d, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        near = np.log(-np.expm1(np.minimum(d, 0.0)))
        far = np.log1p(-np.exp(np.minimum(d, -0.693)))
    return np.where(d > -0.693, near, far)


def log_phi_diff(a, b):
    """log(Phi(b) - Phi(a)) elementwise, a <= b, stable in far tails."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    flip = (a + b) > 0.0                    # reflect so the mass sits in the lower tail
    aa = np.where(flip, -b, a)
    bb = np.where(flip, -a, b)
    u = log_ndtr(bb)
    v = log_ndtr(aa)
    return u + _log1mexp(v - u)


def trunc_std_normal(a, b, w):
    """Quantile w of the standard normal truncated to [a, b], in log space."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.asarray(w, float)
    flip = (a + b) > 0.0
    aa = np.where(flip, -b, a)
    bb = np.where(flip, -a, b)
    ww = np.where(flip, 1.0 - w, w)
    la = log_ndtr(aa)
    lab = log_phi_diff(aa, bb)
    lw = np.log(np.clip(ww, 1e-300, 1.0))
    z = ndtri_exp(np.logaddexp(la, lw + lab))
    z = np.clip(z, aa, bb)
    return np.where(flip, -z, z)


def log_box_mass(mean, cov, lo, hi, rng, n_rand=8, m_pow=11):
    """log P(lo <= Z <= hi), Z ~ N(mean, cov), via log-space Genz SOV.

    Returns (log_mass, spread) where spread is the max-min range of the n_rand
    independent scrambled-Sobol replicate log-estimates (relative-log error proxy).
    """
    mean = np.asarray(mean, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    d = len(mean)
    sd = np.sqrt(np.diag(cov))
    # static Genz ordering: most restrictive marginal first
    marg = log_phi_diff((lo - mean) / sd, (hi - mean) / sd)
    order = np.argsort(marg)
    m, l_, h_ = mean[order], lo[order], hi[order]
    C = cov[np.ix_(order, order)]
    C = 0.5 * (C + C.T)
    jitter = 0.0
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        jitter = 1e-13 * float(np.max(np.diag(C)))
        L = np.linalg.cholesky(C + jitter * np.eye(d))
    reps = []
    npts = 2 ** m_pow
    for _ in range(n_rand):
        sob = qmc.Sobol(max(d - 1, 1), scramble=True, seed=int(rng.integers(2 ** 31)))
        w = sob.random(npts)
        acc = np.zeros(npts)
        zs = np.zeros((npts, d))
        for i in range(d):
            cond = zs[:, :i] @ L[i, :i] if i else 0.0
            a = (l_[i] - m[i] - cond) / L[i, i]
            b = (h_[i] - m[i] - cond) / L[i, i]
            acc += log_phi_diff(a, b)
            if i < d - 1:
                zs[:, i] = trunc_std_normal(a, b, w[:, i])
        reps.append(logsumexp(acc) - np.log(npts))
    reps = np.array(reps)
    return float(logsumexp(reps) - np.log(n_rand)), float(reps.max() - reps.min()), jitter


# ---------------------------------------------------------------------------
# proposal construction (moment match -> kriging -> free coords -> cell masses)
# ---------------------------------------------------------------------------
def gauss_condition(prob, mu, alpha):
    """(m2, V2): the ridge-conditioned Gaussian restricted to the PLB free coords."""
    mu = np.broadcast_to(np.asarray(mu, float), (prob.r,)).copy()
    alpha = np.broadcast_to(np.asarray(alpha, float), (prob.r,))
    v = mu + mu ** 2 / alpha
    A = prob.A.astype(float)
    S = A @ (v[:, None] * A.T)                        # A V A'
    K = (v[:, None] * A.T) @ np.linalg.inv(S)         # V A' S^{-1}
    m_c = mu - K @ (A @ mu - prob.y)
    V_c = np.diag(v) - (v[:, None] * A.T) @ np.linalg.solve(S, A * v[None, :])
    V_c = 0.5 * (V_c + V_c.T)
    cols2 = list(prob.plb_info["cols2"])
    return m_c[cols2], V_c[np.ix_(cols2, cols2)]


def proposal_logq(prob, mu, alpha, rng, m_pow=11):
    """Renormalised log proposal masses over the fibre.

    Returns (logq, feas_mass, max_spread, thin_flag): logq sums to 1 in exp;
    feas_mass = pre-renormalisation Gaussian mass of the feasible cells (the
    complement is the feasibility-rejection rate); max_spread = worst QMC replicate
    range; thin_flag = True when the covariance needed a jitter (deep-mu roundoff).
    """
    m2, V2 = gauss_condition(prob, mu, alpha)
    cols2 = list(prob.plb_info["cols2"])
    X2 = prob.states[:, cols2].astype(float)
    lq = np.empty(prob.fibre_size)
    spread = 0.0
    thin = False
    for i, x2 in enumerate(X2):
        lq[i], sp, jit = log_box_mass(m2, V2, x2 - 0.5, x2 + 0.5, rng, m_pow=m_pow)
        spread = max(spread, sp)
        thin = thin or (jit > 0)
    lz = logsumexp(lq)
    return lq - lz, float(np.exp(lz)), spread, thin


# ---------------------------------------------------------------------------
# Metropolised independence sampler: exact kernel, gap, Liu identity
# ---------------------------------------------------------------------------
def imh_kernel(logpi, logq):
    """Exact IMH transition matrix for normalised log target/proposal."""
    lv = logpi - logq                                  # log importance weights pi/q
    q = np.exp(logq)
    # accept(i->j) = min(1, exp(lv_j - lv_i))
    Acc = np.minimum(1.0, np.exp(lv[None, :] - lv[:, None]))
    K = q[None, :] * Acc
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(K, np.maximum(0.0, 1.0 - K.sum(axis=1)))
    return K


def imh_stats(logpi, logq):
    """gap (eigen), w* (log space), argmin state, DB violation, mean acceptance."""
    pi = np.exp(logpi - logsumexp(logpi))
    K = imh_kernel(logpi, logq)
    F = pi[:, None] * K
    dbe = float(np.abs(F - F.T).max())
    gap = float(slem(K, pi)[1])
    logw = logq - logpi                                # log q/pi
    i_star = int(np.argmin(logw))
    log_wstar = float(logw[i_star])
    lv = -logw
    acc = float(np.sum(pi[:, None] * np.exp(logq)[None, :]
                       * np.minimum(1.0, np.exp(lv[None, :] - lv[:, None]))))
    return gap, log_wstar, i_star, dbe, acc


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def p1_mu(mu5):
    return p1_poisson_theta(mu5)


def table_mu(delta):
    m = np.ones(6)
    m[TABLE23_BOTTLENECK] = delta
    return m


def nb_logpi(prob, mu, alpha=ALPHA):
    lw = nb_log_pmf(prob, mu, alpha)
    return lw - logsumexp(lw)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gate_mass_vs_scipy(prob, mu, rng, out):
    """G0: SOV log masses vs scipy mvn CDF inclusion-exclusion, benign regime."""
    m2, V2 = gauss_condition(prob, mu, ALPHA)
    cols2 = list(prob.plb_info["cols2"])
    d = len(cols2)
    X2 = prob.states[:, cols2].astype(float)
    worst = 0.0
    for x2 in X2[: min(12, len(X2))]:
        lm, _, _ = log_box_mass(m2, V2, x2 - 0.5, x2 + 0.5, rng)
        # inclusion-exclusion over the 2^d corners of the box
        tot = 0.0
        for mask in range(2 ** d):
            corner = np.array([(x2[k] + 0.5) if (mask >> k) & 1 else (x2[k] - 0.5)
                               for k in range(d)])
            sgn = (-1) ** (d - bin(mask).count("1"))
            tot += sgn * multivariate_normal.cdf(corner, mean=m2, cov=V2,
                                                 allow_singular=True)
        if tot > 1e-13:
            worst = max(worst, abs(np.exp(lm) - tot) / tot)
    ok = worst < 5e-3
    out.append(f"  G0 SOV vs scipy mvn ({prob.name or 'prob'}, mu_b={mu:g}): "
               f"max rel diff {worst:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def gate_consistency_limit(prob, name, rng, out):
    """G4: alpha -> inf, all means 5: well-matched relaxation, no collapse."""
    mu = 5.0 * np.ones(prob.r)
    logpi = nb_logpi(prob, mu, alpha=1e6)
    logq, feas, sp, _ = proposal_logq(prob, mu, 1e6, rng)
    gap, lws, i_star, dbe, acc = imh_stats(logpi, logq)
    ok = gap > 0.05
    out.append(f"  G4 consistency limit ({name}): gap={gap:.4f}  w*={np.exp(lws):.4f}  "
               f"feasible mass={feas:.3f}  mean acc={acc:.3f}  "
               f"{'PASS (O(1), no collapse)' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def sweep(prob, name, mu_of, bottleneck, mus, rng, out):
    """mu sweep: Gaussian-IMH gap, w*, Liu identity, argmin state, acceptance."""
    out.append(f"  {'mu':>8} {'gap(eigen)':>12} {'log10 w*':>12} {'|gap-w*|':>10} "
               f"{'feas':>6} {'acc':>8} {'x*_b':>5}  argmin state x*")
    rows = []
    liu_worst = 0.0
    for mu_b in mus:
        mu = mu_of(mu_b)
        logpi = nb_logpi(prob, mu)
        logq, feas, sp, thin = proposal_logq(prob, mu, ALPHA, rng)
        gap, lws, i_star, dbe, acc = imh_stats(logpi, logq)
        wstar = np.exp(lws)
        l10 = lws / np.log(10.0)
        dev = abs(gap - wstar)
        if wstar >= 1e-9:
            liu_worst = max(liu_worst, dev / max(wstar, 1e-300))
        xs = prob.states[i_star]
        gs = f"{gap:.4e}" if wstar >= 1e-13 else "  <1e-13"
        flag = " [thin: jittered]" if thin else ""
        out.append(f"  {mu_b:>8g} {gs:>12} {l10:>12.4g} {dev:>10.1e} "
                   f"{feas:>6.3f} {acc:>8.2e} {int(xs[bottleneck]):>5}  "
                   f"{tuple(int(v) for v in xs)}{flag}")
        rows.append(dict(mu=mu_b, gap=gap, log_wstar=lws, feas=feas, acc=acc,
                         i_star=i_star, x_star=tuple(int(v) for v in xs),
                         dbe=dbe, thin=thin))
    # local slopes of log w* vs log mu (log space, defined even where float gap = 0)
    out.append("  local slopes d(log w*)/d(log mu):")
    sl = []
    for a, b in zip(rows[:-1], rows[1:]):
        s = (a["log_wstar"] - b["log_wstar"]) / (np.log(a["mu"]) - np.log(b["mu"]))
        sl.append(s)
        out.append(f"    {a['mu']:>8g} -> {b['mu']:>8g}: {s:+.4g}")
    out.append(f"  Liu identity (G3), worst rel |gap-w*| where w*>=1e-9: {liu_worst:.2e}")
    return rows, sl, liu_worst


def alpha_sweep(prob, name, mu_of, bottleneck, mu_b, alphas, rng, out):
    """P-G3: fix mu, sweep alpha downward; w* and the argmin state."""
    out.append(f"  {name}, mu_b = {mu_b:g}:")
    out.append(f"  {'alpha':>7} {'log10 w*':>12} {'w*':>12} {'acc':>8} {'x*_b':>5}  argmin x*")
    ws = []
    for al in alphas:
        mu = mu_of(mu_b)
        logpi = nb_logpi(prob, mu, alpha=al)
        logq, feas, sp, _ = proposal_logq(prob, mu, al, rng)
        gap, lws, i_star, dbe, acc = imh_stats(logpi, logq)
        xs = prob.states[i_star]
        out.append(f"  {al:>7g} {lws / np.log(10):>12.4g} {np.exp(lws):>12.3e} "
                   f"{acc:>8.2e} {int(xs[bottleneck]):>5}  {tuple(int(v) for v in xs)}")
        ws.append(lws)
    mono = all(a >= b - 1e-12 for a, b in zip(ws[:-1], ws[1:]))
    out.append(f"  w* monotone decreasing as alpha decreases: {mono}")
    return mono


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    out = []
    out.append("=" * 74)
    out.append("TASK G -- Gaussian-relaxation independence sampler as a teleporter")
    out.append(f"alpha = {ALPHA}, seed = {SEED}, QMC = 8 x 2^11 scrambled Sobol")
    out.append("=" * 74)

    p1 = P1()
    tab = table_2x3()

    # ---------------- gates ----------------
    out.append("")
    out.append("GATES")
    ok = True
    ok &= gate_mass_vs_scipy(p1, 1.0, rng, out)
    ok &= gate_mass_vs_scipy(tab, 1.0, rng, out)
    # G1/G2 at a representative benign point
    for prob, name, mu_of in ((p1, "P1", p1_mu), (tab, "table", table_mu)):
        logpi = nb_logpi(prob, mu_of(0.1))
        logq, feas, sp, _ = proposal_logq(prob, mu_of(0.1), ALPHA, rng)
        gap, lws, i_star, dbe, acc = imh_stats(logpi, logq)
        qsum = float(np.exp(logsumexp(logq)))
        out.append(f"  G1 detailed balance ({name}, mu_b=0.1): max |piK - (piK)'| = {dbe:.2e}  "
                   f"{'PASS' if dbe < 1e-14 else 'FAIL'}")
        out.append(f"  G2 sum_F q = {qsum:.15f}  (pre-renorm feasible mass {feas:.4f}, "
                   f"rejection rate {1 - feas:.4f})  {'PASS' if abs(qsum - 1) < 1e-12 else 'FAIL'}")
        ok &= (dbe < 1e-14) and (abs(qsum - 1) < 1e-12)
    ok &= gate_consistency_limit(p1, "P1", rng, out)
    ok &= gate_consistency_limit(tab, "table", rng, out)
    if not ok:
        out.append("  ** GATE FAILURE -- sweeps below are NOT interpretable **")

    # ---------------- P-G1 / P-G2: mu sweeps ----------------
    out.append("")
    out.append("=" * 74)
    out.append("P-G1 (slope) and P-G2 (Liu identity), mu sweep, alpha = 1.9")
    out.append("=" * 74)
    out.append("")
    out.append(f"P1 (fibre {p1.fibre_size}, free dim {len(p1.plb_info['cols2'])}), "
               f"shrinking mu5:")
    rows_p1, sl_p1, liu_p1 = sweep(p1, "P1", p1_mu, 4, MUS + MUS_DEEP, rng, out)
    out.append("")
    out.append(f"2x3 table (fibre {tab.fibre_size}, free dim {len(tab.plb_info['cols2'])}), "
               f"shrinking delta on coord {TABLE23_BOTTLENECK}:")
    rows_tab, sl_tab, liu_tab = sweep(tab, "table", table_mu, TABLE23_BOTTLENECK,
                                      MUS + MUS_DEEP, rng, out)

    # ---------------- P-G3: alpha sweep ----------------
    out.append("")
    out.append("=" * 74)
    out.append("P-G3 -- dispersion mismatch: alpha sweeps")
    out.append("=" * 74)
    out.append("(a) benign mean mu_b = 1: isolates the dispersion mismatch (the mean-scale")
    out.append("    collapse of P-G1 is switched off).")
    mono_p1 = alpha_sweep(p1, "P1", p1_mu, 4, 1.0, [1.9, 1.0, 0.5, 0.25, 0.1], rng, out)
    mono_tab = alpha_sweep(tab, "table", table_mu, TABLE23_BOTTLENECK, 1.0,
                           [1.9, 1.0, 0.5, 0.25, 0.1], rng, out)
    out.append("(b) collapsed mean mu_b = 1e-2: the mean-scale mismatch dominates w*;")
    out.append("    dispersion mismatch shows in the mean acceptance instead.")
    mono_p1c = alpha_sweep(p1, "P1", p1_mu, 4, 1e-2, [1.9, 1.0, 0.5, 0.25, 0.1], rng, out)
    mono_tabc = alpha_sweep(tab, "table", table_mu, TABLE23_BOTTLENECK, 1e-2,
                            [1.9, 1.0, 0.5, 0.25, 0.1], rng, out)

    # ---------------- P-G4: floor comparison ----------------
    out.append("")
    out.append("=" * 74)
    out.append("P-G4 -- floor comparison at the deep end (mu_b = 1e-16)")
    out.append("=" * 74)
    tg_p1, _ = tempered_gap(p1, nb_log_pmf(p1, p1_mu(1e-16), ALPHA), BETAS3)
    tg_tab, _ = tempered_gap(tab, nb_log_pmf(tab, table_mu(1e-16), ALPHA), BETAS3)
    g_p1 = rows_p1[-1]["log_wstar"] / np.log(10)
    g_tab = rows_tab[-1]["log_wstar"] / np.log(10)
    out.append(f"  {'sampler':<26} {'P1 floor':>14} {'table floor':>14}  work per sweep")
    out.append(f"  {'Gaussian-IMH':<26} {'none':>14} {'none':>14}  1 (r-n)-dim draw + "
               f"1 solve + 2 cell masses; gap -> 0 super-polynomially")
    out.append(f"  {'  (log10 gap at 1e-16)':<26} {g_p1:>14.3g} {g_tab:>14.3g}  "
               f"(thin-direction jitter floor: values are upper bounds)")
    out.append(f"  {'3-rung tempering [1,.5,0]':<26} {tg_p1:>14.5f} {tg_tab:>14.5f}  "
               f"3 ray draws + 1 swap per sweep")
    out.append(f"  {'DA (R->inf), Pred 2':<26} {DA_FLOOR['P1']:>14.4f} {DA_FLOOR['table']:>14.4f}  "
               f"exact x|lambda resample (intractable; R latent draws to approximate)")

    # ---------------- verdicts ----------------
    out.append("")
    out.append("=" * 74)
    out.append("VERDICTS")
    out.append("=" * 74)
    s_p1 = sl_p1[-1]
    s_tab = sl_tab[-1]
    pg1 = (abs(s_p1) <= 0.1) and (abs(s_tab) <= 0.1)
    out.append(f"  P-G1 small-mu slope: P1 {s_p1:+.3g}, table {s_tab:+.3g} "
               f"(bar |slope| <= 0.1): {'HELD' if pg1 else 'FAILED'}")
    if not pg1:
        out.append("       slope is not ~0 and not ~1: |slope| grows without bound as mu")
        out.append("       shrinks (super-polynomial collapse).  Mechanism: the ridge pins the")
        out.append("       bottleneck coordinate, the matched Gaussian has sd ~ sqrt(mu) there,")
        out.append("       so the unit cell one step up the lobe costs exp(-1/(2mu)) under q")
        out.append("       but only Theta(mu) under pi_NB: q starves exactly the lobe states")
        out.append("       the teleporter exists to reach.  w* = min q/pi -> 0 faster than any")
        out.append("       power of mu; the argmin state's ratio is strongly mu-dependent.")
    pg2 = (liu_p1 < 1e-4) and (liu_tab < 1e-4)
    out.append(f"  P-G2 Liu identity gap = w*: worst rel dev P1 {liu_p1:.2e}, "
               f"table {liu_tab:.2e}: {'VERIFIED' if pg2 else 'FAILED'}")
    out.append(f"  P-G3 w* decreases as alpha decreases at benign mean (mu_b=1): "
               f"P1 {mono_p1}, table {mono_tab}: "
               f"{'HELD' if (mono_p1 and mono_tab) else 'FAILED'}")
    out.append(f"       at collapsed mean (mu_b=1e-2) w* is annihilated by the mean-scale")
    out.append(f"       mismatch and is ~flat in alpha (P1 {mono_p1c}, table {mono_tabc}); "
               f"the dispersion mismatch appears as falling mean acceptance instead.")
    out.append("  P-G4: Gaussian-IMH has NO floor (contrast: tempering ~1e-2, DA ~0.6-0.7);")
    out.append("        see table above.")
    out.append("")
    out.append("  Theory note (rem:dichotomy): the argmin ratio scales with mu, so the")
    out.append("  rounded moment-matched Gaussian is NOT a Doeblin teleporter: independence")
    out.append("  proposals teleport only if q dominates pi uniformly on the fibre")
    out.append("  (w* bounded away from 0 in mu).  Tail domination is the missing condition;")
    out.append("  a light-tailed relaxation of a shrinking-mean count target cannot satisfy it.")

    txt = "\n".join(out)
    print(txt)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "taskG_numbers.txt"), "w") as f:
        f.write(txt + "\n")
    print("numbers written: results/taskG_numbers.txt")

    # ---------------- figure (after numbers, Hopper-safe) ----------------
    try:
        make_figure(p1, tab, rows_p1, rows_tab, rng)
        print("figure written: results/fig_taskG_teleport.pdf/.png")
    except Exception as e:                                        # noqa: BLE001
        print(f"figure skipped: {e}")


def make_figure(p1, tab, rows_p1, rows_tab, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus = np.array(MUS)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # P1 panel: the paper's central overlay + the new curve
    ax = axes[0]
    sr_pois, sr_nb, slc, tmp = [], [], [], []
    for mu5 in mus:
        lw_p = log_weights(p1, "poisson", p1_mu(mu5))
        sr_pois.append(slem(sr_transition_matrix(p1, lw_p), normalised_pi(lw_p))[1])
        lw_n = nb_log_pmf(p1, p1_mu(mu5), ALPHA)
        sr_nb.append(slem(sr_transition_matrix(p1, lw_n), normalised_pi(lw_n))[1])
        slc.append(slem(slice_transition_matrix(p1, lw_n), normalised_pi(lw_n))[1])
        tmp.append(tempered_gap(p1, lw_n, BETAS3)[0])
    ga = np.array([np.exp(r["log_wstar"]) for r in rows_p1[: len(mus)]])
    ax.loglog(mus, sr_pois, "o-", label="SR-Pois")
    ax.loglog(mus, sr_nb, "s-", label="SR-NB")
    ax.loglog(mus, slc, "^-", label="slice (NB)")
    ax.loglog(mus, tmp, "d-", label="tempered 3-rung")
    vis = ga > 1e-16
    ax.loglog(mus[vis], ga[vis], "v-", label="Gauss-IMH (w*)")
    if (~vis).any():
        ax.annotate("Gauss-IMH $\\to 0$\nsuper-polynomially",
                    xy=(mus[vis][-1], max(ga[vis][-1], 1e-16)),
                    xytext=(2e-6, 1e-8), fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(r"$\mu_5$")
    ax.set_ylabel("spectral gap")
    ax.set_title("P1 (fibre 55)")
    ax.set_ylim(1e-16, 2.0)
    ax.legend(fontsize=7, loc="lower right")

    # table panel
    ax = axes[1]
    sr_nb_t, tmp_t = [], []
    for d in mus:
        lw_n = nb_log_pmf(tab, table_mu(d), ALPHA)
        sr_nb_t.append(slem(sr_transition_matrix(tab, lw_n), normalised_pi(lw_n))[1])
        tmp_t.append(tempered_gap(tab, lw_n, BETAS3)[0])
    gt = np.array([np.exp(r["log_wstar"]) for r in rows_tab[: len(mus)]])
    ax.loglog(mus, sr_nb_t, "s-", label="SR-NB")
    ax.loglog(mus, tmp_t, "d-", label="tempered 3-rung")
    vis = gt > 1e-16
    ax.loglog(mus[vis], gt[vis], "v-", label="Gauss-IMH (w*)")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("spectral gap")
    ax.set_title("2x3 table (fibre 10)")
    ax.set_ylim(1e-16, 2.0)
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_taskG_teleport.{ext}"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
