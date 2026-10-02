"""Prediction 4 -- the crossover: reflection helps geometry, not loading.

Decoupled design (per the protocol): a block-diagonal product of
  * a GEOMETRY block -- a thin diagonal band (kappa ~ 4-11) whose product-Poisson
    ridge lies along the oblique long axis (moderate mean), so the reflective
    proposal can remove the Theta(kappa) realised-path-length factor; and
  * a LOADING block -- the 2x3 table with a Poisson bottleneck cell whose mean
    mu_* is swept to 0, driving the Theta(mu) loading collapse of Prediction 1/2.

The two mechanisms are independent by construction, so the product gap is
(1/2) min(gap_geom, gap_load).  Reflection scales gap_geom by ~kappa and leaves
gap_load untouched, so the exact-SLEM ratio Ref/SR crosses over from ~kappa (high
mu, geometry-limited) to ~1 (low mu, loading-limited).  Everything is an EXACT
SLEM; every problem is validated exactly reversible first.

Stages:
  1. Geometry de-risk: does reflection help on the band with a *Poisson* ridge on
     the long axis (moderate mean), not just a uniform target?  (Make-or-break.)
  2. The crossover on the product (added after Stage 1 clears).

Run:  python3 experiments/pred4_crossover.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.spectral import normalised_pi, slem, log_weights
from fibresampler.reflective import reflective_transition_matrix
from fibresampler.synthetic import BandProblem, poisson_free_logw, product_problem
from fibresampler.problems import table_2x3, TABLE23_BOTTLENECK


def db_error(Q, pi):
    """max |pi_i Q_ij - pi_j Q_ji| -- the exact-reversibility gate."""
    F = pi[:, None] * Q
    return float(np.abs(F - F.T).max())


def gap_of(prob, log_w, bmax, pi_B=None):
    Q, _ = reflective_transition_matrix(prob, bmax=bmax, pi_B=pi_B, log_w=log_w)
    pi = normalised_pi(log_w)
    dbe = db_error(Q, pi)
    _, gap, _ = slem(Q, pi)
    return gap, dbe


def stage1_geometry_derisk():
    print("=" * 74)
    print("STAGE 1  --  geometry de-risk: does reflection help with a Poisson")
    print("            ridge on the long axis, not only a uniform target?")
    print("=" * 74)
    for (w, C) in [(2, 12), (2, 24), (2, 40)]:
        band = BandProblem(w, C)
        kap, L, a = band.kappa()
        print(f"\nband(w={w}, C={C}):  N={band.fibre_size}, "
              f"kappa=L/a={kap:.2f} (L={L:.2f}, a={a})")
        ridge = C / 4.0   # Poisson mode at (C/4, C/4) = midpoint of the diagonal
        targets = [
            ("uniform",          np.zeros(band.fibre_size)),
            (f"Pois ridge t={ridge:.0f}", poisson_free_logw(band, [ridge, ridge])),
            ("Pois corner t=1",  poisson_free_logw(band, [1.0, 1.0])),
        ]
        print(f"  {'target':>16} {'gap B=0(SR)':>12} {'gap B=2':>9} "
              f"{'gap B=4':>9} {'gap B=6':>9} {'bestRef/SR':>11} {'maxDBerr':>9}")
        for tname, lw in targets:
            g0, e0 = gap_of(band, lw, 0)
            gaps, dbes = {}, [e0]
            for bm in (2, 4, 6):
                g, e = gap_of(band, lw, bm)
                gaps[bm] = g; dbes.append(e)
            best = max(gaps.values())
            ratio = best / g0 if g0 > 0 else float("nan")
            flag = "  <-- helps" if ratio > 1.05 else ("  <-- HURTS" if ratio < 0.95 else "")
            print(f"  {tname:>16} {g0:12.5f} {gaps[2]:9.5f} {gaps[4]:9.5f} "
                  f"{gaps[6]:9.5f} {ratio:11.2f} {max(dbes):9.1e}{flag}")


def table_loading_logw(table, mu_star, base=1.0):
    """Product-Poisson log-target on the 2x3 table with the bottleneck cell mean
    set to mu_star and the remaining cells to `base`."""
    theta = np.full(table.states.shape[1], base, dtype=float)
    theta[TABLE23_BOTTLENECK] = mu_star
    return log_weights(table, "poisson", theta)


def stage1b_loading_block():
    """Sanity: the table's single-ray gap collapses as mu_* -> 0 (loading), and
    reflection does NOT rescue it (no oblique geometry to exploit)."""
    print("\n" + "=" * 74)
    print("STAGE 1b --  loading block (2x3 table): gap collapses as mu_*->0,")
    print("             reflection cannot rescue it")
    print("=" * 74)
    table = table_2x3()
    print(f"  table N={table.fibre_size}")
    print(f"  {'mu_*':>8} {'gap B=0(SR)':>12} {'gap B=2':>9} {'Ref/SR':>8} {'maxDBerr':>9}")
    for mu in (1.0, 0.5, 0.2, 0.1, 0.05, 0.01):
        lw = table_loading_logw(table, mu)
        g0, e0 = gap_of(table, lw, 0)
        g2, e2 = gap_of(table, lw, 2)
        print(f"  {mu:8.2f} {g0:12.5f} {g2:9.5f} {g2/g0:8.2f} {max(e0,e2):9.1e}")


def stage2_crossover(w=2, C=40, ref_bset=(2, 4, 6),
                     mus=(2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)):
    """The crossover on the decoupled product band(geometry) x table(loading)."""
    print("\n" + "=" * 74)
    print("STAGE 2  --  the crossover:  band(geometry) x table(loading)")
    print("=" * 74)
    band = BandProblem(w, C)
    kap, _, _ = band.kappa()
    table = table_2x3()
    prod = product_problem(band, table)
    ridge = C / 4.0
    lw_band = poisson_free_logw(band, [ridge, ridge])
    print(f"  band(w={w},C={C}) kappa={kap:.2f} N={band.fibre_size}  x  "
          f"table N={table.fibre_size}   =>  product N={prod.fibre_size}")
    print(f"  geometry target: Poisson ridge theta={ridge:.0f} on the long axis (fixed)")
    print(f"  Ref = best over B in {ref_bset};  SR = B=0\n")
    print(f"  {'mu_*':>8} {'gap SR':>10} {'gap Ref':>10} {'Ref/SR':>8} "
          f"{'bestB':>5} {'regime':>10} {'maxDBerr':>9}")
    rows = []
    for mu in mus:
        lw_tab = table_loading_logw(table, mu)
        lw = prod.logw_from_blocks(lw_band, lw_tab)
        g_sr, e_sr = gap_of(prod, lw, 0)
        best_g, best_b, e_max = -1.0, None, e_sr
        for bm in ref_bset:
            g, e = gap_of(prod, lw, bm)  # uniform pi_B over {0..bm} (stays ergodic)
            e_max = max(e_max, e)
            if g > best_g:
                best_g, best_b = g, bm
        ratio = best_g / g_sr if g_sr > 0 else float("nan")
        regime = "geometry" if ratio > 1.3 else ("loading" if ratio < 1.1 else "crossover")
        print(f"  {mu:8.2f} {g_sr:10.5f} {best_g:10.5f} {ratio:8.2f} "
              f"{best_b:5d} {regime:>10} {e_max:9.1e}")
        rows.append((mu, g_sr, best_g, ratio, best_b))
    return kap, rows


RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def make_figure(curves):
    """curves = [(kappa, rows), ...]; rows = [(mu, g_sr, g_ref, ratio, bestB), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    # Left: the crossover -- Ref/SR ratio vs mu_*, one curve per kappa.  Largest
    # kappa plotted first so the legend order matches the visual top-to-bottom
    # order; marker/linestyle pairs stay distinguishable in greyscale.
    styles = [("black", "o", "-"), ("0.45", "s", "--"), ("0.65", "^", ":")]
    ordered = sorted(curves, key=lambda kr: -kr[0])
    for (col, mk, ls), (kap, rows) in zip(styles, ordered):
        mu = [r[0] for r in rows]; ratio = [r[3] for r in rows]
        ax[0].plot(mu, ratio, marker=mk, ls=ls, color=col,
                   label=fr"$\kappa={kap:.1f}$")
        # predicted plateau kappa/2 (the bounce-count mixture halves kappa)
        ax[0].axhline(kap / 2.0, ls=":", color=col, lw=0.8, alpha=0.55)
        ax[0].annotate(fr"$\kappa/2={kap/2.0:.2f}$", (0.99, kap / 2.0),
                       xycoords=("axes fraction", "data"), fontsize=7,
                       color=col, ha="right", va="bottom")
    ax[0].axhline(1.0, ls="--", color="gray", lw=0.9)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"loading mean $\mu_\star$")
    ax[0].set_ylabel("exact-SLEM gap ratio  Ref / SR")
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    # data-coordinate placements: clear of both curves, the ratio-1 line, and
    # the kappa/2 guides at every tested rendering
    ax[0].annotate("geometry-limited", (0.55, 2.40), xycoords="data",
                   fontsize=8, color="gray", ha="center")
    ax[0].annotate("loading-limited", (0.0105, 1.12), xycoords="data",
                   fontsize=8, color="gray", ha="left", va="bottom")
    # Right: the two gaps for the largest kappa -- both collapse Theta(mu), Ref above.
    kap, rows = ordered[0]
    mu = [r[0] for r in rows]; gsr = [r[1] for r in rows]; gref = [r[2] for r in rows]
    ax[1].loglog(mu, gref, "o-", color="black",
                 label=fr"Ref (reflective), $\kappa={kap:.1f}$")
    ax[1].loglog(mu, gsr, "s--", color="0.45",
                 label=fr"SR (single ray), $\kappa={kap:.1f}$")
    ax[1].set_xlabel(r"loading mean $\mu_\star$")
    ax[1].set_ylabel(r"spectral gap $1-\lambda_\star$")
    # 1-2-5 ticks with plain decimal labels: the gaps span barely one decade, so
    # base-power labels alone would leave a single labelled tick.
    ax[1].yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[1].legend(frameon=False, fontsize=9)
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS, "fig_pred4_crossover.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"), dpi=130)
    return path


def main(Cs=(24, 32), ref_bset=(2, 4)):
    out = []
    curves = []
    for C in Cs:
        kap, rows = stage2_crossover(w=2, C=C, ref_bset=ref_bset,
                                     mus=(1.0, 0.7, 0.5, 0.3, 0.2, 0.14, 0.1,
                                          0.07, 0.05, 0.03, 0.02, 0.01))
        curves.append((kap, rows))
        out.append(f"band w=2 C={C} kappa={kap:.3f}")
        out.append(f"  {'mu_star':>8} {'gap_SR':>10} {'gap_Ref':>10} {'Ref/SR':>8} {'bestB':>5}")
        for mu, gsr, gref, ratio, b in rows:
            out.append(f"  {mu:8.3f} {gsr:10.6f} {gref:10.6f} {ratio:8.3f} {b:5d}")
        out.append("")
    figpath = None
    try:
        figpath = make_figure(curves)
    except Exception as e:
        out.append(f"[figure skipped: {e}]")
    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "pred4_numbers.txt"), "w") as fh:
        fh.write(text)
    print(text)
    if figpath:
        print("figure ->", figpath)
    print("numbers -> results/pred4_numbers.txt")


if __name__ == "__main__":
    main()
