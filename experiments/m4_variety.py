#!/usr/bin/env python3
"""M4 breadth: the real-data ESS/second protocol across a VARIETY of real
network-tomography problems, not just Auckland SH16.

Martin Hazelton's review asked for ESS/second on a variety of real problems.
We run the identical protocol of m4_realdata_ess.py (obliqueness proxy, then
single-ray vs reflective ESS/second under uniform / floored-posterior / raw-ML
targets) on three real network-tomography fibres of increasing size:

  * P1     -- Hazelton, McVeagh, Tuffley & van Brunt (2024, Bernoulli 30(4)),
              Example 2 / Example 11.  8 routes, 4 links, free dim 4.
  * VARDI  -- Vardi (1996, JASA 91, Example 1), the canonical 4-node network,
              also analysed by Tebaldi & West (1998, JASA 93:442, Fig. 1).
              7 links, 12 OD routes, free dim 5.  Routing matrix and the
              standard example loads Y=(2,28,9,5,34,16,35) are transcribed
              verbatim from Tebaldi & West (1998), eq. in Sec. 2.1; the
              feasibility witness X=(2,2,0,8,5,7,6,4,9,7,17,11) they quote is
              checked in _check_vardi() below.
  * P2     -- Auckland State Highway 16 corridor (Hazelton 2024, Sec. 6).
              7 links, 21 routes, real observed loads, free dim 14.

The larger Monroe NC road network of Tebaldi & West (1998, Sec. 4; 20x64) is a
genuinely real problem but its routing matrix and observed link counts are
printed only as figures (Figs. 6-7), not machine-readable text, so it cannot be
reproduced here without fabricating data; it is noted, not run.

No number is invented: every routing matrix is transcribed from its source and
its feasibility witness re-checked, and every ESS is measured from our own
chains.
"""
import argparse
import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.problems import (A_P1, Y_P1, P1_COLS1,
                                    A_P2, Y_P2, P2_COLS1_PLB1)
from experiments.m4_realdata_ess import (LiteProblem, maxent_theta, logw_uniform,
                                          logw_poisson, run_cell, anisotropy_proxy,
                                          ALPHA)

SEED = 20260709
FLOOR = 25.0

# ---------------------------------------------------------------------------
# Vardi (1996) Example 1 / Tebaldi & West (1998) Figure 1: 4-node network.
# 7 directed links (rows) x 12 OD routes (columns).  Transcribed from
# Tebaldi & West (1998), the 7x12 routing matrix displayed in Section 2.1.
# ---------------------------------------------------------------------------
A_VARDI = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
], dtype=np.int64)
Y_VARDI = np.array([2, 28, 9, 5, 34, 16, 35], dtype=np.int64)
# Tebaldi & West's quoted route counts, used only as a feasibility witness.
X_VARDI_WITNESS = np.array([2, 2, 0, 8, 5, 7, 6, 4, 9, 7, 17, 11], dtype=np.int64)


def _check_vardi():
    assert np.array_equal(A_VARDI @ X_VARDI_WITNESS, Y_VARDI), \
        "Vardi witness does not reproduce Y -- transcription error"


def find_unimodular_cols1(A, y):
    """Find n columns whose square block A1 is unimodular (|det|=1, so A1^{-1}
    is integer) and A1^{-1} y is a non-negative integer vector, i.e. a valid
    PLB with feasible start x0 = [A1^{-1} y ; 0].  Returns the first such tuple
    (columns preferred with the fewest support, giving a short-path partition)."""
    n, r = A.shape
    # prefer columns that are close to singletons first (short-path A1)
    order = sorted(range(r), key=lambda j: int(A[:, j].sum()))
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    best = None
    for combo in combinations(order, n):
        A1 = A[:, combo]
        d = np.linalg.det(A1)
        if abs(abs(d) - 1.0) > 1e-6:
            continue
        xi = np.linalg.solve(A1, y)
        xir = np.rint(xi)
        if np.max(np.abs(xi - xir)) < 1e-6 and (xir >= 0).all():
            best = tuple(sorted(combo))
            break
    if best is None:
        raise RuntimeError("no unimodular non-negative PLB partition found")
    return best


# ---------------------------------------------------------------------------
def run_problem(name, A, y, cols1, n_iter, burn, seed, bmax_sweep=(2, 4)):
    prob = LiteProblem(A, y, cols1, name=name)
    x0 = prob.feasible_start()
    out = [f"\n=== {name}:  r={prob.r} routes, n={prob.n} links, free dim {prob.n_moves} ==="]

    cov, corr = anisotropy_proxy(prob, x0, n_iter, burn, seed)
    out.append(f"  anisotropy: covariance spread {cov:.2f} (route-SCALE); "
               f"correlation spread {corr:.2f} (OBLIQUENESS)")

    theta = maxent_theta(prob.A, prob.y)
    theta_wc = np.maximum(theta, FLOOR)
    n_empty = int((theta < FLOOR).sum())
    out.append(f"  max-ent theta* range {theta.min():.2f}..{theta.max():.1f}; "
               f"{n_empty} route(s) below floor {FLOOR:.0f}")

    targets = [("uniform", logw_uniform()),
               ("pois-wc", logw_poisson(theta_wc)),
               ("pois-me", logw_poisson(theta))]

    hdr = f"  {'target':>9} {'smp':>7} {'us/stp':>7} {'acc':>6} " \
          f"{'ESSmin':>8} {'ESSmin/s':>9} {'ESSmed/s':>9} {'G0':>4}"
    results = {}
    for tname, logw in targets:
        out.append(f"  -- target {tname} --")
        out.append(hdr)
        row = run_cell(prob, logw, x0, "SR", n_iter, burn, seed, 0)
        results[(tname, "SR")] = row
        out.append(_fmt(tname, row))
        sweep = bmax_sweep if tname == "uniform" else (2,)
        for b in sweep:
            row = run_cell(prob, logw, x0, "Ref", n_iter, burn, seed, 0, bmax=b)
            results[(tname, f"Ref/B{b}")] = row
            out.append(_fmt(tname, row))

    # headline ratios
    u_sr = results[("uniform", "SR")]["ess_min_ps"]
    u_rf = results[("uniform", "Ref/B2")]["ess_min_ps"]
    out.append(f"  uniform  Ref/B2 : SR ESS/sec vs single-ray  = {u_rf:.2f} / {u_sr:.2f} "
               f"(ratio {u_rf / u_sr:.2f})")
    me_sr = results[("pois-me", "SR")]["ess_min_ps"]
    out.append(f"  pois-me  worst-coord ESS/sec (loading) = {me_sr:.2f}")
    return "\n".join(out), dict(name=name, free=prob.n_moves, cov=cov, corr=corr,
                                n_empty=n_empty, results=results)


def _fmt(tname, r):
    return (f"  {tname:>9} {r['sampler'] + ('' if r['sampler']=='SR' else '/B'+str(r['bmax'])):>7} "
            f"{r['us_per_step']:7.1f} {r['acc']:6.3f} "
            f"{_n(r['ess_min']):>8} {_n(r['ess_min_ps']):>9} {_n(r['ess_med_ps']):>9} "
            f"{'ok' if r['g0'] else 'BAD':>4}")


def _n(v):
    return "nan" if v != v else (f"{v:.0f}" if v >= 100 else f"{v:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300_000)
    ap.add_argument("--burn", type=int, default=30_000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_iter, burn = (40_000, 4_000) if args.quick else (args.iters, args.burn)

    _check_vardi()
    vardi_cols1 = find_unimodular_cols1(A_VARDI, Y_VARDI)

    header = ("=" * 86 + "\n"
              "M4 breadth -- ESS/second across a VARIETY of real network-tomography problems\n"
              f"  {n_iter} iters + {burn} burn, seed {SEED}; kernels = pred5_ess.py; alpha_NB={ALPHA}\n"
              "  Vardi/Tebaldi-West 4-node (canonical), Hazelton P1, Auckland SH16 (real road)\n"
              f"  Vardi PLB cols1 (0-indexed, unimodular) = {vardi_cols1}\n"
              + "=" * 86)
    blocks = [header]
    summ = []
    t0 = time.time()
    for name, A, y, c1 in [
        ("P1 Hazelton Ex.2/11", A_P1, Y_P1, P1_COLS1),
        ("VARDI 4-node (Vardi96/TW98)", A_VARDI, Y_VARDI, vardi_cols1),
        ("P2 Auckland SH16", A_P2, Y_P2, P2_COLS1_PLB1),
    ]:
        blk, s = run_problem(name, A, y, c1, n_iter, burn, SEED)
        blocks.append(blk)
        summ.append(s)

    blocks.append("\n--- cross-problem summary (obliqueness vs reflection) ---")
    blocks.append(f"  {'problem':>28} {'free':>4} {'corr(obliq)':>11} "
                  f"{'uni Ref/SR':>11} {'loading ESS/s':>13}")
    for s in summ:
        u_sr = s["results"][("uniform", "SR")]["ess_min_ps"]
        u_rf = s["results"][("uniform", "Ref/B2")]["ess_min_ps"]
        me = s["results"][("pois-me", "SR")]["ess_min_ps"]
        ratio = u_rf / u_sr if u_sr == u_sr and u_sr > 0 else float("nan")
        blocks.append(f"  {s['name']:>28} {s['free']:>4} {s['corr']:>11.2f} "
                      f"{ratio:>11.2f} {me:>13.2f}")
    blocks.append(f"\n  (elapsed {time.time() - t0:.0f}s)")

    text = "\n".join(blocks)
    print(text)
    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "m4_variety_numbers.txt")
    with open(outpath, "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
