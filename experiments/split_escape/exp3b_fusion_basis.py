"""Experiment 3b (fusion revisited): does a CHANGE OF BASIS do what the reflective split ladder does?

Same target and fibres as exp3_fusion.py: Uniform(band) x NB(table; bottleneck mean mu, alpha=1.9),
product coordinates (s, t, x12, x13).  Geometry binds on the band (axis defect kappa), loading
binds on the table (bottleneck cell x11 = 4 - x12 - x13).

Exact SLEM of every chain (symmetric eigendecomposition, detailed balance checked):

  SR-direct    single-ray Gibbs in the PLB directions (collapses in mu)
  Split-ray    three-rung split ladder, PLB single-ray fibre moves
  Split-ref    same ladder, reflective (B_max = 2) fibre moves (the paper's fused sampler)
  SR-band      single-ray in the REDUCED band basis {(1,1),(1,0)}, PLB on the table (kappa fixed, loading not)
  SR-basis     single-ray with the reduced band basis AND the hold-fixed table basis {(1,-1),(1,0)}
               (x11 fixed by one move): no ladder, no normalising constants, same four directions
  Split-band   the split ladder with the reduced band basis at every rung
  Split-basis  the split ladder with both recombinations at every rung

The claim under test (Paper II, reflective split sampler): the reflective split ladder is the only chain
whose gap is both flat in mu and immune to kappa.  Output: stdout, saved as
results/refQ1_fusion_basis.txt (python3 exp3b_fusion_basis.py | tee ../../results/refQ1_fusion_basis.txt).
"""
import time

import numpy as np

from fibrelib import *

alpha = 1.9
mus = [1e-1, 1e-2, 1e-3, 1e-4]
bands = [(1, 14), (1, 22)]

tab = table_fibre()
tab_lw_cache = {}


def product_setup(w, C):
    bd = band_fibre(w, C)
    st = [(s, t, a, b) for (s, t) in bd for (a, b) in tab]
    cons = []
    for a, c in band_constraints(w, C):
        cons.append((np.array([a[0], a[1], 0, 0]), c))
    for a, c in TABLE_CONSTRAINTS:
        cons.append((np.array([0, 0, a[0], a[1]]), c))
    return bd, st, cons


def product_logw(st, mu):
    if mu not in tab_lw_cache:
        tab_lw_cache[mu] = {s: v for s, v in zip(tab, table_logw(mu, alpha=alpha))}
    tl = tab_lw_cache[mu]
    return np.array([tl[(a, b)] for (_, _, a, b) in st])


dirs_plb = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
dirs_band = [(1, 1, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
dirs_both = [(1, 1, 0, 0), (1, 0, 0, 0), (0, 0, 1, -1), (0, 0, 1, 0)]

for (w, C) in bands:
    bd, st, cons = product_setup(w, C)
    kap = axis_defect(bd)
    n = len(st)
    print(f"\n=== band (w={w}, C={C}), kappa={kap:.2f}, product fibre {n} states ===")
    zero = np.zeros(n)
    t0 = time.time()
    K_unif = {name: gibbs_ray_kernel(st, zero, d) for name, d in
              (("plb", dirs_plb), ("band", dirs_band), ("both", dirs_both))}
    K_unif_ref = reflective_kernel(st, zero, cons, Bmax=2, dim=4)
    print(f"  [uniform-rung kernels assembled in {time.time() - t0:.1f}s]")
    print(f"  {'mu':>8s} {'SR-direct':>10s} {'Split-ray':>10s} {'Split-ref':>10s} | {'SR-band':>10s} {'SR-basis':>10s} "
          f"{'Split-band':>10s} {'Split-basis':>11s} | {'basis/ref':>9s} {'SRbasis/ref':>11s}")
    for mu in mus:
        lw = product_logw(st, mu)
        lwn = lw - lw.max()
        g_sr, _ = fibre_gap(gibbs_ray_kernel(st, lw, dirs_plb), lw)
        g_band, _ = fibre_gap(gibbs_ray_kernel(st, lw, dirs_band), lw)
        g_basis, _ = fibre_gap(gibbs_ray_kernel(st, lw, dirs_both), lw)
        mmid = 0.5 * (lwn.max() + lwn[np.isfinite(lwn)].min())
        mid = np.where(lwn >= mmid, 0.0, -np.inf)
        rungs = [lw, mid, zero]

        def ladder_gap(dirs_list, key, ref=False):
            if ref:
                ker = [reflective_kernel(st, rungs[0], cons, Bmax=2, dim=4),
                       reflective_kernel(st, rungs[1], cons, Bmax=2, dim=4), K_unif_ref]
            else:
                ker = [gibbs_ray_kernel(st, rungs[0], dirs_list), gibbs_ray_kernel(st, rungs[1], dirs_list),
                       K_unif[key]]
            P, pi, _ = ladder_kernel(st, rungs, ker)
            return slem_gap(P, pi)

        g_ray, db1 = ladder_gap(dirs_plb, "plb")
        g_ref, db2 = ladder_gap(None, None, ref=True)
        g_sb, db3 = ladder_gap(dirs_band, "band")
        g_sbasis, db4 = ladder_gap(dirs_both, "both")
        print(f"  {mu:8.0e} {g_sr:10.4g} {g_ray:10.4g} {g_ref:10.4g} | {g_band:10.4g} {g_basis:10.4g} "
              f"{g_sb:10.4g} {g_sbasis:11.4g} | {g_sbasis / g_ref:9.2f} {g_basis / g_ref:11.2f}   "
              f"[db {max(db1, db2, db3, db4):.0e}]")
