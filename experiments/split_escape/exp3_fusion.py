"""Experiment 3 (fusion, band x table product): prediction P-B.

Target: Uniform(band) x NB(table; bottleneck cell mean mu_star, alpha=1.9).
Geometry binds on the band (axis defect kappa); loading binds on the table.

Chains (exact SLEM):
  SR-direct   : single-ray Gibbs on the target            -> collapses in mu
  Split-ray   : ladder {target, U(A_mid), U(F)}, Gibbs-ray fibre moves
                -> floors in mu, but no kappa gain
  Split-ref   : same ladder, reflective (Bmax=2) fibre moves at every rung
                -> claim: floors in mu AND gains the kappa factor
The P-B claim is that Split-ref is the only chain flat in mu and scaling with kappa.
"""
import numpy as np, time
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

dirs4 = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

for (w, C) in bands:
    bd, st, cons = product_setup(w, C)
    kap = axis_defect(bd)
    n = len(st)
    print(f"\n=== band (w={w}, C={C}), kappa={kap:.2f}, product fibre {n} states ===")
    # rung-2 (U(F)) kernels are mu-independent: cache
    zero = np.zeros(n)
    t0 = time.time()
    K_unif_ray = gibbs_ray_kernel(st, zero, dirs4)
    K_unif_ref = reflective_kernel(st, zero, cons, Bmax=2, dim=4)
    print(f"  [uniform-rung kernels assembled in {time.time()-t0:.1f}s]")
    print(f"  {'mu':>8s} {'SR-direct':>10s} {'Split-ray':>10s} {'Split-ref':>10s} {'ref/ray':>8s}")
    for mu in mus:
        lw = product_logw(st, mu)
        lwn = lw - lw.max()
        # SR direct
        g_sr, _ = fibre_gap(gibbs_ray_kernel(st, lw, dirs4), lw)
        # ladder rungs
        mmid = 0.5 * (lwn.max() + lwn[np.isfinite(lwn)].min())
        mid = np.where(lwn >= mmid, 0.0, -np.inf)
        rungs = [lw, mid, zero]
        ker_ray = [gibbs_ray_kernel(st, r, dirs4) for r in rungs[:2]] + [K_unif_ray]
        P, pi, _ = ladder_kernel(st, rungs, ker_ray)
        g_ray, db1 = slem_gap(P, pi)
        ker_ref = [reflective_kernel(st, rungs[0], cons, Bmax=2, dim=4),
                   reflective_kernel(st, rungs[1], cons, Bmax=2, dim=4),
                   K_unif_ref]
        P, pi, _ = ladder_kernel(st, rungs, ker_ref)
        g_ref, db2 = slem_gap(P, pi)
        print(f"  {mu:8.0e} {g_sr:10.4g} {g_ray:10.4g} {g_ref:10.4g} "
              f"{g_ref/g_ray:8.2f}   [db {max(db1, db2):.0e}]")
