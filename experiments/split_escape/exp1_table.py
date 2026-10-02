"""Experiment 1 (loading, 2x3 table): does the split ladder floor?

Chains (all exact SLEM of the joint operator):
  SR      : single-ray Gibbs targeting pi_NB directly
  T3      : beta-tempering ladder {1, 1/2, 0}         (paper's escape; validation)
  S2      : split ladder {pi_NB, U(F)}                 (hard flatten, one rung)
  S3      : split ladder {pi_NB, U(A_mid), U(F)}       (hard flatten + overlap rung)
  S-keep  : split {pi_NB, U(A(m))}, m = mu^{1.5}-scale (threshold BELOW corridor weight -> corridor kept -> should floor)
  S-excl  : split {pi_NB, U(A(m))}, m = mu^{0.5}-scale (threshold ABOVE corridor weight -> corridor excluded -> should collapse)
"""
import numpy as np
from fibrelib import *

states = table_fibre()
dirs = [(1, 0), (0, 1)]
mus = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
alpha = 1.9

def unif_logw(mask):
    return np.where(mask, 0.0, -np.inf)

rows = {k: [] for k in ["SR", "T3", "S2", "S3", "S-keep", "S-excl"]}
dbmax = 0.0
for mu in mus:
    lw = table_logw(mu, alpha=alpha)
    lwn = lw - lw.max()

    # SR direct
    P = gibbs_ray_kernel(states, lw, dirs)
    g, db = fibre_gap(P, lw); rows["SR"].append(g); dbmax = max(dbmax, db)

    # beta ladder {1, .5, 0}
    rung = [lw, 0.5 * lw, np.zeros_like(lw)]
    ker = [gibbs_ray_kernel(states, r, dirs) for r in rung]
    P, pi, _ = ladder_kernel(states, rung, ker)
    g, db = slem_gap(P, pi); rows["T3"].append(g); dbmax = max(dbmax, db)

    # split S2 {target, U(F)}
    rung = [lw, np.zeros_like(lw)]
    ker = [gibbs_ray_kernel(states, r, dirs) for r in rung]
    P, pi, _ = ladder_kernel(states, rung, ker)
    g, db = slem_gap(P, pi); rows["S2"].append(g); dbmax = max(dbmax, db)

    # split S3 {target, U(A_mid), U(F)}; m_mid = geometric midpoint of weight range
    mmid = 0.5 * (lwn.max() + lwn.min())
    rung = [lw, unif_logw(lwn >= mmid), np.zeros_like(lw)]
    ker = [gibbs_ray_kernel(states, r, dirs) for r in rung]
    P, pi, _ = ladder_kernel(states, rung, ker)
    g, db = slem_gap(P, pi); rows["S3"].append(g); dbmax = max(dbmax, db)

    # thresholded two-rung ladders: corridor weight ~ mu (x11=1 states), lobes ~ 1
    for name, expo in [("S-keep", 1.5), ("S-excl", 0.5)]:
        m = expo * np.log(mu)          # threshold on log-relative weight
        mask = lwn >= m
        rung = [lw, unif_logw(mask)]
        ker = [gibbs_ray_kernel(states, r, dirs) for r in rung]
        P, pi, _ = ladder_kernel(states, rung, ker)
        g, db = slem_gap(P, pi); rows[name].append(g); dbmax = max(dbmax, db)

print(f"max detailed-balance violation across all runs: {dbmax:.2e}\n")
hdr = "mu_star   " + "".join(f"{k:>11s}" for k in rows)
print(hdr)
for t, mu in enumerate(mus):
    print(f"{mu:8.0e}  " + "".join(f"{rows[k][t]:11.4g}" for k in rows))

# fitted slopes over the last three decades
print("\nlog-log slope of gap in mu (last 3 sweep points):")
x = np.log(mus[-3:])
for k, v in rows.items():
    y = np.log(np.array(v[-3:]))
    sl = np.polyfit(x, y, 1)[0]
    print(f"  {k:8s}: {sl:+.3f}")

# corridor membership diagnostic at the two thresholds, smallest mu
mu = mus[-1]
lwn = table_logw(mu, alpha=alpha); lwn -= lwn.max()
x11 = np.array([table_full(s)[0] for s in states])
for name, expo in [("S-keep", 1.5), ("S-excl", 0.5)]:
    mask = lwn >= expo * np.log(mu)
    kept = sorted(set(x11[mask]))
    print(f"{name}: bottom rung keeps x11 values {kept} "
          f"({mask.sum()}/{len(states)} states)")
