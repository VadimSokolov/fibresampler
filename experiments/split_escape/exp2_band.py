"""Experiment 2 (geometry, band family): reflective lattice kernel vs single-ray,
uniform target, exact SLEM. Validates the kappa-scaling mechanism (Prediction 3)
with our independent implementation before fusing with the split ladder."""
import numpy as np
from fibrelib import *

print(f"{'(w,C)':>8s} {'n':>4s} {'kappa':>6s} {'gap SRgibbs':>12s} "
      f"{'gap SRmh':>10s} {'gap Ref(B2)':>12s} {'gap Ref(B4)':>12s} "
      f"{'best/Gibbs':>10s} {'best/MH':>8s}")
for (w, C) in [(1, 10), (1, 18), (1, 30), (2, 18)]:
    st = band_fibre(w, C)
    cons = band_constraints(w, C)
    kap = axis_defect(st)
    lw = np.zeros(len(st))  # uniform
    Pg = gibbs_ray_kernel(st, lw, [(1, 0), (0, 1)])
    gg, db1 = fibre_gap(Pg, lw)
    Pm = reflective_kernel(st, lw, cons, Bmax=0, dim=2)   # B=0 = Metropolis ray
    gm, db2 = fibre_gap(Pm, lw)
    P2 = reflective_kernel(st, lw, cons, Bmax=2, dim=2)
    g2, db3 = fibre_gap(P2, lw)
    P4 = reflective_kernel(st, lw, cons, Bmax=4, dim=2)
    g4, db4 = fibre_gap(P4, lw)
    db = max(db1, db2, db3, db4)
    best = max(g2, g4)
    print(f"({w},{C:3d}) {len(st):4d} {kap:6.2f} {gg:12.4g} {gm:10.4g} "
          f"{g2:12.4g} {g4:12.4g} {best/gg:10.2f} {best/gm:8.2f}   [db {db:.0e}]")
