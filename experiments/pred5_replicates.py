"""Prediction 5 headline-cell seed replicates (editor round 1, M5).

Reruns ONLY the geometry-limited product cell (mu* = 1) of the Prediction-5
grid, all four ablations, across five seeds, and reports the Ref/SR
worst-functional ESS-per-iteration ratio with mean and standard error.  The
first seed (20260701, sid mapping identical to pred5_ess.py) must reproduce
the ledgered ESSmin values exactly; that is the gate that this script runs
the same chains as the main grid.

Everything heavy is imported from pred5_ess (same chains, same IAT
estimator, same functional set, same worst = argmin-ESS rule).  The exact
operator is not re-assembled here: G3 (estimator vs exact operator) was
gated in the main run; replicates only move the Monte Carlo seed.

Output: results/pred5_replicates.txt.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pred5_ess import (BURN, N_ITER, RESULTS, build_problem, iat_geyer,
                       run_ref_chain, run_sr_chain)

SEEDS = [20260701, 20260702, 20260703, 20260704, 20260705]
# sid mapping of the mu*=1 product cells in pred5_ess.main(): (target, sampler)
CELLS = [("pois", "SR", 0), ("pois", "Ref", 1), ("nb", "SR", 2), ("nb", "Ref", 3)]
LEDGER = {("pois", "SR"): 34101, ("pois", "Ref"): 72455,
          ("nb", "SR"): 34821, ("nb", "Ref"): 70181}


def run_cell(job):
    seed, target, sampler, sid = job
    spec = dict(prob="product", mu=1.0, target=target, sampler=sampler, sid=sid)
    prob, lw_vec, fn, rec_cols, longax, names, bcol = build_problem(spec)
    rng = np.random.default_rng(np.random.SeedSequence((seed, sid)))
    x0 = prob.states[int(np.argmax(lw_vec))].astype(np.int64)
    runner = run_sr_chain if sampler == "SR" else run_ref_chain
    trace, secs, extras = runner(prob, fn, x0, N_ITER, BURN, rng, rec_cols, longax)
    ess = []
    for k in range(trace.shape[1]):
        t = iat_geyer(trace[:, k].astype(np.float64))
        if not np.isnan(t):
            ess.append(N_ITER / t)
    return seed, target, sampler, float(min(ess))


def main():
    jobs = [(seed, t, s, sid) for seed in SEEDS for (t, s, sid) in CELLS]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        results = list(ex.map(run_cell, jobs))
    essmin = {(seed, t, s): v for seed, t, s, v in results}

    lines = ["Prediction 5 headline-cell seed replicates (product, mu*=1)",
             f"N = {N_ITER:,} + {BURN:,} burn-in per chain; seeds {SEEDS}",
             "worst-functional ESS/iter, same functional set and rules as the grid",
             "=" * 74]
    gate_ok = all(round(essmin[(SEEDS[0], t, s)]) == LEDGER[(t, s)]
                  for (t, s, _) in CELLS)
    lines.append(f"GATE seed {SEEDS[0]} reproduces the ledgered grid cells "
                 f"(34101/72455/34821/70181): {'PASS' if gate_ok else 'FAIL'}")
    lines.append("")
    lines.append(f"{'seed':>10} {'SR-pois':>9} {'Ref-pois':>9} {'SR-nb':>9} "
                 f"{'Ref-nb':>9} {'ratio pois':>11} {'ratio nb':>9}")
    rp, rn = [], []
    for seed in SEEDS:
        sp = essmin[(seed, "pois", "SR")]; fp = essmin[(seed, "pois", "Ref")]
        sn = essmin[(seed, "nb", "SR")]; fn_ = essmin[(seed, "nb", "Ref")]
        rp.append(fp / sp); rn.append(fn_ / sn)
        lines.append(f"{seed:>10} {sp:9.0f} {fp:9.0f} {sn:9.0f} {fn_:9.0f} "
                     f"{fp/sp:11.3f} {fn_/sn:9.3f}")
    rp, rn = np.array(rp), np.array(rn)
    se = lambda a: a.std(ddof=1) / np.sqrt(len(a))
    lines.append("")
    lines.append(f"Ref/SR worst-functional ESS/iter, mean +/- SE over {len(SEEDS)} seeds:")
    lines.append(f"  Poisson pair: {rp.mean():.3f} +/- {se(rp):.3f}   "
                 f"(range {rp.min():.3f}..{rp.max():.3f})")
    lines.append(f"  NB pair     : {rn.mean():.3f} +/- {se(rn):.3f}   "
                 f"(range {rn.min():.3f}..{rn.max():.3f})")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS, "pred5_replicates.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
