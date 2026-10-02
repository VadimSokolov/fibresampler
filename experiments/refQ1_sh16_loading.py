"""Real-scale loading check: does a hold-fixed recombination of the basis dissolve the
loading obstruction on the Auckland SH16 corridor?

Target: the max-entropy Poisson route model theta* of m4_realdata_ess.py WITHOUT the rate
floor, so that the fitted means of three routes are below 1.1 and two more below 5 (the
genuine loading regime; the floored model of the m4 benchmark removes it).  The PLB of the
paper (short-path partition) leaves most of these cells as pivot coordinates, so almost
every PLB move changes them.

Arms (same start, same iteration budget, same ESS estimator):
  SR (PLB)        single-ray Gibbs in the partition lattice basis
  HF_k            single-ray Gibbs in the hold-fixed recombination U V_k, where V_k is the
                  Hermite reduction of the k smallest-theta rows of U (m - rank moves hold
                  those cells fixed, the rest move them); same m = 14 moves
  WU_k            weighted union: 10 percent PLB moves, 90 percent HF_k moves (the
                  irreducibility-safe deployable form; the PLB is certified augmenting)
  Ref-B2          the reflective sampler (B_max = 2) in the PLB
  WU-LLL-pilot    weighted union of the PLB with the Mahalanobis LLL / surrogate-selected basis from a short
                  PLB pilot (the pilot chain is slow here by design), for contrast

Metric: worst-coordinate ESS over all 21 route counts (Geyer initial positive sequence), per
iteration and per second.  One seed per task (--seed); aggregate with refQ1_sh16_loading_aggregate.py.
Gate G0: A x = y at the end of every chain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.basis import hold_fixed_basis, int_det, recommend_basis                # noqa: E402
from fibresampler.problems import A_P2, P2_COLS1_PLB1, Y_P2                               # noqa: E402
from m4_realdata_ess import LiteProblem, logw_poisson, maxent_theta                       # noqa: E402
from pred5_ess import run_ref_chain, run_sr_chain                                         # noqa: E402
from refQ1_sh16_basis import MoveSet, worst_ess                                           # noqa: E402

KS = [1, 2, 3, 4, 5, 6, 8, 10]
RULE_THETA = 9.2          # P(Poisson(theta) = 0) > 1e-4: the cell can reach the wall x_j = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=300_000)
    ap.add_argument("--burn", type=int, default=30_000)
    ap.add_argument("--pilot", type=int, default=100_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.iters, a.burn, a.pilot = 30_000, 3_000, 15_000

    prob = LiteProblem(A_P2, Y_P2, P2_COLS1_PLB1, name="P2 Auckland SH16")
    theta = maxent_theta(prob.A, prob.y)
    lw = logw_poisson(theta)
    x0 = prob.feasible_start()
    cols2 = list(prob.plb_info["cols2"])
    order = list(np.argsort(theta))
    rule_k = int((theta < RULE_THETA).sum())
    base_seed = 20261002 + 1000 * a.seed
    rec = list(range(prob.r))
    rows = []

    def run(name, mp, sid, extra=None):
        rng = np.random.default_rng(np.random.SeedSequence((base_seed, sid)))
        tr, secs, _ = run_sr_chain(mp, lw, x0, a.iters, a.burn, rng, rec, False)
        g0 = bool(np.array_equal(prob.A @ tr[-1].astype(np.int64), prob.y))
        emin, emed = worst_ess(tr, a.iters)
        row = dict(arm=name, us_per_step=1e6 * secs / (a.iters + a.burn), secs=secs, ess_min=emin, ess_med=emed,
                   ess_min_per_iter=emin / a.iters, ess_min_ps=emin / secs, g0=g0)
        if extra:
            row.update(extra)
        rows.append(row)
        print(json.dumps(row), flush=True)

    run("SR", MoveSet(prob, prob.U), 1)
    for k in KS:
        J = [int(j) for j in order[:k]]
        V, n_slice = hold_fixed_basis(prob.U, J)
        assert abs(int_det(V)) == 1
        U_hf = prob.U @ V
        info = dict(k=k, J=J, n_slice=int(n_slice), max_entry=int(np.abs(U_hf).max()),
                    nnz_per_move=float((U_hf != 0).sum(0).mean()), rule=bool(k == rule_k))
        run(f"HF_{k}", MoveSet(prob, U_hf), 10 + k, info)
        # weighted union: PLB columns once, HF columns nine times, so the PLB share of directions is 10 percent
        U_wu = np.concatenate([prob.U] + [U_hf] * 9, axis=1)
        run(f"WU_{k}", MoveSet(prob, U_wu), 30 + k, dict(k=k))
    # statistical selection from a short PLB pilot (the pilot chain is slow here by design)
    rng = np.random.default_rng(np.random.SeedSequence((base_seed, 0)))
    tr, pilot_secs, _ = run_sr_chain(prob, lw, x0, a.pilot, a.burn, rng, cols2, False)
    V_sel, sel = recommend_basis(tr.astype(float))
    U_sel = prob.U @ V_sel
    run("WU-LLL-pilot", MoveSet(prob, np.concatenate([prob.U] + [U_sel] * 9, axis=1)), 50,
        dict(selector=sel["chosen"], gain=float(sel["gain"]), pilot_secs=pilot_secs))
    rng = np.random.default_rng(np.random.SeedSequence((base_seed, 99)))
    trace, secs, _ = run_ref_chain(prob, lw, x0, a.iters, a.burn, rng, rec, False, bmax=2)
    emin, emed = worst_ess(trace, a.iters)
    rows.append(dict(arm="Ref-B2", us_per_step=1e6 * secs / (a.iters + a.burn), secs=secs, ess_min=emin,
                     ess_med=emed, ess_min_per_iter=emin / a.iters, ess_min_ps=emin / secs,
                     g0=bool(np.array_equal(prob.A @ trace[-1].astype(np.int64), prob.y))))
    print(json.dumps(rows[-1]), flush=True)

    res = dict(seed=a.seed, iters=a.iters, burn=a.burn, theta=[float(t) for t in theta],
               rule_k=rule_k, rule_theta=RULE_THETA, rows=rows)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
