"""Real-scale check of the change-of-basis lever on Auckland SH16 (r=21, n=7, free dim 14).

The exact comparison (refQ1_basis_change.py) is limited to enumerable fibres.  This script
asks the same question where the paper's negative result lives: on the real SH16 corridor,
under the uniform fibre law and a well-conditioned Poisson route model, does a reduced basis
beat single-ray hit-and-run in worst-coordinate ESS per second, and how does it compare
with the reflective sampler?

Procedure (one task = one target and one seed):
  1. PILOT   short single-ray chain in the PLB frame under the target; covariance of the 14
             free coordinates -> LLL in Sigma^{-1} -> unimodular V (frozen afterwards, so the
             main chain is an ordinary fixed-kernel reversible sampler);
  2. ARMS    SR (PLB moves), SR-LLL (moves U V), SR-U (PLB and reduced moves, 28 directions),
             SR-sel (only if the surrogate selector leaves the PLB; else it equals SR),
             Ref-B2 (reflective, B_max = 2), each 330k-iteration chains from the same start;
  3. METRIC  worst-coordinate ESS over all 21 route counts (Geyer initial-sequence IAT, the
             estimator of m4_realdata_ess.py), per iteration and per wall-clock second, with
             the one-off pilot-plus-reduction cost reported separately and also charged.

Gate G0: A x = y holds at the end of every chain.  Output: JSON lines on stdout and, with
--out, one file per task.  Aggregate with experiments/refQ1_sh16_aggregate.py.
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

from fibresampler.basis import int_det, recommend_basis, reduce_basis, to_frame     # noqa: E402
from fibresampler.problems import A_P2, P2_COLS1_PLB1, Y_P2                         # noqa: E402
from m4_realdata_ess import (LiteProblem, logw_poisson, logw_uniform,               # noqa: E402
                             maxent_theta)
from pred5_ess import iat_geyer, run_ref_chain, run_sr_chain                        # noqa: E402

FLOOR = 25.0                       # rate floor of the well-conditioned Poisson model (m4)


class MoveSet:
    """Minimal problem exposing the integer move matrix the chains read."""

    def __init__(self, base, U):
        self.U = np.asarray(U, dtype=np.int64)
        self.A, self.y = base.A, base.y
        self.plb_info = base.plb_info

    @property
    def n_moves(self):
        return self.U.shape[1]

    @property
    def r(self):
        return self.U.shape[0]


def worst_ess(trace, n_iter):
    ess = []
    for k in range(trace.shape[1]):
        col = trace[:, k].astype(np.float64)
        if col.std() < 1e-9:
            continue
        t = iat_geyer(col)
        if np.isfinite(t):
            ess.append(n_iter / t)
    if not ess:                       # every coordinate constant: the chain never left its start
        return 0.0, 0.0
    ess = np.array(ess)
    return float(ess.min()), float(np.median(ess))


def corr_spread(X):
    C = np.atleast_2d(np.cov(np.asarray(X, float).T))
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    ev = np.linalg.eigvalsh(R)
    ev = ev[ev > 1e-9]
    return float(np.sqrt(ev.max() / ev.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["uniform", "poiswc"], default="uniform")
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
    x0 = prob.feasible_start()
    cols2 = list(prob.plb_info["cols2"])
    if a.target == "uniform":
        lw = logw_uniform()
    else:
        theta = np.maximum(maxent_theta(prob.A, prob.y), FLOOR)
        lw = logw_poisson(theta)
    base_seed = 20261002 + 1000 * a.seed

    # ---- 1. pilot (single-ray, PLB frame) and reduction --------------------------------
    rng = np.random.default_rng(np.random.SeedSequence((base_seed, 0)))
    tr, pilot_secs, _ = run_sr_chain(prob, lw, x0, a.pilot, a.burn, rng, cols2, False)
    t0 = time.perf_counter()
    V = reduce_basis(tr.astype(float))
    lll_secs = time.perf_counter() - t0
    assert abs(int_det(V)) == 1
    U_red = prob.U @ V
    cspread0 = corr_spread(tr)
    cspread1 = corr_spread(to_frame(tr.astype(np.int64), V))
    t0 = time.perf_counter()
    V_sel, sel_info = recommend_basis(tr.astype(float))
    sel_secs = time.perf_counter() - t0
    assert abs(int_det(V_sel)) == 1
    cost = dict(pilot_secs=pilot_secs, lll_secs=lll_secs, select_secs=sel_secs, pilot_iters=a.pilot)
    geom = dict(oblq_plb=cspread0, oblq_reduced=cspread1,
                max_entry_V=int(np.abs(V).max()), max_entry_U=int(np.abs(U_red).max()),
                nnz_per_move_plb=float((prob.U != 0).sum(0).mean()),
                nnz_per_move_red=float((U_red != 0).sum(0).mean()),
                selector=dict(chosen=sel_info["chosen"], gain=float(sel_info["gain"]),
                              surrogate={k: float(v) for k, v in sel_info["surrogate"].items()}))

    arms = {
        "SR": MoveSet(prob, prob.U),
        "SR-LLL": MoveSet(prob, U_red),
        "SR-U": MoveSet(prob, np.concatenate([prob.U, U_red], axis=1)),
    }
    if sel_info["chosen"] != "identity":          # otherwise the selected basis IS the PLB
        arms["SR-sel"] = MoveSet(prob, prob.U @ V_sel)
    rows = []
    rec = list(range(prob.r))
    for sid, (name, mp) in enumerate(arms.items(), start=1):
        rng = np.random.default_rng(np.random.SeedSequence((base_seed, sid)))
        trace, secs, _ = run_sr_chain(mp, lw, x0, a.iters, a.burn, rng, rec, False)
        g0 = bool(np.array_equal(prob.A @ trace[-1].astype(np.int64), prob.y))
        emin, emed = worst_ess(trace, a.iters)
        rows.append(dict(arm=name, us_per_step=1e6 * secs / (a.iters + a.burn), secs=secs,
                         ess_min=emin, ess_med=emed, ess_min_per_iter=emin / a.iters,
                         ess_min_ps=emin / secs, g0=g0))
    rng = np.random.default_rng(np.random.SeedSequence((base_seed, 9)))
    trace, secs, extra = run_ref_chain(prob, lw, x0, a.iters, a.burn, rng, rec, False, bmax=2)
    g0 = bool(np.array_equal(prob.A @ trace[-1].astype(np.int64), prob.y))
    emin, emed = worst_ess(trace, a.iters)
    rows.append(dict(arm="Ref-B2", us_per_step=1e6 * secs / (a.iters + a.burn), secs=secs,
                     ess_min=emin, ess_med=emed, ess_min_per_iter=emin / a.iters,
                     ess_min_ps=emin / secs, g0=g0))

    res = dict(target=a.target, seed=a.seed, iters=a.iters, burn=a.burn, cost=cost,
               geom=geom, rows=rows, V=V.tolist(), V_sel=V_sel.tolist())
    text = json.dumps(res)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
