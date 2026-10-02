"""Why does the reflective sampler fall 50x behind the Gibbs ray on SH16 under the fitted Poisson target?

Proposition nohurt bounds the reflective gap against its OWN B = 0 reduction, a Metropolised ray draw
(uniform proposal on the ray, Metropolis-Hastings acceptance), not against the heat-bath (Gibbs) ray.
This probe separates the two effects on the Auckland SH16 corridor under the floored Poisson target of
refQ1_sh16_basis.py: worst-coordinate ESS per iteration of

  SR Gibbs      heat-bath along a PLB ray (the baseline of Table tab:basis-sh16)
  Ref B0        the reflective sampler with B_max = 0 (Metropolised ray, no bounces)
  Ref B2        the reflective sampler with B_max = 2

Three seeds, same chain length and burn-in as the main runs.  Output: results/refQ1_ref_b0_probe.txt.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fibresampler.problems import A_P2, P2_COLS1_PLB1, Y_P2              # noqa: E402
from m4_realdata_ess import LiteProblem, logw_poisson, maxent_theta       # noqa: E402
from pred5_ess import run_ref_chain, run_sr_chain                          # noqa: E402
from refQ1_sh16_basis import FLOOR, worst_ess                              # noqa: E402

OUT = os.path.join(REPO, "results", "refQ1_ref_b0_probe.txt")


def main():
    n_iter, burn = 300_000, 30_000
    prob = LiteProblem(A_P2, Y_P2, P2_COLS1_PLB1, name="P2 Auckland SH16")
    theta = np.maximum(maxent_theta(prob.A, prob.y), FLOOR)
    lw = logw_poisson(theta)
    x0 = prob.feasible_start()
    rec = list(range(prob.r))
    out = ["=" * 110,
           "Metropolised ray (Ref B0) versus heat-bath ray (SR Gibbs) versus Ref B2 on SH16, floored Poisson target",
           f"worst-coordinate ESS per iteration, {n_iter} iterations after {burn} burn-in, three seeds",
           "=" * 110,
           f"  {'seed':>4} {'SR Gibbs':>12} {'Ref B0':>12} {'Ref B2':>12} | {'B0/SR':>7} {'B2/SR':>7} {'B2/B0':>7} | us/step SR, B0, B2"]
    rows = []
    for seed in (1, 2, 3):
        vals, costs = [], []
        for name, bmax in (("SR", None), ("B0", 0), ("B2", 2)):
            rng = np.random.default_rng(np.random.SeedSequence((20261002, 7, seed, 0 if bmax is None else bmax + 1)))
            if bmax is None:
                tr, secs, _ = run_sr_chain(prob, lw, x0, n_iter, burn, rng, rec, False)
            else:
                tr, secs, _ = run_ref_chain(prob, lw, x0, n_iter, burn, rng, rec, False, bmax=bmax)
            vals.append(worst_ess(tr, n_iter)[0] / n_iter)
            costs.append(1e6 * secs / (n_iter + burn))
        rows.append(vals)
        out.append(f"  {seed:>4} {vals[0]:>12.3e} {vals[1]:>12.3e} {vals[2]:>12.3e} | {vals[1] / vals[0]:>7.3f} "
                   f"{vals[2] / vals[0]:>7.3f} {vals[2] / vals[1]:>7.2f} | {costs[0]:.0f}, {costs[1]:.0f}, {costs[2]:.0f}")
        print(out[-1], flush=True)
    r = np.array(rows)
    out.append("")
    out.append(f"mean ratios: B0/SR {np.mean(r[:, 1] / r[:, 0]):.3f}, B2/SR {np.mean(r[:, 2] / r[:, 0]):.3f}, "
               f"B2/B0 {np.mean(r[:, 2] / r[:, 1]):.2f}")
    text = "\n".join(out) + "\n"
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
