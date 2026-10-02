"""Aggregate the Hopper SH16 loading-regime runs (results/q1_sh16_loading/seed*.json).

One row per arm: microseconds per step, worst-coordinate ESS per iteration and per second (mean and standard
error over seeds), and the same quantities as a ratio to single-ray Gibbs in the PLB, taken seed by seed.  The
rule row marks the number of held cells chosen by the dividing-line rule theta_hat < 9.2.  Writes
results/refQ1_sh16_loading.txt.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "results", "refQ1_sh16_loading.txt")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan"))


def main():
    runs = []
    for fn in sorted(glob.glob(os.path.join(REPO, "results", "q1_sh16_loading", "seed*.json"))):
        with open(fn) as fh:
            runs.append(json.loads(fh.readline()))
    theta = np.array(runs[0]["theta"])
    order = np.argsort(theta)
    out = ["=" * 142,
           "SH16 loading regime (Poisson max-entropy theta, NO rate floor): PLB versus hold-fixed recombination, worst-coordinate ESS over 21 routes",
           f"{len(runs)} seeds, {runs[0]['iters']} iterations after {runs[0]['burn']} burn-in; mean (SE); ratios are per-seed paired with SR in the PLB",
           "=" * 142,
           "fitted route means, smallest first: " + ", ".join(f"{theta[j]:.3g}" for j in order[:12]),
           f"cells with theta < {runs[0]['rule_theta']}: {runs[0]['rule_k']} (the rule's k)",
           "",
           f"  {'arm':>13} {'us/step':>8} {'ESS_min/iter':>17} {'ESS_min/s':>15} | {'vs SR /iter':>15} {'vs SR /s':>15} | {'/s incl pilot':>15} | stuck | G0"]
    names = [r["arm"] for r in runs[0]["rows"]]
    for arm in names:
        rows = [next(x for x in r["rows"] if x["arm"] == arm) for r in runs]
        sr = [next(x for x in r["rows"] if x["arm"] == "SR") for r in runs]
        us = mean_se([x["us_per_step"] for x in rows])
        ei = mean_se([x["ess_min_per_iter"] for x in rows])
        es = mean_se([x["ess_min_ps"] for x in rows])
        ri = mean_se([a["ess_min_per_iter"] / b["ess_min_per_iter"] for a, b in zip(rows, sr)])
        rs = mean_se([a["ess_min_ps"] / b["ess_min_ps"] for a, b in zip(rows, sr)])
        stuck = sum(1 for x in rows if x["ess_min"] == 0.0)
        g0 = all(x["g0"] for x in rows)
        mark = "  <- rule" if rows[0].get("rule") else ""
        if arm == "WU-LLL-pilot":
            inc = mean_se([(a["ess_min"] / (a["secs"] + a["pilot_secs"])) / b["ess_min_ps"] for a, b in zip(rows, sr)])
            inc_s = f"{inc[0]:>8.2f} ({inc[1]:>5.2f})"
        else:
            inc_s = f"{'-':>15}"
        out.append(f"  {arm:>13} {us[0]:>8.1f} {ei[0]:>10.2e} ({ei[1]:.1e}) {es[0]:>8.2f} ({es[1]:>5.2f}) | "
                   f"{ri[0]:>9.2f} ({ri[1]:>5.2f}) {rs[0]:>9.2f} ({rs[1]:>5.2f}) | {inc_s:>15} | {stuck:>2}/{len(rows)} | {'ok' if g0 else 'FAIL'}{mark}")
    sel = [next(x for x in r["rows"] if x["arm"] == "WU-LLL-pilot") for r in runs]
    out.append("")
    out.append(f"pilot-selected basis decision per seed: {[x.get('selector') for x in sel]}, surrogate gain {[round(x.get('gain', float('nan')), 3) for x in sel]}, "
               f"pilot time {np.mean([x.get('pilot_secs', float('nan')) for x in sel]):.1f} s")
    from fibresampler.problems import A_P2, P2_COLS1_PLB1, Y_P2
    from m4_realdata_ess import LiteProblem
    U_plb = LiteProblem(A_P2, Y_P2, P2_COLS1_PLB1, name="P2 Auckland SH16").U
    hf = {x["k"]: x for x in runs[0]["rows"] if x["arm"].startswith("HF_")}
    out.append(f"PLB moves U: largest entry {int(np.abs(U_plb).max())}, mean nonzeros per move {float((U_plb != 0).sum(0).mean()):.3f}")
    out.append("hold-fixed moves U V_k (the same on every seed): largest entry "
               + str(max(x["max_entry"] for x in hf.values())) + ", mean nonzeros per move "
               + ", ".join(f"k={k}: {hf[k]['nnz_per_move']:.3f}" for k in sorted(hf)))
    text = "\n".join(out) + "\n"
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
