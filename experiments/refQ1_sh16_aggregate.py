"""Aggregate the Hopper SH16 change-of-basis runs (results/q1_sh16/*.json).

One row per (target, arm): microseconds per step, worst-coordinate ESS per iteration and per
second (mean and standard error over seeds), and the same quantities as a ratio to single-ray
hit-and-run in the PLB frame, taken seed by seed so that the pairing removes common
chain-length noise.  For the reduced-basis arms the one-off pilot and reduction time is also
charged to the wall clock ("ESS/s incl. pilot").  A run whose chain never leaves its start state
(worst-coordinate ESS exactly 0: every move of the arm is infeasible at the vertex start) is counted
as a zero in every mean and reported in the "stuck" column, with the mean over the non-stuck seeds
on a separate line.  Writes results/refQ1_sh16_basis.txt.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "results", "refQ1_sh16_basis.txt")
ARMS = ["SR", "SR-LLL", "SR-U", "SR-sel", "Ref-B2"]


def mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan"))


def main():
    runs = {}
    for fn in sorted(glob.glob(os.path.join(REPO, "results", "q1_sh16", "*.json"))):
        with open(fn) as fh:
            r = json.loads(fh.readline())
        runs.setdefault(r["target"], []).append(r)

    out = ["=" * 118,
           "Change of basis versus reflection on Auckland SH16 (P2, r=21, n=7, free dim 14), worst-coordinate ESS",
           "mean (SE) over seeds; ratios are per-seed paired with SR; pilot = single-ray chain + LLL, charged in the last column",
           "=" * 118]
    for target in ("uniform", "poiswc"):
        rs = runs.get(target, [])
        if not rs:
            continue
        out.append("")
        out.append(f"TARGET {target}: {len(rs)} seeds, {rs[0]['iters']} iterations each, burn {rs[0]['burn']}, "
                   f"pilot {rs[0]['cost']['pilot_iters']}")
        out.append(f"  {'arm':>8} {'us/step':>9} {'ESS_min/iter':>16} {'ESS_min/s':>14} | {'vs SR /iter':>14} "
                   f"{'vs SR /s':>14} | {'vs SR /s incl pilot':>20} | stuck | G0")
        pilot = np.array([r["cost"]["pilot_secs"] + r["cost"]["lll_secs"] for r in rs])
        for arm in ARMS:
            if not all(any(x["arm"] == arm for x in r["rows"]) for r in rs):
                continue
            rows = [next(x for x in r["rows"] if x["arm"] == arm) for r in rs]
            sr = [next(x for x in r["rows"] if x["arm"] == "SR") for r in rs]
            us = mean_se([x["us_per_step"] for x in rows])
            ei = mean_se([x["ess_min_per_iter"] for x in rows])
            es = mean_se([x["ess_min_ps"] for x in rows])
            ri = mean_se([a["ess_min_per_iter"] / b["ess_min_per_iter"] for a, b in zip(rows, sr)])
            rs_ = mean_se([a["ess_min_ps"] / b["ess_min_ps"] for a, b in zip(rows, sr)])
            if arm in ("SR-LLL", "SR-U"):
                inc = mean_se([(a["ess_min"] / (a["secs"] + p)) / b["ess_min_ps"]
                               for a, b, p in zip(rows, sr, pilot)])
                inc_s = f"{inc[0]:>9.2f} ({inc[1]:.2f})"
            else:
                inc_s = f"{'-':>20}"
            g0 = all(x["g0"] for x in rows)
            stuck = sum(1 for x in rows if x["ess_min"] == 0.0)
            out.append(f"  {arm:>8} {us[0]:>9.1f} {ei[0]:>10.2e} ({ei[1]:.1e}) {es[0]:>8.1f} ({es[1]:>4.1f}) | "
                       f"{ri[0]:>8.2f} ({ri[1]:.2f}) {rs_[0]:>8.2f} ({rs_[1]:.2f}) | {inc_s:>20} | {stuck:>2}/{len(rows)} | {'ok' if g0 else 'FAIL'}")
            if stuck:
                ok = [(a, b) for a, b in zip(rows, sr) if a["ess_min"] > 0.0]
                rn_i = mean_se([a["ess_min_per_iter"] / b["ess_min_per_iter"] for a, b in ok])
                rn_s = mean_se([a["ess_min_ps"] / b["ess_min_ps"] for a, b in ok])
                out.append(f"  {'':>8} over the {len(ok)} non-stuck seeds only: vs SR /iter {rn_i[0]:.2f} ({rn_i[1]:.2f}), "
                           f"vs SR /s {rn_s[0]:.2f} ({rn_s[1]:.2f})")
        g = [r["geom"] for r in rs]
        dec = [x["selector"]["chosen"] for x in g]
        out.append(f"  selector decision per seed: {dec}; surrogate gain over the PLB "
                   f"{[round(x['selector']['gain'], 3) for x in g]}")
        out.append(f"  obliqueness (corr-eigenvalue spread): PLB {np.mean([x['oblq_plb'] for x in g]):.2f}, "
                   f"after reduction {np.mean([x['oblq_reduced'] for x in g]):.2f}; "
                   f"max|V| {max(x['max_entry_V'] for x in g)}, max|U V| {max(x['max_entry_U'] for x in g)}; "
                   f"nnz/move PLB {g[0]['nnz_per_move_plb']:.1f}, reduced {np.mean([x['nnz_per_move_red'] for x in g]):.1f}; "
                   f"pilot+LLL {pilot.mean():.1f} s")
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
