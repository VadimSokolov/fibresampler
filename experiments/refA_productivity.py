"""Referee point A / J / K: is a reflective flight PRODUCTIVE, or do its segments cancel?

The referee's arithmetic: a reflective proposal chains K axis-aligned legs of total
length (cost) P = sum_s b_s.  Its *net* displacement is D = || sum_s b_s d^(s) ||_2, the
Euclidean length of the realised move in the free frame.  If D grows like P (coherent /
ballistic), one flight traverses a long fibre chord that single-ray cannot reach in one
step, and the extra walking work buys reach.  If D grows like sqrt(P) (diffusive), the
legs cancel: a K-leg flight costs ~K single rays but moves only sqrt(K) as far, so the
per-cost efficiency is 1/sqrt(K) -- strictly worse, and the no-go proposition holds.

This measures D vs P directly on P3 = the thin diagonal band {|s-t|<=w, s+t<=C} across
the four registered defects kappa in {2.15, 4.26, 7.08, 10.6}, using the ACTUAL sampler
primitives (fibresampler.reflective.ReflectiveSampler.cast/full_move/reflect), starting
from the uniform stationary law.  It reports, per kappa and per leg budget K:

  * mean path P(K) = mean sum_s b_s  (the cost / total unit steps),
  * rms net displacement D(K) = sqrt(E||Delta(s,t)||^2)  (the referee's ||sum b_s d^(s)||),
  * rms long-axis reach   D_long(K)  along sigma = s+t   (the slow, mixing-limiting axis),
  * rms short-axis reach  D_short(K) along s-t          (the fast, decoupled axis),
  * productivity D/P, and the fitted exponent gamma in D ~ P^gamma (pre-peak rising window).

Cost-matched control (the referee's exact comparison): against ONE K-leg reflective
flight we set K INDEPENDENT single-ray heat-bath moves composed as a walk, cost-matched
leg for leg, and report the reach ratio flight/walk.  Coherence wins iff this ratio grows
(~sqrt(K)) rather than sitting at 1.

Deterministic (seeded).  Output: results/refA_productivity.txt.  No fabrication: every
number is an average over seeded stationary starts on the enumerated band.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fibresampler.reflective import ReflectiveSampler
from fibresampler.synthetic import BandProblem

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "refA_productivity.txt")

BANDS = [(2, 12), (2, 24), (2, 40), (1, 30)]     # kappa ~ 2.15, 4.26, 7.08, 10.6
K_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24]          # number of legs in the flight
N_SAMPLES = 6000                                  # stationary starts per (kappa, K)
SEED = 20260812


def flight_to_walls(S, x0, d1, K):
    """Run up to K legs, EACH forced to its wall (deterministic billiard flight).

    Faithful to the sampler: uses S.cast to find the wall, S.full_move to advance,
    S.reflect for the tilted-wall lattice reflection, and stops early on a stuck leg
    (b==0) or a non-invertible reflection (reflect->None), exactly as the proposal
    would.  Returns (path, disp_free, legs_done) where disp_free = (ds, dt)."""
    x = x0.astype(np.int64).copy()
    d = d1.astype(np.int64).copy()
    cols2 = list(S.prob.plb_info["cols2"])
    x0f = x0[cols2].astype(np.int64)
    path = 0
    legs = 0
    for s in range(K):
        b, binding = S.cast(x, d)
        if b is None or b == 0:
            break
        x = x + b * S.full_move(d)
        path += int(b)
        legs += 1
        if s < K - 1:
            dn = S.reflect(d, binding)
            if dn is None:
                break
            d = dn
    disp = x[cols2].astype(np.int64) - x0f
    return path, disp, legs


def single_ray_walk(S, x0, K, rng):
    """K independent single-ray heat-bath moves composed as a walk (the cost-matched
    control: same number of legs, but each leg is an independent Gibbs single ray with
    a fresh uniform axis and a uniform landing point -- no coherence carried across).
    Returns (path, disp_free) with disp_free = (ds, dt)."""
    x = x0.astype(np.int64).copy()
    cols2 = list(S.prob.plb_info["cols2"])
    x0f = x0[cols2].astype(np.int64)
    twom = 2 * S.m
    path = 0
    for _ in range(K):
        d = S.dirs[int(rng.integers(twom))]
        bmax, _ = S.cast(x, d)
        if bmax is None or bmax == 0:
            continue
        b = int(rng.integers(0, bmax + 1))          # uniform on the ray incl. staying
        x = x + b * S.full_move(d)
        path += b
    disp = x[cols2].astype(np.int64) - x0f
    return path, disp


def local_slope(xs, ys):
    """Slope of log ys vs log xs over the first monotone (pre-peak, still-rising) window."""
    lx = np.log(np.asarray(xs, float))
    ly = np.log(np.asarray(ys, float))
    # keep the leading window where displacement is still rising (before the periodic decline)
    keep = [0]
    for i in range(1, len(ly)):
        if ly[i] > ly[keep[-1]] + 1e-3:
            keep.append(i)
        else:
            break
    if len(keep) < 2:
        return float("nan"), keep
    A = np.vstack([lx[keep], np.ones(len(keep))]).T
    slope = np.linalg.lstsq(A, ly[keep], rcond=None)[0][0]
    return float(slope), keep


def run_band(w, C):
    band = BandProblem(w, C)
    kap, L, a = band.kappa()
    S = ReflectiveSampler(band, bmax=max(K_GRID))
    states = band.states
    N = len(states)
    rng = np.random.default_rng(np.random.SeedSequence((SEED, w, C)))

    rows = []
    for K in K_GRID:
        starts = rng.integers(0, N, size=N_SAMPLES)
        dstart = rng.integers(0, 2 * S.m, size=N_SAMPLES)
        paths = np.empty(N_SAMPLES)
        d2 = np.empty(N_SAMPLES)           # ||Delta(s,t)||^2
        long2 = np.empty(N_SAMPLES)        # (Delta sigma)^2,  sigma=s+t
        short2 = np.empty(N_SAMPLES)       # (Delta delta)^2,  delta=s-t
        legs = np.empty(N_SAMPLES)
        wpath = np.empty(N_SAMPLES)        # cost-matched single-ray walk
        wlong2 = np.empty(N_SAMPLES)
        for i in range(N_SAMPLES):
            x0 = states[starts[i]]
            p, disp, nl = flight_to_walls(S, x0, S.dirs[dstart[i]], K)
            paths[i] = p
            ds, dt = int(disp[0]), int(disp[1])
            d2[i] = ds * ds + dt * dt
            long2[i] = (ds + dt) ** 2
            short2[i] = (ds - dt) ** 2
            legs[i] = nl
            wp, wdisp = single_ray_walk(S, x0, K, rng)
            wpath[i] = wp
            wlong2[i] = (int(wdisp[0]) + int(wdisp[1])) ** 2
        rows.append(dict(
            K=K, meanpath=paths.mean(), meanlegs=legs.mean(),
            D=np.sqrt(d2.mean()), Dlong=np.sqrt(long2.mean()),
            Dshort=np.sqrt(short2.mean()),
            walk_long=np.sqrt(wlong2.mean()), walk_path=wpath.mean(),
        ))
    gamma, keep = local_slope([r["meanpath"] for r in rows], [r["D"] for r in rows])
    gamma_long, keepl = local_slope([r["K"] for r in rows], [r["Dlong"] for r in rows])
    return dict(kappa=kap, L=L, a=a, N=N, rows=rows,
                gamma=gamma, keep=keep, gamma_long=gamma_long, keepl=keepl)


def main():
    out = []
    out.append("=" * 86)
    out.append("REFEREE POINT A/J/K -- productivity of the reflective flight on P3 (the band)")
    out.append(f"N_SAMPLES={N_SAMPLES} stationary (uniform) starts per (kappa,K), seed {SEED}")
    out.append("D = rms || sum_s b_s d^(s) ||_2 (net free-frame move);  P = mean sum_s b_s (cost)")
    out.append("sigma=s+t is the slow long axis; delta=s-t the fast short axis.")
    out.append("walk = cost-matched K independent single-ray heat-bath moves (no coherence).")
    out.append("=" * 86)
    results = []
    for (w, C) in BANDS:
        res = run_band(w, C)
        results.append((w, C, res))
        out.append("")
        out.append(f"kappa = {res['kappa']:.2f}   (diameter L={res['L']:.2f}, "
                   f"longest axis chord a={res['a']}, fibre N={res['N']})")
        out.append(f"  {'K':>3} {'legs':>5} {'path P':>8} {'D':>8} {'D_long':>8} "
                   f"{'D_short':>8} {'D/P':>6} {'walk_long':>9} {'flight/walk':>11}")
        for r in res["rows"]:
            ratio = r["Dlong"] / r["walk_long"] if r["walk_long"] > 1e-9 else float("nan")
            out.append(f"  {r['K']:>3d} {r['meanlegs']:>5.1f} {r['meanpath']:>8.2f} "
                       f"{r['D']:>8.2f} {r['Dlong']:>8.2f} {r['Dshort']:>8.2f} "
                       f"{r['D'] / r['meanpath']:>6.3f} "
                       f"{r['walk_long']:>9.2f} {ratio:>11.2f}")
        out.append(f"  fitted exponent gamma in D ~ P^gamma (pre-peak/rising): "
                   f"{res['gamma']:.3f}   [1.0=ballistic/coherent, 0.5=diffusive/cancelling]")
        out.append(f"  long-axis exponent  D_long ~ K^{res['gamma_long']:.3f}  "
                   f"(peaks near ~diameter L={res['L']:.1f} at K~kappa, then declines: billiard periodic)")

    out.append("")
    out.append("=" * 86)
    out.append("SUMMARY across the registered defect ladder")
    out.append("=" * 86)
    out.append(f"  {'kappa':>7} {'gamma(D~P)':>11} {'gamma_long':>11} "
               f"{'peakDlong':>10} {'reach/a':>8} {'max flight/walk':>15}")
    for (w, C, res) in results:
        satD = max(r["Dlong"] for r in res["rows"])
        reach_over_a = satD / res["a"]
        maxfw = max((r["Dlong"] / r["walk_long"] if r["walk_long"] > 1e-9 else 0.0)
                    for r in res["rows"])
        out.append(f"  {res['kappa']:>7.2f} {res['gamma']:>11.3f} "
                   f"{res['gamma_long']:>11.3f} {satD:>10.2f} {reach_over_a:>8.2f} "
                   f"{maxfw:>15.2f}")
    out.append("")
    out.append("READING: gamma near 1.0 => coherent flights (referee resolution (ii): the")
    out.append("reflective move is productive; its long-axis reach grows with the leg budget")
    out.append("and peaks near the fibre diameter at K~kappa then declines (periodic), which")
    out.append("single-ray cannot reach in one step).")
    out.append("gamma near 0.5 => diffusive cancellation (referee resolution (i): no-go).")
    out.append("flight/walk >> 1 quantifies the cost-matched coherence gain over single-ray.")

    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
