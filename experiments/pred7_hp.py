"""Prediction 7: high-probability augmentation on sparse random configurations.

Protocol (manuscript Sec. 9, Prediction 7 / Proposition prop:hp): on (P5), an
ensemble of random binary configuration matrices with controlled row-support
density, the empirical fraction of random invertible partitions yielding an
augmenting PLB increases toward 1 as the support density s -> 0, tracking

    P(augmenting) >= 1 - sum_{k=2}^{n} C(n,k) C(r-n,k) s^{2k},

where s is the support density of the top block of U (fraction of nonzero
entries per row, maximized over rows).  "Augmenting" is certified by the
absence of a c0-Eulerian submatrix in sgn(U) (Hazelton 2024, Thm 4.4).

Design:
  * n=6 links, r=16 routes; A entries iid Bernoulli(p), p swept over
    {0.08,...,0.40}; draws rejected until A has no zero row/column and full
    row rank.
  * Partition: uniform rejection sampling over n-subsets until A_1 is
    invertible; the draw is a PLB iff U is integral (recorded; prop:hp
    conditions on integrality).  Entries beyond {0,+-1} are recorded as a
    separate stratum (Thm 4.4 needs only sgn(U); prop:hp's model is +-1).
  * Certification: c0_eulerian_witness(sgn U) is None.
  * Gate: the GF(2)-kernel checker is cross-validated against a direct
    brute-force submatrix search on a subsample of ensemble draws.

Outputs: results/pred7_numbers.txt, results/fig_pred7_hp.{pdf,png}.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibresampler.eulerian import c0_eulerian_witness, sgn
from fibresampler.fibre import build_plb

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

N_LINKS = 6
R_ROUTES = 16
DENSITIES = (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25,
             0.30, 0.40)
N_DRAWS = 300
SEED = 20260701
SUFFIX = ""

# --shape N R : larger ensembles thin the top block further (the max-row
# density floor scales like 1/n + spread), populating the sparse regime the
# n=6 shape cannot reach.  --suffix tags the output files.
def _parse_cli(argv):
    global N_LINKS, R_ROUTES, DENSITIES, SUFFIX
    if "--shape" in argv:
        i = argv.index("--shape")
        N_LINKS, R_ROUTES = int(argv[i + 1]), int(argv[i + 2])
    if "--densities" in argv:
        i = argv.index("--densities")
        vals = []
        for tok in argv[i + 1:]:
            try:
                vals.append(float(tok))
            except ValueError:
                break
        DENSITIES = tuple(vals)
    if "--suffix" in argv:
        SUFFIX = argv[argv.index("--suffix") + 1]

REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


# ---------------------------------------------------------------------------
# Brute-force cross-check of the c0 search (gate)
# ---------------------------------------------------------------------------
def c0_brute(S: np.ndarray) -> bool:
    """Direct search over all row and column subsets (canonical form: every
    column of the witness nonzero with zero sum, every row sum even)."""
    S = sgn(S)
    n, m = S.shape
    for rmask in range(1, 1 << n):
        I = [i for i in range(n) if (rmask >> i) & 1]
        if len(I) < 2:
            continue
        sub = S[I, :]
        cols = [j for j in range(m)
                if (sub[:, j] != 0).any() and sub[:, j].sum() == 0]
        for cmask in range(1, 1 << len(cols)):
            J = [cols[t] for t in range(len(cols)) if (cmask >> t) & 1]
            block = sub[:, J]
            if (np.abs(block).sum(axis=1) % 2 == 0).all():
                return True
    return False


def gate() -> None:
    say("=" * 72)
    say("GATE: GF(2) c0 checker vs brute force on known and random matrices")
    say("=" * 72)
    E1 = np.array([[1, 1], [-1, -1]])
    E3 = np.array([[0, 1, 1], [1, -1, 0], [-1, 0, -1]])
    NOC0 = np.array([[1, 1, 0], [0, 1, 1]])          # no zero-sum col subset
    fixed = [(E1, True), (E3, True), (NOC0, False)]
    ok = True
    for M, expect in fixed:
        got = c0_eulerian_witness(M) is not None
        brute = c0_brute(M)
        ok &= (got == expect == brute)
    say(f"  fixed cases (E1, E3, control): {'PASS' if ok else 'FAIL'}")

    rng = np.random.default_rng(7)
    agree = 0
    n_pos = 0
    for _ in range(60):
        M = rng.choice([-1, 0, 0, 1], size=(5, 7))
        a = c0_eulerian_witness(M) is not None
        b = c0_brute(M)
        agree += (a == b)
        n_pos += a
    say(f"  random 5x7 sign matrices: 60/60 agreement = {agree == 60} "
        f"({n_pos} contain a c0-Eulerian submatrix)")
    if not ok or agree != 60:
        say("GATE FAILURE: aborting.")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def draw_configuration(rng: np.random.Generator, p: float) -> np.ndarray:
    """Random binary A, iid Bernoulli(p) entries with every column conditioned
    on being nonzero (a route must use at least one link), no zero row, and
    full row rank."""
    for _ in range(2000):
        A = (rng.random((N_LINKS, R_ROUTES)) < p).astype(np.int64)
        for j in range(R_ROUTES):
            while A[:, j].sum() == 0:
                A[:, j] = (rng.random(N_LINKS) < p).astype(np.int64)
        if (A.sum(axis=1) == 0).any():
            continue
        if np.linalg.matrix_rank(A) == N_LINKS:
            return A
    raise RuntimeError(f"no full-rank draw at p={p}")


def random_invertible_partition(rng: np.random.Generator, A: np.ndarray):
    """Uniform over invertible n-subsets by rejection; None if none found."""
    for _ in range(300):
        cols1 = tuple(sorted(rng.choice(R_ROUTES, size=N_LINKS,
                                        replace=False).tolist()))
        if abs(np.linalg.det(A[:, cols1])) > 0.5:
            return cols1
    return None


def lower_bound(s: float) -> float:
    """Proposition prop:hp union bound, clamped at 0."""
    total = sum(comb(N_LINKS, k) * comb(R_ROUTES - N_LINKS, k) * s ** (2 * k)
                for k in range(2, N_LINKS + 1))
    return max(0.0, 1.0 - total)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z ** 2 / n
    cen = (ph + z ** 2 / (2 * n)) / den
    hw = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return (cen - hw, cen + hw)


def run_ensemble(quick: bool = False):
    densities = DENSITIES[:3] if quick else DENSITIES
    n_draws = 20 if quick else N_DRAWS
    rng = np.random.default_rng(SEED)
    records = []          # (p, s_max, s_mean, stratum, certified)
    for p in densities:
        for _ in range(n_draws):
            A = draw_configuration(rng, p)
            cols1 = random_invertible_partition(rng, A)
            if cols1 is None:
                records.append((p, np.nan, np.nan, "no-partition", False))
                continue
            U, info = build_plb(A, cols1)
            if not info["integral"]:
                records.append((p, np.nan, np.nan, "non-integral", False))
                continue
            T = U[list(cols1), :]                    # top block
            dens = (T != 0).mean(axis=1)
            stratum = "pm1" if np.isin(U, (-1, 0, 1)).all() else "big"
            cert = c0_eulerian_witness(U) is None
            records.append((p, float(dens.max()), float(dens.mean()),
                            stratum, cert))
    return records


def summarise(records):
    say("")
    say("=" * 72)
    say(f"ENSEMBLE: n={N_LINKS}, r={R_ROUTES}, {N_DRAWS} draws per density, "
        f"seed {SEED}")
    say("=" * 72)
    say("  by A-density p:")
    say("  p      draws  PLBs  pm1    mean s_max  certified   95% CI")
    by_p: dict[float, list] = {}
    for rec in records:
        by_p.setdefault(rec[0], []).append(rec)
    for p in sorted(by_p):
        rows = by_p[p]
        plbs = [r for r in rows if r[3] in ("pm1", "big")]
        pm1 = [r for r in plbs if r[3] == "pm1"]
        cert = sum(r[4] for r in plbs)
        smax = np.mean([r[1] for r in plbs]) if plbs else np.nan
        frac = cert / len(plbs) if plbs else np.nan
        lo, hi = wilson(cert, len(plbs))
        say(f"  {p:.2f}   {len(rows):4d}  {len(plbs):4d}  {len(pm1):4d}"
            f"   {smax:9.3f}   {frac:8.3f}   [{lo:.3f}, {hi:.3f}]")

    say("")
    say("  binned by measured top-block density s_max (all integral PLBs):")
    say("  s bin        n    certified  95% CI          prop:hp bound")
    plbs = [r for r in records if r[3] in ("pm1", "big")]
    edges = np.concatenate([np.arange(0.0, 0.30, 0.05),
                            np.arange(0.30, 1.05, 0.10)])
    bin_rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        rows = [r for r in plbs if a <= r[1] < b]
        if not rows:
            continue
        cert = sum(r[4] for r in rows)
        frac = cert / len(rows)
        lo, hi = wilson(cert, len(rows))
        lb = lower_bound((a + b) / 2)
        bin_rows.append(((a + b) / 2, len(rows), frac, lo, hi, lb))
        say(f"  [{a:.1f},{b:.1f})   {len(rows):4d}   {frac:8.3f}"
            f"   [{lo:.3f}, {hi:.3f}]   {lb:.3f}")
    return bin_rows


def verdict(bin_rows) -> bool:
    lows = [r for r in bin_rows if r[0] <= 0.35]
    highs = [r for r in bin_rows if r[0] >= 0.55]
    rises = (bool(lows) and bool(highs)
             and min(r[2] for r in lows) >= max(r[2] for r in highs))
    to_one = bool(lows) and lows[0][2] >= 0.95
    above = all(r[2] >= r[5] - 1e-9 for r in bin_rows if r[5] > 0)
    say("")
    say("=" * 72)
    say("VERDICT (Prediction 7)")
    say("=" * 72)
    say(f"  certified fraction rises as s decreases: {rises}")
    say(f"  lowest-density bin reaches >= 0.95: {to_one} "
        f"(value {lows[0][2]:.3f} at s ~ {lows[0][0]:.2f})" if lows else
        "  no low-density bin populated")
    say(f"  empirical fraction >= prop:hp bound wherever the bound is "
        f"positive: {above}")
    held = rises and to_one and above
    say(f"  PREDICTION 7: {'HELD' if held else 'FAILED'}")
    return held


def make_figure(bin_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    s = np.array([r[0] for r in bin_rows])
    f = np.array([r[2] for r in bin_rows])
    lo = np.array([r[3] for r in bin_rows])
    hi = np.array([r[4] for r in bin_rows])
    ax.errorbar(s, f, yerr=[f - lo, hi - f], fmt="o-", color="#0000a7",
                capsize=3, label="empirical certified fraction")
    ss = np.linspace(0.01, 1.0, 200)
    ax.plot(ss, [lower_bound(v) for v in ss], "--", color="#c1272d",
            label="Proposition prop:hp lower bound")
    ax.set_xlabel(r"top-block support density $s$ (max over rows)")
    ax.set_ylabel("fraction of PLBs certified augmenting")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"P5 ensemble: n={N_LINKS}, r={R_ROUTES}")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"fig_pred7_hp{SUFFIX}.{ext}", dpi=200)
    say(f"  figure written: results/fig_pred7_hp{SUFFIX}.pdf/.png")


def main():
    _parse_cli(sys.argv)
    quick = "--quick" in sys.argv
    gate()
    records = run_ensemble(quick=quick)
    bin_rows = summarise(records)
    verdict(bin_rows)
    out = f"pred7_numbers{SUFFIX}.txt"
    (RESULTS / out).write_text("\n".join(REPORT) + "\n")
    print(f"numbers written: results/{out}")
    try:
        make_figure(bin_rows)
    except Exception as exc:                      # no matplotlib on Hopper
        print(f"figure skipped ({exc}); regenerate locally from numbers.txt")


if __name__ == "__main__":
    main()
