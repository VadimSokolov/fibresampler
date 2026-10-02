"""Prediction 6: the balanced extension is real (P4).

Protocol (manuscript Sec. 9, Prediction 6):
  * (P4) is a balanced-but-not-totally-unimodular configuration.  The
    c0-Eulerian check on the original U certifies an augmenting PLB, and the
    resulting sampler exhibits path length <= r-n and no zig-zag stalls,
    matching TU behaviour despite A not being TU.
  * A PLB failing the check should exhibit the zig-zag pathology of
    Hazelton (2024), Example 5.

Stages:
  0. Gates: the Eulerian/balanced/TU checkers reproduce every worked example
     in Hazelton (2024) (E1/E2/E3, Ex.8 printed U + its c0 witness, Ex.9
     printed U + absence, Ex.10 signing, Ex.2 TU).  HALT on any mismatch.
  1. P4 construction and certificates: corridor grid with parallel one-way
     streets; A balanced (exhaustive), not TU (near-pencil minor det -2);
     census of all {0,+-1} PLBs by the c0 check; featured passing PLB (mixed
     signs) and failing PLB.
  2. Positive arm: on fibres i*y0 (i=1,2,3) the passing PLB has all-pairs
     ray-move diameter <= r-n and a uniform/Poisson spectral gap bounded away
     from 0 (flat in the fibre scale), matching the TU control (P1, Ex.9 PLB).
  3. Negative arm (literature-anchored): Hazelton Ex.5 on P1's A with the
     unreordered PLB (Ex.8): shortest walk between the two named states is
     3M, diameter grows linearly, uniform gap collapses; the augmenting Ex.9
     PLB on the SAME fibres keeps diameter <= 4 and a flat gap.  At
     y=(0,1,1,0) the Ex.8 PLB disconnects the fibre (not a Markov basis).
  4. Necessity probe (informative, beyond the registered prediction): for
     every failing PLB on the balanced P4, search (a) column signings for a
     cr0-Eulerian certificate (Hazelton Thm 4.6) and (b) a menu of small
     fibres for an actual augmentation failure (diameter > r-n or
     disconnection).  This measures whether the c0 check has false alarms on
     the balanced class, where Hazelton's converse (Cor. 4.7) is not proved.

Outputs: results/pred6_numbers.txt, results/fig_pred6_balanced.{pdf,png}.
"""

from __future__ import annotations

import sys
from collections import deque
from itertools import combinations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibresampler.eulerian import (c0_eulerian_witness, cr0_eulerian_witness,
                                   is_balanced, is_totally_unimodular,
                                   signing_with_cr0)
from fibresampler.fibre import FibreProblem, build_plb, ray_endpoints
from fibresampler.problems import (A_P1, P1_COLS1, P1_FREE_ORDER,
                                   U_P1_REFERENCE)
from fibresampler.spectral import log_weights, normalised_pi, slem, \
    sr_transition_matrix

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


# ---------------------------------------------------------------------------
# Light problem container (states supplied directly, no generic enumeration)
# ---------------------------------------------------------------------------
class RawProblem:
    """Duck-typed stand-in for FibreProblem when states are enumerated by a
    problem-specific routine (the generic DFS is slow on the 11-column P4)."""

    def __init__(self, states: np.ndarray, U: np.ndarray, name: str = ""):
        self.states = np.asarray(states, dtype=np.int64)
        self.U = np.asarray(U, dtype=np.int64)
        self.index = {tuple(s): i for i, s in enumerate(self.states)}
        self.fibre_size = len(self.states)
        self.n_moves = self.U.shape[1]
        self.name = name

    def ray(self, i: int, j: int):
        x = self.states[i]
        u = self.U[:, j]
        b_min, b_max = ray_endpoints(x, u)
        out = []
        for b in range(b_min, b_max + 1):
            k = self.index.get(tuple(x + b * u))
            if k is not None:
                out.append(k)
        return out


# ---------------------------------------------------------------------------
# P4: corridor grid with parallel one-way streets
# ---------------------------------------------------------------------------
# Links: L1 trunk; mainline L2,L3,L4; stage-2 parallels L5,L7; stage-3
# bypass L6.  Every stage cycle is a pair of parallel links (length 2), so
# all cycles are even.  Routes: one single-link trip per link (columns 0-6)
# plus four compound trips whose pairwise overlaps are exactly the trunk:
#   r1 mainline {1,2,3,4}, r2 short {1,2}, r3 {1,5,3}, r4 {1,7,6,4}.
# On rows (L1..L4) the compound columns realize the near-pencil matrix
#   B = [[1,1,1,1],[1,1,0,0],[1,0,1,0],[1,0,0,1]]  (det -2, balanced),
# so A is balanced but not TU.
COMPOUND = np.array([
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 1],
], dtype=np.int64).T                      # (7 links, 4 compound routes)

A_P4 = np.hstack([np.eye(7, dtype=np.int64), COMPOUND])
N4, R4 = A_P4.shape                       # n=7, r=11, r-n=4
Y0_P4 = A_P4 @ np.ones(R4, dtype=np.int64)   # (5,3,3,3,2,2,2)


def enumerate_p4_fibre(y: np.ndarray) -> np.ndarray:
    """Exact fibre of A_P4: the 7 unit-route counts are determined by the 4
    compound counts, x_unit = y - C x_c >= 0, so enumerate the compound box."""
    y = np.asarray(y, dtype=np.int64)
    caps = [int(min(y[COMPOUND[:, j] > 0])) for j in range(4)]
    states = []
    for xc in product(*(range(c + 1) for c in caps)):
        xu = y - COMPOUND @ np.array(xc, dtype=np.int64)
        if (xu >= 0).all():
            states.append(tuple(xu) + tuple(xc))
    return np.array(sorted(states), dtype=np.int64)


def p4_problem(y: np.ndarray, cols1) -> RawProblem:
    U, info = build_plb(A_P4, cols1)
    assert info["integral"]
    return RawProblem(enumerate_p4_fibre(y), U, name=f"P4 cols1={cols1}")


# ---------------------------------------------------------------------------
# Graph diagnostics on the ray-move graph
# ---------------------------------------------------------------------------
def adjacency(prob) -> list[set]:
    adj: list[set] = [set() for _ in range(prob.fibre_size)]
    for i in range(prob.fibre_size):
        for j in range(prob.n_moves):
            for k in prob.ray(i, j):
                if k != i:
                    adj[i].add(k)
    return adj


def bfs_dist(adj: list[set], src: int) -> np.ndarray:
    dist = np.full(len(adj), -1, dtype=np.int64)
    dist[src] = 0
    q = deque([src])
    while q:
        v = q.popleft()
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                q.append(w)
    return dist


def graph_summary(prob):
    """(diameter over reachable pairs, number of components, adjacency)."""
    adj = adjacency(prob)
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    comps = 0
    diam = 0
    for s in range(n):
        d = bfs_dist(adj, s)
        if not seen[s]:
            comps += 1
            seen[d >= 0] = True
        m = int(d.max())
        if m > diam:
            diam = m
    return diam, comps, adj


def uniform_gap(prob) -> float:
    lw = log_weights(prob, "uniform")
    Q = sr_transition_matrix(prob, lw)
    _, gap, _ = slem(Q, normalised_pi(lw))
    return gap


def poisson_gap(prob, theta) -> float:
    lw = log_weights(prob, "poisson", theta=theta)
    Q = sr_transition_matrix(prob, lw)
    _, gap, _ = slem(Q, normalised_pi(lw))
    return gap


# ---------------------------------------------------------------------------
# Stage 0: gates
# ---------------------------------------------------------------------------
def stage0() -> None:
    say("=" * 72)
    say("STAGE 0: checker gates against Hazelton (2024) worked examples")
    say("=" * 72)
    E1 = np.array([[1, 1], [-1, -1]])
    E2 = np.array([[1, -1], [-1, 1]])
    E3 = np.array([[0, 1, 1], [1, -1, 0], [-1, 0, -1]])
    gates = []
    gates.append(("E1 c0-Eulerian, not cr0",
                  c0_eulerian_witness(E1) is not None
                  and cr0_eulerian_witness(E1) is None))
    gates.append(("E2 cr0-Eulerian",
                  cr0_eulerian_witness(E2) is not None))
    gates.append(("E3 c0 but not cr0; some signing is cr0",
                  c0_eulerian_witness(E3) is not None
                  and cr0_eulerian_witness(E3) is None
                  and signing_with_cr0(E3) is not None))

    U8, info8 = build_plb(A_P1, cols1=(0, 1, 2, 3))
    U8_print = np.array([[0, 1, -1, 1], [1, -1, 0, 0], [-1, 0, 0, -1],
                         [0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 1, 0], [0, 0, 0, 1]])
    w8 = c0_eulerian_witness(U8)
    gates.append(("Ex.8 U reproduces printed matrix",
                  np.array_equal(U8, U8_print) and info8["integral"]))
    gates.append(("Ex.8 c0 witness at rows(1,2,3) x cols(1,2,4)",
                  w8 == ([0, 1, 2], [0, 1, 3])))
    gates.append(("Ex.8 admits a Thm-4.6 signing certificate",
                  signing_with_cr0(U8) is not None))

    U9, _ = build_plb(A_P1, cols1=P1_COLS1, free_order=P1_FREE_ORDER)
    gates.append(("Ex.9 U reproduces printed matrix (eq. 9)",
                  np.array_equal(U9, U_P1_REFERENCE)))
    gates.append(("Ex.9 U passes the c0 check",
                  c0_eulerian_witness(U9) is None))
    gates.append(("Ex.2 A is TU (paper, Ex.10 remark)",
                  is_totally_unimodular(A_P1)[0]))
    B = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])
    tu_b = is_totally_unimodular(B)
    gates.append(("near-pencil B balanced, not TU (det +-2)",
                  is_balanced(B)[0] and not tu_b[0] and abs(tu_b[1][2]) == 2))

    ok = True
    for name, passed in gates:
        say(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    if not ok:
        say("GATE FAILURE: aborting before any Prediction-6 measurement.")
        raise SystemExit(1)
    say("  all gates pass")


# ---------------------------------------------------------------------------
# Stage 1: P4 certificates and PLB census
# ---------------------------------------------------------------------------
def stage1():
    say("")
    say("=" * 72)
    say("STAGE 1: P4 (corridor grid, balanced not TU) and its PLB census")
    say("=" * 72)
    bal, _ = is_balanced(A_P4)
    tu, tw = is_totally_unimodular(A_P4)
    say(f"  A_P4: {N4} links x {R4} routes; balanced = {bal}; TU = {tu}"
        + ("" if tu else
           f" (minor det {tw[2]} on rows {tw[0]} x cols {tw[1]})"))
    assert bal and not tu, "P4 must be balanced and not TU"

    census = {"pass": [], "fail": [], "big": 0, "nonPLB": 0, "singular": 0}
    for cols1 in combinations(range(R4), N4):
        if abs(np.linalg.det(A_P4[:, cols1])) < 0.5:
            census["singular"] += 1
            continue
        U, info = build_plb(A_P4, cols1)
        if not info["integral"]:
            census["nonPLB"] += 1
            continue
        if not np.isin(U, (-1, 0, 1)).all():
            census["big"] += 1
            continue
        T = U[list(cols1), :]
        mixed = bool((T > 0).any() and (T < 0).any())
        target = census["pass" if c0_eulerian_witness(U) is None else "fail"]
        target.append((cols1, mixed))
    say(f"  partitions: {census['singular']} singular, "
        f"{census['nonPLB']} non-integral, {census['big']} with |entries|>1,")
    say(f"              {len(census['pass'])} {{0,+-1}} PLBs PASS the c0 check, "
        f"{len(census['fail'])} FAIL it")

    cols_pass = next(c for c, m in census["pass"] if m)
    U_pass, _ = build_plb(A_P4, cols_pass)
    bal_u, _ = is_balanced(U_pass)
    say(f"  featured passing PLB: A1 columns {cols_pass} (mixed-sign top block)")
    say(f"    sgn(U)=U balanced: {bal_u}; c0-Eulerian submatrix: none "
        f"(Thm~4.4 / thm:balanced certificate)")
    say("    U^T rows (moves):")
    for j in range(U_pass.shape[1]):
        say(f"      u{j + 1} = {U_pass[:, j].tolist()}")

    cols_fail = census["fail"][0][0]
    U_fail, _ = build_plb(A_P4, cols_fail)
    wit = c0_eulerian_witness(U_fail)
    sig = signing_with_cr0(U_fail)
    say(f"  featured failing PLB: A1 columns {cols_fail}")
    say(f"    c0 witness rows {wit[0]} x cols {wit[1]}:")
    for row in U_fail[np.ix_(wit[0], wit[1])]:
        say(f"      {row.tolist()}")
    say(f"    Thm-4.6 signing certificate: "
        f"{'sigma=' + str(sig[0].tolist()) if sig else 'none'}")
    return census, cols_pass, cols_fail


# ---------------------------------------------------------------------------
# Stage 2: positive arm
# ---------------------------------------------------------------------------
def stage2(cols_pass):
    say("")
    say("=" * 72)
    say("STAGE 2: positive arm; passing PLB on P4 vs TU control (P1, Ex.9)")
    say("=" * 72)
    rows = []
    say(f"  P4, passing PLB, r-n = {R4 - N4}: fibres y = i*y0, y0 = "
        f"{Y0_P4.tolist()}")
    for i in (1, 2, 3):
        prob = p4_problem(i * Y0_P4, cols_pass)
        dim = np.linalg.matrix_rank(prob.states - prob.states[0])
        diam, comps, _ = graph_summary(prob)
        g_u = uniform_gap(prob)
        g_p = poisson_gap(prob, np.ones(R4))
        rows.append(("P4-pass", i, prob.fibre_size, diam, comps, g_u, g_p))
        say(f"    i={i}: |F|={prob.fibre_size:4d} (dim {dim}), "
            f"diameter={diam}, components={comps}, "
            f"gap_unif={g_u:.4f}, gap_Pois(1)={g_p:.4f}")
    assert all(r[3] <= R4 - N4 and r[4] == 1 for r in rows), \
        "positive arm: diameter must stay <= r-n on a connected fibre"

    say("  TU control: P1 (Hazelton Ex.2) with the augmenting Ex.9 PLB")
    y_p1 = np.array([4, 4, 2, 2])
    for i in (1, 2, 3):
        prob = FibreProblem(A_P1, i * y_p1, cols1=P1_COLS1,
                            free_order=P1_FREE_ORDER)
        diam, comps, _ = graph_summary(prob)
        g_u = uniform_gap(prob)
        rows.append(("P1-TU", i, prob.fibre_size, diam, comps, g_u, np.nan))
        say(f"    i={i}: |F|={prob.fibre_size:4d}, diameter={diam}, "
            f"components={comps}, gap_unif={g_u:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Stage 3: negative arm (Hazelton Ex.5 zig-zag)
# ---------------------------------------------------------------------------
def stage3():
    say("")
    say("=" * 72)
    say("STAGE 3: negative arm; Hazelton Ex.5 zig-zag on the failing Ex.8 PLB")
    say("=" * 72)
    say("  y=(1,M,M,1); x_M=(1,0,0,1,M,M,0,0) -> x'_M=(1,0,0,1,0,0,0,M);")
    say("  Ex.5: shortest walk with the Ex.8 PLB has length 3M "
        "(repeats -u2-u1+u4 M times)")
    rows = []
    for M in (1, 2, 3, 4, 6, 8):
        y = np.array([1, M, M, 1])
        pf = FibreProblem(A_P1, y, cols1=(0, 1, 2, 3))          # Ex.8 PLB
        pa = FibreProblem(A_P1, y, cols1=P1_COLS1,
                          free_order=P1_FREE_ORDER)             # Ex.9 PLB
        xM = (1, 0, 0, 1, M, M, 0, 0)
        xP = (1, 0, 0, 1, 0, 0, 0, M)
        assert xM in pf.index and xP in pf.index, "named states not on fibre"
        adj_f = adjacency(pf)
        d_named = int(bfs_dist(adj_f, pf.index[xM])[pf.index[xP]])
        diam_f, comps_f, _ = graph_summary(pf)
        diam_a, comps_a, _ = graph_summary(pa)
        g_f = uniform_gap(pf)
        g_a = uniform_gap(pa)
        rows.append((M, pf.fibre_size, d_named, diam_f, comps_f, g_f,
                     diam_a, comps_a, g_a))
        say(f"    M={M}: |F|={pf.fibre_size:4d} | Ex.8 PLB: "
            f"dist(x_M,x'_M)={d_named} (3M={3 * M}), diam={diam_f}, "
            f"gap={g_f:.5f} | Ex.9 PLB: diam={diam_a}, gap={g_a:.4f}")
        assert d_named == 3 * M, "Ex.5 shortest-walk length must equal 3M"
        assert comps_f == 1 and comps_a == 1

    # thin fibre: Ex.8 PLB is not even a Markov basis
    y_thin = np.array([0, 1, 1, 0])
    pf = FibreProblem(A_P1, y_thin, cols1=(0, 1, 2, 3))
    pa = FibreProblem(A_P1, y_thin, cols1=P1_COLS1, free_order=P1_FREE_ORDER)
    _, comps_f, adj_f = graph_summary(pf)
    _, comps_a, _ = graph_summary(pa)
    x1 = (0, 0, 0, 0, 1, 1, 0, 0)
    x2 = (0, 0, 0, 0, 0, 0, 0, 1)
    split = bfs_dist(adjacency(pf), pf.index[x1])[pf.index[x2]] < 0
    say(f"  thin fibre y=(0,1,1,0), |F|={pf.fibre_size}: Ex.8 PLB components="
        f"{comps_f} (named pair disconnected: {split}); "
        f"Ex.9 PLB components={comps_a}")
    assert comps_f > 1 and split and comps_a == 1
    return rows


# ---------------------------------------------------------------------------
# Stage 4: necessity probe on the balanced class
# ---------------------------------------------------------------------------
def fibre_menu(rng: np.random.Generator, n_menu: int = 30,
               size_cap: int = 600) -> list[np.ndarray]:
    """Small full-dimensional P4 fibres: y = A x* for sparse-ish random x*."""
    menu, seen = [], set()
    while len(menu) < n_menu:
        x_star = rng.integers(0, 3, size=R4)
        y = A_P4 @ x_star
        key = tuple(y)
        if key in seen:
            continue
        seen.add(key)
        states = enumerate_p4_fibre(y)
        if not (2 <= len(states) <= size_cap):
            continue
        if np.linalg.matrix_rank(states - states[0]) < R4 - N4:
            continue
        menu.append(y)
    return menu


def stage4(census):
    say("")
    say("=" * 72)
    say("STAGE 4: necessity probe; do failing PLBs on the balanced P4 "
        "actually fail?")
    say("=" * 72)
    rng = np.random.default_rng(20260701)
    menu = fibre_menu(rng)
    say(f"  fibre menu: {len(menu)} distinct full-dimensional fibres, "
        f"sizes {min(len(enumerate_p4_fibre(y)) for y in menu)}-"
        f"{max(len(enumerate_p4_fibre(y)) for y in menu)}")

    n_sign = 0
    n_fib = 0
    neither = []
    for cols1, _ in census["fail"]:
        U, _ = build_plb(A_P4, cols1)
        has_sign = signing_with_cr0(U) is not None
        n_sign += has_sign
        witness = None
        for y in menu:
            prob = RawProblem(enumerate_p4_fibre(y), U)
            diam, comps, _ = graph_summary(prob)
            if comps > 1 or diam > R4 - N4:
                witness = (y, diam, comps)
                break
        if witness is not None:
            n_fib += 1
        elif not has_sign:
            neither.append(cols1)
    total = len(census["fail"])
    say(f"  of {total} failing PLBs: {n_fib} exhibit an augmentation failure "
        f"on the menu")
    say(f"  (diameter > r-n or disconnection); {n_sign} carry a Thm-4.6 "
        f"signing certificate;")
    say(f"  {len(neither)} have neither: {neither if neither else '(none)'}")

    # and the passing PLBs never fail anywhere on the menu
    clean = True
    for cols1, _ in census["pass"]:
        U, _ = build_plb(A_P4, cols1)
        for y in menu:
            prob = RawProblem(enumerate_p4_fibre(y), U)
            diam, comps, _ = graph_summary(prob)
            if comps > 1 or diam > R4 - N4:
                clean = False
                say(f"  UNEXPECTED: passing PLB {cols1} fails at y={y}")
    say(f"  all {len(census['pass'])} passing PLBs stay augmenting "
        f"(diam <= {R4 - N4}, connected) on the menu: {clean}")
    return n_fib, n_sign, neither, clean


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(pos_rows, neg_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    Ms = [r[0] for r in neg_rows]
    ax1.plot(Ms, [r[2] for r in neg_rows], "o-", color="#c1272d",
             label="failing PLB (Ex.8): dist$(x_M,x'_M)$")
    ax1.plot(Ms, [3 * m for m in Ms], "--", color="grey", lw=1,
             label="$3M$ (Ex.5 prediction)")
    ax1.plot(Ms, [r[6] for r in neg_rows], "s-", color="#0000a7",
             label="augmenting PLB (Ex.9): diameter")
    ax1.axhline(4, color="#0000a7", ls=":", lw=1)
    ax1.text(Ms[-1], 4.35, r"$r-n=4$", color="#0000a7", ha="right",
             fontsize=9)
    ax1.set_xlabel(r"$M$  (fibre $y=(1,M,M,1)$)")
    ax1.set_ylabel("ray-move path length")
    ax1.set_title("Zig-zag pathology vs augmenting PLB (P1)")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.semilogy(Ms, [r[5] for r in neg_rows], "o-", color="#c1272d",
                 label="failing PLB (Ex.8), uniform gap")
    ax2.semilogy(Ms, [r[8] for r in neg_rows], "s-", color="#0000a7",
                 label="augmenting PLB (Ex.9), uniform gap")
    p4 = [r for r in pos_rows if r[0] == "P4-pass"]
    ax2.semilogy([r[1] * 2 for r in p4], [r[5] for r in p4], "d-",
                 color="#008176",
                 label="P4 (balanced, not TU), passing PLB")
    ax2.set_xlabel(r"fibre scale ($M$, or $2i$ for P4 at $i\,y_0$)")
    ax2.set_ylabel("uniform spectral gap")
    ax2.set_title("Gap: collapse without augmentation, flat with it")
    ax2.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"fig_pred6_balanced.{ext}", dpi=200)
    say(f"  figure written: results/fig_pred6_balanced.pdf/.png")


def main():
    stage0()
    census, cols_pass, cols_fail = stage1()
    pos_rows = stage2(cols_pass)
    neg_rows = stage3()
    n_fib, n_sign, neither, clean = stage4(census)

    say("")
    say("=" * 72)
    say("VERDICT (Prediction 6)")
    say("=" * 72)
    p4_ok = all(r[3] <= R4 - N4 and r[4] == 1 for r in pos_rows
                if r[0] == "P4-pass")
    gaps = [r[5] for r in pos_rows if r[0] == "P4-pass"]
    say(f"  positive arm: on balanced-not-TU P4 the c0-certified PLB keeps "
        f"diameter <= {R4 - N4}")
    say(f"    on all scaled fibres, connected, uniform gap "
        f"{min(gaps):.4f}-{max(gaps):.4f} (TU control "
        f"{[f'{r[5]:.4f}' for r in pos_rows if r[0] == 'P1-TU']})"
        f" -> {'HELD' if p4_ok else 'FAILED'}")
    zig = all(r[2] == 3 * r[0] for r in neg_rows)
    say(f"  negative arm: failing PLB walks 3M (zig-zag) with gap collapsing "
        f"{neg_rows[0][5]:.4f} -> {neg_rows[-1][5]:.5f};")
    say(f"    same fibres under the augmenting PLB: diameter <= 4, gap flat "
        f"{neg_rows[0][8]:.3f}-{neg_rows[-1][8]:.3f} -> "
        f"{'HELD' if zig else 'FAILED'}")
    say(f"  necessity probe: {n_fib}/{len(census['fail'])} failing PLBs "
        f"exhibit real failures on the fibre menu; {n_sign} carry Thm-4.6 "
        f"certificates; unresolved: {len(neither)}")
    say(f"  PREDICTION 6: {'HELD' if (p4_ok and zig) else 'FAILED'}")

    (RESULTS / "pred6_numbers.txt").write_text("\n".join(REPORT) + "\n")
    print("numbers written: results/pred6_numbers.txt")
    try:
        make_figure(pos_rows, neg_rows)
    except Exception as exc:                      # no matplotlib on Hopper
        print(f"figure skipped ({exc}); regenerate locally from numbers.txt")


if __name__ == "__main__":
    main()
