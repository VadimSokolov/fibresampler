"""Corridor-height check: pairwise k(i,j) = min over paths D_i -> D_j of the
peak x_{j*} en route, for EVERY ordered pair of lobes; k* = max over pairs.

Because min-max path heights concatenate through intermediate lobes, all
pairwise heights <= h iff the height-<=h corridor graph on the lobes is
connected, so k* here equals the bipartition-based corridor height of the
manuscript (Section 5.4), and k* = 1 verifies both the height-one linkage
hypothesis of Theorem prop:loading-obstruction and the k* = 1 claim of the
spectrum-law specialisation.  Same widest-path search as taskV_hardening run
3, run from every source lobe to every target lobe (no early stop).

Also checks the border-drain hypothesis of prop:loading-obstruction: every
state with x_{j*} >= 1 reaches the stratum D by a never-raising path whose
steps are constant-height moves (u_{j*} = 0 rays) or jumps to the minimal-
x_{j*} state of a u_{j*} != 0 ray with strictly smaller x_{j*} (the two step
kinds with Theta(1) selection probability as lambda_{j*} -> 0); reports the
maximal path length d_B and any stuck states.

Output: results/kstar_check.txt.  Deterministic (no RNG).
"""
import heapq
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fibresampler.problems import P1, table_2x3, TABLE23_BOTTLENECK

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "kstar_check.txt")


def corridor_height(prob, jstar):
    """Return (n_lobes, [k from each source lobe]) for the given problem."""
    states = prob.states
    N = prob.fibre_size

    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(prob.n_moves):
            for k in prob.ray(i, j):
                if k != i:
                    adj[i].add(k)

    D = [i for i in range(N) if states[i][jstar] == 0]
    Dset = set(D)
    moves0 = [j for j in range(prob.n_moves) if prob.U[jstar, j] == 0]
    adjD = {i: set() for i in D}
    for i in D:
        for j in moves0:
            for k in prob.ray(i, j):
                if k in Dset and k != i:
                    adjD[i].add(k)

    lobe = {i: i for i in D}

    def find(a):
        while lobe[a] != a:
            lobe[a] = lobe[lobe[a]]
            a = lobe[a]
        return a

    for i in D:
        for k in adjD[i]:
            lobe[find(i)] = find(k)
    comps = {}
    for i in D:
        comps.setdefault(find(i), []).append(i)
    lobe_id = {i: c for c, mem in enumerate(comps.values()) for i in mem}
    nlobes = len(comps)

    INF = 10 ** 9
    pairwise = []
    for src in range(nlobes):
        best = {i: INF for i in range(N)}
        pq = []
        for i in D:
            if lobe_id[i] == src:
                best[i] = 0
                heapq.heappush(pq, (0, i))
        while pq:
            h, i = heapq.heappop(pq)
            if h > best[i]:
                continue
            for k in adj[i]:
                nh = max(h, int(states[k][jstar]))
                if nh < best[k]:
                    best[k] = nh
                    heapq.heappush(pq, (nh, k))
        row = [min((best[i] for i in D if lobe_id[i] == tgt), default=INF)
               for tgt in range(nlobes)]
        pairwise.append(row)
    return nlobes, pairwise


def drain_check(prob, jstar):
    """Return (n_stuck, d_B): BFS from every border state over never-raising
    Theta(1)-probability steps (constant-height moves, or ray-minimum jumps
    with strictly smaller x_{j*}) until the stratum D is reached."""
    states = prob.states
    N = prob.fibre_size
    U = prob.U
    dirs_move = [j for j in range(prob.n_moves) if U[jstar, j] != 0]
    dirs_flat = [j for j in range(prob.n_moves) if U[jstar, j] == 0]

    def cheap_neighbours(i):
        out = set()
        for j in dirs_flat:
            out.update(k for k in prob.ray(i, j) if k != i)
        for j in dirs_move:
            m = min(prob.ray(i, j), key=lambda k: states[k][jstar])
            if states[m][jstar] < states[i][jstar]:
                out.add(m)
        return out

    n_stuck, d_B = 0, 0
    for s in range(N):
        if states[s][jstar] == 0:
            continue
        seen = {s}
        q = deque([(s, 0)])
        found = None
        while q:
            cur, d = q.popleft()
            if states[cur][jstar] == 0:
                found = d
                break
            for nb in cheap_neighbours(cur):
                if nb not in seen and states[nb][jstar] <= states[cur][jstar]:
                    seen.add(nb)
                    q.append((nb, d + 1))
        if found is None:
            n_stuck += 1
        else:
            d_B = max(d_B, found)
    return n_stuck, d_B


def main():
    lines = []
    lines.append("corridor-height check: pairwise k(i,j) = min over paths of max x_{j*};"
                 " k* = max over pairs")
    lines.append("=" * 70)
    kstars, drains = [], []
    for name, prob, jstar in [
        ("P1 road network", P1(), 4),
        ("2x3 table", table_2x3(), TABLE23_BOTTLENECK),
    ]:
        nlobes, pw = corridor_height(prob, jstar)
        kstar = max(pw[i][j] for i in range(nlobes) for j in range(nlobes)
                    if i != j)
        kstars.append(kstar)
        n_stuck, d_B = drain_check(prob, jstar)
        drains.append((n_stuck, d_B))
        lines.append(f"{name}: fibre={prob.fibre_size}, lobes={nlobes}, "
                     f"pairwise k(i,j) matrix = {pw}, k* = {kstar}; "
                     f"drain: stuck={n_stuck}, d_B={d_B}")
    ok = all(k == 1 for k in kstars) and all(s == 0 for s, _ in drains)
    verdict = ("k*=1 on BOTH problems (all pairs) and border drains with no "
               "stuck states: the height-one linkage and drain hypotheses of "
               "prop:loading-obstruction and the k*=1 specialisation of "
               "thm:spectrum are verified directly"
               if ok else
               "hypothesis FAILS somewhere: adjust theorem hypothesis")
    lines.append(verdict)
    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
