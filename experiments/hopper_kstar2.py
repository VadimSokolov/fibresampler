"""Parallel worker: search for an enumerable fibre with corridor height k*=2.

One SLURM array task per seed.  Scans random integer configuration matrices (and a
few structured 'bridge' designs), enumerates each fibre, and for every bottleneck
coordinate computes the corridor height with kstar_check.corridor_height.  Writes any
k*=2 hit (with full A, y, cols1, jstar) and the k* histogram to
results/kstar2_hits_<seed>.json.

Usage:  python3 experiments/hopper_kstar2.py <seed> <ntry>
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fibresampler.fibre import FibreProblem, enumerate_fibre
from kstar_check import corridor_height


def nlobes_fast(prob, jstar):
    """Cheap count of ground-stratum (x_jstar=0) lobes under flat moves only.
    Avoids building the full all-states adjacency; used to bail on the common
    single-lobe case before the expensive widest-path search."""
    states = prob.states
    D = [i for i in range(prob.fibre_size) if states[i][jstar] == 0]
    if len(D) < 2:
        return len(D)
    Dset = set(D)
    moves0 = [j for j in range(prob.n_moves) if prob.U[jstar, j] == 0]
    if not moves0:
        return len(D)
    parent = {i: i for i in D}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in D:
        for j in moves0:
            for k in prob.ray(i, j):
                if k in Dset and k != i:
                    parent[find(i)] = find(k)
    return len({find(i) for i in D})


def first_invertible_cols1(A):
    n_rank = int(np.linalg.matrix_rank(A.astype(float)))
    chosen, M = [], np.zeros((A.shape[0], 0))
    for j in range(A.shape[1]):
        cand = np.column_stack([M, A[:, j].astype(float)])
        if np.linalg.matrix_rank(cand) > M.shape[1]:
            M = cand
            chosen.append(j)
        if len(chosen) == n_rank:
            break
    return tuple(chosen), n_rank


def eval_candidate(A, y, hist, hits, maxfib=150):
    # cheap pre-filter: keep margins small so enumeration never blows up
    if y.max() > 6 or int(y.sum()) > 18:
        return
    # box upper bound on |fibre| WITHOUT enumerating: x_j <= min_{i:A_ij>0} y_i
    ub = 1
    for j in range(A.shape[1]):
        rowsj = [y[i] for i in range(A.shape[0]) if A[i, j] > 0]
        uj = min(rowsj) if rowsj else 0
        ub *= (int(uj) + 1)
        if ub > 1500:
            return
    if np.linalg.matrix_rank(A.astype(float)) < A.shape[0]:
        return
    cols1, rank = first_invertible_cols1(A)
    if rank < A.shape[0] or len(cols1) < A.shape[0]:
        return
    try:
        states = enumerate_fibre(A, y)
    except Exception:
        return
    if len(states) < 6 or len(states) > maxfib:
        return
    try:
        prob = FibreProblem(A, y, cols1=cols1, name="rand")
    except Exception:
        return
    for jstar in range(A.shape[1]):
        col = prob.states[:, jstar]
        if col.min() != 0 or col.max() == 0:
            continue
        # cheap bail: only multi-lobe ground strata can have k*>=2
        try:
            if nlobes_fast(prob, jstar) < 2:
                continue
            nlobes, pw = corridor_height(prob, jstar)
        except Exception:
            continue
        if nlobes < 2:
            continue
        ks = [pw[i][j] for i in range(nlobes) for j in range(nlobes) if i != j]
        kstar = max(ks)
        if kstar >= 10 ** 8:
            hist["disc"] += 1
            continue
        hist[int(kstar)] += 1
        if kstar == 2:
            hits.append({
                "A": A.tolist(), "y": [int(v) for v in y],
                "cols1": list(cols1), "jstar": int(jstar),
                "fibre": int(len(states)), "lobes": int(nlobes),
                "pairwise": pw,
            })


def structured_bridge(rng):
    """A structured family: a bottleneck column appearing in every row (a shared
    'bridge' edge) plus random light columns, which biases toward corridors that
    must climb the shared coordinate."""
    n = int(rng.integers(2, 4))
    r = int(rng.integers(n + 2, n + 5))
    A = rng.integers(0, 3, size=(n, r))
    A[:, 0] = 1                       # column 0 is the shared bridge (all rows)
    base = rng.integers(0, 3, size=r)
    return A, A @ base


def _flush(out, seed, ntry, done, hist, hits):
    """Write current progress atomically, so a SLURM timeout still leaves the
    histogram and any hits found so far."""
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"seed": seed, "ntry": ntry, "done": done,
                   "hist": {str(k): v for k, v in hist.items()},
                   "n_hits": len(hits), "hits": hits[:5]}, fh, indent=1)
    os.replace(tmp, out)


def scan(seed, ntry, out):
    rng = np.random.default_rng(seed)
    hist = Counter()
    hits = []
    for t in range(ntry):
        if t % 3 == 0:
            A, y = structured_bridge(rng)
        else:
            n = int(rng.integers(2, 4))
            r = int(rng.integers(n + 2, n + 6))
            hi = int(rng.integers(3, 5))          # entries 0..hi-1 (up to 0..3)
            A = rng.integers(0, hi, size=(n, r))
            base = rng.integers(0, 3, size=r)
            y = A @ base
        nhits_before = len(hits)
        eval_candidate(A, y, hist, hits)
        # flush every 300 tries, and immediately whenever a hit is recorded
        if t % 300 == 299 or len(hits) > nhits_before:
            _flush(out, seed, ntry, t + 1, hist, hits)
        if len(hits) >= 25:
            break
    _flush(out, seed, ntry, ntry, hist, hits)
    return hist, hits


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ntry = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"kstar2_hits_{seed}.json")
    hist, hits = scan(seed, ntry, out)
    print(f"seed={seed}: hist={dict(hist)} hits={len(hits)} -> {os.path.basename(out)}")
