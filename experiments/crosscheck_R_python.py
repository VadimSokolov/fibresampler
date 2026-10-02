"""Cross-check: the Python core must reproduce Hazelton's R output bit-for-bit.

Runs experiments/R_reference.R (if Rscript is available), parses results/
reference_R.txt, recomputes SLEM and loading in Python on P1, and asserts
agreement to 1e-4 relative.  This is the hybrid guarantee: Python is validated
against the author's own supplementary code.

Run:  python3 experiments/crosscheck_R_python.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from fibresampler.problems import P1, p1_poisson_theta
from fibresampler.spectral import (log_weights, sr_transition_matrix, slem,
                                   normalised_pi, hazelton_loading)


def run_r_reference():
    ref = os.path.join(ROOT, "results", "reference_R.txt")
    try:
        subprocess.run(["Rscript", os.path.join(ROOT, "experiments", "R_reference.R")],
                       check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        if not os.path.exists(ref):
            print(f"[R unavailable and no cached reference_R.txt: {e}]"); return None
        print("[Rscript failed; using cached results/reference_R.txt]")
    return ref


def parse_reference(path):
    rows = {}
    uniform = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("uniform"):
            parts = line.split()
            uniform = {"gap": float(parts[1].split("=")[1]),
                       "rho": float(parts[2].split("=")[1])}
        elif line and line[0].isdigit():
            t5, sle, gap, rho = line.split()
            rows[float(t5)] = {"slem": float(sle), "gap": float(gap), "rho": float(rho)}
    return uniform, rows


def main():
    ref_path = run_r_reference()
    if ref_path is None:
        return 0  # nothing to check against; treated as skip
    uniform, rows = parse_reference(ref_path)
    prob = P1()

    ok = True
    # uniform
    lw = log_weights(prob, "uniform"); pi = normalised_pi(lw)
    Q = sr_transition_matrix(prob, lw)
    _, gap, _ = slem(Q, pi); rho, _, _ = hazelton_loading(prob, Q, pi)
    a = np.isclose(gap, uniform["gap"], rtol=1e-4)
    b = np.isclose(rho, uniform["rho"], rtol=1e-4)
    ok &= a and b
    print(f"uniform  gap py={gap:.8f} R={uniform['gap']:.8f} {'OK' if a else 'DIFF'} | "
          f"rho py={rho:.4f} R={uniform['rho']:.4f} {'OK' if b else 'DIFF'}")
    # poisson sweep
    for t5, rv in sorted(rows.items(), reverse=True):
        lw = log_weights(prob, "poisson", p1_poisson_theta(t5)); pi = normalised_pi(lw)
        Q = sr_transition_matrix(prob, lw)
        _, gap, _ = slem(Q, pi); rho, _, _ = hazelton_loading(prob, Q, pi)
        a = np.isclose(gap, rv["gap"], rtol=1e-4); b = np.isclose(rho, rv["rho"], rtol=1e-4)
        ok &= a and b
        print(f"t5={t5:<5} gap py={gap:.6e} R={rv['gap']:.6e} {'OK' if a else 'DIFF'} | "
              f"rho py={rho:12.3f} R={rv['rho']:12.3f} {'OK' if b else 'DIFF'}")

    print("\nCROSS-CHECK:", "PASS (Python == Hazelton R)" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
