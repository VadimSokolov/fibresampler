"""Cost-adjusted comparison of the reduced basis and the reflective sampler (derived, no new runs).

Reads results/refQ1_basis_change.txt (exact gaps, uniform target, and the measured per-step costs
of the same implementation) and reports, per fibre, the gap per microsecond of the selected basis
relative to the single-ray sampler and relative to the better of the two reflective arms
(B_max = 2, 4), by gap per microsecond.  A fibre where the selector kept the PLB has ratio 1
against SR by construction.  Output: results/refQ1_cost_adjusted.txt.
"""

from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "results", "refQ1_basis_change.txt")
OUT = os.path.join(REPO, "results", "refQ1_cost_adjusted.txt")


def parse():
    text = open(SRC).read().split("TARGET = pois1")[0]           # uniform block only
    gaps = {}
    for line in text.splitlines():
        m = re.match(r"\s+(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+(.*?)\s+\|", line)
        if not m:
            continue
        name = m.group(1)
        vals = m.group(6).split()
        if len(vals) != 8:
            continue
        gaps[name] = [float(v) if v != "nan" else float("nan") for v in vals]      # SR LLL sel U.5 U.1 orc B2 B4
    costs = {}
    cblock = open(SRC).read().split("MEASURED COST PER STEP")[1]
    for line in cblock.splitlines():
        m = re.match(r"\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(nan|[\d.]+)\s+(nan|[\d.]+)\s+\|", line)
        if m:
            costs[m.group(1)] = [float(m.group(k)) if m.group(k) != "nan" else float("nan") for k in range(2, 7)]
    return gaps, costs


def main():
    gaps, costs = parse()
    W = 118
    out = ["=" * W,
           "Gap per microsecond (exact uniform-target gap over measured time per step, same implementation), relative ratios",
           "sel = recommend_basis choice (identity when the surrogate gain is below 1.10); cost(sel) = cost of SR-LLL (same single-ray step)",
           "best Ref = the better of Ref-B2 / Ref-B4 by gap per microsecond",
           "=" * W,
           f"  {'fibre':>22} | {'gap sel/SR':>10} {'cost LLL/SR':>11} {'(g/c) sel/SR':>12} | {'g sel/best Ref':>14} {'cost Ref/LLL':>12} {'(g/c) sel/Ref':>13}"]
    ratios = []
    for name, g in gaps.items():
        if name not in costs:
            continue
        c_sr, c_lll, _, c_b2, c_b4 = costs[name]
        g_sr, _, g_sel, _, _, _, g_b2, g_b4 = g
        if g_b2 != g_b2 or g_b4 != g_b4:
            continue
        eff_b2, eff_b4 = g_b2 / c_b2, g_b4 / c_b4
        eff_ref, c_ref, g_ref = (eff_b2, c_b2, g_b2) if eff_b2 >= eff_b4 else (eff_b4, c_b4, g_b4)
        eff_sel = g_sel / c_lll
        out.append(f"  {name:>22} | {g_sel / g_sr:>10.2f} {c_lll / c_sr:>11.2f} {eff_sel / (g_sr / c_sr):>12.2f} | "
                   f"{g_sel / g_ref:>14.2f} {c_ref / c_lll:>12.2f} {eff_sel / eff_ref:>13.2f}")
        ratios.append((name, eff_sel / eff_ref, g_sel / g_sr))
    thin = [r for r in ratios if r[2] > 2.0]
    out.append("")
    out.append("thin tilted fibres where the selector moved off the PLB (gap ratio over SR above 2): "
               f"cost-adjusted advantage over the best reflective arm {min(r[1] for r in thin):.1f}x to {max(r[1] for r in thin):.1f}x "
               f"across {len(thin)} fibres")
    text = "\n".join(out) + "\n"
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
