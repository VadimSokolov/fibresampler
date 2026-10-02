# fibresampler

Code behind two papers on MCMC samplers for count-data linear inverse problems, where the unknown is a vector of nonnegative integer counts `x` with `A x = y` (network tomography, contingency tables, latent multinomials) and the sampler moves over the integer points of that fibre. It implements the single-ray hit-and-run sampler of Hazelton et al. (2024), the reflective lattice sampler, the negative-binomial (gamma-augmented) chain, simulated tempering and split ladders, and a change-of-basis layer (LLL reduction, a pilot-based selector, a union kernel, and hold-fixed recombination), together with the exact spectral-gap experiments and the real-data benchmarks that the papers report.

The two manuscripts split the problem along the two factors of Sinclair's bound.

- Paper I, "Reflective Lattice Samplers and Augmenting Bases Beyond Total Unimodularity": the path-length factor (geometry, reflection, augmenting bases, and a change of basis as a second lever).
- Paper II, "Conservation of Difficulty: Loading Obstructions and Their Escapes in Fibre Sampling": the edge-loading factor (the loading obstruction, tempering and split-ladder escapes, the rim, and the structural escape by recombining the basis).

Authors of both: L. Fromm, M. L. Hazelton, N. G. Polson, V. Sokolov.

## Install

```bash
git clone <this repository> && cd fibresampler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
pytest -q
```

Python 3.10 or later with numpy, scipy and matplotlib. Optional: `numba` (only `experiments/refQ2_tempering_tuning.py`), `sympy`, `mpmath` and `networkx` (only some `helpers/`), and the `4ti2` executables (only `experiments/fmb_reference.py`, skipped if absent). `requirements-pinned.txt` records the exact versions of the original runs; the results are exact spectral computations and do not depend on them.

## Quick start

```python
import numpy as np
from fibresampler.problems import P1, p1_poisson_theta
from fibresampler.spectral import log_weights, normalised_pi, slem, sr_transition_matrix
from fibresampler.basis import RayView, with_basis, hold_fixed_basis, free_points, recommend_basis
from fibresampler.synthetic import BandProblem


def gap(view, lw):                              # exact spectral gap: 1 - SLEM of the assembled transition matrix
    return slem(sr_transition_matrix(view, lw), normalised_pi(lw))[1]


prob = P1()                                     # Hazelton et al. (2024) network: 8 routes, 4 links, 55 states
view = RayView(prob.states, prob.U, index=prob.index, plb_info=prob.plb_info)
print(gap(view, log_weights(prob, "uniform")))                        # 0.1198

# loading: the Poisson mean of one cell shrinks to 1e-6; recombining the basis so that one move set holds it fixed
lw = log_weights(prob, "poisson", p1_poisson_theta(1e-6))
V, n_slice = hold_fixed_basis(prob.U, [4])
print(gap(view, lw), gap(with_basis(view, V), lw))                    # 1.3e-07 against 0.1197

# geometry: a thin tilted band, where the selector replaces the partition lattice basis by an LLL-reduced one
band = BandProblem(2, 24)
bview = RayView(band.states, band.U, index=band.index, plb_info=band.plb_info)
lw = np.zeros(band.fibre_size)
V, info = recommend_basis(free_points(band), normalised_pi(lw))
print(gap(bview, lw), info["chosen"], gap(with_basis(bview, V), lw))  # 0.0325, lll, 0.3341
```

## What is where

| path | contents |
|---|---|
| `fibresampler/fibre.py`, `problems.py` | fibre enumeration, the partition lattice basis `U = [-A1^-1 A2; I]`, ray endpoints; the test problems P1 (Hazelton et al. 2024 network), P2 (Auckland SH16), the 2x3 table |
| `fibresampler/spectral.py` | the single-ray (SR) Gibbs hit-and-run transition matrix, exact SLEM, Sinclair loading, a port of Hazelton's `MaxEdgeLoading` |
| `fibresampler/reflective.py` | the reflective lattice sampler, with the exact reverse-path Metropolis rule |
| `fibresampler/augment.py` | the negative-binomial / gamma-augmented operator and its quadrature version |
| `fibresampler/basis.py` | change of basis: LLL in the Mahalanobis metric, the pilot-based selector, the union kernel, hold-fixed recombination, irreducibility and augmentation checks |
| `fibresampler/eulerian.py`, `synthetic.py` | the `c0`-Eulerian certificate for augmenting bases; band, sloped-band and product fibres |
| `experiments/pred*.py` | the seven registered predictions of the first programme (loading divergence, loading floor, geometry, crossover, ESS grid, balanced bases, high-probability augmentation) |
| `experiments/taskV_*.py`, `taskS_*.py`, `taskG_*.py`, `taskZ_*.py` | tempering, slice sampling, Gaussian-relaxation independence sampler, robustness to normalising-constant error |
| `experiments/split_escape/` | the split-ladder experiments and the fusion test |
| `experiments/refQ1_*.py` | change of basis versus reflection, the loading obstruction as a property of the move set, the dividing line over partitions, SH16 benchmarks |
| `experiments/refQ2_tempering_tuning.py` | a tuning study of simulated tempering on P1 |
| `experiments/m4_*.py` | real-data ESS-per-second benchmarks (Auckland SH16 and other networks) |
| `helpers/` | exact-SLEM and enumeration helpers for the rim, counting and mixture-sampler experiments |
| `hopper/`, `experiments/*.sbatch` | SLURM templates (site independent: set `PYTHON`, submit from the repository root) |
| `results/` | reference outputs of every script (text, json, figures), tracked so that a re-run can be compared with `git diff`; `results/numbers.txt` is the ledger of the values quoted in the papers, one line per value with the script that produced it, an identifier and the date |
| `tests/` | fast regression tests of the exact numbers quoted in the papers |

`REPRODUCE.md` lists the scripts behind the results in the papers, with their output files and run times.

## Conventions that make runs comparable

- Exact first. Wherever a fibre can be enumerated, report `1 - SLEM` of the assembled transition matrix (`fibresampler/spectral.py`), not an autocorrelation estimate.
- Effective sample size is the worst coordinate over the route counts, by Geyer's initial positive monotone sequence, reported per iteration and per second on the same machine, with the one-off pilot charged where a pilot is used (columns labelled "incl. pilot").
- Seeds. The chain scripts take a seed argument and draw their streams from `np.random.SeedSequence((base_seed, stream_id))`; the base seeds are in each script.
- Feasibility gate. Every chain checks `A x = y` at its end (the `G0` column in the tables). A chain that never leaves its start counts as ESS zero in the means and is reported, not dropped.
- Machine dependence. Gaps, slopes, lobe counts and ESS per iteration do not depend on the machine (the chain quantities up to Monte Carlo error). Microseconds per step and ESS per second do.

## Compute

The exact computations (fibres of up to a few thousand states) take seconds to minutes on a laptop. The chain experiments on the Auckland SH16 corridor (21 routes, free dimension 14, too large to enumerate) take a few minutes to half an hour per seed and core and were run as SLURM arrays; see `hopper/`.

## Acknowledgements and licence

`fibresampler.spectral.hazelton_loading` is a port of the `MaxEdgeLoading` routine in the supplementary code of Hazelton, McVeagh, Tuffley and van Brunt (2024), `experiments/pred1_control.py` compares with the values that code produces (embedded as constants), and `experiments/R_reference.R` with `experiments/crosscheck_R_python.py` re-run it (put the supplement's functions at `reference/hazelton_functions.R`; they are not redistributed here). Released under the MIT licence (see `LICENSE`).
