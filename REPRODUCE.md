# Reproducing the results

All commands run from the repository root. Times are wall clock on a single core of an Apple M2 laptop unless a cluster is named.

Every script writes its numbers to a file in `results/`, and those files are tracked. After a re-run, `git diff results/` shows exactly what changed. What to expect:

- Exact quantities (gaps, slopes, lobe counts, loadings) reproduce to the printed digits on any machine. Three things can still differ in the last digit or in a label: a gap that is exactly zero prints as noise of order `1e-16` (for example `5.551e-16` against `2.220e-16` on a reducible chain), a slope fitted to numerically zero gaps is reported as `nan`, and when two candidate bases tie in the selector's surrogate score the label of the winner (`lll` or `climb(I)`) can depend on the BLAS build, while the gap does not.
- Monte Carlo quantities (chain ESS, and the pilot replicates in `refQ2_tempering_tuning.py`) reproduce up to Monte Carlo error for the same seeds, and exactly on the same machine and library versions: a different numpy release can change the random streams, which moves these summaries in the last digits, while the exact gaps beside them do not move.
- Timing columns (microseconds per step, ESS per second) depend on the machine and on what else it is running.

## 0. Set up and check

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest numba sympy mpmath networkx
pytest -q                      # 6 tests, a few seconds: exact gaps quoted in the papers
```

## 1. Change of basis versus reflection (Paper I, Section 6; Paper II, Section 4)

| script | what it produces | output | time |
|---|---|---|---|
| `experiments/refQ1_basis_change.py` | exact gaps on 11 enumerable fibres: partition basis, LLL, selected, union, search oracle, reflective; pilot robustness; measured cost per step | `results/refQ1_basis_change.txt` | 4 minutes |
| `experiments/refQ1_cost_adjusted.py` | gap per measured microsecond, derived from the file above (no new runs) | `results/refQ1_cost_adjusted.txt` | seconds |
| `experiments/refQ1_band_bound.py` | check of the band proposition on the grid `w in {1,2,3,5,8}`, `4w <= C <= 400` | `results/refQ1_band_bound.txt` | seconds |
| `experiments/refQ1_fibonacci_slope.py` | near-golden-slope bands, where reflection might beat a change of basis | `results/refQ1_fibonacci_slope.txt` | seconds |
| `experiments/refQ1_sh16_basis.py` | Auckland SH16 chains, one seed and one target per call (`--target uniform` or `poiswc`, `--seed`, `--out`) | `results/q1_sh16/*.json` | 3 to 8 minutes each |
| `experiments/refQ1_sh16_aggregate.py` | aggregates the SH16 json files into the paper's table | `results/refQ1_sh16_basis.txt` | seconds |
| `experiments/refQ1_ref_b0_probe.py` | why reflection loses on SH16: Metropolised ray against heat-bath ray | `results/refQ1_ref_b0_probe.txt` | minutes |

## 2. The loading obstruction as a property of the move set (Paper II, Section 4)

| script | what it produces | output | time |
|---|---|---|---|
| `experiments/refQ1_loading_basis.py` | does a change of basis touch the loading obstruction (P1) | `results/refQ1_loading_basis.txt` | 40 seconds |
| `experiments/refQ1_loading_generality.py` | exact gaps in the partition basis, the hold-fixed basis, the selected basis and the union, on P1, the table and the six `k*=2` fibres | `results/refQ1_loading_generality.txt` | seconds |
| `experiments/refQ1_verify_hold_fixed.py` | the converse's hypotheses on the hold-fixed basis (lobes, border drain, move sizes) | `results/refQ1_hold_fixed_assumptions.txt` | seconds |
| `experiments/refQ1_partition_dividing_line.py` | every integral partition of P1 and of the 2x3 table: lobe count against slope | `results/refQ1_partition_dividing_line.txt` | seconds |
| `experiments/refQ1_nb_augmented_basis.py` | the gamma-augmented chain in the partition and the hold-fixed basis | `results/refQ1_nb_augmented_basis.txt` | seconds |
| `experiments/split_escape/exp3b_fusion_basis.py` | fusion test: change of basis against the reflective split ladder; prints to stdout, so `python3 experiments/split_escape/exp3b_fusion_basis.py \| tee results/refQ1_fusion_basis.txt` | `results/refQ1_fusion_basis.txt` | 10 seconds |
| `experiments/refQ1_sh16_loading.py` | SH16 without the rate floor, the genuine loading regime (`--seed`, `--out`) | `results/q1_sh16_loading/*.json` | 27 minutes each |
| `experiments/refQ1_sh16_loading_aggregate.py` | aggregates the loading json files | `results/refQ1_sh16_loading.txt` | seconds |

## 3. Simulated tempering (Paper II, Section 5)

| script | what it produces | output | time |
|---|---|---|---|
| `experiments/refQ2_tempering_tuning.py` | tuning study on P1, exact joint operators: number of rungs, spacing, floor, level-move probability, weights, pilots, overlap, parallel tempering (needs `numba`) | `results/refQ2_tempering_tuning.txt` | 5 minutes |
| `experiments/taskV_tempering.py`, `taskV_hardening.py` | the tempering floor, the spectrum law, and its hardening | `results/taskV_numbers.txt`, `results/taskV_hardening_numbers.txt` | seconds |
| `experiments/taskZ_robustness.py` | robustness of the three-rung floor to normalising-constant error | `results/taskZ_robustness.txt` | seconds |
| `experiments/taskS_slice.py`, `taskG_gaussian_teleport.py` | slice sampling and a Gaussian-relaxation independence sampler are also conserved | `results/taskS_numbers.txt`, `results/taskG_numbers.txt` | seconds to a minute |
| `experiments/split_escape/exp*.py` | the split ladder and the fusion test (print to stdout) | `results/split_escape_numbers.txt` | seconds |

## 4. The seven registered predictions of the first programme

| prediction | script | output |
|---|---|---|
| 1. Poisson loading divergence on P1 (positive control) | `experiments/pred1_control.py` | `results/pred1_numbers.txt` |
| 2. the loading floor (keystone) | `experiments/pred2_keystone.py`, `pred2_generality.py`, `pred2_secondseed.py` | `results/pred2_numbers.txt`, `pred2_generality.txt`, `pred2_secondseed.txt` |
| 3. reflection helps geometry | `experiments/pred3_bands.py` | `results/pred3_bands.txt` |
| 4. the crossover | `experiments/pred4_crossover.py`, `pred4_mechanism_ablation.py` | `results/pred4_numbers.txt`, `pred4_ablation.txt` |
| 5. the ESS-per-second grid | `experiments/pred5_ess.py`, `pred5_replicates.py` | `results/pred5_numbers.txt`, `pred5_replicates.txt` |
| 6. balanced bases | `experiments/pred6_balanced.py`, `pred6_ubalance_check.py` | `results/pred6_numbers.txt`, `pred6_ubalance.txt` |
| 7. high-probability augmentation | `experiments/pred7_hp.py` (`--quick` for a short run) | `results/pred7_numbers.txt` |

Further checks: `experiments/validate_reflective.py` (correctness gates of the reflective sampler), `kstar_check.py`, `verify_assumptions.py`, `verify_assumption_lobes.py` (the geometric hypotheses), `refA_productivity.py`, `refD_sh16_bmax.py`, `refF_guard.py`, `refK_obliqueness.py`, `m9_fit_uncertainty.py`, `ref_kstar2_slice.py`, `ref_3x3x2_rim.py`, `fmb_reference.py` (needs 4ti2), `counting_estimators.py`, and the real-data benchmarks `m4_realdata_ess.py` and `m4_variety.py`. Each one's docstring says what it checks and where its output goes.

The rim and counting helpers live in `helpers/` and are meant to be run from inside it (`cd helpers && python3 fig_rim.py`, `python3 verify_rim_precision.py`, about two minutes, needs `mpmath`).

## 5. On a SLURM cluster

The SH16 chains and the full-length benchmarks are cluster jobs. The templates in `hopper/` and `experiments/*.sbatch` are site independent: submit from the repository root, create the log directory first, and set `PYTHON` if the default `python3` on the compute nodes lacks numpy and scipy.

```bash
mkdir -p slurm_logs
sbatch hopper/q1_sh16.slurm            # 10 tasks: 5 seeds x {uniform, fitted Poisson}, 3 to 8 minutes each
sbatch hopper/q1_sh16_loading.slurm    # 5 tasks, 27 minutes each
sbatch hopper/kstar2.slurm             # search for fibres with corridor height 2 (64 tasks)
sbatch experiments/hopper_m4.sbatch    # full-length SH16 benchmark (time limit 6 hours)
# when the arrays finish:
python3 experiments/refQ1_sh16_aggregate.py
python3 experiments/refQ1_sh16_loading_aggregate.py
```

## 6. Conventions behind the numbers

- The partition basis is `U = [-A1^-1 A2; I]` for a column subset with `|det A1| = 1`; the experiments name the columns explicitly (`cols1`).
- A gap is `1 - SLEM` of the assembled transition matrix, with the matrix symmetrised by `D^(1/2) Q D^(-1/2)` before the eigensolve, and detailed balance checked to machine precision.
- Chain seeds: `np.random.SeedSequence((base_seed, stream_id))` with base seed 20261002, offset by 1000 times the seed number in the SH16 scripts, and one stream id per arm.
- Worst-coordinate ESS uses Geyer's initial positive monotone sequence on each route count; a coordinate that never moves is skipped and a chain with no moving coordinate has ESS 0.
- A pilot (for the basis selector) is charged to the per-second figure in the columns labelled "incl. pilot".
