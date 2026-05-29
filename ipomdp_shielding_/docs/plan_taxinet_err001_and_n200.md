# Plan: TaxiNet `error=0.01` Consistency + Uniform `n=200` Sampling

Author intent: bring the supplement-only TaxiNet experiments onto the same
TaxiNet dynamics as the main paper (`error=0.01`, γ=0.99) and standardise the
Monte Carlo budget at **200 trials / trajectories per cell** across every
experiment that feeds a table or figure.

## Motivation

After auditing the paper against the latest `sweep_v7` rerun (TaxiNet with
`error=0.01`, γ=0.99), two supplement studies were found to still reference
the old TaxiNet dynamics (`error=0.1`, γ=0.9):

- α-sweep — supplement §VII, Figs S-3/S-4/S-5 and the paragraph after Fig S-5.
- Perception-variability — supplement §VII, Fig S-6 and surrounding paragraph.

Their JSON sources are internally consistent but use a *different* TaxiNet
kernel than Table I in the main paper. We will rebuild both with `error=0.01`.

We are also unifying the trial budget at **`n=200`**. `sweep_v7` already runs
at 200; the alpha sweep, perception-variability sweep, coarseness experiments,
and the TaxiNetV2 sweeps still run at 10–100. Standardising at 200 closes the
Clopper-Pearson intervals to a consistent width across the paper.

## Phase 1 — Config edits (TaxiNet error=0.01 + n=200)

All paths relative to the `ipomdp_shielding_` repo root.

### 1a. Alpha sweep — TaxiNet kernel + budget

File: `ipomdp_shielding/experiments/sweeps/rl_alpha_sweep_taxinet.py`

```diff
-    seeds=[42, 123, 456, 789, 1024],
-    num_trials=20,
+    seeds=[42, 123, 456, 789, 1024],
+    num_trials=40,                       # 5 × 40 = 200 trials per cell
     trial_length=20,
@@
     ipomdp_base_kwargs={
         "confidence_method": "Clopper_Pearson",
         "train_fraction": 0.8,
-        "error": 0.1,
+        "error": 0.01,
         "smoothing": True,
     },
```

Rationale for `num_trials=40`: keeping the 5 seeds unchanged gives pooled
`n=200` per `(α, β, perception, shield)` cell with the existing aggregation
logic in `rl_alpha_sweep.py::_aggregate_sweep`, no other code changes.

### 1b. Perception-variability sweep — TaxiNet kernel + budget

File: `ipomdp_shielding/experiments/configs/perception_variability_taxinet.py`

```diff
     seed=42,
-    num_trajectories=100,
+    num_trajectories=200,
     trajectory_length=20,
     initial_state=(0, 0),
     initial_action=0,
     sampler_budget=500,
     sampler_k=100,
     sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
     sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
+    ipomdp_kwargs={"error": 0.01},
     results_path="results/final/perception_variability/perception_variability_taxinet_results.json",
```

Note: `CoarseExperimentConfig.__post_init__` already coerces `None` →
`{}`, so an explicit `ipomdp_kwargs={"error": 0.01}` is safe.

### 1c. Coarseness — bump every case study to n=200

The supplement tables (S-I, S-II) are populated by the coarseness experiments
in `configs/coarse_*_final.py`. Bump `num_trajectories` to 200 in all four:

| file                          | current → new |
|-------------------------------|---------------|
| `coarse_taxinet_final.py`     | 100 → 200     |
| `coarse_taxinet_v2_final.py`  | 100 → 200     |
| `coarse_obstacle_final.py`    | 50  → 200     |
| `coarse_cartpole_final.py`    | 30  → 200     |
| `coarse_refuel_final.py`      | 10  → 200     |

Also add `ipomdp_kwargs={"error": 0.01}` to `coarse_taxinet_final.py` (TaxiNetV2
config is independent and unaffected). `trajectory_length` stays as-is.

### 1d. (Optional) TaxiNetV2 / conformal — n=200

These power the revision letter only, but the same consistency argument applies.

| file                                              | current → new |
|---------------------------------------------------|---------------|
| `configs/rl_shield_taxinet_v2_final.py`           | 100 → 200     |
| `configs/rl_shield_taxinet_v2_comparison.py`      | 100 → 200     |
| `configs/rl_shield_taxinet_v2_comparison_conf99.py`  | check + 200 |
| `configs/rl_shield_taxinet_v2_comparison_conf995.py` | check + 200 |

If we bump 1d the revision-letter numbers (7/93, 32/46, 39/41, 37, 29, 38, 41)
will shift slightly. **Decision point: include 1d in the rerun or skip?**
Recommendation: include, since the user asked for "200 across all experiments
to be consistent".

## Phase 2 — Reruns

All commands assume `cd` into the repo root and the project's standard
environment.

### 2a. Alpha sweep (TaxiNet)

```bash
# Invalidate cached adversarial realisations & RL agent because dynamics changed
rm -f results/sweep/rl_alpha_taxinet_v2/cache_alpha*_rl_agent.pt
rm -f results/sweep/rl_alpha_taxinet_v2/cache_alpha*_opt_realization*.json

python -m ipomdp_shielding.experiments.sweeps.rl_alpha_sweep \
    --config sweeps.rl_alpha_sweep_taxinet
```

Outputs (will be overwritten):
- `results/sweep/rl_alpha_taxinet_v2/results_tidy.csv`
- `results/sweep/rl_alpha_taxinet_v2/sweep_summary.json`
- `results/sweep/rl_alpha_taxinet_v2/figures/{alpha_vs_fail,alpha_vs_stuck,alpha_vs_safe,pareto_alpha}.png`

Also copy the `alpha_vs_*` and `pareto_alpha` PNGs back into
`results/alpha_sweep/` if that is the canonical paper-figure location.

### 2b. Perception variability (TaxiNet)

```bash
python -m ipomdp_shielding.experiments.sweeps.perception_variability_sweep \
    configs.perception_variability_taxinet
```

Outputs:
- `results/final/perception_variability/perception_variability_taxinet_results.json`
- `results/final/perception_variability/perception_variability_taxinet_results_tidy.csv`
- `results/final/perception_variability/figures/perception_variability_taxinet.png`

### 2c. Coarseness (all four)

```bash
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_final
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_v2_final
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_obstacle_final
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_cartpole_final
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_refuel_final
```

Outputs land under `results/final/coarse_*`.

### 2d. (Optional, if Phase 1d applied) TaxiNetV2 / conformal

```bash
# Operating Pareto sweep (revision-letter source)
python -m ipomdp_shielding.experiments.run_taxinet_v2_operating_pareto_sweep \
    --num-trials 200 --trial-length 20

# Cross-confidence comparison (Table S-III)
python -m ipomdp_shielding.experiments.run_taxinet_v2_comparison \
    --config configs.rl_shield_taxinet_v2_comparison
python -m ipomdp_shielding.experiments.run_taxinet_v2_comparison \
    --config configs.rl_shield_taxinet_v2_comparison_conf99
python -m ipomdp_shielding.experiments.run_taxinet_v2_comparison \
    --config configs.rl_shield_taxinet_v2_comparison_conf995
```

Verify CLI flag names against the runner before kicking off; some have
`--num-trials`, others read everything from the config dataclass.

### 2e. Sweep_v7 — no rerun needed

`run_v7_all_sweeps.py` already enforces `num_trials=200` for every case
study via `THRESHOLD_PARAMS` / `OBS_PARAMS` / `FS_PARAMS` / `CARR_PARAMS`,
overriding the lower per-config defaults. The 2026-05-28 TaxiNet rerun
already used `error=0.01`. **Nothing to do here.**

## Phase 3 — Paper-side sync (after data lands)

Files in the outer paper repo:

1. `emsoft_source/tex/supplement_evaluation.tex`
   - Line ~180 (alpha-sweep paragraph) — replace the three "36/17/47", "41/9/50", "45/3/52" triples with the values from the new `sweep_summary.json` at α=0.15, β=0.9, adversarial.
   - Line ~196 (perception-variability paragraph) — replace the six summary statistics (`0.353 / 0.258 / 0.808 / 0.286 / 0.193 / 0.730` and `0.0545 / 0.0394`) with the new aggregates from the regenerated JSON.
   - Tables S-I and S-II — re-syncing entries against new coarseness JSONs.

2. `revisions/revision_letter.tex` (only if Phase 1d/2d applied)
   - Update the TaxiNetSim conformal comparison numbers to match the new
     `operating_pareto_sweep/results.json` aggregates.

3. Sanity check that `pareto_v7_taxinet.png`, `summary_v7.1_bars.png`,
   `summary_v7.1_safe_bars.png` (already 2026-05-28) still agree with
   `sweep_v7/threshold/taxinet_sweep.json` — they should, since no
   sweep_v7 data is changing in this rerun.

## Phase 4 — Audit checklist

After all reruns, verify before re-submitting:

- [ ] Every TaxiNet config that the paper references shows `error: 0.01` in
      its on-disk metadata (`results/.../metadata.ipomdp_kwargs.error == 0.01`
      or `0.01`-derived γ in setup info).
- [ ] Every JSON whose numbers are quoted in `.tex` was modified after the
      Phase 2 rerun.
- [ ] `n=200` appears in the metadata of every results JSON cited by the
      paper or supplement.
- [ ] No remaining figures dated before the rerun in `emsoft_source/figures/`
      or supplement figures directory.
- [ ] `(βγ)^H` arithmetic in Table I still uses γ=0.99 (it does — no change
      needed because sweep_v7 TaxiNet is unchanged).

## Estimated runtime

Rough estimates on the EMSOFT-paper Skua image (single GPU, 8 cores):

| Phase | Job                                | Estimate |
|-------|------------------------------------|----------|
| 2a    | TaxiNet alpha sweep (8α × 3β × 5seeds × 40trials × 3shields × 2perc) | ~3.5 h |
| 2b    | Perception variability (200 traj × 20 steps) | ~25 min |
| 2c    | Coarseness × 5 case studies @ n=200 | ~1.5 h total |
| 2d    | TaxiNetV2 operating sweep @ n=200  | ~1 h     |
| 2d    | Three cross-confidence runs @ n=200| ~30 min  |
| **Total** | sequential                     | **~6.5–7 h** |

Phases 2a/2b/2c are independent and can run in parallel if the box has
the cores; 2d depends on the TaxiNetV2 RL/adversarial cache which is shared
across the three confidence runs, so run those in sequence.
