# Alpha Sweep: TaxiNet under Fixed-Realization Uniform Perception

This report summarizes the refreshed TaxiNet alpha sweep after changing the
`uniform` perception regime to sample one interval realization per trajectory
and hold it fixed for the whole trajectory.

- Sweep: 8 `alpha` values × 3 `beta` values × 5 seeds
- Trials: 20 per seed-cell, so 100 trials per `(alpha, beta, perception, shield)`
- Shields: `single_belief`, `envelope`, `forward_sampling`
- Perception regimes: `uniform`, `adversarial_opt`
- Total runtime: `11247.2 s` (about `3.1 h`)

The fresh machine-readable outputs are:

- `results_tidy.csv`
- `sweep_summary.json`
- `figures/pareto_alpha.png`
- `figures/alpha_vs_fail.png`
- `figures/alpha_vs_stuck.png`

For convenience, the top-level figure copies below were also refreshed.

## Figures

![TaxiNet Pareto scatter](pareto_alpha.png)

![Fail rate vs alpha](alpha_vs_fail.png)

![Stuck rate vs alpha](alpha_vs_stuck.png)

## Headline Findings

1. `beta` is still the dominant operating-point knob. Moving from `beta=0.7`
   to `beta=0.9` materially lowers fail rate for `envelope` and
   `forward_sampling`, but increases stuck.
2. Under the new fixed-realization uniform semantics, `envelope` is now the
   lowest-fail shield on average for every `(perception, beta)` row. In the
   previous checked-in run, `forward_sampling` had been slightly ahead of
   `envelope` for `uniform, beta=0.8` and `uniform, beta=0.9`.
3. The adversarial-perception ordering did not change: `envelope` remains best
   on fail, `forward_sampling` remains second, and `single_belief` remains the
   most failure-prone but least stuck.
4. The trade-off pattern is otherwise unchanged:
   `single_belief` is still the low-stuck method,
   `envelope` is still the low-fail method,
   and `forward_sampling` sits between them.

## Average Ordering by Fail Rate

The table below averages across all `alpha` values and seeds inside each
`(perception, beta, shield)` bucket.

| Perception | Beta | 1st | 2nd | 3rd |
|---|---:|---|---|---|
| uniform | 0.7 | envelope | forward_sampling | single_belief |
| uniform | 0.8 | envelope | forward_sampling | single_belief |
| uniform | 0.9 | envelope | forward_sampling | single_belief |
| adversarial_opt | 0.7 | envelope | forward_sampling | single_belief |
| adversarial_opt | 0.8 | envelope | forward_sampling | single_belief |
| adversarial_opt | 0.9 | envelope | forward_sampling | single_belief |

## Average Metrics by Beta

Values are `fail% / stuck%`, averaged over all `alpha` values and seeds.

### Uniform perception

| Beta | single_belief | envelope | forward_sampling |
|---|---:|---:|---:|
| 0.7 | 66.4 / 0.1 | 58.5 / 5.0 | 63.4 / 0.6 |
| 0.8 | 65.1 / 1.3 | 52.9 / 7.3 | 56.8 / 1.3 |
| 0.9 | 55.5 / 1.0 | 43.8 / 15.6 | 48.4 / 6.0 |

### Adversarial perception

| Beta | single_belief | envelope | forward_sampling |
|---|---:|---:|---:|
| 0.7 | 61.8 / 0.0 | 56.3 / 5.1 | 60.6 / 0.9 |
| 0.8 | 60.6 / 0.9 | 54.2 / 6.5 | 58.5 / 2.0 |
| 0.9 | 54.3 / 2.4 | 44.3 / 16.7 | 46.0 / 8.7 |

## Best Cells

Best means lowest fail rate, tie-broken by lower stuck.

### Uniform perception

| Beta | single_belief | envelope | forward_sampling |
|---|---|---|---|
| 0.7 | `a=0.30`: 60.0 / 1.0 | `a=0.075`: 54.0 / 6.0 | `a=0.20`: 57.0 / 2.0 |
| 0.8 | `a=0.10`: 57.0 / 3.0 | `a=0.025`: 43.0 / 13.0 | `a=0.075`: 48.0 / 0.0 |
| 0.9 | `a=0.075`: 48.0 / 2.0 | `a=0.15`: 35.0 / 17.0 | `a=0.01`: 36.0 / 14.0 |

### Adversarial perception

| Beta | single_belief | envelope | forward_sampling |
|---|---|---|---|
| 0.7 | `a=0.01`: 57.0 / 0.0 | `a=0.10`: 43.0 / 6.0 | `a=0.10`: 54.0 / 0.0 |
| 0.8 | `a=0.01`: 57.0 / 0.0 | `a=0.30`: 46.0 / 8.0 | `a=0.01`: 52.0 / 3.0 |
| 0.9 | `a=0.15`: 45.0 / 3.0 | `a=0.15`: 36.0 / 17.0 | `a=0.15`: 41.0 / 9.0 |

## What Changed Relative to the Previous Checked-In Run

Using the previously checked-in `results_tidy.csv` as the baseline:

1. The only clear ordering flips are in the `uniform` rows at `beta=0.8` and
   `beta=0.9`, where `envelope` overtook `forward_sampling` on average fail
   rate.
2. Those flips are not tiny:
   at `uniform, beta=0.8`, the average fail rates moved from
   `forward_sampling 55.4%` vs `envelope 56.1%`
   to `envelope 52.9%` vs `forward_sampling 56.8%`.
3. At `uniform, beta=0.9`, the average fail rates moved from
   `forward_sampling 46.7%` vs `envelope 47.4%`
   to `envelope 43.8%` vs `forward_sampling 48.4%`.
4. No adversarial-perception ordering changed in the alpha sweep.

## Reproducibility

This refreshed run was written directly into `results/alpha_sweep/`.

```bash
python3 - <<'PY'
import dataclasses
from ipomdp_shielding.experiments.sweeps.rl_alpha_sweep import run_alpha_sweep
from ipomdp_shielding.experiments.sweeps.rl_alpha_sweep_taxinet import config as base_config

config = dataclasses.replace(base_config, results_dir='results/alpha_sweep')
run_alpha_sweep(config)
PY
```
