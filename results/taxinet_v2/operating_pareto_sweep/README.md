# TaxiNetV2 Operating Pareto Sweep

This experiment compares TaxiNetV2 RL tradeoff curves under:

- point shields swept over `beta`:
  - `single_belief`
  - `envelope`
  - `forward_sampling`
- conformal shielding swept over:
  - `confidence_level`
  - `action_filter`

## Published Runs

Canonical preserved runs live under:

```text
results/taxinet_v2/operating_pareto_sweep/runs/<version>/
```

The experiment root may contain regenerated working outputs, but any run worth
keeping must also be copied into a versioned run directory with:

- `manifest.json`
- primary JSON / CSV outputs
- summary markdown
- `figures/`

## Experiment Rule

This sweep must reuse:

- one shared RL controller cache
- one shared adversarial realization cache

across every `beta`, `conf`, and `action_filter` operating point. Only the
runtime shield parameters vary across the sweep.

## Current Canonical Run

See `LATEST_RUN`.
