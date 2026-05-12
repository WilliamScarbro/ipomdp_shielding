# TaxiNetV2 Conformal RL Sweep

This directory is the script output root for TaxiNetV2 conformal RL comparisons.

## Published Runs

Preserved runs live under:

```text
results/taxinet_v2/conformal_rl_sweep/runs/<version>/
```

The root directory may contain ad hoc generated files from scripts. Those are not the canonical published location unless they have also been copied into `runs/<version>/`.

The summary markdown belongs inside the published run directory next to the
JSON/CSV outputs and `figures/`.

## Required Publish Step

After generating a run that should be kept:

1. choose a version like `vYYYY-MM-DD-<slug>`
2. create `runs/<version>/`
3. copy in:
   - headline JSON/CSV
   - sweep JSON/CSV
   - preflight or decision note
   - summary markdown
   - figures
4. write `manifest.json`
5. update `LATEST_RUN`

## Current Canonical Run

See `LATEST_RUN`.
