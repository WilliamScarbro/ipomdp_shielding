# Results Layout

This repository has two result trees:

- `results/`: newer experiment bundles and cached search artifacts
- `ipomdp_shielding_/results/`: legacy project outputs kept in place for reproducibility

## Rule

For any result bundle worth keeping, publish it under a versioned run directory instead of leaving the only copy at an experiment root.
The summary markdown for that run must live in the same directory as the copied
results it describes.

Use this layout:

```text
results/
  <case-study>/
    <experiment>/
      README.md
      LATEST_RUN
      runs/
        <version>/
          manifest.json
          summary files
          figures/
```

## Version Format

Use:

```text
vYYYY-MM-DD-<short-slug>
```

Examples:

- `v2026-05-12-earlystop-seed42`
- `v2026-05-13-paper-model-rejected`

## What Goes In A Run Directory

Each published run directory must contain:

- `manifest.json`
- the primary JSON/CSV outputs for the run
- the markdown summary
- rendered figures under `figures/`

`manifest.json` should record at least:

- `version`
- `created_utc`
- `case_study`
- `experiment`
- `source_checkpoint`
- measured accuracy / provenance fields
- key sweep settings
- paths to the copied artifacts in that run directory

## What Stays Out

Do not commit large transient training caches by default:

- intermediate checkpoints
- RL cache files
- scratch search directories

Only commit those when the user explicitly wants model artifacts preserved.

## Compatibility

Some scripts still write directly into experiment roots such as:

- `results/taxinet_v2/conformal_rl_sweep/`

That is acceptable for generation, but after the run finishes:

1. copy the preserved artifacts into `runs/<version>/`
2. update `LATEST_RUN`
3. update the experiment `README.md` if conventions changed

Future agents should follow this policy unless a user explicitly requests a different layout.
