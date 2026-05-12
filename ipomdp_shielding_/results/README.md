# Legacy Results Layout

This tree contains older experiment outputs that predate the versioned publish
scheme in the repo-root `results/` directory.

## Rule

Summaries should live with the result bundle they describe.

Preferred layout:

```text
ipomdp_shielding_/results/
  <experiment>/
    summary markdown
    JSON / CSV outputs
    figures/
```

If a summary spans multiple experiment directories and there is no single clear
home, place it in:

```text
ipomdp_shielding_/results/summaries/
```

## Applied Conventions

- `prelim/`, `final/`, `alpha_sweep/`, `taxinet_v2/`, `threshold_sweep/`,
  `threshold_sweep_expanded/`, `observation_shield_sweep/`, `sweep_v5/`,
  `sweep_v6/`, and `sweep_v7/` now keep their summary markdown in the same
  directory as the results they describe.
- `summaries/` is reserved for cross-experiment overviews such as
  `experiments_summary.md`.

Future agents should not add new top-level `evaluation_summary*.md` files under
`ipomdp_shielding_/`.
