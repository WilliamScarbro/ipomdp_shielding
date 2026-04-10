# Artifact Reconstruction Plan

This plan is for recreating the lost clean artifact repository from the current source repository at `/home/dev/ipomdp_shielding`.

The source repo still contains the code and results needed to rebuild the artifact. The intended final artifact repo is versionless:

- repo directory: `ipomdp_shielding_artifact`
- no `v7` in public-facing file names, script names, result directory names, README text, or generated summary names
- the artifact should contain only the single experiment bundle, not broader development history

## Source of Truth

Work from:

- source repo root: `/home/dev/ipomdp_shielding`
- project subtree: `/home/dev/ipomdp_shielding/ipomdp_shielding_`

Relevant source files still present:

- `ipomdp_shielding_/ipomdp_shielding/experiments/run_v7_all_sweeps.py`
- `ipomdp_shielding_/ipomdp_shielding/experiments/plot_sweep_v7.py`
- `ipomdp_shielding_/ipomdp_shielding/experiments/configs/rl_shield_*_v7.py`
- `ipomdp_shielding_/results/sweep_v7/`
- `ipomdp_shielding_/results/cache/`
- `ipomdp_shielding_/evaluation_summary_threshold_sweep_7.md`

## Target Artifact Repo

Create a fresh sibling repo:

- `/home/dev/ipomdp_shielding_artifact`

Do not preserve source git history. Initialize a brand-new git repository with a clean history.

## Files To Copy Into The New Artifact Repo

Copy these from `ipomdp_shielding_/` into the new repo root:

- `ipomdp_shielding/`
- `pyproject.toml`
- `evaluation_summary_threshold_sweep_7.md`

Copy the result bundle:

- `results/sweep_v7/` -> temporary copy first, then rename later

Copy only these cache files into `results/cache/`:

- `prelim_rl_shield_taxinet_agent.pt`
- `prelim_rl_shield_obstacle_agent.pt`
- `prelim_rl_shield_cartpole3_agent.pt`
- `lowacc_rl_shield_cartpole_agent.pt`
- `v2_rl_shield_refuel_agent.pt`
- `v7_rl_shield_taxinet_opt_realization.json`
- `v7_rl_shield_obstacle_opt_realization.json`
- `v7_rl_shield_cartpole3_sb_opt_realization_single_belief.json`
- `v7_lowacc_rl_shield_cartpole_opt_realization_single_belief.json`
- `v7_rl_shield_refuel_v2_sb_opt_realization_single_belief.json`

Do not copy the rest of `results/`, docs archives, preliminary/full result trees, or unrelated caches.

## Files To Add In The Artifact Repo

Add:

- `README.md`
- `requirements.txt`
- `Dockerfile`
- `run_experiments.sh`
- `scripts/smoke_test.py`
- `.gitignore`

Also update `pyproject.toml`.

## Required Artifact Layout

The final artifact repo should look like this at top level:

- `Dockerfile`
- `README.md`
- `evaluation_summary.md`
- `ipomdp_shielding/`
- `pyproject.toml`
- `requirements.txt`
- `results/cache/`
- `results/experiment/`
- `run_experiments.sh`
- `scripts/smoke_test.py`

## Required Renames

### Repo / top-level

- `evaluation_summary_threshold_sweep_7.md` -> `evaluation_summary.md`
- `run_v7_experiments.sh` style naming should not exist; use `run_experiments.sh`
- `results/sweep_v7/` -> `results/experiment/`

### Experiment runners

Rename:

- `ipomdp_shielding/experiments/run_v7_all_sweeps.py` -> `ipomdp_shielding/experiments/run_all_sweeps.py`
- `ipomdp_shielding/experiments/plot_sweep_v7.py` -> `ipomdp_shielding/experiments/plot_experiment_summary.py`

### Config files

Rename:

- `rl_shield_taxinet_v7.py` -> `rl_shield_taxinet_artifact.py`
- `rl_shield_obstacle_v7.py` -> `rl_shield_obstacle_artifact.py`
- `rl_shield_cartpole_v7.py` -> `rl_shield_cartpole_artifact.py`
- `rl_shield_cartpole_lowacc_v7.py` -> `rl_shield_cartpole_lowacc_artifact.py`
- `rl_shield_refuel_v2_v7.py` -> `rl_shield_refuel_v2_artifact.py`

### Cache file names

Rename copied artifact-specific opt-realization caches:

- `v7_rl_shield_taxinet_opt_realization.json` -> `rl_shield_taxinet_opt_realization.json`
- `v7_rl_shield_obstacle_opt_realization.json` -> `rl_shield_obstacle_opt_realization.json`
- `v7_rl_shield_cartpole3_sb_opt_realization_single_belief.json` -> `rl_shield_cartpole3_sb_opt_realization_single_belief.json`
- `v7_lowacc_rl_shield_cartpole_opt_realization_single_belief.json` -> `lowacc_rl_shield_cartpole_opt_realization_single_belief.json`
- `v7_rl_shield_refuel_v2_sb_opt_realization_single_belief.json` -> `rl_shield_refuel_v2_sb_opt_realization_single_belief.json`

### Figure names

Rename bundled figures in `results/experiment/`:

- `summary_v7_bars.png` -> `summary_experiment_bars.png`
- `barchart_v7_taxinet.png` -> `barchart_experiment_taxinet.png`
- `barchart_v7_obstacle.png` -> `barchart_experiment_obstacle.png`
- `barchart_v7_cartpole_lowacc.png` -> `barchart_experiment_cartpole_lowacc.png`
- `barchart_v7_refuel_v2.png` -> `barchart_experiment_refuel_v2.png`
- `pareto_v7_taxinet.png` -> `pareto_experiment_taxinet.png`
- `pareto_v7_obstacle.png` -> `pareto_experiment_obstacle.png`

## Code Changes Required

### `pyproject.toml`

Keep the package metadata minimal. Do not encode heavyweight dependencies in `project.dependencies`; install them from `requirements.txt` instead.

Use:

- package name: `ipomdp-shielding`
- version can remain `0.1.0`

Reason: `pip install -e .` should not trigger a second dependency resolution after `requirements.txt` installs the scientific stack.

### `requirements.txt`

Use pinned dependencies with CPU-only torch:

```text
--extra-index-url https://download.pytorch.org/whl/cpu
numpy==1.26.4
scipy==1.13.1
statsmodels==0.14.2
matplotlib==3.9.2
torch==2.4.1+cpu
```

### `Dockerfile`

Base image:

- `python:3.11-slim`

Environment:

- `PYTHONDONTWRITEBYTECODE=1`
- `PYTHONUNBUFFERED=1`
- `MPLBACKEND=Agg`

Install flow:

1. copy repo contents into `/artifact`
2. `pip install --upgrade pip`
3. `pip install -r requirements.txt`
4. `pip install --no-deps -e .`

Default command:

- `python scripts/smoke_test.py`

### `run_experiments.sh`

Provide three commands:

- `smoke`
- `plot`
- `reproduce`

Implementation:

- `smoke` -> `python scripts/smoke_test.py`
- `plot` -> `python -m ipomdp_shielding.experiments.plot_experiment_summary`
- `reproduce` -> run `python -m ipomdp_shielding.experiments.run_all_sweeps` and then `python -m ipomdp_shielding.experiments.plot_experiment_summary`

### `scripts/smoke_test.py`

Smoke test should:

1. create a temporary directory
2. copy `results/` there
3. run from that temp directory
4. execute a tiny real threshold sweep for `cartpole_lowacc`
5. save the tiny sweep into `results/experiment/threshold`
6. run `plot_experiment_summary`
7. assert that these files exist and are non-empty:
   - `results/experiment/summary_experiment_bars.png`
   - `results/experiment/barchart_experiment_cartpole_lowacc.png`
   - `evaluation_summary.md`

Use:

- `num_trials = 2`
- `trial_length = 5`
- `exclude_envelope = True`
- config module `rl_shield_cartpole_lowacc_artifact`

Validate the sweep result keys:

- thresholds should match `run_threshold_sweep.THRESHOLDS`
- combos should be exactly:
  - `uniform/rl/single_belief`
  - `adversarial_opt/rl/single_belief`

### `run_all_sweeps.py`

Modify the copied `run_v7_all_sweeps.py` as follows:

- rename module to `run_all_sweeps.py`
- rename output base from `results/sweep_v7` to `results/experiment`
- replace all `rl_shield_*_v7` config names with `rl_shield_*_artifact`
- replace all human-facing `V7` strings with versionless text
- replace `V7:` note prefixes in saved metadata with plain text

Keep runtime estimate and sweep logic otherwise unchanged.

### `plot_experiment_summary.py`

Modify the copied `plot_sweep_v7.py` as follows:

- rename module to `plot_experiment_summary.py`
- read from:
  - `results/experiment/threshold`
  - `results/experiment/obs`
  - `results/experiment/fs`
  - `results/experiment/carr`
- write markdown to `evaluation_summary.md`
- set `_v5.VERSION = "experiment"`
- keep using `plot_sweep_v5` as the underlying plot generator
- override markdown title replacements so final heading is `# Evaluation Summary`

Because `plot_sweep_v5` formats filenames as `barchart_{VERSION}_...`, `pareto_{VERSION}_...`, and `summary_{VERSION}_bars.png`, this produces:

- `barchart_experiment_*.png`
- `pareto_experiment_*.png`
- `summary_experiment_bars.png`

### Artifact config modules

Each renamed `*_artifact.py` config file should:

- import the corresponding base config
- use `dataclasses.replace(...)`
- point `opt_cache_path` to the renamed artifact cache file

Exact target paths:

- taxinet -> `results/cache/rl_shield_taxinet_opt_realization.json`
- obstacle -> `results/cache/rl_shield_obstacle_opt_realization.json`
- cartpole -> `results/cache/rl_shield_cartpole3_sb_opt_realization_single_belief.json`
- cartpole_lowacc -> `results/cache/lowacc_rl_shield_cartpole_opt_realization_single_belief.json`
- refuel_v2 -> `results/cache/rl_shield_refuel_v2_sb_opt_realization.json`

For standard CartPole artifact config, keep:

- `adversarial_opt_targets=["single_belief"]`

## Text / Metadata Cleanup

The final artifact must not expose experiment-number naming.

Remove or rewrite:

- `v7` in README
- `v7` in module docstrings
- `v7` in command names
- `v7` in output paths
- `v7` in markdown title
- `V7:` prefixes inside bundled JSON metadata strings

It is acceptable that some historical helper modules still reference `v5` or `v6` internally if they are not part of the artifact’s public surface. The artifact-facing names and docs should be versionless.

## README Requirements

The artifact `README.md` should state:

- this is a clean, single-history artifact repo
- how to build Docker
- how to install locally
- how to run the smoke test
- how to regenerate figures/summary from bundled JSON
- how to rerun the full experiment suite
- estimated runtime:
  - smoke: about 10-60 seconds
  - plot-only: under 1 minute
  - full reproduce: about 8 hours on CPU

Important wording requirement:

- do not refer to “v7” anywhere
- present it as the only experiment bundle in the artifact

## `.gitignore`

Include at least:

```text
__pycache__/
*.pyc
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

## Git Procedure For The Recreated Artifact Repo

Inside `/home/dev/ipomdp_shielding_artifact`:

1. `git init`
2. `git add .`
3. `git commit -m "Initial artifact"`
4. `git branch -m main`

After the versionless rename cleanup, make a second commit if needed.

## Validation Steps

At minimum, validate:

```bash
bash -n run_experiments.sh
python3 -m py_compile scripts/smoke_test.py \
  ipomdp_shielding/experiments/run_all_sweeps.py \
  ipomdp_shielding/experiments/plot_experiment_summary.py
grep -RIn --exclude-dir=.git -E '\bv7\b|V7|sweep_v7|run_v7|threshold_sweep_7' .
```

Expected:

- shell syntax passes
- python compilation passes
- grep finds nothing

If Docker is available, also run:

```bash
docker build -t ipomdp-shielding-artifact .
docker run --rm ipomdp-shielding-artifact
```

## Notes From The Lost Artifact Build

Two implementation details mattered:

1. The artifact repo was deliberately rebuilt from selected files rather than copied wholesale from the source repo.
2. CPU-only torch was necessary in `requirements.txt`; the generic `torch==2.4.1` wheel tried to pull CUDA packages and was too large.

## Deliverable

The recreated deliverable should be a standalone repo at:

- `/home/dev/ipomdp_shielding_artifact`

with:

- a fresh git history
- versionless naming
- Dockerfile
- smoke test
- README
- only the necessary source/results/caches to reproduce the paper experiment bundle
