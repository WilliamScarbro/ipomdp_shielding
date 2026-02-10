# Quick Reference Guide

## Run All Preliminary Experiments (Fast)

```bash
cd /home/scarbro/claude/ipomdp_shielding/experiments
./prelim.sh
```

**Runs:**
- Coarse + TaxiNet (10 traj × 10 steps)
- Coarse + CartPole (10 traj × 10 steps)
- RL Shield + TaxiNet (10 trials × 10 steps)
- RL Shield + CartPole (10 trials × 10 steps)

**Output:** `./data/prelim/`

---

## Run All Full Experiments (Comprehensive)

```bash
cd /home/scarbro/claude/ipomdp_shielding/experiments
./full.sh
```

**Runs:**
- Coarse + TaxiNet (100 traj × 20 steps)
- Coarse + CartPole (100 traj × 20 steps)
- RL Shield + TaxiNet (30 trials × 20 steps)
- RL Shield + CartPole (30 trials × 20 steps)

**Output:** `./data/full/`

---

## Run Individual Experiments

### Coarse Experiments

```bash
# TaxiNet Preliminary
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_prelim

# TaxiNet Full
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_full

# CartPole Preliminary
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_cartpole_prelim

# CartPole Full
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_cartpole_full
```

### RL Shield Experiments

```bash
# TaxiNet Preliminary
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim

# TaxiNet Full
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_full

# CartPole Preliminary
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_cartpole_prelim

# CartPole Full
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_cartpole_full
```

---

## Output Files

### Coarse Experiments

```
./data/prelim/coarse_taxinet_results.json    # Coarseness metrics
./data/prelim/coarse_taxinet_results.png     # Coarseness plot
./data/prelim/coarse_cartpole_results.json
./data/prelim/coarse_cartpole_results.png

./data/full/coarse_taxinet_results.json
./data/full/coarse_taxinet_results.png
./data/full/coarse_cartpole_results.json
./data/full/coarse_cartpole_results.png
```

### RL Shield Experiments

```
./data/prelim/rl_shield_taxinet_results.json         # Safety metrics
./data/prelim/rl_shield_taxinet_figures/             # 6 figures
    ├── uniform_fail.png
    ├── uniform_stuck.png
    ├── uniform_safe.png
    ├── adversarial_opt_fail.png
    ├── adversarial_opt_stuck.png
    └── adversarial_opt_safe.png

./data/prelim/rl_shield_cartpole_results.json
./data/prelim/rl_shield_cartpole_figures/

./data/full/rl_shield_taxinet_results.json
./data/full/rl_shield_taxinet_figures/
./data/full/rl_shield_cartpole_results.json
./data/full/rl_shield_cartpole_figures/
```

---

## Cache Files (Auto-generated)

RL shield experiments cache trained models to `/tmp/`:

```
/tmp/prelim_rl_shield_taxinet_agent.pt              # RL agent
/tmp/prelim_rl_shield_taxinet_opt_realization.json  # Optimized realization
/tmp/prelim_rl_shield_cartpole_agent.pt
/tmp/prelim_rl_shield_cartpole_opt_realization.json

/tmp/full_rl_shield_taxinet_agent.pt
/tmp/full_rl_shield_taxinet_opt_realization.json
/tmp/full_rl_shield_cartpole_agent.pt
/tmp/full_rl_shield_cartpole_opt_realization.json
```

**To retrain from scratch:** Delete the relevant cache file

---

## Configuration Files

All configs in: `ipomdp_shielding/experiments/configs/`

### Coarse Experiment Configs
- `coarse_taxinet_prelim.py`
- `coarse_taxinet_full.py`
- `coarse_cartpole_prelim.py`
- `coarse_cartpole_full.py`

### RL Shield Experiment Configs
- `rl_shield_taxinet_prelim.py`
- `rl_shield_taxinet_full.py`
- `rl_shield_cartpole_prelim.py`
- `rl_shield_cartpole_full.py`

---

## Creating Custom Configs

1. Copy an existing config file
2. Modify parameters
3. Save with new name (e.g., `my_custom_config.py`)
4. Run: `python -m ipomdp_shielding.experiments.run_<experiment_type> configs.my_custom_config`

Example:
```python
# configs/coarse_taxinet_custom.py
from .base_config import CoarseExperimentConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy

config = CoarseExperimentConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    seed=123,              # Custom seed
    num_trajectories=50,   # Custom count
    trajectory_length=15,  # Custom length
    initial_state=(0, 0),
    initial_action=0,
    sampler_budget=150,
    sampler_k=15,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="./data/custom/my_results.json",
)
```

Run:
```bash
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_custom
```

---

## Troubleshooting

### Import Error
```bash
# Make sure package is installed
cd /home/scarbro/claude
pip install -e .
```

### Out of Memory
Reduce parameters in config file:
- Coarse: Lower `num_trajectories`, `sampler_budget`, `sampler_k`
- RL Shield: Lower `num_trials`, `rl_episodes`, `opt_candidates`

### Cache Issues
Delete cache files to retrain:
```bash
rm /tmp/prelim_rl_shield_*
rm /tmp/full_rl_shield_*
```

---

## Documentation

- **QUICK_REFERENCE.md** (this file): Quick command reference
- **EXPERIMENTS_README.md**: Comprehensive guide with details
- **MODULARIZATION_SUMMARY.md**: Implementation details and architecture

---

## Original Experiments (Still Work)

```bash
# Original coarse experiment (TaxiNet only)
python -m ipomdp_shielding.experiments.coarse_experiment

# Original RL shield experiment (TaxiNet only)
python -m ipomdp_shielding.experiments.rl_shielding_experiment
```

These remain unchanged for backward compatibility.
