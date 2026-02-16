# Modular Experiments Framework

This directory contains a modular framework for running coarseness and RL shielding experiments across different case studies (TaxiNet, CartPole) and experiment sizes (preliminary, full).

## Structure

```
experiments/
├── configs/                          # Configuration files
│   ├── __init__.py
│   ├── base_config.py               # Base config dataclasses
│   ├── coarse_taxinet_prelim.py     # Prelim coarse config for TaxiNet
│   ├── coarse_taxinet_full.py       # Full coarse config for TaxiNet
│   ├── coarse_cartpole_prelim.py    # Prelim coarse config for CartPole
│   ├── coarse_cartpole_full.py      # Full coarse config for CartPole
│   ├── rl_shield_taxinet_prelim.py  # Prelim RL shield config for TaxiNet
│   ├── rl_shield_taxinet_full.py    # Full RL shield config for TaxiNet
│   ├── rl_shield_cartpole_prelim.py # Prelim RL shield config for CartPole
│   └── rl_shield_cartpole_full.py   # Full RL shield config for CartPole
├── run_coarse_experiment.py         # Coarse experiment runner
├── run_rl_shield_experiment.py      # RL shield experiment runner
├── prelim.sh                        # Run all preliminary experiments
├── full.sh                          # Run all full experiments
└── EXPERIMENTS_README.md            # This file
```

## Quick Start

### Run Preliminary Experiments (Fast)

```bash
cd /home/scarbro/claude/ipomdp_shielding/experiments
./prelim.sh
```

This runs 4 experiments:
1. Coarse + TaxiNet (10 trajectories, length 10)
2. Coarse + CartPole (10 trajectories, length 10, 625 states)
3. RL Shield + TaxiNet (10 trials, length 10)
4. RL Shield + CartPole (10 trials, length 10, 625 states)

Results saved to `./data/prelim/`

### Run Full Experiments (Comprehensive)

```bash
cd /home/scarbro/claude/ipomdp_shielding/experiments
./full.sh
```

This runs 4 experiments:
1. Coarse + TaxiNet (100 trajectories, length 20)
2. Coarse + CartPole (100 trajectories, length 20, 625 states)
3. RL Shield + TaxiNet (30 trials, length 20)
4. RL Shield + CartPole (30 trials, length 20, 625 states)

Results saved to `./data/full/`

## Running Individual Experiments

### Coarse Experiment

```bash
python -m ipomdp_shielding.experiments.run_coarse_experiment <config_module>
```

Examples:
```bash
# Preliminary TaxiNet
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_prelim

# Full CartPole
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_cartpole_full
```

### RL Shield Experiment

```bash
python -m ipomdp_shielding.experiments.run_rl_shield_experiment <config_module>
```

Examples:
```bash
# Preliminary TaxiNet
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim

# Full CartPole
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_cartpole_full
```

## Experiment Types

### Coarseness Experiment

**Purpose:** Measures the gap between the LFP over-approximation (BeliefPolytope via LP) and the forward-sampled under-approximation (concrete belief points).

**Key Parameters:**
- `num_trajectories`: Number of trajectories to evaluate
- `trajectory_length`: Length of each trajectory
- `sampler_budget`: Budget for forward sampling
- `sampler_k`: Number of samples to keep (K)
- `sampler_likelihood_strategy`: Likelihood sampling strategy (HYBRID, UNIFORM, etc.)
- `sampler_pruning_strategy`: Pruning strategy (FARTHEST_POINT, etc.)

**Output:**
- JSON file with coarseness metrics (overall max/mean gap, per-timestep gaps)
- PNG plot showing coarseness over time

### RL Shielding Experiment

**Purpose:** Evaluates RL action selection under four shielding strategies, with uniform-random and adversarial-optimized perception realizations.

**Factors:**
1. **Perception Realization:** Uniform Random, Adversarial Optimized
2. **Action Selection:** Random, Best (shield-informed), RL (learned policy)
3. **Shielding Strategy:** None, Observation, Single-Belief, Envelope

**Key Parameters:**
- `num_trials`: Number of Monte Carlo trials
- `trial_length`: Length of each trial
- `rl_episodes`: Number of RL training episodes
- `opt_candidates`: Number of candidates for optimized realization
- `opt_iterations`: Number of optimization iterations
- `shield_threshold`: Threshold for belief-based shields

**Output:**
- JSON file with safety metrics (fail/stuck/safe rates)
- 6 PNG figures (3 outcomes x 2 perceptions)

## CartPole Discretization

CartPole experiments use a configurable discretization that reduces the state space from 7^4=2,401 states to 5^4=625 states, making LFP propagation feasible.

**Current Configuration:**
- All CartPole configs use `ipomdp_kwargs={"num_bins": 5}`
- This creates 625 states instead of 2,401 (4× reduction)
- Makes LFP propagator practical for both coarseness and RL shielding experiments

**Alternative Configurations:**

```python
# Uniform discretization: 4 bins per dimension = 256 states (fastest)
ipomdp_kwargs={"num_bins": 4}

# Non-uniform: higher precision on position/angle = 1,225 states
ipomdp_kwargs={"num_bins": [7, 5, 7, 5]}  # [x, x_dot, theta, theta_dot]
```

**Regenerating Data:**
If you change `num_bins`, you must regenerate the CartPole data files:

```bash
python -m ipomdp_shielding.CaseStudies.CartPole.data_preparation
```

See `ipomdp_shielding/CaseStudies/CartPole/DISCRETIZATION_CONFIG.md` for detailed documentation.

## Configuration Files

Each configuration file defines experiment parameters for a specific case study and size.

### Example: Coarse TaxiNet Preliminary

```python
# configs/coarse_taxinet_prelim.py
from .base_config import CoarseExperimentConfig
from ..CaseStudies.Taxinet import build_taxinet_ipomdp
from ..Propagators import LikelihoodSamplingStrategy, PruningStrategy

config = CoarseExperimentConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    seed=42,
    num_trajectories=10,
    trajectory_length=10,
    initial_state=(0, 0),
    initial_action=0,
    sampler_budget=100,
    sampler_k=10,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="./data/prelim/coarse_taxinet_results.json",
)
```

## Creating Custom Configurations

To create a custom configuration:

1. Create a new config file in `configs/`
2. Import the appropriate base config class (`CoarseExperimentConfig` or `RLShieldExperimentConfig`)
3. Import the case study builder function
4. Define a `config` variable with your parameters
5. Run with: `python -m ipomdp_shielding.experiments.run_<experiment_type> configs.<your_config>`

### Example: Custom Coarse Experiment

```python
# configs/coarse_taxinet_custom.py
from .base_config import CoarseExperimentConfig
from ..CaseStudies.Taxinet import build_taxinet_ipomdp
from ..Propagators import LikelihoodSamplingStrategy, PruningStrategy

config = CoarseExperimentConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    seed=123,
    num_trajectories=50,
    trajectory_length=15,
    initial_state=(0, 0),
    initial_action=0,
    sampler_budget=150,
    sampler_k=15,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.UNIFORM,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="./data/custom/coarse_taxinet_custom.json",
)
```

Run with:
```bash
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_custom
```

## Parameter Tuning Guide

### Preliminary vs Full

**Preliminary (Fast):**
- Purpose: Quick validation, debugging, testing
- Trajectories/Trials: 10
- Length: 10
- Sampler budget: 100 (coarse), RL episodes: 100 (RL shield)
- Runtime: Minutes per experiment

**Full (Comprehensive):**
- Purpose: Publication-quality results
- Trajectories/Trials: 30-100
- Length: 20
- Sampler budget: 200 (coarse), RL episodes: 1000 (RL shield)
- Runtime: Hours per experiment

### Coarse Experiment Parameters

- **`num_trajectories`**: More trajectories = better statistics (10 prelim, 100 full)
- **`trajectory_length`**: Longer = tests long-term propagation (10 prelim, 20 full)
- **`sampler_budget`**: Higher = better under-approximation (100 prelim, 200 full)
- **`sampler_k`**: More samples = finer resolution (10 prelim, 20 full)

### RL Shield Experiment Parameters

- **`num_trials`**: More trials = better statistics (10 prelim, 30 full)
- **`trial_length`**: Longer = tests long-term safety (10 prelim, 20 full)
- **`rl_episodes`**: More episodes = better RL agent (100 prelim, 1000 full)
- **`opt_candidates`**: More candidates = better optimization (5 prelim, 10 full)
- **`opt_iterations`**: More iterations = convergence (5 prelim, 10 full)
- **`adversarial_opt_targets`**: Which shield(s) to optimize the fixed adversarial realization against.
  - Default: `["envelope"]` (legacy behavior).
  - Optional: include `"single_belief"` to train a separate realization optimized against the single-belief shield.
  - When multiple targets are enabled, the `adversarial_opt` perception realization is selected per shield when available.

## Output Files

### Coarse Experiment

```
data/
├── prelim/
│   ├── coarse_taxinet_results.json
│   ├── coarse_taxinet_results.png
│   ├── coarse_cartpole_results.json
│   └── coarse_cartpole_results.png
└── full/
    ├── coarse_taxinet_results.json
    ├── coarse_taxinet_results.png
    ├── coarse_cartpole_results.json
    └── coarse_cartpole_results.png
```

### RL Shield Experiment

```
data/
├── prelim/
│   ├── rl_shield_taxinet_results.json
│   ├── rl_shield_taxinet_figures/
│   │   ├── uniform_fail.png
│   │   ├── uniform_stuck.png
│   │   ├── uniform_safe.png
│   │   ├── adversarial_opt_fail.png
│   │   ├── adversarial_opt_stuck.png
│   │   └── adversarial_opt_safe.png
│   ├── rl_shield_cartpole_results.json
│   └── rl_shield_cartpole_figures/...
└── full/
    ├── rl_shield_taxinet_results.json
    ├── rl_shield_taxinet_figures/...
    ├── rl_shield_cartpole_results.json
    └── rl_shield_cartpole_figures/...
```

## Cache Files

RL shield experiments cache trained models to avoid retraining:

```
/tmp/
├── prelim_rl_shield_taxinet_agent.pt
├── prelim_rl_shield_taxinet_opt_realization.json
├── prelim_rl_shield_cartpole_agent.pt
├── prelim_rl_shield_cartpole_opt_realization.json
├── full_rl_shield_taxinet_agent.pt
├── full_rl_shield_taxinet_opt_realization.json
├── full_rl_shield_cartpole_agent.pt
└── full_rl_shield_cartpole_opt_realization.json
```

To retrain from scratch, delete the relevant cache files.

If `adversarial_opt_targets` contains multiple entries, additional cache files are created by suffixing the base `opt_cache_path`, e.g.:

```
/tmp/full_rl_shield_taxinet_opt_realization.json               # target=envelope
/tmp/full_rl_shield_taxinet_opt_realization_single_belief.json # target=single_belief
```

## Troubleshooting

### Import Errors

Make sure the package is installed:
```bash
cd /home/scarbro/claude
pip install -e .
```

### Out of Memory

Reduce parameters:
- Coarse: Lower `num_trajectories`, `sampler_budget`, `sampler_k`
- RL Shield: Lower `num_trials`, `rl_episodes`, `opt_candidates`

### Slow Performance

Use preliminary configs first to validate setup, then run full experiments overnight or on a compute cluster.

## Migration from Original Experiments

The original `coarse_experiment.py` and `rl_shielding_experiment.py` have been preserved for reference. The new modular structure offers:

1. **Separation of concerns:** Config vs execution
2. **Easy case study switching:** Change one import
3. **Consistent parameter management:** All configs inherit from base classes
4. **Batch execution:** Shell scripts run multiple experiments
5. **No code duplication:** Same runner for all variations

To migrate existing code:
1. Extract your configuration into a new config file
2. Run with the appropriate runner
3. No changes needed to core experiment logic
