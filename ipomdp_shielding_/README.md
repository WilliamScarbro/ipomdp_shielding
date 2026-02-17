# IPOMDP Shielding

Runtime shielding for systems with imprecise perception, using Interval POMDPs (IPOMDPs). Compares belief-envelope shielding (lifted shield) against support-based shielding (Carr et al.) on the TaxiNet and CartPole case studies.

## Installation

```bash
pip install -e .
pip install numpy scipy statsmodels matplotlib torch
```

## Running Experiments

All experiments are run from the project root. Results are written to `results/`.

```bash
# Quick preliminary experiments (small parameters, fast)
./run_experiments.sh prelim

# Full experiments (publication parameters)
./run_experiments.sh full

# Carr shield comparison
./run_experiments.sh carr

# Generate paper artifacts from existing results
./run_experiments.sh artifacts
```

To run individual experiments:

```bash
# Coarseness experiment with a specific config
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_prelim

# RL shielding experiment
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim

# Carr comparison
python -m ipomdp_shielding.experiments.run_carr_comparison default
```

## Project Structure

```
ipomdp_shielding_/
├── ipomdp_shielding/           # Python package
│   ├── Models/                 # IPOMDP, POMDP, MDP, IMDP models
│   ├── Propagators/            # Belief propagation (LFP, exact HMM, min-max)
│   ├── Evaluation/             # Shield evaluation, runtime shields, metrics
│   ├── MonteCarlo/             # Monte Carlo safety evaluation, RL agents
│   ├── CaseStudies/            # TaxiNet and CartPole case studies
│   └── experiments/            # Experiment runners and configs
│       ├── configs/            # All experiment configurations
│       └── sweeps/             # Parameter sweep experiments
├── docs/                       # Documentation (see below)
│   └── archive/               # Implementation summaries and references
├── scripts/                    # Standalone debug/test scripts
├── results/                    # Experiment outputs (not tracked in git)
│   ├── prelim/                 # Preliminary experiment results
│   ├── full/                   # Full experiment results
│   ├── carr_comparison/        # Carr comparison results
│   └── cache/                  # RL agent and realization caches
├── models/                     # Trained model weights
├── run_experiments.sh          # Single entry point for all experiments
└── pyproject.toml
```

## Documentation

Each file in `docs/` corresponds to a code module:

| Document | Code Module |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Overall system design |
| [models.md](docs/models.md) | `Models/` (MDP, IMDP, POMDP, IPOMDP) |
| [belief_propagation.md](docs/belief_propagation.md) | `Propagators/` (LFP, HMM, sampling) |
| [runtime_shielding.md](docs/runtime_shielding.md) | `Evaluation/runtime_shield.py` |
| [coarseness_measurement.md](docs/coarseness_measurement.md) | `Evaluation/coarseness_evaluator.py` |
| [monte_carlo.md](docs/monte_carlo.md) | `MonteCarlo/` (simulation, evaluation) |
| [neural_action_selector.md](docs/neural_action_selector.md) | `MonteCarlo/neural_action_selector.py` |
| [experiments.md](docs/experiments.md) | `experiments/` (configs, runners) |
| [cartpole.md](docs/cartpole.md) | `CaseStudies/CartPole/` |

`docs/archive/` contains implementation summaries, reference papers, and historical notes.
