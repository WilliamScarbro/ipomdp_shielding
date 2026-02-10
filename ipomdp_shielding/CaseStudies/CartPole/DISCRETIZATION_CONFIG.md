# Configurable Discretization for CartPole

## Overview

The CartPole case study now supports configurable discretization with different numbers of bins per state dimension. This allows you to trade off state space size (and computational cost) against precision.

## State Space Size

The state space grows exponentially with the number of bins:
- **Original (7×7×7×7)**: 2,401 states → **Too slow for LFP propagator**
- **Recommended (5×5×5×5)**: 625 states → **Feasible**
- **Very coarse (4×4×4×4)**: 256 states → **Fast**
- **Minimal (3×3×3×3)**: 81 states → **Very fast**

## Usage

### Uniform Discretization

Use the same number of bins for all dimensions:

```python
from ipomdp_shielding.CaseStudies.CartPole import build_cartpole_ipomdp

# 5 bins per dimension → 625 states
ipomdp, pp_shield, test_data, _ = build_cartpole_ipomdp(
    num_bins=5,
    seed=42
)
```

### Non-Uniform Discretization

Specify different numbers of bins per dimension for targeted precision:

```python
# Higher precision on position and angle, lower on velocities
# Format: [n_x, n_xdot, n_theta, n_thetadot]
num_bins_config = [7, 5, 7, 5]  # → 1,225 states

ipomdp, pp_shield, test_data, _ = build_cartpole_ipomdp(
    num_bins=num_bins_config,
    seed=42
)
```

**Rationale**: Position and angle are typically more critical for safety predicates than velocities in CartPole.

## Generating Data Files

When using non-standard discretization, you must regenerate the data files:

```python
from ipomdp_shielding.CaseStudies.CartPole.data_preparation import prepare_all_data

# Generate data with 5 bins per dimension
prepare_all_data(
    perception_episodes=200,
    dynamics_episodes=10000,
    num_bins=5,  # Or [7, 5, 7, 5] for non-uniform
    epochs=20,
    seed=0,
    device="cuda"
)
```

This will create:
- `artifacts/bin_edges.npy`: Bin edges for each dimension
- `artifacts/x_confusion.npy`: Confusion matrix for x dimension
- `artifacts/x_dot_confusion.npy`: Confusion matrix for x_dot dimension
- `artifacts/theta_confusion.npy`: Confusion matrix for theta dimension
- `artifacts/theta_dot_confusion.npy`: Confusion matrix for theta_dot dimension
- `artifacts/cartpole_state_net.pt`: Trained perception model
- `artifacts/dynamics_mdp.pkl`: Empirical dynamics MDP

## Experiment Configurations

Updated experiment configs use coarser discretization:

```python
# experiments/configs/coarse_cartpole_prelim.py
config = CoarseExperimentConfig(
    # ... other params ...
    initial_state=(2, 2, 2, 2),  # Center state for 5 bins
    ipomdp_kwargs={"num_bins": 5},  # 625 states
)
```

## Complexity Analysis for LFP Propagator

The LFP propagator's computational cost scales with:
1. **LP dimension**: `4n` variables where `n` = number of states
2. **Number of LPs**: `2K` per belief update (K = number of templates)

### Recommended Upper Limits

| Use Case | Max States | Example Config |
|----------|------------|----------------|
| Real-time/Interactive | ≤ 500 | 5×5×5×4 = 500 |
| Batch experiments | ≤ 1000 | 7×5×7×5 = 1,225 |
| Overnight jobs (max) | ≤ 1500 | 6×6×6×7 = 1,512 |

**The original 2,401 states is ~50-100× too slow.**

## Discretization Strategies

### Strategy 1: Uniform Coarsening
- Use same bins for all dimensions
- Simple and predictable
- **Recommended for initial experiments**: `num_bins=5` (625 states)

### Strategy 2: Prioritize Safety-Critical Dimensions
- Higher precision on position (`x`) and angle (`theta`)
- Lower precision on velocities (`x_dot`, `theta_dot`)
- **Example**: `[7, 4, 7, 4]` → 784 states

### Strategy 3: Adaptive Around Safety Boundary
- Use finer bins near termination thresholds:
  - `|x| ≤ 2.4` (position limit)
  - `|θ| ≤ 0.209` rad (angle limit ~12°)
- Currently requires custom bin edge generation

## Example Script

Run `example_coarse_discretization.py` to see different configurations:

```bash
python -m ipomdp_shielding.CaseStudies.CartPole.example_coarse_discretization
```

Output shows state space sizes and feasibility estimates for each configuration.

## Validation

After changing discretization:
1. Regenerate all data files with `data_preparation.py`
2. Run perception bounds test: `cartpole_model_test.py`
3. Verify violation rate ≤ α (typically 5%)

## Notes

- **Bin edges** are computed from training data to cover observed ranges
- **Confusion matrices** adapt to per-dimension bin counts
- **Dynamics MDP** is collected using the same discretization
- **Perception IMDP** is factored across dimensions (independence assumption)

## Future Work

- [ ] Adaptive discretization based on belief density
- [ ] Non-uniform bin widths (e.g., finer near safety boundary)
- [ ] Dimension reduction via PCA before discretization
- [ ] Automatic bin count selection based on sample size
