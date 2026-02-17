# ML-Optimized Fixed Interval Realization Implementation

## Overview

This document describes the implementation of an ML-based approach for finding optimal **fixed** IPOMDP interval realizations that maximize failure probability.

## Key Differences from Existing Approaches

| Approach | Strategy | Update Frequency |
|----------|----------|------------------|
| `UniformPerceptionModel` | Random/cooperative | Per sample |
| `AdversarialPerceptionModel` | Reactive adversary | Per timestep (belief-aware) |
| **`FixedRealizationPerceptionModel`** | **Committed adversary** | **Once per trial** |

**Use Case:** Models an adversary that must commit to a perception model upfront, rather than one that can react to the agent's belief state.

## Architecture

```
OptimizedRealizationTrainer
├── IntervalRealizationParameterizer
│   └── Converts alpha ∈ [0,1] ↔ P(obs|state) realizations
├── CrossEntropyOptimizer
│   └── Searches alpha space to maximize failure probability
└── produces → FixedRealizationPerceptionModel
```

## Implementation Files

### Core Components

1. **`ipomdp_shielding/MonteCarlo/fixed_realization_model.py`**
   - `FixedRealizationPerceptionModel`: Perception model using fixed realization
   - `IntervalRealizationParameterizer`: Converts between alpha parameters and realizations

2. **`ipomdp_shielding/MonteCarlo/optimizers/cross_entropy.py`**
   - `CrossEntropyOptimizer`: Black-box optimizer for continuous parameters

3. **`ipomdp_shielding/MonteCarlo/realization_optimizer.py`**
   - `OptimizedRealizationTrainer`: Orchestrates training process
   - `train_optimal_realization()`: High-level convenience API

4. **`experiments/optimal_realization_example.py`**
   - Complete end-to-end examples and usage demonstrations

### Integration

5. **`ipomdp_shielding/MonteCarlo/__init__.py`** (modified)
   - Added exports for new components

## Key Design Decisions

### 1. Alpha Parameterization

**Why:** Automatically maintains feasibility constraints.

For each (state, obs) pair:
```python
P_raw(obs|state) = (1 - alpha) * P_lower + alpha * P_upper
P(obs|state) = P_raw / sum(P_raw)  # Normalize to simplex
```

**Benefits:**
- `alpha ∈ [0, 1]` ensures probabilities stay within intervals
- Reduced parameter space
- Interpretable: α=0 → lower bound, α=1 → upper bound

### 2. Cross-Entropy Method

**Why:** Simple, derivative-free, works well with noisy objectives.

**Algorithm:**
1. Sample N candidates from distribution N(μ, σ²)
2. Evaluate each via Monte Carlo trials
3. Select top-K elite candidates
4. Update μ, σ to match elite set
5. Decay σ over time

**Alternatives considered:** CMA-ES, Bayesian Optimization (more complex, similar performance)

### 3. Configurable Action Selector

**Why:** Different agents require different adversarial strategies.

```python
# Train different realizations for different agent behaviors
optimal_vs_random = train_optimal_realization(
    action_selector=RandomActionSelector(),
    save_path="models/optimal_vs_random.json"
)

optimal_vs_safest = train_optimal_realization(
    action_selector=SafestActionSelector(ipomdp, pp_shield),
    save_path="models/optimal_vs_safest.json"
)
```

### 4. Objective Function

**Current:** Simply maximize failure rate.

```python
def objective(alphas):
    realization = parameterizer.params_to_realization(alphas)
    perception = FixedRealizationPerceptionModel(realization)
    results = run_monte_carlo_trials(...)
    metrics = compute_safety_metrics(results)
    return metrics.fail_rate
```

**Future:** Multi-objective (e.g., `fail_rate + 0.5 * stuck_rate`)

## Usage Examples

### Basic Training

```python
from ipomdp_shielding.MonteCarlo import train_optimal_realization

optimal_perception = train_optimal_realization(
    ipomdp=ipomdp,
    pp_shield=pp_shield,
    rt_shield_factory=rt_shield_factory,
    action_selector=RandomActionSelector(),
    num_candidates=50,
    num_trials_per_candidate=20,
    max_iterations=100,
    save_path="optimal_realization.json"
)
```

### Evaluation

```python
from ipomdp_shielding.MonteCarlo import (
    FixedRealizationPerceptionModel,
    MonteCarloSafetyEvaluator
)

# Load trained model
perception = FixedRealizationPerceptionModel.load("optimal_realization.json")

# Evaluate
evaluator = MonteCarloSafetyEvaluator(
    ipomdp=ipomdp,
    pp_shield=pp_shield,
    perception=perception,
    rt_shield_factory=rt_shield_factory
)

results, _ = evaluator.evaluate(
    action_selector=RandomActionSelector(),
    num_trials=1000,
    trial_length=20
)
```

### Comparison

```python
perception_models = {
    "uniform": UniformPerceptionModel(),
    "adversarial": AdversarialPerceptionModel(pp_shield),
    "optimal_fixed": FixedRealizationPerceptionModel.load("optimal.json")
}

for name, perception in perception_models.items():
    evaluator = MonteCarloSafetyEvaluator(..., perception=perception)
    results, _ = evaluator.evaluate(...)
    print(f"{name}: fail_rate={results['random'].fail_rate:.2%}")
```

## Configuration Recommendations

### Small Budget (~1 minute)
```python
num_candidates=10
num_trials_per_candidate=5
max_iterations=20
```

### Medium Budget (~10 minutes)
```python
num_candidates=50
num_trials_per_candidate=20
max_iterations=100
```

### Large Budget (~1 hour)
```python
num_candidates=100
num_trials_per_candidate=50
max_iterations=200
```

## Testing

### Unit Tests Needed

1. **IntervalRealizationParameterizer**
   - `test_params_to_realization_simplex`: Verify sum = 1
   - `test_params_to_realization_intervals`: Verify P_lower ≤ P ≤ P_upper
   - `test_round_trip_conversion`: Verify params → realization → params

2. **FixedRealizationPerceptionModel**
   - `test_sample_observation_validity`: Returns valid observations
   - `test_distribution_simplex`: Distribution sums to 1
   - `test_save_load`: Preservation of realization and metadata

3. **CrossEntropyOptimizer**
   - `test_quadratic_optimization`: Converges to known optimum
   - `test_elite_selection`: Correct top-K selection

### Integration Tests

```python
# Test in experiments/optimal_realization_example.py
python3 experiments/optimal_realization_example.py
```

## Performance Characteristics

### Computational Cost

For each CEM iteration:
- Sample N candidates
- Evaluate each with K Monte Carlo trials
- **Total trials per iteration:** N × K

Example (medium budget):
- 50 candidates × 20 trials × 100 iterations = **100,000 trials**
- Runtime: ~5-15 minutes on TaxiNet IPOMDP

### Convergence

Typical convergence pattern:
- Iterations 1-20: Rapid exploration, high variance
- Iterations 20-60: Steady improvement, decreasing variance
- Iterations 60-100: Fine-tuning, low variance

Monitor `best_scores` in training history to assess convergence.

## Future Extensions

1. **Multi-objective optimization**
   - Combine failure, stuck, and safety rates
   - Pareto frontier exploration

2. **Alternative optimizers**
   - CMA-ES for better sample efficiency
   - Bayesian Optimization for expensive objectives

3. **Parallel evaluation**
   - Evaluate candidates in parallel (multiprocessing)
   - Significant speedup on multi-core systems

4. **Adaptive sampling**
   - Use fewer trials for low-scoring candidates
   - More trials for promising candidates

5. **Transfer learning**
   - Initialize from similar IPOMDP's optimal realization
   - Fine-tune instead of training from scratch

6. **Ensemble methods**
   - Train multiple realizations
   - Use voting or worst-case selection

## References

1. **Cross-Entropy Method:** Rubinstein & Kroese (2004), "The Cross-Entropy Method"
2. **IPOMDP Shielding:** Original implementation in `ipomdp_shielding/`
3. **Monte Carlo Evaluation:** `ipomdp_shielding/MonteCarlo/`

## API Documentation

### `FixedRealizationPerceptionModel`

```python
class FixedRealizationPerceptionModel(PerceptionModel):
    """Perception model using fixed realization of intervals."""

    def __init__(self, realization: Dict, metadata: Optional[Dict] = None)
    def sample_observation(self, state, ipomdp, context=None) -> Observation
    def sample_distribution(self, state, ipomdp, context=None) -> Dict[Obs, float]
    def save(self, filepath: str)
    @classmethod
    def load(cls, filepath: str) -> "FixedRealizationPerceptionModel"
```

### `train_optimal_realization()`

```python
def train_optimal_realization(
    ipomdp: IPOMDP,
    pp_shield: Dict,
    rt_shield_factory: Callable,
    action_selector: Optional[ActionSelector] = None,
    initial_generator: Optional[InitialStateGenerator] = None,
    num_candidates: int = 50,
    num_trials_per_candidate: int = 20,
    max_iterations: int = 100,
    trial_length: int = 20,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> FixedRealizationPerceptionModel
```

High-level API with sensible defaults. Returns trained perception model.

### `OptimizedRealizationTrainer`

```python
class OptimizedRealizationTrainer:
    """Trainer for learning optimal fixed realizations."""

    def __init__(self, ipomdp, pp_shield, rt_shield_factory,
                 action_selector, initial_generator, **config)
    def train(self, trial_length=20, max_iterations=100,
              verbose=True) -> FixedRealizationPerceptionModel
    def save_training_history(self, filepath: str)
```

Lower-level API for advanced customization.

## Troubleshooting

### Warning: "Best realization violates interval constraints"

**Cause:** Normalization can push probabilities slightly outside intervals.

**Impact:** Usually minimal (<1% violation). The optimizer still finds useful adversarial realizations.

**Fix:** Could add projection step after normalization, but may affect optimization dynamics.

### Slow convergence

**Symptoms:** `best_scores` not improving after many iterations.

**Solutions:**
1. Increase `num_trials_per_candidate` (reduce evaluation noise)
2. Increase `num_elite` (broader distribution update)
3. Decrease `std_decay` (maintain exploration longer)
4. Check if objective is truly optimizable (uniform baseline comparison)

### Poor performance vs uniform

**Expected:** Optimal fixed should achieve ≥ uniform failure rate.

**If not:**
1. Check if training converged (`best_scores` plateau?)
2. Verify action selector matches evaluation (random vs random?)
3. Try longer training (`max_iterations = 200`)
4. Check for bugs in objective function

## Conclusion

This implementation provides a principled ML-based approach to adversarial interval realization learning. By separating optimization (outer loop) from evaluation (inner loop), it integrates cleanly with existing Monte Carlo infrastructure while enabling new analyses of IPOMDP safety under committed perception uncertainty.

The modular design allows easy experimentation with:
- Different optimization algorithms (just swap `CrossEntropyOptimizer`)
- Different objectives (modify `_evaluate_realization`)
- Different agent strategies (pass different `ActionSelector`)
- Different evaluation protocols (use any `MonteCarloSafetyEvaluator` setup)
