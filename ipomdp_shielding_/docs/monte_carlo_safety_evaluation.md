# Monte Carlo Safety Evaluation for Shielding

## Overview

The Monte Carlo safety evaluation module provides a simulation-based framework for measuring the safety and effectiveness of runtime shielding strategies in Interval POMDPs (IPOMDPs). It complements analytical approaches by empirically testing shields across diverse scenarios through randomized trials.

## Motivation

While perfect-perception shields provide theoretical safety guarantees, runtime shields operating under imperfect perception may:
- **Fail**: Allow transitions to unsafe states
- **Get stuck**: Over-constrain actions, leaving no valid moves
- **Succeed**: Maintain safety while allowing progress

Monte Carlo simulation quantifies these outcomes across best-case, average-case, and worst-case scenarios.

## Architecture

### 1. Action Selection Strategies

The framework uses a **Strategy Pattern** for action selection, allowing seamless integration with reinforcement learning policies or simple baselines.

**Base Interface:**
```python
class ActionSelector(ABC):
    def select(self, history, allowed_actions) -> action
```

**Implementations:**
- `RandomActionSelector`: Uniform random selection from allowed actions (baseline)
- `UniformFallbackSelector`: Random selection with fallback to all actions if shield gets stuck

The `history` parameter provides the full observation-action sequence, enabling RL policies to make informed decisions while respecting shield constraints.

### 2. Initial State Sampling

Different sampling strategies enable **best/worst/average case analysis**:

| Strategy | Purpose | Selection Criterion |
|----------|---------|---------------------|
| `RandomInitialState` | Average case | Uniform sampling from all states |
| `SafeInitialState` | Best case | States with maximum safe actions |
| `BoundaryInitialState` | Worst case | States with minimum safe actions |

This design isolates shield performance under varying levels of constraint.

### 3. Core Trial Execution

**Single Trial Flow (`run_single_trial`):**

```
1. Initialize state and action
2. For each step (up to trial_length):
   a. Check if state == "FAIL" → outcome = "fail"
   b. Obtain observation via perception function
   c. Query shield for allowed_actions
   d. If no allowed actions → outcome = "stuck"
   e. Select action using action_selector
   f. Evolve state via IPOMDP dynamics
3. Return SafetyTrialResult (outcome, steps, stuck_count, trajectory)
```

**Key Design Decisions:**
- Each trial uses a **fresh shield instance** (via factory pattern) to ensure independence
- The shield is restarted between trials to clear any accumulated history
- Trajectories are optionally stored (memory vs. detail tradeoff)

### 4. Metrics Aggregation

The `MCSafetyMetrics` dataclass captures:
- **Fail rate**: Fraction of trials reaching FAIL state
- **Stuck rate**: Fraction with no allowed actions
- **Safe rate**: Fraction completing without issues
- **Mean steps**: Average trajectory length
- **Mean stuck count**: Average temporary stuck events per trial
- **Fail step distribution**: Histogram of when failures occur

## Usage Example

```python
from ipomdp_shielding.Evaluation.monte_carlo_safety import (
    MonteCarloSafetyEvaluator,
    RandomActionSelector,
    plot_safety_metrics
)

# Setup
evaluator = MonteCarloSafetyEvaluator(
    ipomdp=my_ipomdp,
    pp_shield=perfect_perception_shield,
    perception=lambda s: my_perception_model(s),
    rt_shield_factory=lambda: RuntimeImpShield(my_ipomdp, imperfect_model)
)

# Run evaluation across scenarios
results = evaluator.evaluate(
    action_selector=RandomActionSelector(),
    num_trials=1000,
    trial_length=50,
    sampling_modes=["random", "best_case", "worst_case"],
    seed=42
)

# Display results
for mode, metrics in results.items():
    print(f"\n{mode}:")
    print(metrics)

# Visualize
plot_safety_metrics(results, save_path="images/safety_eval.png")
```

## Interpretation Guide

### Safety Rates

- **High fail rate**: Shield is too permissive (under-constraining)
- **High stuck rate**: Shield is too conservative (over-constraining)
- **High safe rate**: Shield achieves good balance

### Sampling Mode Comparison

- **Best case** should show lowest fail/stuck rates (validates shield works in favorable conditions)
- **Worst case** reveals shield limitations near safety boundaries
- **Random** provides expected performance under typical operation

### Failure Step Distribution

- **Early failures** (low step count): Shield fails quickly, possibly due to poor initialization
- **Late failures** (high step count): Shield degrades over time, possibly due to accumulated uncertainty
- **Uniform distribution**: Failures not correlated with trajectory length

## Design Patterns

1. **Strategy Pattern**: Action selectors are interchangeable
2. **Factory Pattern**: Shield instances created per-trial via factory function
3. **Template Method**: Initial state generators follow common interface
4. **Dataclass Pattern**: Immutable result structures for clean data flow

## Extensions

The modular design supports:
- **Custom action selectors**: Integrate trained RL policies
- **Custom initial generators**: Add domain-specific sampling (e.g., near obstacles)
- **Custom metrics**: Extend `SafetyTrialResult` with task-specific measurements
- **Parallel execution**: Trials are independent and can be distributed

## Relationship to Other Components

- **Input**: Requires `RuntimeImpShield` (from `runtime_shield.py`) and perfect-perception shield
- **Comparison**: Results can be compared against analytical template comparisons (see `template_comparison.py`)
- **Ground truth**: Provides empirical validation for theoretical safety bounds

## Performance Considerations

- **Memory**: Set `store_trajectories=False` for large-scale trials (only metrics are retained)
- **Computation**: Trials are embarrassingly parallel; consider multiprocessing for >10k trials
- **Randomness**: Always set `seed` parameter for reproducible results

## Validation

The framework tracks **stuck count** separately from outcome to distinguish:
- **Temporary stuck**: Shield recovers after fallback (if using `UniformFallbackSelector`)
- **Terminal stuck**: No recovery possible (trial ends with outcome="stuck")

This provides insight into shield robustness vs. irrecoverable failure modes.
