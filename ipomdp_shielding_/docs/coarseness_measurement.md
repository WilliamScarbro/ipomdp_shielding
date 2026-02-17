# Coarseness Measurement

This document describes the system for measuring how coarse the LFP over-approximation is, by comparing it against a sampling-based under-approximation.

## Motivation

The `LFPPropagator` computes an **over-approximation** of the true reachable belief set. This means its probability queries are conservative: `min_allowed_lfp <= min_allowed_true`. But *how* conservative? If the gap is large, the shield rejects actions that are actually safe, leading to the agent getting "stuck."

By running a `ForwardSampledBelief` in parallel, we obtain an **under-approximation** whose queries bound the true values from the other side. The gap between the two is a sound upper bound on the true coarseness.

## Mathematical Basis

Let `P_lfp`, `P_true`, and `P_sampled` denote the belief sets from the LFP over-approximation, the true reachable set, and the sampled under-approximation, respectively.

**Set containment**:
```
P_sampled  ⊆  P_true  ⊆  P_lfp
```

**Probability ordering** (for an allowed state set):
```
min_allowed_lfp  ≤  min_allowed_true  ≤  min_allowed_sampled
```

The minimum is taken over a larger set on the left, so the value can only decrease.

**Probability ordering** (for a disallowed state set):
```
max_disallowed_sampled  ≤  max_disallowed_true  ≤  max_disallowed_lfp
```

The maximum is taken over a larger set on the right, so the value can only increase.

**Coarseness gaps** (sound upper bounds on true approximation error):
```
safe_gap   = min_allowed_sampled - min_allowed_lfp      ≥ 0
unsafe_gap = max_disallowed_lfp  - max_disallowed_sampled  ≥ 0
```

These gaps are always non-negative by construction. They bound how much the true values could differ from the LFP values. As the sampled set grows (larger budget), `P_sampled -> P_true` and the gaps tighten.

## Components

### ForwardSampledBelief

**File**: `Propagators/forward_sampled_belief.py`

The under-approximation propagator. See [belief_propagation.md](belief_propagation.md) for full details.

Key parameters affecting coarseness measurement:
- **`budget`**: Number of belief points. Larger budget -> tighter under-approximation -> smaller measured gaps. Typical values: 100-1000.
- **`K_samples`**: Likelihood samples per step. More samples explore more of the observation uncertainty. Typical values: 10-50.
- **`likelihood_strategy`**: `HYBRID` (default) gives good coverage with structured + random samples.
- **`pruning_strategy`**: `COORDINATE_EXTREMAL` (default) preserves extreme points that drive min/max queries.

### Coarseness Data Classes

**File**: `Evaluation/coarseness_evaluator.py`

#### CoarsenessStepResult

Per-action result at a single timestep:

```python
@dataclass
class CoarsenessStepResult:
    action: Any
    min_allowed_lfp: float
    min_allowed_sampled: float
    max_disallowed_lfp: float
    max_disallowed_sampled: float

    safe_gap: float    # property: max(0, min_allowed_sampled - min_allowed_lfp)
    unsafe_gap: float  # property: max(0, max_disallowed_lfp - max_disallowed_sampled)
```

#### CoarsenessSnapshot

Per-timestep collection across all actions:

```python
@dataclass
class CoarsenessSnapshot:
    step: int
    action_results: List[CoarsenessStepResult]

    max_safe_gap: float     # property: max across actions
    max_unsafe_gap: float   # property: max across actions
    mean_safe_gap: float    # property: mean across actions
    mean_unsafe_gap: float  # property: mean across actions
```

#### CoarsenessReport

Full trajectory of snapshots:

```python
@dataclass
class CoarsenessReport:
    snapshots: List[CoarsenessSnapshot]

    # Time series accessors
    max_safe_gaps: List[float]       # per-timestep max safe gap
    max_unsafe_gaps: List[float]     # per-timestep max unsafe gap
    mean_safe_gaps: List[float]      # per-timestep mean safe gap

    # Summary statistics
    overall_max_safe_gap: float      # max across all timesteps and actions
    overall_max_unsafe_gap: float
    overall_mean_safe_gap: float
    overall_mean_unsafe_gap: float
```

### CoarsenessEvaluator

Drives both propagators in tandem:

```python
evaluator = CoarsenessEvaluator(lfp, sampler, pp_shield)

# Step-by-step
snapshot = evaluator.step(action, obs)

# Or run a full trajectory
report = evaluator.run_trajectory([(obs0, action0), (obs1, action1), ...])
```

Internally:
1. Inverts `pp_shield` to get `action -> safe state indices` and `action -> unsafe state indices`.
2. On each `step()`, propagates both LFP and sampler, then queries both for each action.
3. Computes gaps and assembles results.

### CoarsenessMetricsCollector

Adapts coarseness measurement to the `MetricsCollector` interface used by the evaluation infrastructure:

```python
collector = CoarsenessMetricsCollector(sampler, pp_shield)

# Called by the evaluation harness after each shield step
metrics = collector.compute(rt_shield, step)
# Returns: max_safe_gap, max_unsafe_gap, mean_safe_gap, num_sample_points
```

**Important**: The caller must ensure `sampler.propagate(action, obs)` is called with the same evidence as the shield's propagator before calling `compute()`.

Emitted metrics:

| Metric | Description |
|---|---|
| `max_safe_gap` | Max safe gap across all actions at this step |
| `max_unsafe_gap` | Max unsafe gap across all actions at this step |
| `mean_safe_gap` | Mean safe gap across all actions at this step |
| `num_sample_points` | Current number of sample points (= budget) |

## Usage Example

```python
import numpy as np
from ipomdp_shielding.Models import IPOMDP
from ipomdp_shielding.Propagators import (
    LFPPropagator, ForwardSampledBelief, BeliefPolytope
)
from ipomdp_shielding.Propagators.lfp_propagator import default_solver, TemplateFactory
from ipomdp_shielding.Evaluation.coarseness_evaluator import CoarsenessEvaluator

# Setup
n = len(ipomdp.states)
solver = default_solver()
template = TemplateFactory.canonical(n)

lfp = LFPPropagator(
    ipomdp=ipomdp,
    template=template,
    solver=solver,
    belief=BeliefPolytope.uniform_prior(n),
)

sampler = ForwardSampledBelief(
    ipomdp=ipomdp,
    budget=200,
    K_samples=20,
    rng=np.random.default_rng(42),
)

evaluator = CoarsenessEvaluator(lfp, sampler, pp_shield)

# Run trajectory
history = [(obs_0, action_0), (obs_1, action_1), ...]
report = evaluator.run_trajectory(history)

# Analyze
print(f"Overall max safe gap: {report.overall_max_safe_gap:.4f}")
print(f"Overall max unsafe gap: {report.overall_max_unsafe_gap:.4f}")

# Per-timestep
for snap in report.snapshots:
    print(f"Step {snap.step}: safe_gap={snap.max_safe_gap:.4f}")
```

## Interpreting Results

| Gap value | Interpretation |
|---|---|
| ~0.0 | LFP is tight -- little room for improvement |
| 0.01 - 0.05 | Mild coarseness -- LFP is reasonably precise |
| 0.05 - 0.20 | Moderate coarseness -- consider richer templates |
| > 0.20 | High coarseness -- LFP is rejecting many safe actions |

**Caveat**: These gaps are *upper bounds* on true coarseness. The true gap may be smaller if the sampled set hasn't fully converged to the true set. Increase `budget` and `K_samples` to tighten.

## Tuning for Tighter Measurement

1. **Increase `budget`** (e.g., 500-1000): More points -> better coverage of the true belief set.
2. **Increase `K_samples`** (e.g., 30-50): More likelihood samples per step -> more posterior diversity.
3. **Use `EXTREME_POINTS` strategy**: For small state spaces (n <= 10), enumerating corners gives maximal spread.
4. **Use `FARTHEST_POINT` pruning**: Maximizes coverage of the belief simplex.
5. **Run multiple seeds**: Average gaps across random seeds to reduce variance.
