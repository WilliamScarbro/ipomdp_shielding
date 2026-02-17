# Runtime Shielding

This document describes the runtime safety shield that enforces safety at execution time under imperfect perception.

## Overview

The runtime shield ensures that the agent only takes actions that are safe with high probability, even when the agent cannot observe its true state. It combines:

1. A **perfect-perception shield** (`pp_shield`): precomputed mapping `state -> set of safe actions`, assuming the agent knows its exact state.
2. A **belief propagator** (`IPOMDP_Belief`): tracks what the agent can infer about its state from observation history.
3. A **probability threshold** (`action_shield`): minimum required probability for an action to be considered safe.

## RuntimeImpShield

**File**: `Evaluation/runtime_shield.py`

### Construction

```python
from ipomdp_shielding.Evaluation import RuntimeImpShield

shield = RuntimeImpShield(
    pp_shield=pp_shield,         # dict: state -> set of safe actions
    ipomdp_belief=propagator,    # any IPOMDP_Belief subclass
    action_shield=0.9,           # probability threshold
    default_action=None,         # fallback when no action passes
)
```

### Shield Inversion

During initialization, the shield inverts `pp_shield` to create per-action state index sets:

```python
# inv_shield[a] = list of state indices where action a is safe
self.inv_shield = {
    a: [state_to_idx[s] for s in states if a in pp_shield[s]]
    for a in actions
}

# inv_shield_compliment[a] = list of state indices where action a is unsafe
self.inv_shield_compliment = {
    a: [state_to_idx[s] for s in states if a not in pp_shield[s]]
    for a in actions
}
```

These index lists are passed directly to the propagator's probability queries.

### Action Filtering (`next_actions`)

The main interface. Called each timestep with the evidence `(observation, previous_action)`:

```python
allowed_actions = shield.next_actions((obs, prev_action))
```

Steps:
1. **Propagate belief**: `propagator.propagate(prev_action, obs)`
   - If this returns `False` (numerical failure): increment `error_count`, return fallback.
2. **Query per-action probabilities** via `get_action_probs()`:
   - `allowed_prob = propagator.minimum_allowed_probability(inv_shield[a])`
   - `disallowed_prob = propagator.maximum_disallowed_probability(inv_shield_compliment[a])`
3. **Filter**: keep actions where `allowed_prob >= threshold` OR `disallowed_prob <= 1 - threshold`.
4. **Handle empty result**: if no actions pass, increment `stuck_count` and return fallback.

### Probability Queries (Pure)

`get_action_probs()` computes probabilities without side effects on the belief state:

```python
probs = shield.get_action_probs()
# Returns: [(action, allowed_prob, disallowed_prob), ...]
```

This is useful for analysis and metrics collection without triggering a belief update.

### State Tracking

The shield tracks two failure modes:
- `stuck_count`: number of times no action passed the threshold.
- `error_count`: number of times belief propagation failed (numerical issues).

Both reset on `shield.restart()`.

## Interface Compatibility

Any `IPOMDP_Belief` subclass can be used as the belief propagator. The shield relies on:

| Method | Signature | What the shield expects |
|---|---|---|
| `propagate` | `(action, obs) -> bool` | Update belief, return success |
| `minimum_allowed_probability` | `(List[int]) -> float` | Min probability of being in allowed states |
| `maximum_disallowed_probability` | `(List[int]) -> float` | Max probability of being in disallowed states |
| `restart` | `() -> None` | Reset to initial belief |
| `ipomdp` | attribute | Access to the IPOMDP model |

Verified compatible propagators:
- `LFPPropagator` (over-approximation, LP-based queries)
- `ForwardSampledBelief` (under-approximation, O(N) queries)
- `ExactIHMMBelief` (exact, for non-interval models)
- `MinMaxIHMMBelief` (conservative bounds)

## The pp_shield

The perfect-perception shield is a dictionary mapping each state to the set of actions that are safe to take from that state:

```python
pp_shield = {
    state_0: {action_a, action_b},    # a and b are safe from state_0
    state_1: {action_a},              # only a is safe from state_1
    state_2: {action_a, action_b},    # both safe from state_2
    ...
}
```

This is typically computed offline via MDP model checking -- verifying that an action from a given state cannot lead to a FAIL state (or satisfies some temporal logic property).

## Usage Pattern

```python
# 1. Build model and compute pp_shield
ipomdp = build_ipomdp(...)
pp_shield = compute_pp_shield(ipomdp)  # e.g., from MDP model checking

# 2. Create belief propagator
from ipomdp_shielding.Propagators import LFPPropagator, BeliefPolytope
from ipomdp_shielding.Propagators.lfp_propagator import default_solver, TemplateFactory

n = len(ipomdp.states)
propagator = LFPPropagator(
    ipomdp=ipomdp,
    template=TemplateFactory.canonical(n),
    solver=default_solver(),
    belief=BeliefPolytope.uniform_prior(n),
)

# 3. Create shield
shield = RuntimeImpShield(pp_shield, propagator, action_shield=0.9)

# 4. Run
shield.initialize(initial_state)
for step in range(horizon):
    obs = environment.observe()
    allowed = shield.next_actions((obs, prev_action))
    action = agent.choose(allowed)
    environment.step(action)
    prev_action = action

# 5. Check
print(f"Stuck {shield.stuck_count} times, {shield.error_count} errors")
```

## Metrics Integration

The `Evaluation/metrics.py` module provides `MetricsCollector` subclasses that extract diagnostics from the shield state after each step:

- `ApproximationMetrics_1`: template spread, volume proxy, safest action probability
- `GroundTruthComparisonMetrics`: per-action min safe probabilities, per-state bounds
- `CoarsenessMetricsCollector`: gaps between LFP over-approximation and sampled under-approximation

All collectors implement:
```python
metrics = collector.compute(rt_shield, step)  # returns Dict[str, MetricValue]
```

They read from `rt_shield.ipomdp_belief` (the propagator) and `rt_shield.inv_shield` (the index maps).
