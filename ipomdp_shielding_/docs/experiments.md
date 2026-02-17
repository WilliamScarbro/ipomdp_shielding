# Experiments

All experiment scripts live in `experiments/` and write JSON results to `./data/`.

## Coarseness Experiment

**Script:** `experiments/coarse_experiment.py`

Measures how much the LFP (Linear Fractional Programming) belief propagator over-approximates the true reachable belief set on the TaxiNet model.

### Background

The LFP propagator maintains a convex polytope that over-approximates the set of reachable beliefs. This over-approximation is conservative — it may declare actions unsafe that are actually safe, leading to unnecessary "stuck" outcomes. The **coarseness** of this approximation quantifies how much slack the polytope introduces.

To measure coarseness, we compare two propagators run in tandem on identical (observation, action) trajectories:

- **LFP (over-approximation):** Maintains a `BeliefPolytope` using LP-based template bounds. Provides a lower bound on `min P(allowed states)` and an upper bound on `max P(disallowed states)`.
- **Forward-sampled belief (under-approximation):** Maintains N concrete belief points. Provides an upper bound on `min P(allowed states)` and a lower bound on `max P(disallowed states)`.

Since `P_sampled ⊆ P_true ⊆ P_lfp`, the gap between them is a sound upper bound on the true coarseness:

```
gap = min_allowed_sampled - min_allowed_lfp >= 0
```

The safe and unsafe gaps are symmetric by construction, so we only track one.

### What it does

1. Builds the TaxiNet IPOMDP (16 states + FAIL, 3 actions, interval perception)
2. Creates an LFP propagator with canonical templates (one per state coordinate)
3. Creates a ForwardSampledBelief with configurable budget, sampling strategy, and pruning
4. Generates 100 trajectories under perfect-perception shielding with uniform perception
5. For each trajectory, propagates both propagators in tandem and computes per-action coarseness gaps at every timestep
6. Aggregates: overall max/mean gap with std, per-timestep gap time series

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `NUM_TRAJECTORIES` | 100 | Number of trajectories to evaluate |
| `TRAJECTORY_LENGTH` | 20 | Steps per trajectory |
| `SAMPLER_BUDGET` | 200 | Number of belief points maintained |
| `SAMPLER_K` | 20 | Observation likelihood samples per propagation step |
| `SAMPLER_LIKELIHOOD_STRATEGY` | HYBRID | How to sample likelihood vectors (EXTREME_POINTS, UNIFORM_RANDOM, HYBRID) |
| `SAMPLER_PRUNING_STRATEGY` | FARTHEST_POINT | How to prune candidates (COORDINATE_EXTREMAL, FARTHEST_POINT, RANDOM) |

### Output

- `./data/coarseness_results.json` — full results with per-timestep gap time series
- `./data/coarseness_results.png` — plot of coarseness over time (if matplotlib available)

### Interpretation

- **Gap near 0:** The LFP polytope is tight — minimal conservatism.
- **Gap near 1:** The LFP polytope is very loose — the over-approximation introduces substantial conservatism that may cause unnecessary stuck outcomes.
- **Gap increasing with timestep:** Coarseness accumulates over time as the polytope relaxes.

---

## RL Shielding Experiment

**Script:** `experiments/rl_shielding_experiment.py`

Compares four shielding strategies for RL-based action selection on TaxiNet, evaluating the safety/liveness tradeoff under benign and adversarial perception.

### Background

A runtime shield filters the actions proposed by an agent, blocking any that might lead to unsafe states. Different shields make different tradeoffs:

- **No Shield:** All actions allowed. Maximum liveness but no safety guarantee.
- **Observation Shield:** Looks up `pp_shield[observation]` directly. Simple but assumes perfect perception.
- **Single-Belief Shield:** Maintains a POMDP point-belief and filters by `P(action safe | belief) >= threshold`. Accounts for perception uncertainty under a fixed (known) observation model.
- **Envelope Shield:** Maintains a belief polytope via LFP, accounting for interval perception uncertainty. Most conservative but sound under all realizations in the interval.

### What it does

The experiment crosses three orthogonal factors:

1. **Perception** (Nature's strategy): Uniform Random vs Adversarial Optimized
2. **Action Selection** (Agent's strategy): Random vs Best (shield-informed) vs RL (trained DQN)
3. **Shield Strategy**: None vs Observation vs Single-Belief vs Envelope

This produces a 2 x 3 x 4 = 24-cell grid. For each cell:

1. Runs 30 Monte Carlo trials of 20 timesteps each
2. Tracks three outcomes: fail (reached FAIL state), stuck (no allowed actions), safe (survived)
3. Computes per-timestep cumulative outcome probabilities

### Setup

The experiment first trains (or loads cached):
- A **DQN agent** (NeuralActionSelector) trained with adversarial-greedy perception
- An **adversarial-optimized perception realization** that maximizes failure rate against the envelope shield

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `NUM_TRIALS` | 30 | Monte Carlo trials per grid cell |
| `TRIAL_LENGTH` | 20 | Steps per trial |
| `SHIELD_THRESHOLD` | 0.8 | `P(safe)` threshold for belief-based shields |
| `RL_EPISODES` | 1000 | DQN training episodes |

### Output

- `./data/rl_shielding_results.json` — aggregate metrics for all 24 combinations
- `./data/rl_shielding_figures/` — 6 plots (3 outcomes x 2 perceptions) showing per-timestep curves

### Key questions answered

- Does the envelope shield reduce failure rate compared to observation-only shielding?
- How much liveness (stuck rate) does the envelope shield cost?
- Does the advantage of envelope shielding grow under adversarial perception?
- How does RL action selection interact with different shield strategies?

---

## Running

From the repository root:

```bash
# Coarseness experiment
python experiments/coarse_experiment.py

# RL shielding experiment
python experiments/rl_shielding_experiment.py
```

Results are written to `./data/`. The `data/` directory is created automatically if it does not exist.
