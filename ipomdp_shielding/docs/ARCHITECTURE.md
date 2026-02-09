# Architecture Overview

This document is the main entry point for understanding the `ipomdp_shielding` codebase. It describes the package structure, core abstractions, data flow, and points to detailed documentation for each subsystem.

## Purpose

This library implements **runtime safety shielding** for systems with **imperfect perception** modeled as **Interval POMDPs (IPOMDPs)**. The core idea:

1. A **perfect-perception shield** (`pp_shield`) precomputes which actions are safe in each state, assuming the agent knows its exact state.
2. At runtime the agent does *not* know its exact state -- it receives noisy observations with interval-bounded uncertainty.
3. A **belief propagator** tracks a set of possible belief distributions consistent with the observation history.
4. A **runtime shield** filters actions by checking: "under every belief consistent with the history, is the probability of being in a safe state high enough?"

## Package Structure

```
ipomdp_shielding/
  Models/              # Core mathematical models
    mdp.py               MDP (Markov Decision Process)
    imdp.py              IMDP (Interval MDP -- interval transition probabilities)
    pomdp.py             POMDP (Partially Observable MDP -- point observations)
    ipomdp.py            IPOMDP (Interval POMDP -- interval observation bounds)
    Confidence/          Confidence interval estimation from data

  Propagators/         # Belief propagation algorithms
    belief_base.py       IPOMDP_Belief abstract base class
    belief_polytope.py   BeliefPolytope representation and volume computation
    lfp_propagator.py    LFPPropagator -- template-based over-approximation via LFP
    forward_sampled_belief.py  ForwardSampledBelief -- sampling-based under-approximation
    exact_hmm.py         ExactIHMMBelief -- exact interval HMM belief (exponential)
    minmax_hmm.py        MinMaxIHMMBelief -- min/max interval HMM belief
    approx_belief.py     IPOMDP_ApproxBelief -- approximate belief
    utils.py             Shared utilities (tightened_likelihood_bounds, transition_update)

  Evaluation/          # Shield construction, evaluation, and metrics
    runtime_shield.py    RuntimeImpShield -- runtime safety enforcement
    metrics.py           MetricsCollector infrastructure and concrete collectors
    coarseness_evaluator.py  CoarsenessEvaluator -- LFP vs sampled gap measurement
    template_comparison.py   Template comparison and ground-truth evaluation
    shield_evaluator.py      Shield evaluation harness
    lfp_reporters.py         LFP comparison reporting
    report_runner.py         ReportRunner and ScriptedReportRunner
    script_library.py        RunScript and ScriptLibrary for scripted experiments
    single_run.py            Single-run debug utilities

  MonteCarlo/          # Monte Carlo safety simulation
    action_selectors.py      Strategy pattern: Random, Belief, Shielded, RL selectors
    perception_models.py     Nature's strategy: Uniform, Adversarial perception
    initial_states.py        Initial state sampling: Random, Safe, Boundary
    simulation.py            Core trial execution (run_single_trial, run_monte_carlo_trials)
    evaluator.py             MonteCarloSafetyEvaluator high-level API
    experiment_runner.py     ExperimentConfig and ExperimentRunner
    neural_action_selector.py  DQN-based neural action selector
    fixed_realization_model.py  Fixed interval realization perception model
    realization_optimizer.py    Cross-entropy optimization for realizations
    visualization.py         Plotting functions
    data_structures.py       SafetyTrialResult, MCSafetyMetrics, TimestepMetrics
    experiments.py           Pre-built experiment functions (Taxinet, etc.)

  CaseStudies/         # Domain-specific applications
    Taxinet/             TaxiNet aircraft taxiing case study
```

## Core Abstractions

### Model Hierarchy

```
MDP  -->  IMDP   (add interval transition uncertainty)
  |         |
  v         v
POMDP --> IPOMDP  (add interval observation uncertainty)
```

- **MDP**: States, actions, exact transition probabilities `T(s'|s,a)`.
- **IMDP**: Like MDP but transitions are intervals `[T_lo, T_hi]`.
- **POMDP**: MDP + observations with exact observation probabilities `Z(o|s)`.
- **IPOMDP**: MDP + observations with interval bounds `Z(o|s) in [P_lower, P_upper]`.

The main model used throughout is **IPOMDP**. See [models.md](models.md).

### Belief Propagation

All belief propagators implement `IPOMDP_Belief` (defined in `belief_base.py`):

```python
class IPOMDP_Belief:
    ipomdp          # The IPOMDP model
    restart()                                  # Reset to initial belief
    propagate(action, obs) -> bool             # Update belief, return success
    minimum_allowed_probability(allowed) -> float
    maximum_disallowed_probability(disallowed) -> float
```

Two primary propagators:

| Propagator | Type | Cost per step | Use case |
|---|---|---|---|
| `LFPPropagator` | Over-approximation (polytope) | O(K) LPs per template direction | Runtime shielding (sound) |
| `ForwardSampledBelief` | Under-approximation (point set) | O(N*K) arithmetic | Coarseness measurement |

See [belief_propagation.md](belief_propagation.md) and [coarseness_measurement.md](coarseness_measurement.md).

### Runtime Shielding

`RuntimeImpShield` (in `Evaluation/runtime_shield.py`) ties everything together:

```
Evidence (obs, action)
    |
    v
ipomdp_belief.propagate(action, obs)      # Update belief
    |
    v
For each candidate action a:
    min P(safe states for a)  >= threshold?  # Check safety
    |
    v
Return list of allowed actions
```

See [runtime_shielding.md](runtime_shielding.md).

### Monte Carlo Evaluation

The MonteCarlo package implements a **2-player game framework**:
- **Player 1 (Agent)**: Chooses actions (strategy pattern via `ActionSelector`)
- **Player 2 (Nature)**: Chooses observation probabilities within IPOMDP intervals (`PerceptionModel`)

Outcomes per trial: `safe`, `fail` (reached FAIL state), or `stuck` (no allowed actions).

See [monte_carlo_safety_evaluation.md](monte_carlo_safety_evaluation.md).

## Data Flow: Runtime Shield Execution

```
1. Build IPOMDP model from data/domain
2. Compute pp_shield (state -> safe actions) via MDP model checking
3. Create belief propagator (LFPPropagator or ForwardSampledBelief)
4. Create RuntimeImpShield(pp_shield, propagator, threshold)

At each timestep:
5. Agent acts, environment transitions, agent receives observation
6. shield.next_actions((obs, prev_action))
   a. propagator.propagate(prev_action, obs)  -- update belief
   b. For each action: query min_allowed / max_disallowed probability
   c. Filter by threshold
7. Agent picks from allowed actions
```

## Data Flow: Coarseness Measurement

```
1. Create LFPPropagator (over-approximation) and ForwardSampledBelief (under-approximation)
2. Create CoarsenessEvaluator(lfp, sampler, pp_shield)
3. For each (obs, action) in history:
   a. Propagate both in tandem
   b. Query both for min_allowed / max_disallowed per action
   c. Compute gaps: safe_gap, unsafe_gap
4. Report aggregate statistics
```

See [coarseness_measurement.md](coarseness_measurement.md).

## Key Mathematical Concepts

- **Belief polytope**: The set of probability distributions consistent with observations. Represented as `{b | Ab <= d, b >= 0, 1^T b = 1}`.
- **Template-based abstraction**: Bound `v^T b` for template directions `v` to over-approximate the belief set.
- **Linear Fractional Programming (LFP)**: Optimizing `v^T x / sum(x)` (normalized posterior) via Charnes-Cooper transform.
- **McCormick relaxation**: Linearizing the bilinear product `x = w * y` (likelihood times predicted belief).
- **Tightened likelihood bounds**: Using the row-sum constraint `sum_o Z(o|s) = 1` to tighten interval bounds.

See [minicourse.md](minicourse.md) for the full mathematical treatment.

## Documentation Index

| Document | Description |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | This file -- top-level orientation |
| [models.md](models.md) | Models package: MDP, IMDP, POMDP, IPOMDP |
| [belief_propagation.md](belief_propagation.md) | Propagators: LFP, forward sampled, exact, minmax |
| [runtime_shielding.md](runtime_shielding.md) | RuntimeImpShield and action filtering |
| [coarseness_measurement.md](coarseness_measurement.md) | Coarseness evaluation: LFP vs sampled gaps |
| [monte_carlo_safety_evaluation.md](monte_carlo_safety_evaluation.md) | Monte Carlo 2-player game framework |
| [minicourse.md](minicourse.md) | Mathematical foundations (convex algebras, LFP) |
| [neural_action_selector_readme.md](neural_action_selector_readme.md) | DQN-based neural action selector |
| [FIXED_REALIZATION_IMPLEMENTATION.md](FIXED_REALIZATION_IMPLEMENTATION.md) | Fixed realization optimization |
| [references.txt](references.txt) | Academic references |
| [archive/](archive/) | Historical implementation artifacts |
