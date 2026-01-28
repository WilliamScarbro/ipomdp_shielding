"""Monte Carlo Safety Evaluation for Shielding Strategies.

This package provides tools for evaluating the safety of shielding strategies
through Monte Carlo simulation. It implements a 2-player game framework:

- Player 1 (Agent): Chooses actions from the shield (best/worst/random strategies)
- Player 2 (Nature): Chooses perception probabilities within IPOMDP intervals

The package measures safety through simulation, tracking three outcomes:
- fail: State reached "FAIL"
- stuck: No allowed actions available
- safe: Completed trial without failure

Supports best/worst/average case analysis via modular initial state sampling.

Modules
-------
action_selectors
    Action selection strategies (random, safest, riskiest, RL-based)
perception_models
    Perception models (uniform, adversarial, legacy adapter)
initial_states
    Initial state sampling strategies (random, safe interior, boundary)
simulation
    Core Monte Carlo simulation functions
evaluator
    High-level MonteCarloSafetyEvaluator class
visualization
    Plotting functions for results
data_structures
    SafetyTrialResult and MCSafetyMetrics dataclasses
tests
    Test functions for Taxinet case study
"""

# Data structures
from .data_structures import SafetyTrialResult, MCSafetyMetrics

# Action selectors
from .action_selectors import (
    ActionSelector,
    RandomActionSelector,
    UniformFallbackSelector,
    SafestActionSelector,
    RiskiestActionSelector,
    BeliefSelector,
    RLActionSelector,
    QLearningActionSelector,
    create_rl_action_selector,
)

# Perception models
from .perception_models import (
    PerceptionModel,
    UniformPerceptionModel,
    AdversarialPerceptionModel,
    LegacyPerceptionAdapter,
)

# Initial state generators
from .initial_states import (
    InitialStateGenerator,
    RandomInitialState,
    SafeInitialState,
    BoundaryInitialState,
)

# Core simulation
from .simulation import (
    run_single_trial,
    run_monte_carlo_trials,
    compute_safety_metrics,
)

# High-level API
from .evaluator import MonteCarloSafetyEvaluator

# Visualization
from .visualization import (
    plot_safety_metrics,
    plot_rl_training_curves,
    plot_two_player_game_results,
)

# Tests
from .tests import (
    test_taxinet_monte_carlo_safety,
    test_two_player_game,
    test_rl_two_player_game,
)

__all__ = [
    # Data structures
    "SafetyTrialResult",
    "MCSafetyMetrics",
    # Action selectors
    "ActionSelector",
    "RandomActionSelector",
    "UniformFallbackSelector",
    "SafestActionSelector",
    "RiskiestActionSelector",
    "BeliefSelector",
    "RLActionSelector",
    "QLearningActionSelector",
    "create_rl_action_selector",
    # Perception models
    "PerceptionModel",
    "UniformPerceptionModel",
    "AdversarialPerceptionModel",
    "LegacyPerceptionAdapter",
    # Initial state generators
    "InitialStateGenerator",
    "RandomInitialState",
    "SafeInitialState",
    "BoundaryInitialState",
    # Core simulation
    "run_single_trial",
    "run_monte_carlo_trials",
    "compute_safety_metrics",
    # High-level API
    "MonteCarloSafetyEvaluator",
    # Visualization
    "plot_safety_metrics",
    "plot_rl_training_curves",
    "plot_two_player_game_results",
    # Tests
    "test_taxinet_monte_carlo_safety",
    "test_two_player_game",
    "test_rl_two_player_game",
]
