"""High-level Monte Carlo safety evaluation API.

This module provides the MonteCarloSafetyEvaluator class for running
comprehensive safety evaluations across different scenarios.
"""

from typing import Any, Callable, Dict, List, Optional, Set
import random
import numpy as np

from ..Models.ipomdp import IPOMDP
from ..Evaluation.runtime_shield import RuntimeImpShield

from .data_structures import MCSafetyMetrics
from .action_selectors import (
    ActionSelector,
    RandomActionSelector,
    SafestActionSelector,
    RiskiestActionSelector,
    QLearningActionSelector,
    create_rl_action_selector,
)
from .perception_models import (
    PerceptionModel,
    UniformPerceptionModel,
    AdversarialPerceptionModel,
    LegacyPerceptionAdapter,
)
from .initial_states import (
    InitialStateGenerator,
    RandomInitialState,
    SafeInitialState,
    BoundaryInitialState,
)
from .simulation import run_monte_carlo_trials, compute_safety_metrics


class MonteCarloSafetyEvaluator:
    """High-level interface for Monte Carlo safety evaluation.

    Evaluates shielding strategies across best/worst/average case scenarios
    by varying initial state sampling strategies and perception models.

    This implements a 2-player game framework:
    - Player 1 (Agent): Chooses actions from the shield
    - Player 2 (Nature): Chooses perception probabilities within intervals

    Nature can be cooperative (random perception) or adversarial (maximizing failure).
    """

    def __init__(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]],
        perception: "PerceptionModel | Callable[[Any], Any]",
        rt_shield_factory: Callable[[], RuntimeImpShield]
    ):
        """Initialize evaluator.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions
        perception : PerceptionModel or callable
            Default perception model or legacy function
        rt_shield_factory : callable
            Factory function returning fresh RuntimeImpShield instance
        """
        self.ipomdp = ipomdp
        self.pp_shield = pp_shield
        self.rt_shield_factory = rt_shield_factory

        # Convert legacy perception to PerceptionModel
        if callable(perception) and not isinstance(perception, PerceptionModel):
            self.perception = LegacyPerceptionAdapter(perception)
        else:
            self.perception = perception

    def evaluate(
        self,
        action_selector: ActionSelector,
        num_trials: int = 100,
        trial_length: int = 20,
        sampling_modes: Optional[List[str]] = None,
        store_trajectories: bool = False,
        seed: Optional[int] = None,
        perception_model: Optional[PerceptionModel] = None
    ) -> Dict[str, MCSafetyMetrics]:
        """Run Monte Carlo evaluation across sampling modes.

        Parameters
        ----------
        action_selector : ActionSelector
            Strategy for selecting actions
        num_trials : int
            Number of trials per sampling mode
        trial_length : int
            Maximum steps per trial
        sampling_modes : list of str, optional
            Sampling modes to evaluate. Defaults to ["random", "best_case", "worst_case"]
        store_trajectories : bool
            Whether to store full trajectories (memory intensive)
        seed : int, optional
            Random seed for reproducibility
        perception_model : PerceptionModel, optional
            Override the default perception model for this evaluation

        Returns
        -------
        dict
            Mapping from sampling mode to MCSafetyMetrics
        """
        if sampling_modes is None:
            sampling_modes = ["random", "best_case", "worst_case"]

        # Use provided perception model or default
        perception = perception_model if perception_model is not None else self.perception

        # Map mode names to generator classes
        generator_map = {
            "random": RandomInitialState(),
            "best_case": SafeInitialState(),
            "worst_case": BoundaryInitialState()
        }

        results_by_mode = {}

        for mode in sampling_modes:
            if mode not in generator_map:
                raise ValueError(f"Unknown sampling mode: {mode}")

            generator = generator_map[mode]

            # Run trials for this mode
            results = run_monte_carlo_trials(
                ipomdp=self.ipomdp,
                pp_shield=self.pp_shield,
                perception=perception,
                rt_shield_factory=self.rt_shield_factory,
                action_selector=action_selector,
                initial_generator=generator,
                num_trials=num_trials,
                trial_length=trial_length,
                store_trajectories=store_trajectories,
                seed=seed
            )

            # Compute metrics
            metrics = compute_safety_metrics(results)
            results_by_mode[mode] = metrics

        return results_by_mode

    def evaluate_two_player_game(
        self,
        num_trials: int = 100,
        trial_length: int = 20,
        seed: Optional[int] = None,
        use_rl: bool = False,
        rl_training_episodes: int = 500
    ) -> Dict[str, Dict[str, MCSafetyMetrics]]:
        """Evaluate all combinations of agent and nature strategies.

        This runs the full 2-player game analysis:
        - Agent strategies: random, best (safest action), worst (riskiest action)
        - Nature strategies: cooperative (uniform), adversarial

        The agent strategies are now achieved through action selection from the
        shield's allowed actions, not through initial state selection.

        Parameters
        ----------
        num_trials : int
            Number of trials per combination
        trial_length : int
            Maximum steps per trial
        seed : int, optional
            Random seed for reproducibility
        use_rl : bool
            If True, use RL-trained selectors for best/worst instead of heuristics
        rl_training_episodes : int
            Number of episodes for RL training (if use_rl=True)

        Returns
        -------
        dict
            Nested dict: nature_strategy -> agent_strategy -> MCSafetyMetrics
        """
        results = {}

        # Define perception models (Nature's strategies)
        perception_models = {
            "cooperative": UniformPerceptionModel(),
            "adversarial": AdversarialPerceptionModel(self.pp_shield)
        }

        # Define action selectors (Agent's strategies)
        if use_rl:
            print("\nTraining RL action selectors...")

            # Train best (maximize safety) selector
            best_selector = create_rl_action_selector(
                ipomdp=self.ipomdp,
                pp_shield=self.pp_shield,
                rt_shield_factory=self.rt_shield_factory,
                perception=perception_models["adversarial"],
                maximize_safety=True,
                num_episodes=rl_training_episodes,
                episode_length=trial_length
            )

            # Train worst (minimize safety) selector
            worst_selector = create_rl_action_selector(
                ipomdp=self.ipomdp,
                pp_shield=self.pp_shield,
                rt_shield_factory=self.rt_shield_factory,
                perception=perception_models["adversarial"],
                maximize_safety=False,
                num_episodes=rl_training_episodes,
                episode_length=trial_length
            )

            action_selectors = {
                "random": RandomActionSelector(),
                "best": best_selector,
                "worst": worst_selector
            }
        else:
            # Use heuristic-based selectors
            action_selectors = {
                "random": RandomActionSelector(),
                "best": SafestActionSelector(self.ipomdp, self.pp_shield),
                "worst": RiskiestActionSelector(self.ipomdp, self.pp_shield)
            }

        # Use random initial state for all evaluations
        initial_generator = RandomInitialState()

        for nature_name, perception_model in perception_models.items():
            print(f"\nEvaluating nature strategy: {nature_name}")
            results[nature_name] = {}

            for agent_name, action_selector in action_selectors.items():
                print(f"  Agent strategy: {agent_name}...")

                # Reset RL selector state if needed
                if hasattr(action_selector, 'reset'):
                    action_selector.reset()

                trial_results = run_monte_carlo_trials(
                    ipomdp=self.ipomdp,
                    pp_shield=self.pp_shield,
                    perception=perception_model,
                    rt_shield_factory=self.rt_shield_factory,
                    action_selector=action_selector,
                    initial_generator=initial_generator,
                    num_trials=num_trials,
                    trial_length=trial_length,
                    store_trajectories=False,
                    seed=seed
                )

                metrics = compute_safety_metrics(trial_results)
                results[nature_name][agent_name] = metrics

                print(f"    fail={metrics.fail_rate:.2%}, "
                      f"stuck={metrics.stuck_rate:.2%}, "
                      f"safe={metrics.safe_rate:.2%}")

        return results

    def evaluate_with_trained_rl(
        self,
        num_trials: int = 100,
        trial_length: int = 20,
        training_episodes: int = 500,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate using RL-trained action selectors with training curves.

        Trains RL agents to maximize (best) and minimize (worst) safety,
        then evaluates them against cooperative and adversarial nature.

        Parameters
        ----------
        num_trials : int
            Number of evaluation trials per combination
        trial_length : int
            Maximum steps per trial
        training_episodes : int
            Number of RL training episodes
        seed : int, optional
            Random seed

        Returns
        -------
        dict
            Contains:
            - "evaluation": nested dict of MCSafetyMetrics
            - "training": dict with training curves for best/worst selectors
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        training_metrics = {}

        # Train best selector
        print("\n" + "=" * 50)
        print("Training BEST agent (maximize safety)")
        print("=" * 50)
        best_selector = QLearningActionSelector(
            actions=list(self.ipomdp.actions),
            maximize_safety=True,
            exploration_rate=0.3,
            learning_rate=0.1
        )
        training_metrics["best"] = best_selector.train(
            ipomdp=self.ipomdp,
            pp_shield=self.pp_shield,
            rt_shield_factory=self.rt_shield_factory,
            perception=UniformPerceptionModel(),
            num_episodes=training_episodes,
            episode_length=trial_length
        )
        best_selector.exploration_rate = 0.0  # Evaluation mode

        # Train worst selector
        print("\n" + "=" * 50)
        print("Training WORST agent (minimize safety)")
        print("=" * 50)
        worst_selector = QLearningActionSelector(
            actions=list(self.ipomdp.actions),
            maximize_safety=False,
            exploration_rate=0.3,
            learning_rate=0.1
        )
        training_metrics["worst"] = worst_selector.train(
            ipomdp=self.ipomdp,
            pp_shield=self.pp_shield,
            rt_shield_factory=self.rt_shield_factory,
            perception=UniformPerceptionModel(),
            num_episodes=training_episodes,
            episode_length=trial_length
        )
        worst_selector.exploration_rate = 0.0  # Evaluation mode

        # Evaluate
        print("\n" + "=" * 50)
        print("Evaluating trained agents")
        print("=" * 50)

        action_selectors = {
            "random": RandomActionSelector(),
            "best_rl": best_selector,
            "worst_rl": worst_selector
        }

        perception_models = {
            "cooperative": UniformPerceptionModel(),
            "adversarial": AdversarialPerceptionModel(self.pp_shield)
        }

        initial_generator = RandomInitialState()
        evaluation_results = {}

        for nature_name, perception_model in perception_models.items():
            print(f"\nNature: {nature_name}")
            evaluation_results[nature_name] = {}

            for agent_name, selector in action_selectors.items():
                if hasattr(selector, 'reset'):
                    selector.reset()

                trial_results = run_monte_carlo_trials(
                    ipomdp=self.ipomdp,
                    pp_shield=self.pp_shield,
                    perception=perception_model,
                    rt_shield_factory=self.rt_shield_factory,
                    action_selector=selector,
                    initial_generator=initial_generator,
                    num_trials=num_trials,
                    trial_length=trial_length,
                    store_trajectories=False,
                    seed=None  # Don't reset seed for each combo
                )

                metrics = compute_safety_metrics(trial_results)
                evaluation_results[nature_name][agent_name] = metrics
                print(f"  {agent_name}: fail={metrics.fail_rate:.2%}, "
                      f"safe={metrics.safe_rate:.2%}")

        return {
            "evaluation": evaluation_results,
            "training": training_metrics
        }
