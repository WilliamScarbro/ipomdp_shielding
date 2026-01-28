"""Test functions for Monte Carlo safety evaluation."""

from typing import Optional
import random

from .action_selectors import RandomActionSelector
from .perception_models import UniformPerceptionModel, AdversarialPerceptionModel
from .evaluator import MonteCarloSafetyEvaluator
from .visualization import (
    plot_safety_metrics,
    plot_two_player_game_results,
    plot_rl_training_curves,
)


def test_taxinet_monte_carlo_safety(
    num_trials: int = 100,
    trial_length: int = 20,
    seed: Optional[int] = 42,
    save_path: Optional[str] = None
):
    """Simple test of Monte Carlo safety evaluation on Taxinet model.

    Parameters
    ----------
    num_trials : int
        Number of Monte Carlo trials to run
    trial_length : int
        Maximum steps per trial
    seed : int, optional
        Random seed for reproducibility
    save_path : str, optional
        Path to save visualization (e.g., "images/mc_safety_test.png")

    Returns
    -------
    dict
        Results by sampling mode
    """
    from ..CaseStudies.Taxinet import build_taxinet_ipomdp
    from ..Evaluation.runtime_shield import RuntimeImpShield

    print("=" * 60)
    print("TAXINET MONTE CARLO SAFETY TEST")
    print(f"Trials: {num_trials}, Length: {trial_length}")
    print("=" * 60)

    # Setup Taxinet model
    ipomdp, dyn_shield, test_cte_model, test_he_model = build_taxinet_ipomdp()

    def perception(state):
        if state == "FAIL":
            return "FAIL"
        return (random.choice(test_cte_model[state[0]]), random.choice(test_he_model[state[1]]))

    # Create runtime shield factory
    def rt_shield_factory():
        from ..Propagators import LFPPropagator, BeliefPolytope, TemplateFactory
        from ..Propagators.lfp_propagator import default_solver

        n = len(ipomdp.states)
        template = TemplateFactory.canonical(n)
        polytope = BeliefPolytope.uniform_prior(n)
        propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)

        return RuntimeImpShield(dyn_shield, propagator, action_shield=0.8)

    # Create evaluator
    evaluator = MonteCarloSafetyEvaluator(
        ipomdp=ipomdp,
        pp_shield=dyn_shield,
        perception=perception,
        rt_shield_factory=rt_shield_factory
    )

    # Run evaluation with random action selection
    action_selector = RandomActionSelector()

    results = evaluator.evaluate(
        action_selector=action_selector,
        num_trials=num_trials,
        trial_length=trial_length,
        sampling_modes=["random", "best_case", "worst_case"],
        seed=seed
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for mode, metrics in results.items():
        print(f"\n{mode.upper()}:")
        print(metrics)

    # Plot results
    if save_path:
        plot_safety_metrics(results, save_path=save_path, show=False)
    else:
        plot_safety_metrics(results, show=True)

    return results


def test_two_player_game(
    num_trials: int = 50,
    trial_length: int = 20,
    seed: Optional[int] = 42,
    save_path: Optional[str] = None,
    use_rl: bool = False,
    rl_training_episodes: int = 500
):
    """Test the 2-player game evaluation with cooperative vs adversarial nature.

    This demonstrates the full game-theoretic analysis:
    - Player 1 (Agent): Chooses actions from shield (best/worst/random)
    - Player 2 (Nature): Chooses perception probabilities

    Agent strategies are determined by ACTION SELECTION from allowed actions:
    - best: Selects safest action (maximizes expected safety)
    - worst: Selects riskiest action (minimizes expected safety)
    - random: Uniform random selection

    Parameters
    ----------
    num_trials : int
        Number of Monte Carlo trials per combination
    trial_length : int
        Maximum steps per trial
    seed : int, optional
        Random seed for reproducibility
    save_path : str, optional
        Path to save visualization
    use_rl : bool
        If True, use RL-trained action selectors instead of heuristics
    rl_training_episodes : int
        Number of RL training episodes (if use_rl=True)

    Returns
    -------
    dict
        Nested results: nature_strategy -> agent_strategy -> MCSafetyMetrics
    """
    from ..CaseStudies.Taxinet import build_taxinet_ipomdp
    from ..Evaluation.runtime_shield import RuntimeImpShield

    print("=" * 60)
    print("2-PLAYER GAME: AGENT vs NATURE")
    print(f"Trials: {num_trials}, Length: {trial_length}")
    print("=" * 60)
    print("\nPlayer 1 (Agent): Selects actions from shield's allowed actions")
    print("  - best: Selects safest action (maximizes safety)")
    print("  - worst: Selects riskiest action (minimizes safety)")
    print("  - random: Uniform random selection")
    if use_rl:
        print(f"  (Using RL-trained selectors with {rl_training_episodes} training episodes)")
    else:
        print("  (Using heuristic-based selectors)")
    print("\nPlayer 2 (Nature): Chooses perception probabilities within intervals")
    print("  - cooperative: Random perception (uniform within intervals)")
    print("  - adversarial: Maximizes failure probability")

    # Setup Taxinet model
    ipomdp, dyn_shield, _, _ = build_taxinet_ipomdp()

    # Create runtime shield factory
    def rt_shield_factory():
        from ..Propagators import LFPPropagator, BeliefPolytope, TemplateFactory
        from ..Propagators.lfp_propagator import default_solver

        n = len(ipomdp.states)
        template = TemplateFactory.canonical(n)
        polytope = BeliefPolytope.uniform_prior(n)
        propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)

        return RuntimeImpShield(dyn_shield, propagator, action_shield=0.8)

    # Use uniform perception as default
    default_perception = UniformPerceptionModel()

    # Create evaluator
    evaluator = MonteCarloSafetyEvaluator(
        ipomdp=ipomdp,
        pp_shield=dyn_shield,
        perception=default_perception,
        rt_shield_factory=rt_shield_factory
    )

    # Run 2-player game evaluation
    results = evaluator.evaluate_two_player_game(
        num_trials=num_trials,
        trial_length=trial_length,
        seed=seed,
        use_rl=use_rl,
        rl_training_episodes=rl_training_episodes
    )

    # Print summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: COOPERATIVE vs ADVERSARIAL NATURE")
    print("=" * 60)

    for nature_strategy in results:
        print(f"\n{nature_strategy.upper()} NATURE:")
        for agent_mode, metrics in results[nature_strategy].items():
            print(f"  {agent_mode}: "
                  f"fail={metrics.fail_rate:.1%}, "
                  f"stuck={metrics.stuck_rate:.1%}, "
                  f"safe={metrics.safe_rate:.1%}")

    # Compute impact of adversarial perception
    agent_modes = list(results["cooperative"].keys())
    print("\n" + "-" * 40)
    print("IMPACT OF ADVERSARIAL PERCEPTION:")
    for agent_mode in agent_modes:
        coop_fail = results["cooperative"][agent_mode].fail_rate
        adv_fail = results["adversarial"][agent_mode].fail_rate
        increase = adv_fail - coop_fail
        print(f"  {agent_mode}: +{increase:.1%} failure rate")

    # Compute impact of agent strategy
    print("\n" + "-" * 40)
    print("IMPACT OF AGENT STRATEGY (vs random):")
    for nature_name in results:
        random_fail = results[nature_name]["random"].fail_rate
        if "best" in results[nature_name]:
            best_fail = results[nature_name]["best"].fail_rate
            print(f"  {nature_name}/best: {best_fail - random_fail:+.1%} failure rate")
        if "worst" in results[nature_name]:
            worst_fail = results[nature_name]["worst"].fail_rate
            print(f"  {nature_name}/worst: {worst_fail - random_fail:+.1%} failure rate")

    # Plot comparison
    if save_path:
        plot_two_player_game_results(results, save_path=save_path, show=False)

    return results


def test_rl_two_player_game(
    num_trials: int = 50,
    trial_length: int = 20,
    training_episodes: int = 500,
    seed: Optional[int] = 42,
    save_path: Optional[str] = None
):
    """Test the 2-player game with RL-trained action selectors and training curves.

    Parameters
    ----------
    num_trials : int
        Number of Monte Carlo trials per combination
    trial_length : int
        Maximum steps per trial
    training_episodes : int
        Number of RL training episodes
    seed : int, optional
        Random seed for reproducibility
    save_path : str, optional
        Path to save visualization

    Returns
    -------
    dict
        Contains evaluation results and training curves
    """
    from ..CaseStudies.Taxinet import build_taxinet_ipomdp

    print("=" * 60)
    print("RL-TRAINED 2-PLAYER GAME")
    print(f"Training episodes: {training_episodes}")
    print(f"Evaluation trials: {num_trials}, Length: {trial_length}")
    print("=" * 60)

    # Setup Taxinet model
    ipomdp, dyn_shield, _, _ = build_taxinet_ipomdp()

    # Create runtime shield factory
    def rt_shield_factory():
        from ..Propagators import LFPPropagator, BeliefPolytope, TemplateFactory
        from ..Propagators.lfp_propagator import default_solver

        n = len(ipomdp.states)
        template = TemplateFactory.canonical(n)
        polytope = BeliefPolytope.uniform_prior(n)
        propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)

        from ..Evaluation.runtime_shield import RuntimeImpShield
        return RuntimeImpShield(dyn_shield, propagator, action_shield=0.8)

    # Create evaluator
    evaluator = MonteCarloSafetyEvaluator(
        ipomdp=ipomdp,
        pp_shield=dyn_shield,
        perception=UniformPerceptionModel(),
        rt_shield_factory=rt_shield_factory
    )

    # Run evaluation with training
    full_results = evaluator.evaluate_with_trained_rl(
        num_trials=num_trials,
        trial_length=trial_length,
        training_episodes=training_episodes,
        seed=seed
    )

    # Plot training curves if save_path provided
    if save_path:
        plot_rl_training_curves(full_results["training"], save_path=save_path)
        # Also plot evaluation results
        eval_save_path = save_path.replace(".png", "_eval.png")
        plot_two_player_game_results(
            full_results["evaluation"],
            save_path=eval_save_path,
            show=False
        )

    return full_results


if __name__ == "__main__":
    # Run 2-player game test
    print("\n" + "=" * 70)
    print("Running 2-player game evaluation...")
    test_two_player_game(
        num_trials=10,
        trial_length=20,
        seed=42,
        save_path="images/two_player_game.png"
    )

    # Run RL 2-player game test
    print("\n" + "=" * 70)
    print("Running RL 2-player game evaluation...")
    test_rl_two_player_game(
        num_trials=10,
        trial_length=20,
        seed=42,
        save_path="images/rl_two_player_game.png"
    )
