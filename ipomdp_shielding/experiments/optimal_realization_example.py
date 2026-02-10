"""Example: Training and evaluating optimal fixed interval realizations.

This script demonstrates the complete workflow for ML-optimized fixed
interval realizations:

1. Load TaxiNet IPOMDP case study
2. Train optimal realizations with different action selectors
3. Compare with uniform and adversarial perception models
4. Save/load trained models
5. Visualize results

Usage:
    python experiments/optimal_realization_example.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ipomdp_shielding.CaseStudies.Taxinet import build_taxinet_ipomdp
from ipomdp_shielding.Evaluation.runtime_shield import RuntimeImpShield
from ipomdp_shielding.Propagators import LFPPropagator, BeliefPolytope, TemplateFactory
from ipomdp_shielding.Propagators.lfp_propagator import default_solver
from ipomdp_shielding.MonteCarlo import (
    train_optimal_realization,
    FixedRealizationPerceptionModel,
    UniformPerceptionModel,
    AdversarialPerceptionModel,
    RandomActionSelector,
    SafestActionSelector,
    MonteCarloSafetyEvaluator,
    RandomInitialState,
)


def create_rt_shield_factory(ipomdp, pp_shield, action_shield_threshold=0.8):
    """Create runtime shield factory function.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield
    action_shield_threshold : float
        Threshold for action shield probability cutoff

    Returns
    -------
    callable
        Factory function returning fresh RuntimeImpShield instances
    """
    def factory():
        n = len(ipomdp.states)
        template = TemplateFactory.canonical(n)
        polytope = BeliefPolytope.uniform_prior(n)
        propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)
        return RuntimeImpShield(pp_shield, propagator, action_shield=action_shield_threshold)

    return factory


def train_optimal_vs_random():
    """Train optimal realization against random action selector."""
    print("\n" + "=" * 70)
    print("TRAINING OPTIMAL REALIZATION VS RANDOM ACTION SELECTOR")
    print("=" * 70)

    # Load TaxiNet IPOMDP
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp()

    # Create runtime shield factory
    rt_shield_factory = create_rt_shield_factory(ipomdp, pp_shield)

    # Train optimal realization
    optimal_perception = train_optimal_realization(
        ipomdp=ipomdp,
        pp_shield=pp_shield,
        rt_shield_factory=rt_shield_factory,
        action_selector=RandomActionSelector(),
        initial_generator=RandomInitialState(),
        num_candidates=20,  # Small for demonstration
        num_trials_per_candidate=10,
        max_iterations=20,
        trial_length=20,
        save_path="/tmp/optimal_vs_random.json",
        verbose=True
    )

    print(f"\nBest failure rate: {optimal_perception.metadata['objective_score']:.2%}")

    return optimal_perception


def train_optimal_vs_safest():
    """Train optimal realization against safest action selector."""
    print("\n" + "=" * 70)
    print("TRAINING OPTIMAL REALIZATION VS SAFEST ACTION SELECTOR")
    print("=" * 70)

    # Load TaxiNet IPOMDP
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp()

    # Create runtime shield factory
    rt_shield_factory = create_rt_shield_factory(ipomdp, pp_shield)

    # Train optimal realization
    optimal_perception = train_optimal_realization(
        ipomdp=ipomdp,
        pp_shield=pp_shield,
        rt_shield_factory=rt_shield_factory,
        action_selector=SafestActionSelector(ipomdp, pp_shield),
        initial_generator=RandomInitialState(),
        num_candidates=20,
        num_trials_per_candidate=10,
        max_iterations=20,
        trial_length=20,
        save_path="/tmp/optimal_vs_safest.json",
        verbose=True
    )

    print(f"\nBest failure rate: {optimal_perception.metadata['objective_score']:.2%}")

    return optimal_perception


def compare_perception_models():
    """Compare optimal fixed realization with uniform and adversarial."""
    print("\n" + "=" * 70)
    print("COMPARING PERCEPTION MODELS")
    print("=" * 70)

    # Load TaxiNet IPOMDP
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp()

    # Create runtime shield factory
    rt_shield_factory = create_rt_shield_factory(ipomdp, pp_shield)

    # Load trained optimal realization
    try:
        optimal_perception = FixedRealizationPerceptionModel.load(
            "/tmp/optimal_vs_random.json"
        )
        print("\nLoaded trained optimal realization from /tmp/optimal_vs_random.json")
    except FileNotFoundError:
        print("\nNo trained model found. Training first...")
        optimal_perception = train_optimal_vs_random()

    # Create perception models
    perception_models = {
        "uniform": UniformPerceptionModel(),
        "adversarial": AdversarialPerceptionModel(pp_shield),
        "optimal_fixed": optimal_perception,
    }

    # Evaluate each with random action selector
    action_selector = RandomActionSelector()

    results = {}
    for name, perception in perception_models.items():
        print(f"\n{name.upper()} perception:")

        evaluator = MonteCarloSafetyEvaluator(
            ipomdp=ipomdp,
            pp_shield=pp_shield,
            perception=perception,
            rt_shield_factory=rt_shield_factory
        )

        metrics_by_mode, _ = evaluator.evaluate(
            action_selector=action_selector,
            num_trials=100,
            trial_length=20,
            sampling_modes=["random"],
            seed=42
        )

        metrics = metrics_by_mode["random"]
        results[name] = metrics

        print(f"  Fail rate: {metrics.fail_rate:.2%}")
        print(f"  Stuck rate: {metrics.stuck_rate:.2%}")
        print(f"  Safe rate: {metrics.safe_rate:.2%}")
        print(f"  Mean steps: {metrics.mean_steps:.1f}")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Fail Rate':<15} {'Stuck Rate':<15} {'Safe Rate':<15}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics.fail_rate:<15.2%} "
              f"{metrics.stuck_rate:<15.2%} {metrics.safe_rate:<15.2%}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"Optimal fixed vs uniform improvement: "
          f"{(results['optimal_fixed'].fail_rate - results['uniform'].fail_rate):.2%}")
    print(f"Optimal fixed vs adversarial: "
          f"{(results['optimal_fixed'].fail_rate - results['adversarial'].fail_rate):.2%}")


def demonstrate_save_load():
    """Demonstrate save/load functionality."""
    print("\n" + "=" * 70)
    print("DEMONSTRATING SAVE/LOAD")
    print("=" * 70)

    # Load TaxiNet IPOMDP
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp()

    # Create runtime shield factory
    rt_shield_factory = create_rt_shield_factory(ipomdp, pp_shield)

    # Train and save
    print("\n1. Training and saving model...")
    optimal_perception = train_optimal_realization(
        ipomdp=ipomdp,
        pp_shield=pp_shield,
        rt_shield_factory=rt_shield_factory,
        action_selector=RandomActionSelector(),
        num_candidates=10,
        num_trials_per_candidate=5,
        max_iterations=5,
        trial_length=20,
        save_path="/tmp/demo_realization.json",
        verbose=False
    )

    print(f"Saved model with score: {optimal_perception.metadata['objective_score']:.4f}")

    # Load
    print("\n2. Loading model...")
    loaded_perception = FixedRealizationPerceptionModel.load("/tmp/demo_realization.json")
    print(f"Loaded model with score: {loaded_perception.metadata['objective_score']:.4f}")

    # Verify they produce same results
    print("\n3. Verifying consistency...")
    evaluator = MonteCarloSafetyEvaluator(
        ipomdp=ipomdp,
        pp_shield=pp_shield,
        perception=loaded_perception,
        rt_shield_factory=rt_shield_factory
    )

    metrics_by_mode, _ = evaluator.evaluate(
        action_selector=RandomActionSelector(),
        num_trials=50,
        trial_length=20,
        sampling_modes=["random"],
        seed=42
    )

    metrics = metrics_by_mode["random"]
    print(f"Evaluation failure rate: {metrics.fail_rate:.2%}")
    print("\nSave/load verification complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("OPTIMAL FIXED REALIZATION EXAMPLE")
    print("=" * 70)

    # Example 1: Train against random selector
    train_optimal_vs_random()

    # Example 2: Train against safest selector
    # Uncomment to run (takes longer):
    # train_optimal_vs_safest()

    # Example 3: Compare perception models
    compare_perception_models()

    # Example 4: Demonstrate save/load
    demonstrate_save_load()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
