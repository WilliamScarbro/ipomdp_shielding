"""
Run Carr comparison experiment with different configurations.
"""

from .carr_comparison_experiment import CarrComparisonExperiment
from .configs.carr_comparison_config import CarrComparisonConfig


def run_default_experiment():
    """Run with default parameters."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Default Configuration")
    print("="*80)

    config = CarrComparisonConfig(
        num_trials=100,
        trial_length=50,
        lifted_shield_threshold=0.8,
        realization_strategy="midpoint",
        seed=42,
        results_path="results/carr_comparison_default.json"
    )

    experiment = CarrComparisonExperiment(config)
    experiment.run()


def run_conservative_lifted():
    """Run with more conservative lifted shield."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Conservative Lifted Shield (threshold=0.95)")
    print("="*80)

    config = CarrComparisonConfig(
        num_trials=100,
        trial_length=50,
        lifted_shield_threshold=0.95,
        realization_strategy="midpoint",
        seed=42,
        results_path="results/carr_comparison_conservative.json"
    )

    experiment = CarrComparisonExperiment(config)
    experiment.run()


def run_low_error():
    """Run with lower dynamics error."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Low Dynamics Error (error=0.01)")
    print("="*80)

    config = CarrComparisonConfig(
        num_trials=100,
        trial_length=50,
        lifted_shield_threshold=0.9,
        realization_strategy="midpoint",
        seed=42,
        results_path="results/carr_comparison_low_error.json",
        ipomdp_kwargs={
            "confidence_method": "Clopper_Pearson",
            "alpha": 0.05,
            "train_fraction": 0.8,
            "error": 0.01,  # Much lower error
            "smoothing": True,
        }
    )

    experiment = CarrComparisonExperiment(config)
    experiment.run()


def run_pessimistic_realization():
    """Run with pessimistic realization."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Pessimistic Realization (lower bounds)")
    print("="*80)

    config = CarrComparisonConfig(
        num_trials=100,
        trial_length=50,
        lifted_shield_threshold=0.9,
        realization_strategy="lower",
        seed=42,
        results_path="results/carr_comparison_pessimistic.json"
    )

    experiment = CarrComparisonExperiment(config)
    experiment.run()


def main():
    """Run all experiments."""
    import sys

    experiments = {
        "default": run_default_experiment,
        "conservative": run_conservative_lifted,
        "low_error": run_low_error,
        "pessimistic": run_pessimistic_realization,
        "all": lambda: [run_default_experiment(), run_conservative_lifted(),
                        run_low_error(), run_pessimistic_realization()]
    }

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        if exp_name in experiments:
            experiments[exp_name]()
        else:
            print(f"Unknown experiment: {exp_name}")
            print(f"Available experiments: {list(experiments.keys())}")
    else:
        print("Available experiments:")
        for name in experiments.keys():
            print(f"  - {name}")
        print("\nUsage: python -m ipomdp_shielding.experiments.run_carr_comparison <experiment>")
        print("\nRunning default experiment...")
        run_default_experiment()


if __name__ == "__main__":
    main()
