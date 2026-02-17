"""
Carr vs Lifted Shield Comparison Experiment.

Compares support-based shielding (Carr et al.) with belief-envelope shielding (Lifted shield).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import asdict

from .experiment_io import build_metadata, add_rate_cis, save_experiment_results

from ..Models.ipomdp import IPOMDP
from ..Models.pomdp import POMDP, State, Action, Observation
from ..CaseStudies.Taxinet.taxinet import build_taxinet_ipomdp, taxinet_next_state, taxinet_safe
from ..CaseStudies.Taxinet.taxinet_pomdp_adapter import (
    convert_taxinet_to_pomdp,
    get_taxinet_avoid_states,
    get_taxinet_initial_support,
)
from ..Evaluation.carr_shield import CarrShield
from ..Evaluation.runtime_shield import RuntimeImpShield
from ..Propagators.lfp_propagator import LFPPropagator, TemplateFactory, default_solver
from ..Propagators.belief_polytope import BeliefPolytope
from .configs.carr_comparison_config import CarrComparisonConfig


class CarrComparisonExperiment:
    """
    Experiment comparing Carr shield with Lifted shield on TaxiNet.
    """

    def __init__(self, config: CarrComparisonConfig):
        """
        Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        random.seed(config.seed)

        # Build models
        print("Building TaxiNet IPOMDP...")
        self.ipomdp, self.dynamic_shield, self.test_cte_model, self.test_he_model = \
            build_taxinet_ipomdp(seed=config.seed, **config.ipomdp_kwargs)

        print(f"Converting IPOMDP to POMDP (realization: {config.realization_strategy})...")
        self.pomdp = convert_taxinet_to_pomdp(
            self.ipomdp,
            realization=config.realization_strategy,
            seed=config.seed
        )

        # Get avoid states and initial support
        self.avoid_states = get_taxinet_avoid_states()
        self.initial_support = get_taxinet_initial_support(self.ipomdp)
        self.initial_state = config.initial_state

        print(f"Number of states: {len(self.ipomdp.states)}")
        print(f"Number of observations: {len(self.ipomdp.observations)}")
        print(f"Number of actions: {len(self.ipomdp.actions)}")
        print(f"Avoid states: {self.avoid_states}")
        print(f"Initial support size: {len(self.initial_support)}")

        # Build shields
        print("\nBuilding Carr shield (support-based)...")
        self.carr_shield = CarrShield(
            self.pomdp,
            self.avoid_states,
            self.initial_support
        )
        print(f"Support-MDP statistics: {self.carr_shield.get_statistics()}")

        print("\nBuilding Lifted shield (belief-envelope)...")
        # Create LFP propagator for IPOMDP
        n_states = len(self.ipomdp.states)
        if config.template_type == "canonical":
            template = TemplateFactory.canonical(n_states)
        elif config.template_type == "safe_set_indicators":
            # Map state to index
            state_to_idx = {s: i for i, s in enumerate(self.ipomdp.states)}
            safe_indices = [state_to_idx[s] for s in self.ipomdp.states if s not in self.avoid_states]
            safe_sets = {"safe": safe_indices}
            template = TemplateFactory.safe_set_indicators(n_states, safe_sets)
        else:
            raise ValueError(f"Unknown template type: {config.template_type}")

        # Create initial belief polytope (uniform prior)
        initial_polytope = BeliefPolytope.uniform_prior(n_states)

        # Create LFP propagator
        self.lfp_propagator = LFPPropagator(
            ipomdp=self.ipomdp,
            template=template,
            solver=default_solver(),
            belief=initial_polytope
        )

        # Create perfect-perception shield for Lifted shield
        safe_states = set(self.ipomdp.states) - self.avoid_states
        perfect_perception_shield = {
            s: set(self.ipomdp.actions) if s in safe_states else set()
            for s in self.ipomdp.states
        }

        self.lifted_shield = RuntimeImpShield(
            perfect_perception_shield,
            self.lfp_propagator,
            config.lifted_shield_threshold,
            default_action=None
        )

        # Results storage
        self.results: Dict[str, Any] = {
            "config": asdict(config),
            "carr_trials": [],
            "lifted_trials": [],
        }

    def run_single_trajectory(
        self,
        shield_name: str,
        shield: Any,
        max_steps: int,
        track_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single trajectory with the given shield.

        Args:
            shield_name: Name of shield ("carr" or "lifted")
            shield: Shield object (CarrShield or RuntimeImpShield)
            max_steps: Maximum timesteps
            track_metrics: Whether to track detailed metrics

        Returns:
            Dictionary with trial outcome
        """
        # Initialize
        state = self.initial_state
        if shield_name == "carr":
            shield.restart()
            shield.initialize()
        else:  # lifted
            shield.initialize(state)

        outcome = {
            "success": False,
            "stuck": False,
            "failed": False,
            "timestep": 0,
            "trajectory_length": 0,
            "support_sizes": [] if track_metrics else None,
            "belief_masses": [] if track_metrics else None,
        }

        # Run trajectory
        prev_action = None  # Initialize for first iteration
        for t in range(max_steps):
            # Get allowed actions
            if t == 0:
                # First timestep - no prior evidence
                if shield_name == "carr":
                    allowed_actions = shield.next_actions(None)
                else:  # lifted
                    # Skip propagation on first call
                    allowed_actions = list(shield.actions)
            else:
                # For TaxiNet, observation is the estimated state
                obs = self._get_observation(state)
                allowed_actions = shield.next_actions((obs, prev_action))

            # Track metrics
            if track_metrics:
                if shield_name == "carr":
                    support_size = len(shield.propagator.get_support())
                    outcome["support_sizes"].append(support_size)
                elif shield_name == "lifted":
                    # Track belief envelope mass on avoid states
                    avoid_indices = [shield.state_to_idx[s] for s in self.avoid_states if s in shield.state_to_idx]
                    avoid_prob = shield.ipomdp_belief.maximum_disallowed_probability(avoid_indices)
                    outcome["belief_masses"].append(avoid_prob)

            # Check if stuck
            if not allowed_actions:
                outcome["stuck"] = True
                outcome["timestep"] = t
                outcome["trajectory_length"] = t
                return outcome

            # Select action randomly from allowed
            action = random.choice(allowed_actions)
            prev_action = action  # Store for next iteration

            # Take action
            next_state = taxinet_next_state(state, action)

            # Check if failed
            if not taxinet_safe(next_state):
                outcome["failed"] = True
                outcome["timestep"] = t + 1
                outcome["trajectory_length"] = t + 1
                return outcome

            state = next_state

        # Completed successfully
        outcome["success"] = True
        outcome["trajectory_length"] = max_steps
        return outcome

    def _get_observation(self, state: State) -> Observation:
        """
        Get observation for TaxiNet state.

        Args:
            state: Current state (cte, he)

        Returns:
            Observation (estimated state)
        """
        if state == "FAIL":
            return "FAIL"

        cte, he = state
        # Sample from test models (these are dicts mapping state component to list of observations)
        cte_options = self.test_cte_model.get(cte, [])
        he_options = self.test_he_model.get(he, [])

        # Fallback to true state if no observations available
        cte_obs = random.choice(cte_options) if cte_options else cte
        he_obs = random.choice(he_options) if he_options else he

        return (cte_obs, he_obs)

    def run_trials(self):
        """Run Monte Carlo trials for both shields."""
        print("\n" + "="*60)
        print("Running Carr shield trials...")
        print("="*60)

        for trial_idx in range(self.config.num_trials):
            if (trial_idx + 1) % 10 == 0:
                print(f"Carr trial {trial_idx + 1}/{self.config.num_trials}")

            outcome = self.run_single_trajectory(
                "carr",
                self.carr_shield,
                self.config.trial_length,
                track_metrics=self.config.track_support_size
            )
            self.results["carr_trials"].append(outcome)

        print("\n" + "="*60)
        print("Running Lifted shield trials...")
        print("="*60)

        for trial_idx in range(self.config.num_trials):
            if (trial_idx + 1) % 10 == 0:
                print(f"Lifted trial {trial_idx + 1}/{self.config.num_trials}")

            outcome = self.run_single_trajectory(
                "lifted",
                self.lifted_shield,
                self.config.trial_length,
                track_metrics=self.config.track_belief_mass
            )
            self.results["lifted_trials"].append(outcome)

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics from trials.

        Returns:
            Dictionary with comparison metrics
        """
        carr_trials = self.results["carr_trials"]
        lifted_trials = self.results["lifted_trials"]

        def compute_shield_metrics(trials: List[Dict]) -> Dict[str, Any]:
            n_trials = len(trials)
            success = sum(1 for t in trials if t["success"])
            stuck = sum(1 for t in trials if t["stuck"])
            failed = sum(1 for t in trials if t["failed"])

            avg_length = sum(t["trajectory_length"] for t in trials) / n_trials if n_trials > 0 else 0

            stuck_timesteps = [t["timestep"] for t in trials if t["stuck"]]

            return {
                "total_trials": n_trials,
                "success_count": success,
                "stuck_count": stuck,
                "failed_count": failed,
                "success_rate": success / n_trials if n_trials > 0 else 0,
                "stuck_rate": stuck / n_trials if n_trials > 0 else 0,
                "fail_rate": failed / n_trials if n_trials > 0 else 0,
                "avg_trajectory_length": avg_length,
                "stuck_timesteps_distribution": stuck_timesteps,
            }

        metrics = {
            "carr": compute_shield_metrics(carr_trials),
            "lifted": compute_shield_metrics(lifted_trials),
        }

        # Compute comparison
        carr_stuck_rate = metrics["carr"]["stuck_rate"]
        lifted_stuck_rate = metrics["lifted"]["stuck_rate"]

        metrics["comparison"] = {
            "stuck_rate_difference": carr_stuck_rate - lifted_stuck_rate,
            "stuck_rate_ratio": carr_stuck_rate / lifted_stuck_rate if lifted_stuck_rate > 0 else float('inf'),
        }

        return metrics

    def print_results(self, metrics: Dict[str, Any]):
        """Print comparison results."""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        print("\nCarr Shield (Support-based):")
        print(f"  Success rate: {metrics['carr']['success_rate']:.2%}")
        print(f"  Stuck rate:   {metrics['carr']['stuck_rate']:.2%}")
        print(f"  Fail rate:    {metrics['carr']['fail_rate']:.2%}")
        print(f"  Avg trajectory length: {metrics['carr']['avg_trajectory_length']:.1f}")

        print("\nLifted Shield (Belief-envelope):")
        print(f"  Success rate: {metrics['lifted']['success_rate']:.2%}")
        print(f"  Stuck rate:   {metrics['lifted']['stuck_rate']:.2%}")
        print(f"  Fail rate:    {metrics['lifted']['fail_rate']:.2%}")
        print(f"  Avg trajectory length: {metrics['lifted']['avg_trajectory_length']:.1f}")

        print("\nComparison:")
        print(f"  Stuck rate difference (Carr - Lifted): {metrics['comparison']['stuck_rate_difference']:.2%}")
        if metrics['comparison']['stuck_rate_ratio'] != float('inf'):
            print(f"  Stuck rate ratio (Carr / Lifted): {metrics['comparison']['stuck_rate_ratio']:.2f}x")
        else:
            print(f"  Stuck rate ratio (Carr / Lifted): ∞ (Lifted never stuck)")

        print("\n" + "="*60)

    def save_results(self, metrics: Dict[str, Any]):
        """Save results to JSON file with full metadata and CIs."""
        # Add CIs to each shield's metrics
        for shield_name in ["carr", "lifted"]:
            if shield_name in metrics:
                add_rate_cis(metrics[shield_name], metrics[shield_name]["total_trials"])

        self.results["metrics"] = metrics

        # Build tidy rows for CSV output
        tidy_rows = []
        for shield_name in ["carr", "lifted"]:
            if shield_name in metrics:
                m = metrics[shield_name]
                tidy_rows.append({
                    "shield": shield_name,
                    "num_trials": m["total_trials"],
                    "fail_rate": m["fail_rate"],
                    "stuck_rate": m["stuck_rate"],
                    "success_rate": m["success_rate"],
                    "avg_trajectory_length": m["avg_trajectory_length"],
                    "fail_rate_ci_low": m.get("fail_rate_ci_low", ""),
                    "fail_rate_ci_high": m.get("fail_rate_ci_high", ""),
                    "stuck_rate_ci_low": m.get("stuck_rate_ci_low", ""),
                    "stuck_rate_ci_high": m.get("stuck_rate_ci_high", ""),
                })

        extra = {
            "carr_statistics": self.carr_shield.get_statistics(),
        }
        metadata = build_metadata(self.config, extra=extra)
        save_experiment_results(
            self.config.results_path, self.results, metadata, tidy_rows
        )
        print(f"\nResults saved to: {self.config.results_path}")

    def run(self):
        """Run complete experiment."""
        print("\n" + "="*60)
        print("CARR VS LIFTED SHIELD COMPARISON EXPERIMENT")
        print("="*60)

        # Run trials
        self.run_trials()

        # Compute metrics
        metrics = self.compute_metrics()

        # Print results
        self.print_results(metrics)

        # Save results
        self.save_results(metrics)


def main():
    """Main entry point."""
    config = CarrComparisonConfig(
        num_trials=100,
        trial_length=50,
        lifted_shield_threshold=0.8,
        realization_strategy="midpoint",
        seed=42
    )

    experiment = CarrComparisonExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
