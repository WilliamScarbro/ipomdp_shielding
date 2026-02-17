"""
Run Carr vs Lifted comparison with perfect observations.

This should show a more interesting comparison where both shields operate.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

from ipomdp_shielding.Models.pomdp import POMDP
from ipomdp_shielding.Models.ipomdp import IPOMDP
from ipomdp_shielding.CaseStudies.Taxinet.taxinet import (
    taxinet_states, taxinet_dynamics_prob, taxinet_actions,
    taxinet_next_state, taxinet_safe
)
from ipomdp_shielding.Evaluation.carr_shield import CarrShield
from ipomdp_shielding.Evaluation.runtime_shield import RuntimeImpShield
from ipomdp_shielding.Propagators.lfp_propagator import LFPPropagator, TemplateFactory, default_solver
from ipomdp_shielding.Propagators.belief_polytope import BeliefPolytope


def build_perfect_obs_pomdp(error=0.01):
    """Build TaxiNet POMDP with perfect observations."""
    states = taxinet_states(with_fail=True)
    actions_dict = taxinet_actions()
    all_actions = list(set(a for acts in actions_dict.values() for a in acts))

    # Use TaxiNet dynamics
    mdp = taxinet_dynamics_prob(error=error)

    # Perfect observations: P(o=s | s) = 1.0
    P = {}
    for s in states:
        P[s] = {obs: 1.0 if obs == s else 0.0 for obs in states}

    return POMDP(
        states=states,
        observations=states,
        actions=all_actions,
        T=mdp.P,
        P=P
    )


def pomdp_to_ipomdp(pomdp):
    """Convert POMDP to IPOMDP with exact bounds."""
    P_lower = {s: dict(pomdp.P[s]) for s in pomdp.states}
    P_upper = {s: dict(pomdp.P[s]) for s in pomdp.states}

    return IPOMDP(
        states=pomdp.states,
        observations=pomdp.observations,
        actions=pomdp.actions,
        T=pomdp.T,
        P_lower=P_lower,
        P_upper=P_upper
    )


def run_trajectory(shield, shield_name, initial_state, max_steps, avoid_states):
    """Run a single trajectory."""
    state = initial_state

    if shield_name == "carr":
        shield.restart()
        shield.initialize()
    else:
        shield.initialize(state)

    for t in range(max_steps):
        # Get allowed actions
        if t == 0:
            if shield_name == "carr":
                allowed_actions = shield.next_actions(None)
            else:
                allowed_actions = list(shield.actions)
        else:
            obs = state  # Perfect observation
            allowed_actions = shield.next_actions((obs, action))

        # Check if stuck
        if not allowed_actions:
            return {"success": False, "stuck": True, "failed": False, "length": t}

        # Select action
        action = random.choice(allowed_actions)

        # Take action
        next_state = taxinet_next_state(state, action)

        # Check if failed
        if not taxinet_safe(next_state):
            return {"success": False, "stuck": False, "failed": True, "length": t + 1}

        state = next_state

    return {"success": True, "stuck": False, "failed": False, "length": max_steps}


def main():
    """Run comparison experiment."""
    print("="*80)
    print("CARR VS LIFTED: PERFECT OBSERVATIONS EXPERIMENT")
    print("="*80)

    # Config
    seed = 42
    num_trials = 100
    trial_length = 50
    threshold = 0.9
    error = 0.01
    initial_state = (0, 0)

    random.seed(seed)

    # Build models
    print("\nBuilding TaxiNet with perfect observations...")
    pomdp = build_perfect_obs_pomdp(error=error)
    ipomdp = pomdp_to_ipomdp(pomdp)

    avoid_states = frozenset(["FAIL"])
    safe_states = frozenset(s for s in pomdp.states if s not in avoid_states)
    initial_support = frozenset([initial_state])

    print(f"States: {len(pomdp.states)}, Actions: {len(pomdp.actions)}")
    print(f"Dynamics error: {error}")
    print(f"Initial state: {initial_state}")

    # Build Carr shield
    print("\nBuilding Carr shield...")
    carr_shield = CarrShield(pomdp, avoid_states, initial_support)
    stats = carr_shield.get_statistics()
    print(f"Support-MDP: {stats['total_supports']} supports, {stats['winning_supports']} winning")

    # Build Lifted shield
    print("\nBuilding Lifted shield...")
    n_states = len(ipomdp.states)
    state_to_idx = {s: i for i, s in enumerate(ipomdp.states)}
    safe_indices = [state_to_idx[s] for s in safe_states]
    template = TemplateFactory.safe_set_indicators(n_states, {"safe": safe_indices})
    initial_polytope = BeliefPolytope.uniform_prior(n_states)
    lfp_propagator = LFPPropagator(ipomdp, template, default_solver(), initial_polytope)

    pp_shield = {
        s: set(ipomdp.actions) if s in safe_states else set()
        for s in ipomdp.states
    }
    lifted_shield = RuntimeImpShield(pp_shield, lfp_propagator, threshold, default_action=None)

    # Run trials
    print("\n" + "="*80)
    print("Running Carr shield trials...")
    print("="*80)
    carr_results = []
    for i in range(num_trials):
        if (i + 1) % 20 == 0:
            print(f"Trial {i+1}/{num_trials}")
        result = run_trajectory(carr_shield, "carr", initial_state, trial_length, avoid_states)
        carr_results.append(result)

    print("\n" + "="*80)
    print("Running Lifted shield trials...")
    print("="*80)
    lifted_results = []
    for i in range(num_trials):
        if (i + 1) % 20 == 0:
            print(f"Trial {i+1}/{num_trials}")
        result = run_trajectory(lifted_shield, "lifted", initial_state, trial_length, avoid_states)
        lifted_results.append(result)

    # Compute metrics
    def compute_metrics(results):
        n = len(results)
        success = sum(1 for r in results if r["success"])
        stuck = sum(1 for r in results if r["stuck"])
        failed = sum(1 for r in results if r["failed"])
        avg_length = sum(r["length"] for r in results) / n

        return {
            "success_rate": success / n,
            "stuck_rate": stuck / n,
            "fail_rate": failed / n,
            "avg_length": avg_length
        }

    carr_metrics = compute_metrics(carr_results)
    lifted_metrics = compute_metrics(lifted_results)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print("\nCarr Shield (Support-based):")
    print(f"  Success rate: {carr_metrics['success_rate']:.1%}")
    print(f"  Stuck rate:   {carr_metrics['stuck_rate']:.1%}")
    print(f"  Fail rate:    {carr_metrics['fail_rate']:.1%}")
    print(f"  Avg length:   {carr_metrics['avg_length']:.1f}")

    print("\nLifted Shield (Belief-envelope):")
    print(f"  Success rate: {lifted_metrics['success_rate']:.1%}")
    print(f"  Stuck rate:   {lifted_metrics['stuck_rate']:.1%}")
    print(f"  Fail rate:    {lifted_metrics['fail_rate']:.1%}")
    print(f"  Avg length:   {lifted_metrics['avg_length']:.1f}")

    print("\nComparison:")
    print(f"  Stuck rate diff: {carr_metrics['stuck_rate'] - lifted_metrics['stuck_rate']:.1%}")
    print(f"  Fail rate diff:  {carr_metrics['fail_rate'] - lifted_metrics['fail_rate']:.1%}")

    # Save results
    results = {
        "config": {
            "seed": seed,
            "num_trials": num_trials,
            "trial_length": trial_length,
            "threshold": threshold,
            "error": error,
            "initial_state": initial_state,
            "observation_model": "perfect"
        },
        "carr": {"trials": carr_results, "metrics": carr_metrics},
        "lifted": {"trials": lifted_results, "metrics": lifted_metrics}
    }

    output_path = Path("results/carr_comparison_perfect_obs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
