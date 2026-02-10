"""Full experiment: Perception × Action Selection × Shielding Strategy.

Evaluates all combinations from the experiment plan:

Perception Realization (Nature's strategy):
  - Uniform Random (UniformPerceptionModel)
  - Adversarial Greedy (AdversarialPerceptionModel)
  - Adversarial Optimized (OptimizedRealizationTrainer)

Action Selection (Agent's strategy):
  - Uniform Random Shielded (RandomActionSelector)
  - Safest Shielded Action (BeliefSelector mode="best")
  - RL UnShielded (NeuralActionSelector)
  - RL Envolope Shielded (BeliefShieldedActionSelector)
  - RL Point Shielded (SingleBeliefActionSelector)

Shielding Strategies:
  - Baseline PP - apply pp_shield directly to state estimate
  - Belief Shield - LP belief propagation via RuntimeImpShield

Usage:
    python -m ipomdp_shielding.experiments.full_experiment
"""

import os
import json
import time

from ..CaseStudies.Taxinet import build_taxinet_ipomdp
from ..Evaluation.runtime_shield import RuntimeImpShield
from ..Propagators import LFPPropagator, BeliefPolytope, TemplateFactory
from ..Propagators.lfp_propagator import default_solver
from ..MonteCarlo import (
    UniformPerceptionModel,
    AdversarialPerceptionModel,
    FixedRealizationPerceptionModel,
    train_optimal_realization,
    RandomActionSelector,
    BeliefSelector,
    BeliefShieldedActionSelector,
    SingleBeliefShieldedActionSelector,
    NeuralActionSelector,
    RandomInitialState,
    run_monte_carlo_trials,
    compute_safety_metrics,
)


# ============================================================
# Configuration — adjust for demo vs production
# ============================================================

NUM_TRIALS = 30
TRIAL_LENGTH = 20
SEED = 42

# RL training
RL_EPISODES = 400
RL_EPISODE_LENGTH = 20

# Optimized realization training
OPT_CANDIDATES = 10
OPT_TRIALS_PER_CANDIDATE = 5
OPT_ITERATIONS = 10

# Cache paths
RL_CACHE_PATH = "/tmp/full_exp_rl_agent.pt"
OPT_CACHE_PATH = "/tmp/full_exp_optimal_realization.json"
RESULTS_PATH = "./data/full_experiment_results.json"


# ============================================================
# Lightweight shield wrappers
# ============================================================

class PPDirectShield:
    """Baseline shield: applies pp_shield directly to observed state.

    No belief propagation — just looks up pp_shield[obs].
    """

    def __init__(self, pp_shield):
        self.pp_shield = pp_shield
        self.stuck_count = 0
        self.error_count = 0

    def next_actions(self, evidence):
        obs, _action = evidence
        allowed = list(self.pp_shield.get(obs, set()))
        if not allowed:
            self.stuck_count += 1
        return allowed

    def get_action_probs(self):
        return []

    def restart(self):
        self.stuck_count = 0
        self.error_count = 0

    def initialize(self, _initial_state):
        self.restart()


class NoShield:
    """Passthrough shield: allows all actions (for unshielded RL)."""

    def __init__(self, all_actions):
        self.all_actions = list(all_actions)
        self.stuck_count = 0
        self.error_count = 0

    def next_actions(self, _evidence):
        return list(self.all_actions)

    def get_action_probs(self):
        return []

    def restart(self):
        self.stuck_count = 0
        self.error_count = 0

    def initialize(self, _initial_state):
        self.restart()


# ============================================================
# Shield factory builders
# ============================================================

def create_belief_shield_factory(ipomdp, pp_shield, threshold=0.8):
    """Factory producing RuntimeImpShield with LFP belief propagation."""
    def factory():
        n = len(ipomdp.states)
        template = TemplateFactory.canonical(n)
        polytope = BeliefPolytope.uniform_prior(n)
        propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)
        return RuntimeImpShield(pp_shield, propagator, action_shield=threshold)
    return factory


def create_pp_direct_shield_factory(pp_shield):
    """Factory producing PPDirectShield (baseline PP lookup)."""
    def factory():
        return PPDirectShield(pp_shield)
    return factory


def create_no_shield_factory(all_actions):
    """Factory producing NoShield (passthrough for unshielded RL)."""
    def factory():
        return NoShield(all_actions)
    return factory


# ============================================================
# Setup: load model, train/cache RL agent & optimized realization
# ============================================================

def setup(ipomdp, pp_shield):
    """Train or load RL agent and optimized realization.

    Returns (rl_selector, optimized_perception).
    """
    # --- RL Agent ---
    print("\n" + "=" * 70)
    print("SETUP: RL AGENT")
    print("=" * 70)

    if os.path.exists(RL_CACHE_PATH):
        print(f"Loading cached RL agent from {RL_CACHE_PATH}")
        rl_selector = NeuralActionSelector.load(RL_CACHE_PATH, ipomdp)
    else:
        print(f"Training RL agent ({RL_EPISODES} episodes)...")
        rl_selector = NeuralActionSelector(
            actions=list(ipomdp.actions),
            observations=ipomdp.observations,
            maximize_safety=True,
        )
        train_metrics = rl_selector.train(
            ipomdp=ipomdp,
            perception=UniformPerceptionModel(),
            num_episodes=RL_EPISODES,
            episode_length=RL_EPISODE_LENGTH,
            verbose=True,
        )
        rl_selector.save(RL_CACHE_PATH)
        print(f"RL agent saved to {RL_CACHE_PATH}")
        print(f"  Final safe rate: {train_metrics['final_safe_rate']:.2%}")

    # Set to evaluation mode
    rl_selector.exploration_rate = 0.0

    # --- Optimized Realization ---
    print("\n" + "=" * 70)
    print("SETUP: OPTIMIZED REALIZATION")
    print("=" * 70)

    if os.path.exists(OPT_CACHE_PATH):
        print(f"Loading cached optimized realization from {OPT_CACHE_PATH}")
        optimized_perception = FixedRealizationPerceptionModel.load(OPT_CACHE_PATH)
    else:
        print(f"Training optimized realization ({OPT_ITERATIONS} iterations)...")
        rt_shield_factory = create_belief_shield_factory(ipomdp, pp_shield)
        optimized_perception = train_optimal_realization(
            ipomdp=ipomdp,
            pp_shield=pp_shield,
            rt_shield_factory=rt_shield_factory,
            action_selector=RandomActionSelector(),
            initial_generator=RandomInitialState(),
            num_candidates=OPT_CANDIDATES,
            num_trials_per_candidate=OPT_TRIALS_PER_CANDIDATE,
            max_iterations=OPT_ITERATIONS,
            trial_length=TRIAL_LENGTH,
            save_path=OPT_CACHE_PATH,
            verbose=True,
        )
        print(f"Optimized realization saved to {OPT_CACHE_PATH}")
        score = optimized_perception.metadata.get('objective_score', 'N/A')
        print(f"  Best score: {score}")

    return rl_selector, optimized_perception


# ============================================================
# Build experiment grid
# ============================================================

def build_grid(ipomdp, pp_shield, rl_selector, optimized_perception):
    """Build the full experiment grid.

    Returns list of (perception_name, selector_name, shield_name,
                     perception, selector, shield_factory) tuples.
    """
    all_actions = list(ipomdp.actions)

    # Perception models
    perceptions = {
        "uniform": UniformPerceptionModel(),
        "adversarial": AdversarialPerceptionModel(pp_shield),
        "optimized": optimized_perception,
    }

    # Shield factories
    belief_factory = create_belief_shield_factory(ipomdp, pp_shield)
    pp_factory = create_pp_direct_shield_factory(pp_shield)
    no_shield_factory = create_no_shield_factory(all_actions)

    grid = []

    pomdp = ipomdp.to_pomdp()
    
    for p_name, perception in perceptions.items():
        # Shielded selectors × both shield strategies
        shielded_selectors = {
            "random": RandomActionSelector(),
            "safest": BeliefSelector(mode="best"),
            "rl_envolope_shielded": BeliefShieldedActionSelector(rl_selector, all_actions),
            "rl_point_shielded" : SingleBeliefShieldedActionSelector(rl_selector, all_actions, pomdp, pp_shield),
        }

        for s_name, selector in shielded_selectors.items():
            for sh_name, sh_factory in [("baseline_pp", pp_factory),
                                         ("belief", belief_factory)]:
                grid.append((p_name, s_name, sh_name,
                             perception, selector, sh_factory))

        # RL unshielded — no shield, one run per perception
        grid.append((p_name, "rl_unshielded", "none",
                     perception, rl_selector, no_shield_factory))

    return grid


# ============================================================
# Run experiment
# ============================================================

def run_experiment(ipomdp, pp_shield, grid):
    """Run all grid combinations and collect results.

    Returns dict mapping (perception, selector, shield) → MCSafetyMetrics.
    """
    results = {}
    total = len(grid)

    for i, (p_name, s_name, sh_name, perception, selector, sh_factory) in enumerate(grid):
        label = f"{p_name}/{s_name}/{sh_name}"
        print(f"\n[{i+1}/{total}] Running: {label} ...", end=" ", flush=True)

        t0 = time.time()
        trial_results = run_monte_carlo_trials(
            ipomdp=ipomdp,
            pp_shield=pp_shield,
            perception=perception,
            rt_shield_factory=sh_factory,
            action_selector=selector,
            initial_generator=RandomInitialState(),
            num_trials=NUM_TRIALS,
            trial_length=TRIAL_LENGTH,
            seed=SEED,
        )
        metrics = compute_safety_metrics(trial_results)
        elapsed = time.time() - t0

        results[(p_name, s_name, sh_name)] = metrics
        print(f"fail={metrics.fail_rate:.1%}  stuck={metrics.stuck_rate:.1%}  "
              f"safe={metrics.safe_rate:.1%}  ({elapsed:.1f}s)")

    return results


# ============================================================
# Print and save results
# ============================================================

def print_results_table(results):
    """Print formatted comparison table."""
    print("\n" + "=" * 85)
    print("FULL EXPERIMENT RESULTS")
    print("=" * 85)
    header = (f"{'Perception':<14} {'Action Sel':<16} {'Shield':<13} "
              f"{'Fail%':>7} {'Stuck%':>7} {'Safe%':>7} {'Steps':>7}")
    print(header)
    print("-" * 85)

    # Sort by perception, then selector, then shield
    for key in sorted(results.keys()):
        p_name, s_name, sh_name = key
        m = results[key]
        print(f"{p_name:<14} {s_name:<16} {sh_name:<13} "
              f"{m.fail_rate:>6.1%} {m.stuck_rate:>6.1%} {m.safe_rate:>6.1%} "
              f"{m.mean_steps:>7.1f}")

    # Analysis: shield comparison
    print("\n" + "=" * 85)
    print("ANALYSIS: BELIEF SHIELD vs BASELINE PP")
    print("=" * 85)
    for p_name in ["uniform", "adversarial", "optimized"]:
        for s_name in ["random", "safest", "rl_shielded"]:
            pp_key = (p_name, s_name, "baseline_pp")
            belief_key = (p_name, s_name, "belief")
            if pp_key in results and belief_key in results:
                pp_fail = results[pp_key].fail_rate
                belief_fail = results[belief_key].fail_rate
                diff = belief_fail - pp_fail
                print(f"  {p_name}/{s_name}: belief - pp = {diff:+.1%} fail rate")

    # Analysis: perception comparison
    print("\n" + "=" * 85)
    print("ANALYSIS: PERCEPTION MODEL IMPACT (belief shield)")
    print("=" * 85)
    for s_name in ["random", "safest", "rl_shielded", "rl_unshielded"]:
        sh_name = "belief" if s_name != "rl_unshielded" else "none"
        u_key = ("uniform", s_name, sh_name)
        a_key = ("adversarial", s_name, sh_name)
        o_key = ("optimized", s_name, sh_name)
        if u_key in results and a_key in results and o_key in results:
            u_fail = results[u_key].fail_rate
            a_fail = results[a_key].fail_rate
            o_fail = results[o_key].fail_rate
            print(f"  {s_name}: uniform={u_fail:.1%}  adversarial={a_fail:.1%}  "
                  f"optimized={o_fail:.1%}")


def save_results(results):
    """Save results to JSON."""
    serializable = {}
    for (p, s, sh), m in results.items():
        key = f"{p}/{s}/{sh}"
        serializable[key] = {
            "fail_rate": m.fail_rate,
            "stuck_rate": m.stuck_rate,
            "safe_rate": m.safe_rate,
            "mean_steps": m.mean_steps,
            "num_trials": m.num_trials,
        }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("FULL EXPERIMENT: Perception x Action Selection x Shielding")
    print(f"Trials: {NUM_TRIALS}, Length: {TRIAL_LENGTH}, Seed: {SEED}")
    print("=" * 70)

    # Load TaxiNet IPOMDP
    print("\nLoading TaxiNet IPOMDP...")
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp()
    print(f"  States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    # Setup: train/load RL agent and optimized realization
    rl_selector, optimized_perception = setup(ipomdp, pp_shield)

    # Build experiment grid
    grid = build_grid(ipomdp, pp_shield, rl_selector, optimized_perception)
    print(f"\nExperiment grid: {len(grid)} combinations")

    # Run all combinations
    t0 = time.time()
    results = run_experiment(ipomdp, pp_shield, grid)
    total_time = time.time() - t0

    # Print and save results
    print_results_table(results)
    save_results(results)

    print(f"\nTotal experiment time: {total_time:.1f}s")
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
