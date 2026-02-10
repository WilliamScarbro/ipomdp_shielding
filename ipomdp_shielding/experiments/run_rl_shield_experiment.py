"""RL Shielding Experiment: Compare shield strategies for RL action selection.

Evaluates RL action selection under four shielding strategies, with
uniform-random and adversarial-optimized perception realizations.

Usage:
    python -m ipomdp_shielding.experiments.run_rl_shield_experiment <config_module>

Example:
    python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim
"""

import os
import sys
import json
import time
import importlib
import random as random_module

from ..Evaluation.runtime_shield import RuntimeImpShield
from ..Propagators import LFPPropagator, BeliefPolytope, TemplateFactory
from ..Propagators.lfp_propagator import default_solver
from ..Models.pomdp import POMDP_Belief
from ..MonteCarlo import (
    UniformPerceptionModel,
    AdversarialPerceptionModel,
    FixedRealizationPerceptionModel,
    train_optimal_realization,
    RandomActionSelector,
    BeliefSelector,
    NeuralActionSelector,
    ActionSelector,
    RandomInitialState,
    run_monte_carlo_trials,
    compute_safety_metrics,
)


# ============================================================
# Shield classes
# ============================================================

class NoShield:
    """Passthrough shield: allows all actions."""

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


class ObservationShield:
    """Baseline shield: applies pp_shield directly to observed state.

    No belief propagation -- just looks up pp_shield[obs].
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


class SingleBeliefShield:
    """Shield using single-point POMDP belief propagation.

    Maintains a standard POMDP belief (point estimate) and filters
    actions based on P(action allowed | belief) >= threshold.
    """

    def __init__(self, pomdp, pp_shield, threshold=0.8):
        self.pomdp = pomdp
        self.pp_shield = pp_shield
        self.threshold = threshold
        self.belief = POMDP_Belief(pomdp)
        self.stuck_count = 0
        self.error_count = 0

    def _allowance_probability(self, action):
        """P(action allowed | belief) = sum_{s: action in pp_shield[s]} belief(s)."""
        return sum(
            self.belief.belief.get(s, 0.0)
            for s, allowed_set in self.pp_shield.items()
            if action in allowed_set
        )

    def next_actions(self, evidence):
        self.belief.propogate(evidence)
        allowed = [
            a for a in self.pomdp.actions
            if self._allowance_probability(a) >= self.threshold
        ]
        if not allowed:
            self.stuck_count += 1
        return allowed

    def get_action_probs(self):
        """Return (action, allowed_prob, disallowed_prob) triples."""
        return [
            (a, self._allowance_probability(a), 1.0 - self._allowance_probability(a))
            for a in self.pomdp.actions
        ]

    def restart(self):
        self.belief.restart()
        self.stuck_count = 0
        self.error_count = 0

    def initialize(self, _initial_state):
        self.restart()


# ============================================================
# Shield factories
# ============================================================

def create_no_shield_factory(all_actions):
    def factory():
        return NoShield(all_actions)
    return factory


def create_observation_shield_factory(pp_shield):
    def factory():
        return ObservationShield(pp_shield)
    return factory


def create_single_belief_shield_factory(pomdp, pp_shield, threshold=0.8):
    def factory():
        return SingleBeliefShield(pomdp, pp_shield, threshold)
    return factory


def create_envelope_shield_factory(ipomdp, pp_shield, threshold=0.8):
    def factory():
        n = len(ipomdp.states)
        template = TemplateFactory.canonical(n)
        polytope = BeliefPolytope.uniform_prior(n)
        propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)
        return RuntimeImpShield(pp_shield, propagator, action_shield=threshold)
    return factory


# ============================================================
# Action selector wrapper for RL
# ============================================================

class ShieldCompliantSelector(ActionSelector):
    """Wraps a selector that ignores allowed_actions (e.g. NeuralActionSelector)
    to respect the shield's filtering.

    If the primary selector's preferred action is in allowed_actions, use it.
    Otherwise, fall back to random choice from allowed_actions.

    This keeps action selection cleanly separated from shielding --
    the fallback is shield-agnostic.
    """

    def __init__(self, primary_selector, all_actions):
        self.primary = primary_selector
        self.all_actions = all_actions
        self.primary_count = 0
        self.fallback_count = 0

    def select(self, history, allowed_actions, context=None):
        if not allowed_actions:
            return self.primary.select(history, self.all_actions, context)

        preferred = self.primary.select(history, self.all_actions, context)
        if preferred in allowed_actions:
            self.primary_count += 1
            return preferred
        self.fallback_count += 1
        return random_module.choice(allowed_actions)

    def reset_stats(self):
        self.primary_count = 0
        self.fallback_count = 0


# ============================================================
# Setup: train/cache RL agent & optimized realization
# ============================================================

def setup(ipomdp, pp_shield, config):
    """Train or load RL agent and optimized realization.

    RL agent is trained with adversarial-greedy perception (per plan).
    Returns (rl_selector, optimized_perception).
    """
    # --- RL Agent (trained with adversarial-greedy perception) ---
    print("\n" + "=" * 70)
    print("SETUP: RL AGENT")
    print("=" * 70)

    if os.path.exists(config.rl_cache_path):
        print(f"Loading cached RL agent from {config.rl_cache_path}")
        rl_selector = NeuralActionSelector.load(config.rl_cache_path, ipomdp)
    else:
        print(f"Training RL agent ({config.rl_episodes} episodes, adversarial-greedy perception)...")
        rl_selector = NeuralActionSelector(
            actions=list(ipomdp.actions),
            observations=ipomdp.observations,
            maximize_safety=True,
        )
        train_metrics = rl_selector.train(
            ipomdp=ipomdp,
            perception=AdversarialPerceptionModel(pp_shield),
            num_episodes=config.rl_episodes,
            episode_length=config.rl_episode_length,
            verbose=True,
        )
        rl_selector.save(config.rl_cache_path)
        print(f"RL agent saved to {config.rl_cache_path}")
        print(f"  Final safe rate: {train_metrics['final_safe_rate']:.2%}")

    rl_selector.exploration_rate = 0.0

    # --- Optimized Realization ---
    print("\n" + "=" * 70)
    print("SETUP: OPTIMIZED REALIZATION")
    print("=" * 70)

    if os.path.exists(config.opt_cache_path):
        print(f"Loading cached optimized realization from {config.opt_cache_path}")
        optimized_perception = FixedRealizationPerceptionModel.load(config.opt_cache_path)
    else:
        print(f"Training optimized realization ({config.opt_iterations} iterations)...")
        rt_shield_factory = create_envelope_shield_factory(ipomdp, pp_shield)
        optimized_perception = train_optimal_realization(
            ipomdp=ipomdp,
            pp_shield=pp_shield,
            rt_shield_factory=rt_shield_factory,
            action_selector=RandomActionSelector(),
            initial_generator=RandomInitialState(),
            num_candidates=config.opt_candidates,
            num_trials_per_candidate=config.opt_trials_per_candidate,
            max_iterations=config.opt_iterations,
            trial_length=config.trial_length,
            save_path=config.opt_cache_path,
            verbose=True,
        )
        print(f"Optimized realization saved to {config.opt_cache_path}")
        score = optimized_perception.metadata.get('objective_score', 'N/A')
        print(f"  Best score: {score}")

    return rl_selector, optimized_perception


# ============================================================
# Build experiment grid
# ============================================================

def build_grid(ipomdp, pp_shield, rl_selector, optimized_perception, config):
    """Build the 3-factor experiment grid.

    Factors:
      - Perception: uniform, adversarial_opt
      - Selector: random, best, rl
      - Shield: none, observation, single_belief, envelope

    Returns list of (p_name, s_name, sh_name,
                     perception, selector, shield_factory) tuples.
    """
    all_actions = list(ipomdp.actions)
    pomdp = ipomdp.to_pomdp()

    # Factor 1: Perception models
    perceptions = {
        "uniform": UniformPerceptionModel(),
        "adversarial_opt": optimized_perception,
    }

    # Factor 2: Action selectors (independent of shield)
    selectors = {
        "random": RandomActionSelector(),
        "best": BeliefSelector(mode="best"),
        "rl": ShieldCompliantSelector(rl_selector, all_actions),
    }

    # Factor 3: Shield strategies (independent of selector)
    shields = {
        "none": create_no_shield_factory(all_actions),
        "observation": create_observation_shield_factory(pp_shield),
        "single_belief": create_single_belief_shield_factory(
            pomdp, pp_shield, config.shield_threshold),
        "envelope": create_envelope_shield_factory(
            ipomdp, pp_shield, config.shield_threshold),
    }

    grid = []
    for p_name, perception in perceptions.items():
        for s_name, selector in selectors.items():
            for sh_name, shield_factory in shields.items():
                grid.append((p_name, s_name, sh_name,
                             perception, selector, shield_factory))

    return grid


# ============================================================
# Run experiment
# ============================================================

def run_experiment(ipomdp, pp_shield, grid, config):
    """Run all grid combinations, collecting per-trial results.

    Returns (results, trial_data) where:
      - results: dict mapping (perception, selector, shield) -> MCSafetyMetrics
      - trial_data: dict mapping same keys -> list of SafetyTrialResult
    """
    results = {}
    trial_data = {}
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
            num_trials=config.num_trials,
            trial_length=config.trial_length,
            seed=config.seed,
        )
        metrics = compute_safety_metrics(trial_results)
        elapsed = time.time() - t0

        key = (p_name, s_name, sh_name)
        results[key] = metrics
        trial_data[key] = trial_results

        print(f"fail={metrics.fail_rate:.1%}  stuck={metrics.stuck_rate:.1%}  "
              f"safe={metrics.safe_rate:.1%}  ({elapsed:.1f}s)")

    return results, trial_data


# ============================================================
# Per-timestep analysis
# ============================================================

def compute_timestep_outcomes(trial_results, trial_length):
    """Compute per-timestep cumulative outcome probabilities.

    At timestep t:
      P(fail) = fraction of trials that failed at or before step t
      P(stuck) = fraction of trials that got stuck at or before step t
      P(safe) = 1 - P(fail) - P(stuck)

    Returns dict with keys 'fail', 'stuck', 'safe', each a list of
    length trial_length.
    """
    n = len(trial_results)
    if n == 0:
        return {"fail": [0.0] * trial_length,
                "stuck": [0.0] * trial_length,
                "safe": [1.0] * trial_length}

    fail_by_t = [0] * trial_length
    stuck_by_t = [0] * trial_length

    for trial in trial_results:
        if trial.outcome == "fail" and trial.fail_step is not None:
            for t in range(trial.fail_step, trial_length):
                fail_by_t[t] += 1
        elif trial.outcome == "stuck":
            for t in range(trial.steps_completed, trial_length):
                stuck_by_t[t] += 1

    return {
        "fail": [f / n for f in fail_by_t],
        "stuck": [s / n for s in stuck_by_t],
        "safe": [1.0 - fail_by_t[t] / n - stuck_by_t[t] / n
                 for t in range(trial_length)],
    }


# ============================================================
# Plotting
# ============================================================

def plot_results(trial_data, config):
    """Generate 6 figures: 3 outcomes x 2 perceptions.

    Each figure shows RL lines (solid) for all 4 shield strategies,
    plus random baseline (dashed) for reference.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(config.figures_dir, exist_ok=True)

    perceptions = ["uniform", "adversarial_opt"]
    outcomes = ["fail", "stuck", "safe"]
    shield_order = ["none", "observation", "single_belief", "envelope"]

    shield_labels = {
        "none": "No Shield",
        "observation": "Observation Shield",
        "single_belief": "Single-Belief Shield",
        "envelope": "Envelope Shield",
    }
    shield_colors = {
        "none": "gray",
        "observation": "orange",
        "single_belief": "blue",
        "envelope": "green",
    }
    perception_labels = {
        "uniform": "Uniform Random",
        "adversarial_opt": "Adversarial Optimized",
    }
    selector_styles = {
        "rl": ("-", 2.0, 1.0),       # solid, thick, full opacity
        "best": ("--", 1.5, 0.8),     # dashed
        "random": (":", 1.2, 0.5),    # dotted, thin, faded
    }

    timesteps = list(range(config.trial_length))

    for p_name in perceptions:
        for outcome in outcomes:
            fig, ax = plt.subplots(figsize=(10, 6))

            for sh_name in shield_order:
                color = shield_colors[sh_name]
                for s_name, (ls, lw, alpha) in selector_styles.items():
                    key = (p_name, s_name, sh_name)
                    if key not in trial_data:
                        continue
                    ts = compute_timestep_outcomes(trial_data[key], config.trial_length)
                    label = f"{s_name.upper()} + {shield_labels[sh_name]}"
                    ax.plot(timesteps, ts[outcome],
                            label=label, color=color,
                            linestyle=ls, linewidth=lw, alpha=alpha)

            ax.set_xlabel("Timestep")
            ax.set_ylabel(f"P({outcome})")
            ax.set_title(f"{outcome.capitalize()} Rate by Timestep\n"
                         f"Perception: {perception_labels[p_name]} ({config.case_study_name.upper()})")
            ax.legend(loc="best", fontsize=7)
            ax.set_xlim(0, config.trial_length - 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            fname = f"{p_name}_{outcome}.png"
            fig.savefig(os.path.join(config.figures_dir, fname),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {fname}")


# ============================================================
# Results table and analysis
# ============================================================

def print_results_table(results, config):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"RL SHIELDING EXPERIMENT RESULTS - {config.case_study_name.upper()}")
    print("=" * 90)
    header = (f"{'Perception':<17} {'Selector':<10} {'Shield':<16} "
              f"{'Fail%':>7} {'Stuck%':>7} {'Safe%':>7} {'Steps':>7}")
    print(header)
    print("-" * 90)

    for key in sorted(results.keys()):
        p_name, s_name, sh_name = key
        m = results[key]
        print(f"{p_name:<17} {s_name:<10} {sh_name:<16} "
              f"{m.fail_rate:>6.1%} {m.stuck_rate:>6.1%} {m.safe_rate:>6.1%} "
              f"{m.mean_steps:>7.1f}")

    # Shield comparison for RL
    print("\n" + "=" * 90)
    print("ANALYSIS: SHIELD COMPARISON FOR RL")
    print("=" * 90)
    for p_name in ["uniform", "adversarial_opt"]:
        print(f"\n  Perception: {p_name}")
        for sh_name in ["none", "observation", "single_belief", "envelope"]:
            key = (p_name, "rl", sh_name)
            if key in results:
                m = results[key]
                print(f"    {sh_name:<16} fail={m.fail_rate:.1%}  "
                      f"stuck={m.stuck_rate:.1%}  safe={m.safe_rate:.1%}")

    # Envelope vs single-belief under different perceptions
    print("\n" + "=" * 90)
    print("ANALYSIS: ENVELOPE vs SINGLE-BELIEF (expected: envelope better under adversarial)")
    print("=" * 90)
    for s_name in ["rl", "random", "best"]:
        for p_name in ["uniform", "adversarial_opt"]:
            sb_key = (p_name, s_name, "single_belief")
            env_key = (p_name, s_name, "envelope")
            if sb_key in results and env_key in results:
                sb_fail = results[sb_key].fail_rate
                env_fail = results[env_key].fail_rate
                diff = env_fail - sb_fail
                print(f"  {s_name}/{p_name}: envelope - single_belief = "
                      f"{diff:+.1%} fail rate")


def save_results(results, config):
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

    os.makedirs(os.path.dirname(config.results_path), exist_ok=True)
    with open(config.results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {config.results_path}")


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ipomdp_shielding.experiments.run_rl_shield_experiment <config_module>")
        print("Example: python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim")
        sys.exit(1)

    # Import config
    config_module_name = sys.argv[1]
    try:
        config_module = importlib.import_module(f".{config_module_name}", package="ipomdp_shielding.experiments")
        config = config_module.config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    print("=" * 70)
    print(f"RL SHIELDING EXPERIMENT - {config.case_study_name.upper()}")
    print(f"Trials: {config.num_trials}, Length: {config.trial_length}, Seed: {config.seed}")
    print(f"Shield threshold: {config.shield_threshold}")
    print("=" * 70)

    # Load IPOMDP
    print(f"\nLoading {config.case_study_name.upper()} IPOMDP...")
    ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(**config.ipomdp_kwargs)
    print(f"  States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    # Setup: train/load RL agent and optimized realization
    rl_selector, optimized_perception = setup(ipomdp, pp_shield, config)

    # Build 3-factor grid
    grid = build_grid(ipomdp, pp_shield, rl_selector, optimized_perception, config)
    print(f"\nExperiment grid: {len(grid)} combinations "
          f"(2 perceptions x 3 selectors x 4 shields)")

    # Run all combinations
    t0 = time.time()
    results, trial_data = run_experiment(ipomdp, pp_shield, grid, config)
    total_time = time.time() - t0

    # Results
    print_results_table(results, config)
    save_results(results, config)

    # Plots
    print("\nGenerating figures...")
    plot_results(trial_data, config)

    print(f"\nTotal experiment time: {total_time:.1f}s")
    print(f"Figures saved to {config.figures_dir}")
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
