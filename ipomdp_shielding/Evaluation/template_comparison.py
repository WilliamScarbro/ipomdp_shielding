"""
Comparative analysis of template functions for LFP belief propagation.

Compares different template strategies by measuring:
1. Approximation decay: growth of BeliefPolytope bounds over time
2. Shield permissivity: how many actions remain allowed under the approximation

Templates compared:
- Canonical: standard basis vectors (bound each state probability)
- Safe-set indicators: bound probability mass in action-safe sets
- Hybrid: combination of canonical and safe-set indicators
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
import random

from ..CaseStudies.Taxinet import build_taxinet_ipomdp
from ..Models import IPOMDP
from ..Propagators import (
    LFPPropagator,
    BeliefPolytope,
    Template,
    TemplateFactory,
)
from ..Propagators.lfp_propagator import default_solver

from .report_runner import ReportRunner
from .runtime_shield import RuntimeImpShield

@dataclass
class ApproximationMetrics:
    """Metrics for measuring approximation quality at a single step."""
    step: int
    template_spread: float  # Sum of (upper - lower) for each template direction
    volume_proxy: float     # Product of spreads (proxy for volume)
    min_state_bounds: Dict  # Lower bounds on each state probability
    max_state_bounds: Dict  # Upper bounds on each state probability
    safest_action_prob: float  # Max over actions of min safe probability
    action_safe_probs: Dict  # Min safe probability for each action


@dataclass
class ComparisonResult:
    """Results from comparing templates on a single run."""
    template_name: str
    template: Template
    history: List[Tuple]
    metrics: List[ApproximationMetrics]
    final_polytope: BeliefPolytope


def compute_template_spread(polytope: BeliefPolytope, template: Template) -> Tuple[float, Dict, Dict]:
    """
    Compute the spread (upper - lower) for each template direction.

    Returns:
        (total_spread, min_bounds, max_bounds)
    """
    n = polytope.n
    min_bounds = {}
    max_bounds = {}
    total_spread = 0.0

    for k in range(template.K):
        v = template.V[k]

        # Find min and max of v^T b over the polytope
        try:
            max_val = polytope.maximize_linear(v)
            min_val = -polytope.maximize_linear(-v)
        except RuntimeError:
            # Polytope might be infeasible
            max_val = 1.0
            min_val = 0.0

        spread = max_val - min_val
        total_spread += spread

        name = template.names[k] if template.names else f"template_{k}"
        min_bounds[name] = min_val
        max_bounds[name] = max_val

    return total_spread, min_bounds, max_bounds


def compute_volume_proxy(polytope: BeliefPolytope, template: Template) -> float:
    """
    Compute a proxy for polytope volume using template bounds.

    Uses the product of spreads in each template direction.
    This is exact for axis-aligned boxes and provides a rough
    estimate for general polytopes.
    """
    volume = 1.0
    for k in range(template.K):
        v = template.V[k]
        try:
            max_val = polytope.maximize_linear(v)
            min_val = -polytope.maximize_linear(-v)
            spread = max(max_val - min_val, 1e-10)
        except RuntimeError:
            spread = 1.0
        volume *= spread
    return volume


def compute_shield_permissivity(
    polytope: BeliefPolytope,
    ipomdp: IPOMDP,
) -> Tuple[float, Dict]:
    """
    Compute the safest action probability.

    For each action, compute the minimum probability of being in a safe state.
    Returns the maximum of these (the probability threshold needed for at least
    one action to be allowed).

    Returns:
        (safest_action_prob, action_safe_probs)
        - safest_action_prob: max over actions of min safe probability
        - action_safe_probs: dict mapping action -> min safe probability
    """
    states = list(ipomdp.states)
    actions = list(ipomdp.actions) if isinstance(ipomdp.actions, (list, set)) else list(ipomdp.actions)
    n = len(states)

    action_safe_probs = {}

    for action in actions:
        # Find states where this action is safe
        safe_states = []
        for i, s in enumerate(states):
            # Check if action leads to a safe state
            next_states = ipomdp.T.get((s, action), {})
            # Action is safe from state s if it doesn't lead to FAIL
            if "FAIL" not in next_states or next_states.get("FAIL", 0) < 0.5:
                safe_states.append(i)

        if not safe_states:
            action_safe_probs[action] = 0.0
            continue

        # Compute minimum probability of being in safe states
        indicator = np.zeros(n)
        for i in safe_states:
            indicator[i] = 1.0

        try:
            min_safe_prob = -polytope.maximize_linear(-indicator)
        except RuntimeError:
            min_safe_prob = 0.0

        action_safe_probs[action] = max(0.0, min(1.0, min_safe_prob))

    # Safest action probability is the max of all action safe probs
    safest_action_prob = max(action_safe_probs.values()) if action_safe_probs else 0.0

    return safest_action_prob, action_safe_probs


class LFPReporter(ReportRunner):
    
    # requires rt_shield.ipomdp_belief : LFPPropagator
    def collect_metrics(self, rt_shield, step):
        assert isinstance(rt_shield.ipomdp_belief, LFPPropagator)

        polytope = rt_shield.ipomdp_belief.belief
        template = rt_shield.ipomdp_belief.template
        ipomdp = rt_shield.ipomdp_belief.ipomdp

        spread, min_b, max_b = compute_template_spread(polytope, template)
        vol = compute_volume_proxy(polytope, template)
        safest_prob, action_probs = compute_shield_permissivity(polytope, ipomdp)

        metric = ApproximationMetrics(
            step=step + 1,
            template_spread=spread,
            volume_proxy=vol,
            min_state_bounds=min_b,
            max_state_bounds=max_b,
            safest_action_prob=safest_prob,
            action_safe_probs=action_probs
        )

        return metric

    def initialize(self):
        self.metrics = []
        self.history = []
        
    def report(self, step, state, obs, action, rt_shield):
        self.metrics.append(self.collect_metrics(rt_shield, step))
        self.history.append((step, obs, action))

    def final(self, rt_shield):
        assert isinstance(rt_shield.ipomdp_belief, LFPPropagator)

        template = rt_shield.ipomdp_belief.template
        polytope = rt_shield.ipomdp_belief.belief
        
        return  ComparisonResult(
            template_name=getattr(template, 'name', 'unnamed'),
            template=template,
            history=self.history,
            metrics=self.metrics,
            final_polytope=polytope
        )

def run_lfp_propagation(
        ipomdp: IPOMDP,
        pp_shield : "Dict State -> {Action}",
        template : Template,
        perception : "State -> Obs",
        initial : "State x Action",
        verbose : bool = False,
        action_shield : float = 0.8,
        length : int = 20
):

    n = len(ipomdp.states)
    polytope = BeliefPolytope.uniform_prior(n)
    propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)
    rt_shield = RuntimeImpShield(pp_shield, propagator, action_shield)

    all_actions = list(ipomdp.actions)
    action_selector = lambda actions: random.choice(actions) if actions else random.choice(all_actions)
    lfp_reporter = LFPReporter(ipomdp, perception, rt_shield, action_selector, length, initial)

    return lfp_reporter.run()

    
                
# def run_lfp_propagation(
#     ipomdp: IPOMDP,
#     template: Template,
#     initial: State,
#     # history: List[Tuple],
#     verbose: bool = False
# ) -> ComparisonResult:
#     """
#     Run LFP propagation with a specific template and collect metrics.
#     """
#     n = len(ipomdp.states)
#     solver = default_solver()
#     propagator = LFPPropagator(model=ipomdp, template=template, solver=solver)

#     polytope = BeliefPolytope.uniform_prior(n)
#     metrics = []

#     # Initial metrics
#     spread, min_b, max_b = compute_template_spread(polytope, template)
#     vol = compute_volume_proxy(polytope, template)
#     safest_prob, action_probs = compute_shield_permissivity(polytope, ipomdp)

#     metrics.append(ApproximationMetrics(
#         step=0,
#         template_spread=spread,
#         volume_proxy=vol,
#         min_state_bounds=min_b,
#         max_state_bounds=max_b,
#         safest_action_prob=safest_prob,
#         action_safe_probs=action_probs
#     ))

#     if verbose:
#         print(f"Step 0: spread={spread:.4f}, volume={vol:.6f}, safest_prob={safest_prob:.4f}")

#     current_state = initial
#     for step in range(length):

        
#         polytope = propagator.propagate(polytope, action, obs)

#         spread, min_b, max_b = compute_template_spread(polytope, template)
#         vol = compute_volume_proxy(polytope, template)
#         safest_prob, action_probs = compute_shield_permissivity(polytope, ipomdp)

#         metrics.append(ApproximationMetrics(
#             step=step + 1,
#             template_spread=spread,
#             volume_proxy=vol,
#             min_state_bounds=min_b,
#             max_state_bounds=max_b,
#             safest_action_prob=safest_prob,
#             action_safe_probs=action_probs
#         ))

#         if verbose:
#             print(f"Step {step+1}: spread={spread:.4f}, volume={vol:.6f}, safest_prob={safest_prob:.4f}")

#     return ComparisonResult(
#         template_name=getattr(template, 'name', 'unnamed'),
#         template=template,
#         history=history,
#         metrics=metrics,
#         final_polytope=polytope
#     )


def create_templates_for_ipomdp(
    ipomdp: IPOMDP,
    include_canonical: bool = True,
    include_safe_sets: bool = True,
    include_hybrid: bool = True
) -> Dict[str, Template]:
    """
    Create different template types for comparison.
    """
    n = len(ipomdp.states)
    states = list(ipomdp.states)
    actions = list(ipomdp.actions) if isinstance(ipomdp.actions, (list, set)) else list(ipomdp.actions)

    templates = {}

    if include_canonical:
        canonical = TemplateFactory.canonical(n)
        canonical.name = "canonical"
        templates["canonical"] = canonical

    if include_safe_sets:
        # Build safe sets for each action
        safe_sets = {}
        for action in actions:
            safe_indices = []
            for i, s in enumerate(states):
                next_states = ipomdp.T.get((s, action), {})
                if "FAIL" not in next_states or next_states.get("FAIL", 0) < 0.5:
                    safe_indices.append(i)
            if safe_indices:
                safe_sets[f"safe_{action}"] = safe_indices

        if safe_sets:
            safe_template = TemplateFactory.safe_set_indicators(n, safe_sets)
            safe_template.name = "safe_sets"
            templates["safe_sets"] = safe_template

    if include_hybrid and "canonical" in templates and "safe_sets" in templates:
        hybrid = TemplateFactory.hybrid([templates["canonical"], templates["safe_sets"]])
        hybrid.name = "hybrid"
        templates["hybrid"] = hybrid

    return templates


def compare_templates(
        ipomdp: IPOMDP,
        pp_shield : "Dict (State -> Col Action)",
        perception : "State -> Obs",
        initial : "State x Action",
        templates: Optional[Dict[str, Template]] = None,
        verbose: bool = False
) -> Dict[str, ComparisonResult]:
    """
    Compare multiple template strategies on the same IPOMDP and history.
    """
    if templates is None:
        templates = create_templates_for_ipomdp(ipomdp)

    results = {}
    for name, template in templates.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running template: {name}")
            print(f"{'='*60}")

        result = run_lfp_propagation(
            ipomdp,
            pp_shield,
            template,
            perception,
            initial,
        )
        result.template_name = name
        results[name] = result

    return results


def plot_comparison(
    results: Dict[str, ComparisonResult],
    title: str = "Template Comparison",
    save_path: Optional[str] = None
):
    """
    Plot comparison of template strategies.

    Creates a figure with:
    - Template spread over time
    - Volume proxy over time (log scale)
    - Safest action probability over time
    - Spread normalized by initial
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Plot 1: Template spread
    ax1 = axes[0, 0]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        spreads = [m.template_spread for m in result.metrics]
        ax1.plot(steps, spreads, 'o-', label=name, color=color)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Template Spread')
    ax1.set_title('Approximation Growth (Template Spread)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Volume proxy (log scale)
    ax2 = axes[0, 1]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        volumes = [max(m.volume_proxy, 1e-20) for m in result.metrics]  # Avoid log(0)
        ax2.semilogy(steps, volumes, 's-', label=name, color=color)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Volume Proxy (log scale)')
    ax2.set_title('Approximation Growth (Volume Proxy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Safest action probability
    ax3 = axes[1, 0]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        probs = [m.safest_action_prob for m in result.metrics]
        ax3.plot(steps, probs, '^-', label=name, color=color, markersize=8)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Safest Action Probability')
    ax3.set_title('Shield Permissivity (higher = more permissive)')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Spread normalized by step 1 (skip step 0 which is often 0)
    ax4 = axes[1, 1]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        spreads = [m.template_spread for m in result.metrics]
        # Normalize by first non-zero spread
        initial = next((s for s in spreads if s > 0), 1.0)
        normalized = [s / initial for s in spreads]
        ax4.plot(steps, normalized, 'd-', label=name, color=color)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Normalized Spread')
    ax4.set_title('Relative Approximation Decay')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


@dataclass
class AveragedMetrics:
    """Averaged metrics across multiple runs."""
    step: int
    template_spread_mean: float
    template_spread_std: float
    volume_proxy_mean: float
    volume_proxy_std: float
    safest_action_prob_mean: float
    safest_action_prob_std: float


@dataclass
class AveragedComparisonResult:
    """Averaged results from multiple comparison runs."""
    template_name: str
    num_runs: int
    num_steps: int
    metrics: List[AveragedMetrics]


def run_multiple_comparisons(
        ipomdp: IPOMDP,
        pp_shield,
        perceptor,
        initial_state,
        templates: Optional[Dict[str, Template]] = None,
        num_runs: int = 10,
        verbose: bool = False
) -> Dict[str, AveragedComparisonResult]:
    """
    Run template comparison multiple times and average results.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    history_generator : callable
        Function that takes ipomdp and returns a history (list of (obs, action) tuples)
    templates : dict, optional
        Templates to compare. If None, creates default templates.
    num_runs : int
        Number of runs to average over
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Mapping from template name to AveragedComparisonResult
    """
    if templates is None:
        templates = create_templates_for_ipomdp(ipomdp)

    # Collect results from all runs
    all_results = {name: [] for name in templates}

    for run in range(num_runs):
        if verbose:
            print(f"Run {run + 1}/{num_runs}")

        results = compare_templates(
            ipomdp,
            pp_shield,
            perceptor,
            initial_state,
            templates,
            verbose=False)

        for name, result in results.items():
            all_results[name].append(result)

    # Average the results
    averaged_results = {}
    for name, runs in all_results.items():
        if not runs:
            continue

        # Use minimum length across all runs to handle early terminations
        num_steps = min(len(r.metrics) for r in runs)
        if num_steps == 0:
            continue

        averaged_metrics = []

        for step in range(num_steps):
            spreads = [r.metrics[step].template_spread for r in runs]
            volumes = [r.metrics[step].volume_proxy for r in runs]
            probs = [r.metrics[step].safest_action_prob for r in runs]

            averaged_metrics.append(AveragedMetrics(
                step=step,
                template_spread_mean=np.mean(spreads),
                template_spread_std=np.std(spreads),
                volume_proxy_mean=np.mean(volumes),
                volume_proxy_std=np.std(volumes),
                safest_action_prob_mean=np.mean(probs),
                safest_action_prob_std=np.std(probs)
            ))

        averaged_results[name] = AveragedComparisonResult(
            template_name=name,
            num_runs=num_runs,
            num_steps=num_steps,
            metrics=averaged_metrics
        )

    return averaged_results


def plot_averaged_comparison(
    results: Dict[str, AveragedComparisonResult],
    title: str = "Template Comparison (Averaged)",
    save_path: Optional[str] = None
):
    """
    Plot averaged comparison with error bands.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Plot 1: Template spread with std bands
    ax1 = axes[0, 0]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        means = [m.template_spread_mean for m in result.metrics]
        stds = [m.template_spread_std for m in result.metrics]
        ax1.plot(steps, means, 'o-', label=name, color=color)
        ax1.fill_between(steps,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.2, color=color)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Template Spread')
    ax1.set_title('Approximation Growth (Template Spread)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Volume proxy with std bands (log scale)
    ax2 = axes[0, 1]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        means = [max(m.volume_proxy_mean, 1e-20) for m in result.metrics]
        ax2.semilogy(steps, means, 's-', label=name, color=color)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Volume Proxy (log scale)')
    ax2.set_title('Approximation Growth (Volume Proxy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Safest action probability with std bands
    ax3 = axes[1, 0]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        means = [m.safest_action_prob_mean for m in result.metrics]
        stds = [m.safest_action_prob_std for m in result.metrics]
        ax3.plot(steps, means, '^-', label=name, color=color, markersize=8)
        ax3.fill_between(steps,
                         [max(0, m - s) for m, s in zip(means, stds)],
                         [min(1, m + s) for m, s in zip(means, stds)],
                         alpha=0.2, color=color)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Safest Action Probability')
    ax3.set_title('Shield Permissivity (higher = more permissive)')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Normalized spread with std bands
    ax4 = axes[1, 1]
    for (name, result), color in zip(results.items(), colors):
        steps = [m.step for m in result.metrics]
        means = [m.template_spread_mean for m in result.metrics]
        stds = [m.template_spread_std for m in result.metrics]
        # Normalize by first non-zero mean
        initial = next((m for m in means if m > 0), 1.0)
        norm_means = [m / initial for m in means]
        norm_stds = [s / initial for s in stds]
        ax4.plot(steps, norm_means, 'd-', label=name, color=color)
        ax4.fill_between(steps,
                         [m - s for m, s in zip(norm_means, norm_stds)],
                         [m + s for m, s in zip(norm_means, norm_stds)],
                         alpha=0.2, color=color)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Normalized Spread')
    ax4.set_title('Relative Approximation Decay')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


# ============================================================
# Test Models
# ============================================================

def create_toy_model() -> Tuple[IPOMDP, List[Tuple]]:
    """
    Create a simple 3-state toy IPOMDP for testing.
    """
    states = [0, 1, 2]
    observations = [0, 1, 2]
    actions = [0, 1]

    # Dynamics: action 0 stays, action 1 moves right
    dynamics = {}
    for s in states:
        dynamics[(s, 0)] = {s: 1.0}
        next_s = min(s + 1, 2)
        if next_s == s:
            dynamics[(s, 1)] = {s: 1.0}
        else:
            dynamics[(s, 1)] = {s: 0.2, next_s: 0.8}

    # Interval observation model
    P_low = {
        0: {0: 0.6, 1: 0.1, 2: 0.0},
        1: {0: 0.1, 1: 0.5, 2: 0.1},
        2: {0: 0.0, 1: 0.1, 2: 0.6},
    }
    P_high = {
        0: {0: 0.9, 1: 0.3, 2: 0.1},
        1: {0: 0.3, 1: 0.8, 2: 0.3},
        2: {0: 0.1, 1: 0.3, 2: 0.9},
    }

    ipomdp = IPOMDP(states, observations, actions, dynamics, P_low, P_high)

    # Sample history
    history = [(0, 0), (1, 1), (1, 0), (2, 1), (1, 1), (2, 0)]

    return ipomdp, history



# ============================================================
# Main test runners
# ============================================================

def run_toy_comparison(verbose: bool = True, save_path: Optional[str] = None):
    """Run template comparison on the toy model."""
    print("=" * 60)
    print("TOY MODEL TEMPLATE COMPARISON")
    print("=" * 60)

    ipomdp, history = create_toy_model()
    templates = create_templates_for_ipomdp(ipomdp)

    print(f"\nStates: {ipomdp.states}")
    print(f"Actions: {ipomdp.actions}")
    print(f"History length: {len(history)}")
    print(f"Templates: {list(templates.keys())}")

    results = compare_templates(ipomdp, history, templates, verbose=verbose)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        final = result.metrics[-1]
        print(f"\n{name}:")
        print(f"  Final spread: {final.template_spread:.4f}")
        print(f"  Final volume proxy: {final.volume_proxy:.6f}")
        print(f"  Safest action prob: {final.safest_action_prob:.4f}")

    plot_comparison(results, "Toy Model: Template Comparison", save_path)
    return results


def run_toy_comparison_averaged(
    num_runs: int = 10,
    verbose: bool = True,
    save_path: Optional[str] = None
):
    """Run averaged template comparison on the toy model."""
    print("=" * 60)
    print(f"TOY MODEL TEMPLATE COMPARISON (AVERAGED, {num_runs} runs)")
    print("=" * 60)

    ipomdp, _ = create_toy_model()
    templates = create_templates_for_ipomdp(ipomdp)

    def history_generator(ipomdp):
        import random
        # Generate random history
        actions = list(ipomdp.actions)
        obs_list = list(ipomdp.observations)
        return [(random.choice(obs_list), random.choice(actions)) for _ in range(6)]

    print(f"\nStates: {ipomdp.states}")
    print(f"Actions: {ipomdp.actions}")
    print(f"Templates: {list(templates.keys())}")

    results = run_multiple_comparisons(
        ipomdp, templates, num_runs=num_runs, verbose=verbose
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (AVERAGED)")
    print("=" * 60)
    for name, result in results.items():
        final = result.metrics[-1]
        print(f"\n{name}:")
        print(f"  Final spread: {final.template_spread_mean:.4f} +/- {final.template_spread_std:.4f}")
        print(f"  Final volume: {final.volume_proxy_mean:.6f} +/- {final.volume_proxy_std:.6f}")
        print(f"  Safest prob:  {final.safest_action_prob_mean:.4f} +/- {final.safest_action_prob_std:.4f}")

    plot_averaged_comparison(results, "Toy Model: Template Comparison (Averaged)", save_path)
    return results


def run_taxinet_comparison(verbose: bool = True, save_path: Optional[str] = None):
    """Run template comparison on the Taxinet model."""
    print("=" * 60)
    print("TAXINET MODEL TEMPLATE COMPARISON")
    print("=" * 60)

    initial = (0,0), 0
    ipomdp, dyn_shield, test_cte_model, test_he_model = build_taxinet_ipomdp()

    def perception(state):
        if state == "FAIL":
            return "FAIL"
        return (random.choice(test_cte_model[state[0]]), random.choice(test_he_model[state[1]]))

    templates = create_templates_for_ipomdp(ipomdp)

    print(f"\nStates: {len(ipomdp.states)} states")
    print(f"Actions: {ipomdp.actions}")
    print(f"Templates: {list(templates.keys())}")
    
    results = compare_templates(
        ipomdp,
        dyn_shield,
        perception,
        initial,
        templates,
        verbose=verbose)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        final = result.metrics[-1]
        print(f"\n{name}:")
        print(f"  Final spread: {final.template_spread:.4f}")
        print(f"  Final volume proxy: {final.volume_proxy:.6e}")
        print(f"  Safest action prob: {final.safest_action_prob:.4f}")

    plot_comparison(results, "Taxinet: Template Comparison", save_path)
    return results


def run_taxinet_comparison_averaged(
    num_runs: int = 10,
    verbose: bool = True,
    save_path: Optional[str] = None
):
    """Run averaged template comparison on the Taxinet model."""
    print("=" * 60)
    print(f"TAXINET MODEL TEMPLATE COMPARISON (AVERAGED, {num_runs} runs)")
    print("=" * 60)

    initial_state = ((0,0), 0)
    ipomdp, dyn_shield, test_cte_model, test_he_model = build_taxinet_ipomdp()

    def perception(state):
        if state == "FAIL":
            return "FAIL"
        return (random.choice(test_cte_model[state[0]]), random.choice(test_he_model[state[1]]))

    templates = create_templates_for_ipomdp(ipomdp)

    # ipomdp, _ = create_taxinet_model()
    # templates = create_templates_for_ipomdp(ipomdp)

    # # Get non-FAIL states and observations for random history generation
    # states_no_fail = [s for s in ipomdp.states if s != "FAIL"]
    # obs_no_fail = [o for o in ipomdp.observations if o != "FAIL"]

    # def history_generator(ipomdp):
    #     import random
    #     actions = list(ipomdp.actions) if isinstance(ipomdp.actions, (list, set)) else list(ipomdp.actions)
    #     return [(random.choice(obs_no_fail), random.choice(actions)) for _ in range(6)]

    print(f"\nStates: {len(ipomdp.states)} states")
    print(f"Actions: {ipomdp.actions}")
    print(f"Templates: {list(templates.keys())}")

    results = run_multiple_comparisons(
        ipomdp,
        dyn_shield,
        perception,
        initial_state,
        templates,
        verbose=verbose,
        num_runs=num_runs
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (AVERAGED)")
    print("=" * 60)
    for name, result in results.items():
        final = result.metrics[-1]
        print(f"\n{name}:")
        print(f"  Final spread: {final.template_spread_mean:.4f} +/- {final.template_spread_std:.4f}")
        print(f"  Final volume: {final.volume_proxy_mean:.6e} +/- {final.volume_proxy_std:.6e}")
        print(f"  Safest prob:  {final.safest_action_prob_mean:.4f} +/- {final.safest_action_prob_std:.4f}")

    plot_averaged_comparison(results, "Taxinet: Template Comparison (Averaged)", save_path)
    return results


def run_all_comparisons(save_dir: Optional[str] = None, num_runs: int = 10):
    """Run all template comparisons (single and averaged)."""
    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        toy_path = os.path.join(save_dir, "toy_comparison.png")
        toy_avg_path = os.path.join(save_dir, "toy_comparison_averaged.png")
        taxinet_path = os.path.join(save_dir, "taxinet_comparison.png")
        taxinet_avg_path = os.path.join(save_dir, "taxinet_comparison_averaged.png")
    else:
        toy_path = toy_avg_path = taxinet_path = taxinet_avg_path = None

    all_results = {}

    # print("\n" + "#" * 60)
    # print("# RUNNING TOY MODEL COMPARISON (SINGLE RUN)")
    # print("#" * 60 + "\n")
    # all_results["toy_single"] = run_toy_comparison(verbose=True, save_path=toy_path)

    # print("\n" + "#" * 60)
    # print(f"# RUNNING TOY MODEL COMPARISON (AVERAGED, {num_runs} runs)")
    # print("#" * 60 + "\n")
    # all_results["toy_averaged"] = run_toy_comparison_averaged(
    #     num_runs=num_runs, verbose=True, save_path=toy_avg_path
    # )

    print("\n" + "#" * 60)
    print("# RUNNING TAXINET MODEL COMPARISON (SINGLE RUN)")
    print("#" * 60 + "\n")
    all_results["taxinet_single"] = run_taxinet_comparison(verbose=True, save_path=taxinet_path)

    print("\n" + "#" * 60)
    print(f"# RUNNING TAXINET MODEL COMPARISON (AVERAGED, {num_runs} runs)")
    print("#" * 60 + "\n")
    all_results["taxinet_averaged"] = run_taxinet_comparison_averaged(
        num_runs=num_runs, verbose=True, save_path=taxinet_avg_path
    )

    return all_results


if __name__ == "__main__":
    run_all_comparisons(num_runs=5)
