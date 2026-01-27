"""Monte Carlo Safety Evaluation for Shielding Strategies.

Measures safety of shielding through simulation, tracking three outcomes:
- fail: State reached "FAIL"
- stuck: No allowed actions available
- safe: Completed trial without failure

Supports best/worst/average case analysis via modular initial state sampling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import random
import numpy as np
import matplotlib.pyplot as plt

from ..Models.ipomdp import IPOMDP
from .runtime_shield import RuntimeImpShield


# ============================================================
# Action Selector Abstraction (Modular for RL)
# ============================================================

class ActionSelector(ABC):
    """Base class for action selection strategies.

    Type signature: Callable[[List[Tuple[obs, action]], List[action]], action]

    The history parameter provides full observation-action sequence for RL policies.
    The allowed_actions ensure safety (pre-filtered by shield).
    """

    @abstractmethod
    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any]
    ) -> Any:
        """Select an action from allowed actions given history.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Full observation-action sequence up to current step
        allowed_actions : list
            Actions permitted by the shield

        Returns
        -------
        action
            Selected action from allowed_actions
        """
        pass


class RandomActionSelector(ActionSelector):
    """Baseline random selection from allowed actions."""

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any]
    ) -> Any:
        """Randomly select from allowed actions."""
        if not allowed_actions:
            raise ValueError("No allowed actions to select from")
        return random.choice(allowed_actions)


class UniformFallbackSelector(ActionSelector):
    """Random selection with fallback to all actions if stuck.

    If no allowed actions are provided (shield stuck), falls back to
    selecting uniformly from all available actions.
    """

    def __init__(self, all_actions: List[Any]):
        """Initialize with complete action set for fallback.

        Parameters
        ----------
        all_actions : list
            Complete set of actions in the IPOMDP
        """
        self.all_actions = all_actions

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any]
    ) -> Any:
        """Select from allowed actions, or all actions if empty."""
        if allowed_actions:
            return random.choice(allowed_actions)
        else:
            return random.choice(self.all_actions)


# ============================================================
# Initial State Sampling (Best/Worst/Average Cases)
# ============================================================

class InitialStateGenerator(ABC):
    """Base class for initial state sampling strategies."""

    @abstractmethod
    def generate(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]]
    ) -> Tuple[Any, Any]:
        """Generate initial (state, action) pair.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions

        Returns
        -------
        tuple
            (initial_state, initial_action)
        """
        pass


class RandomInitialState(InitialStateGenerator):
    """Average case: uniform random sampling from all states."""

    def generate(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]]
    ) -> Tuple[Any, Any]:
        """Sample uniformly from all states and actions."""
        state = random.choice(ipomdp.states)
        action = random.choice(list(ipomdp.actions))
        return state, action


class SafeInitialState(InitialStateGenerator):
    """Best case: sample from safe interior regions.

    Selects states with maximum number of safe actions (most flexibility).
    """

    def generate(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]]
    ) -> Tuple[Any, Any]:
        """Sample from states with most safe actions."""
        # Count safe actions per state
        state_safety = {}
        for state in ipomdp.states:
            if state == "FAIL":
                continue
            safe_actions = pp_shield.get(state, set())
            state_safety[state] = len(safe_actions)

        if not state_safety:
            # Fallback to random if no safe states
            return RandomInitialState().generate(ipomdp, pp_shield)

        # Find states with maximum safe actions
        max_safety = max(state_safety.values())
        safest_states = [s for s, count in state_safety.items() if count == max_safety]

        state = random.choice(safest_states)
        safe_actions = list(pp_shield.get(state, ipomdp.actions))
        action = random.choice(safe_actions) if safe_actions else random.choice(list(ipomdp.actions))

        return state, action


class BoundaryInitialState(InitialStateGenerator):
    """Worst case: sample near safety boundary.

    Selects states with minimum number of safe actions (most constrained).
    """

    def generate(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]]
    ) -> Tuple[Any, Any]:
        """Sample from states with fewest safe actions."""
        # Count safe actions per state
        state_safety = {}
        for state in ipomdp.states:
            if state == "FAIL":
                continue
            safe_actions = pp_shield.get(state, set())
            state_safety[state] = len(safe_actions)

        if not state_safety:
            # Fallback to random if no safe states
            return RandomInitialState().generate(ipomdp, pp_shield)

        # Find states with minimum safe actions (but > 0)
        min_safety = min(state_safety.values())
        boundary_states = [s for s, count in state_safety.items() if count == min_safety]

        state = random.choice(boundary_states)
        safe_actions = list(pp_shield.get(state, ipomdp.actions))
        action = random.choice(safe_actions) if safe_actions else random.choice(list(ipomdp.actions))

        return state, action


# ============================================================
# Data Structures
# ============================================================

@dataclass
class SafetyTrialResult:
    """Result from a single Monte Carlo trial.

    Attributes
    ----------
    trial_id : int
        Trial identifier
    outcome : str
        One of: "fail", "stuck", or "safe"
    steps_completed : int
        Number of steps executed before termination
    stuck_count : int
        Number of times shield had no allowed actions
    fail_step : int or None
        Step at which failure occurred (None if didn't fail)
    trajectory : list of (state, obs, action) tuples
        Complete trajectory history
    """
    trial_id: int
    outcome: str  # "fail", "stuck", or "safe"
    steps_completed: int
    stuck_count: int
    fail_step: Optional[int]
    trajectory: List[Tuple[Any, Any, Any]] = field(default_factory=list)


@dataclass
class MCSafetyMetrics:
    """Aggregated metrics from Monte Carlo safety evaluation.

    Attributes
    ----------
    num_trials : int
        Total number of trials executed
    fail_rate : float
        Fraction of trials ending in FAIL state
    stuck_rate : float
        Fraction of trials that got stuck (no allowed actions)
    safe_rate : float
        Fraction of trials completing safely
    mean_steps : float
        Average trajectory length
    mean_stuck_count : float
        Average number of stuck events per trial
    fail_step_distribution : list of int
        Steps at which failures occurred (for histogram)
    by_sampling_mode : dict, optional
        Nested results by sampling mode (best/worst/average)
    """
    num_trials: int
    fail_rate: float
    stuck_rate: float
    safe_rate: float
    mean_steps: float
    mean_stuck_count: float
    fail_step_distribution: List[int] = field(default_factory=list)
    by_sampling_mode: Dict[str, "MCSafetyMetrics"] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format metrics for display."""
        lines = [
            "Monte Carlo Safety Metrics",
            "=" * 40,
            f"Trials: {self.num_trials}",
            f"Fail Rate: {self.fail_rate:.2%}",
            f"Stuck Rate: {self.stuck_rate:.2%}",
            f"Safe Rate: {self.safe_rate:.2%}",
            f"Mean Steps: {self.mean_steps:.2f}",
            f"Mean Stuck Count: {self.mean_stuck_count:.2f}",
        ]

        if self.by_sampling_mode:
            lines.append("\nBy Sampling Mode:")
            for mode, metrics in self.by_sampling_mode.items():
                lines.append(f"\n{mode}:")
                lines.append(f"  Fail: {metrics.fail_rate:.2%}")
                lines.append(f"  Stuck: {metrics.stuck_rate:.2%}")
                lines.append(f"  Safe: {metrics.safe_rate:.2%}")

        return "\n".join(lines)


# ============================================================
# Core Evaluation Functions
# ============================================================

def run_single_trial(
    trial_id: int,
    ipomdp: IPOMDP,
    initial_state: Any,
    initial_action: Any,
    rt_shield: RuntimeImpShield,
    perception: Callable[[Any], Any],
    action_selector: ActionSelector,
    trial_length: int,
    store_trajectory: bool = False
) -> SafetyTrialResult:
    """Run a single Monte Carlo trial.

    Parameters
    ----------
    trial_id : int
        Trial identifier
    ipomdp : IPOMDP
        The interval POMDP model
    initial_state : any
        Starting state
    initial_action : any
        Initial action
    rt_shield : RuntimeImpShield
        Runtime shield instance (should be fresh/restarted)
    perception : callable
        Function mapping state to observation
    action_selector : ActionSelector
        Strategy for selecting actions
    trial_length : int
        Maximum number of steps
    store_trajectory : bool
        Whether to store full trajectory (memory intensive)

    Returns
    -------
    SafetyTrialResult
        Complete trial result
    """
    state = initial_state
    action = initial_action
    history = []  # List of (obs, action) pairs
    trajectory = []

    outcome = "safe"  # Default outcome
    fail_step = None
    steps_completed = 0

    for step in range(trial_length):
        # Check for failure
        if state == "FAIL":
            outcome = "fail"
            fail_step = step
            break

        # Get observation
        obs = perception(state)

        # Store in history
        history.append((obs, action))

        if store_trajectory:
            trajectory.append((state, obs, action))

        # Get allowed actions from shield
        allowed_actions = rt_shield.next_actions((obs, action))

        # Check if stuck (no allowed actions)
        if not allowed_actions:
            outcome = "stuck"
            steps_completed = step
            break

        # Select next action
        action = action_selector.select(history, allowed_actions)

        # Evolve state
        state = ipomdp.evolve(state, action)
        steps_completed = step + 1

    # If loop completed without failure or stuck, outcome remains "safe"

    result = SafetyTrialResult(
        trial_id=trial_id,
        outcome=outcome,
        steps_completed=steps_completed,
        stuck_count=rt_shield.stuck_count,
        fail_step=fail_step,
        trajectory=trajectory if store_trajectory else []
    )

    return result


def run_monte_carlo_trials(
    ipomdp: IPOMDP,
    pp_shield: Dict[Any, Set[Any]],
    perception: Callable[[Any], Any],
    rt_shield_factory: Callable[[], RuntimeImpShield],
    action_selector: ActionSelector,
    initial_generator: InitialStateGenerator,
    num_trials: int,
    trial_length: int,
    store_trajectories: bool = False,
    seed: Optional[int] = None
) -> List[SafetyTrialResult]:
    """Run multiple Monte Carlo trials.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions
    perception : callable
        Function mapping state to observation
    rt_shield_factory : callable
        Factory function returning fresh RuntimeImpShield instance
    action_selector : ActionSelector
        Strategy for selecting actions
    initial_generator : InitialStateGenerator
        Strategy for sampling initial states
    num_trials : int
        Number of trials to run
    trial_length : int
        Maximum steps per trial
    store_trajectories : bool
        Whether to store full trajectories (memory intensive)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list of SafetyTrialResult
        Results from all trials
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results = []

    for trial_id in range(num_trials):
        # Generate initial state
        initial_state, initial_action = initial_generator.generate(ipomdp, pp_shield)

        # Create fresh shield instance
        rt_shield = rt_shield_factory()
        rt_shield.restart()

        # Run trial
        result = run_single_trial(
            trial_id=trial_id,
            ipomdp=ipomdp,
            initial_state=initial_state,
            initial_action=initial_action,
            rt_shield=rt_shield,
            perception=perception,
            action_selector=action_selector,
            trial_length=trial_length,
            store_trajectory=store_trajectories
        )

        results.append(result)

    return results


def compute_safety_metrics(
    results: List[SafetyTrialResult]
) -> MCSafetyMetrics:
    """Compute aggregated metrics from trial results.

    Parameters
    ----------
    results : list of SafetyTrialResult
        Results from Monte Carlo trials

    Returns
    -------
    MCSafetyMetrics
        Aggregated safety metrics
    """
    if not results:
        return MCSafetyMetrics(
            num_trials=0,
            fail_rate=0.0,
            stuck_rate=0.0,
            safe_rate=0.0,
            mean_steps=0.0,
            mean_stuck_count=0.0
        )

    num_trials = len(results)

    # Count outcomes
    fail_count = sum(1 for r in results if r.outcome == "fail")
    stuck_count = sum(1 for r in results if r.outcome == "stuck")
    safe_count = sum(1 for r in results if r.outcome == "safe")

    # Compute rates
    fail_rate = fail_count / num_trials
    stuck_rate = stuck_count / num_trials
    safe_rate = safe_count / num_trials

    # Aggregate trajectory metrics
    mean_steps = sum(r.steps_completed for r in results) / num_trials
    mean_stuck_count = sum(r.stuck_count for r in results) / num_trials

    # Build failure step distribution
    fail_step_distribution = [r.fail_step for r in results if r.fail_step is not None]

    return MCSafetyMetrics(
        num_trials=num_trials,
        fail_rate=fail_rate,
        stuck_rate=stuck_rate,
        safe_rate=safe_rate,
        mean_steps=mean_steps,
        mean_stuck_count=mean_stuck_count,
        fail_step_distribution=fail_step_distribution
    )


# ============================================================
# High-Level API
# ============================================================

class MonteCarloSafetyEvaluator:
    """High-level interface for Monte Carlo safety evaluation.

    Evaluates shielding strategies across best/worst/average case scenarios
    by varying initial state sampling strategies.
    """

    def __init__(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]],
        perception: Callable[[Any], Any],
        rt_shield_factory: Callable[[], RuntimeImpShield]
    ):
        """Initialize evaluator.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions
        perception : callable
            Function mapping state to observation
        rt_shield_factory : callable
            Factory function returning fresh RuntimeImpShield instance
        """
        self.ipomdp = ipomdp
        self.pp_shield = pp_shield
        self.perception = perception
        self.rt_shield_factory = rt_shield_factory

    def evaluate(
        self,
        action_selector: ActionSelector,
        num_trials: int = 100,
        trial_length: int = 20,
        sampling_modes: Optional[List[str]] = None,
        store_trajectories: bool = False,
        seed: Optional[int] = None
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

        Returns
        -------
        dict
            Mapping from sampling mode to MCSafetyMetrics
        """
        if sampling_modes is None:
            sampling_modes = ["random", "best_case", "worst_case"]

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
                perception=self.perception,
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


# ============================================================
# Visualization
# ============================================================

def plot_safety_metrics(
    metrics_by_mode: Dict[str, MCSafetyMetrics],
    save_path: Optional[str] = None,
    show: bool = True
):
    """Visualize safety metrics across sampling modes.

    Creates a 3-panel figure:
    1. Bar chart: fail/stuck/safe rates per sampling mode
    2. Histogram: failure step distribution across modes
    3. Bar chart: mean trajectory length per mode

    Parameters
    ----------
    metrics_by_mode : dict
        Mapping from sampling mode to MCSafetyMetrics
    save_path : str, optional
        Path to save figure (e.g., "images/safety_comparison.png")
    show : bool
        Whether to display the figure
    """
    if not metrics_by_mode:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    modes = list(metrics_by_mode.keys())

    # Panel 1: Outcome rates
    ax = axes[0]
    x = np.arange(len(modes))
    width = 0.25

    fail_rates = [metrics_by_mode[m].fail_rate for m in modes]
    stuck_rates = [metrics_by_mode[m].stuck_rate for m in modes]
    safe_rates = [metrics_by_mode[m].safe_rate for m in modes]

    ax.bar(x - width, fail_rates, width, label='Fail', color='red', alpha=0.7)
    ax.bar(x, stuck_rates, width, label='Stuck', color='orange', alpha=0.7)
    ax.bar(x + width, safe_rates, width, label='Safe', color='green', alpha=0.7)

    ax.set_xlabel('Sampling Mode')
    ax.set_ylabel('Rate')
    ax.set_title('Safety Outcome Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Failure step distribution
    ax = axes[1]
    for mode in modes:
        fail_steps = metrics_by_mode[mode].fail_step_distribution
        if fail_steps:
            ax.hist(fail_steps, bins=20, alpha=0.5, label=mode)

    ax.set_xlabel('Step')
    ax.set_ylabel('Failure Count')
    ax.set_title('Failure Step Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Mean trajectory length
    ax = axes[2]
    mean_steps = [metrics_by_mode[m].mean_steps for m in modes]

    ax.bar(modes, mean_steps, color='steelblue', alpha=0.7)
    ax.set_xlabel('Sampling Mode')
    ax.set_ylabel('Mean Steps')
    ax.set_title('Mean Trajectory Length')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# Test Functions
# ============================================================

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

    print("=" * 60)
    print(f"TAXINET MONTE CARLO SAFETY TEST")
    print(f"Trials: {num_trials}, Length: {trial_length}")
    print("=" * 60)

    # Setup Taxinet model
    initial = ((0, 0), 0)
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


if __name__ == "__main__":
    # Run simple test
    test_taxinet_monte_carlo_safety(
        num_trials=100,
        trial_length=20,
        seed=42,
        save_path="images/mc_safety_test.png"
    )
