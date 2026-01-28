"""Data structures for Monte Carlo safety evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
