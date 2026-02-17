"""Configuration for Carr shield comparison experiments."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class CarrComparisonConfig:
    """
    Configuration for comparing Carr shield vs Lifted shield.

    Attributes:
        case_study: Case study to use ("taxinet" is currently supported)
        realization_strategy: How to convert IPOMDP to POMDP:
            - "midpoint": Average of lower and upper bounds
            - "lower": Pessimistic (lower bound)
            - "upper": Optimistic (upper bound)
            - "random": Random point in interval
        seed: Random seed for reproducibility
        num_trials: Number of Monte Carlo trials per shield
        trial_length: Maximum timesteps per trial
        lifted_shield_threshold: Probability threshold for lifted shield (1 - risk tolerance)
        track_support_size: Whether to track support size over time
        track_belief_mass: Whether to track belief mass on avoid states
        results_path: Path to save results JSON
        figures_dir: Directory to save figures
        template_type: Template type for LFP propagator ("canonical" or "safe_set_indicators")
        ipomdp_kwargs: Additional kwargs for TaxiNet builder
    """
    # Case study
    case_study: str = "taxinet"

    # POMDP realization
    realization_strategy: str = "midpoint"

    # Monte Carlo parameters
    seed: int = 42
    num_trials: int = 100
    trial_length: int = 50

    # Shield parameters
    lifted_shield_threshold: float = 0.8

    # Tracking
    track_support_size: bool = True
    track_belief_mass: bool = True

    # Output paths
    results_path: str = "results/carr_comparison.json"
    figures_dir: str = "figures/carr_comparison"

    # LFP propagator settings
    template_type: str = "safe_set_indicators"

    # TaxiNet-specific settings
    ipomdp_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "confidence_method": "Clopper_Pearson",
        "alpha": 0.05,
        "train_fraction": 0.8,
        "error": 0.1,
        "smoothing": True,
    })

    # Initial state
    initial_state: Optional[Any] = None  # Use (0, 0) for TaxiNet by default

    def __post_init__(self):
        """Validate configuration."""
        if self.case_study != "taxinet":
            raise ValueError(f"Only 'taxinet' case study is currently supported, got: {self.case_study}")

        if self.realization_strategy not in ["midpoint", "lower", "upper", "random"]:
            raise ValueError(f"Invalid realization strategy: {self.realization_strategy}")

        if not 0.0 < self.lifted_shield_threshold <= 1.0:
            raise ValueError(f"Threshold must be in (0, 1], got: {self.lifted_shield_threshold}")

        # Set default initial state for TaxiNet
        if self.initial_state is None and self.case_study == "taxinet":
            self.initial_state = (0, 0)
