"""Smaller alpha/beta sweep config for TaxiNet (fast, paper-dev friendly).

This config targets quick iteration while still showing how alpha (interval width)
and beta (shield threshold) influence safety/liveness.

Outputs to the same directory as the default artifact pipeline so
`make_paper_artifacts.py` can pick it up without changes.
"""

from .rl_alpha_beta_sweep import AlphaBetaSweepConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp


config = AlphaBetaSweepConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    # Keep the grid small but informative
    alphas=[0.05, 0.2],
    betas=[0.7, 0.8, 0.9],
    seeds=[42, 123],
    # Moderate eval size
    num_trials=20,
    trial_length=20,
    # Training per alpha (minimal strategy in sweep script)
    rl_episodes=200,
    rl_episode_length=20,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=5,
    baseline_beta=0.8,
    shields=["single_belief", "envelope"],
    perceptions=["uniform", "adversarial_opt"],
    # Use default artifact location
    results_dir="./data/sweep/rl_alpha_beta_taxinet",
    ipomdp_base_kwargs={
        "confidence_method": "Clopper_Pearson",
        "train_fraction": 0.8,
        "error": 0.1,
        "smoothing": True,
    },
)

