"""Default alpha/beta sweep config for TaxiNet."""

from .rl_alpha_beta_sweep import AlphaBetaSweepConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp

config = AlphaBetaSweepConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    alphas=[0.01, 0.05, 0.1, 0.2],
    betas=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    seeds=[42, 123, 456, 789, 1024],
    num_trials=30,
    trial_length=20,
    rl_episodes=500,
    rl_episode_length=20,
    opt_candidates=20,
    opt_trials_per_candidate=10,
    opt_iterations=10,
    baseline_beta=0.8,
    shields=["single_belief", "envelope"],
    perceptions=["uniform", "adversarial_opt"],
    results_dir="./data/sweep/rl_alpha_beta_taxinet",
    ipomdp_base_kwargs={
        "confidence_method": "Clopper_Pearson",
        "train_fraction": 0.8,
        "error": 0.1,
        "smoothing": True,
    },
)
