"""Default alpha sweep config for TaxiNet (fixed beta)."""

from .rl_alpha_sweep import AlphaSweepConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp

config = AlphaSweepConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    alphas=[0.01, 0.05, 0.1, 0.2],
    beta=0.8,
    seeds=[42, 123, 456, 789, 1024],
    num_trials=30,
    trial_length=20,
    rl_episodes=500,
    rl_episode_length=20,
    opt_candidates=20,
    opt_trials_per_candidate=10,
    opt_iterations=10,
    shields=["single_belief", "envelope", "forward_sampling"],
    perceptions=["uniform", "adversarial_opt"],
    results_dir="./data/sweep/rl_alpha_taxinet",
    ipomdp_base_kwargs={
        "confidence_method": "Clopper_Pearson",
        "train_fraction": 0.8,
        "error": 0.1,
        "smoothing": True,
    },
)
