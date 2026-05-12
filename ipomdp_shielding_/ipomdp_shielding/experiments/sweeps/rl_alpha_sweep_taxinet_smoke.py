"""Tiny alpha sweep smoke-test config for TaxiNet (fast pipeline check)."""

from .rl_alpha_sweep import AlphaSweepConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp

config = AlphaSweepConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    alphas=[0.05, 0.2],
    beta=0.8,
    seeds=[42, 123],
    num_trials=5,
    trial_length=10,
    rl_episodes=50,
    rl_episode_length=10,
    opt_candidates=4,
    opt_trials_per_candidate=3,
    opt_iterations=2,
    shields=["single_belief", "envelope", "forward_sampling"],
    perceptions=["uniform", "adversarial_opt"],
    results_dir="./data/sweep/rl_alpha_taxinet_smoke",
    ipomdp_base_kwargs={
        "confidence_method": "Clopper_Pearson",
        "train_fraction": 0.8,
        "error": 0.1,
        "smoothing": True,
    },
)
