"""Preliminary RL shielding experiment configuration for TaxiNet."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    seed=42,
    num_trials=10,  # Reduced for prelim
    trial_length=10,  # Reduced for prelim
    rl_episodes=100,  # Reduced for prelim
    rl_episode_length=10,  # Reduced for prelim
    opt_candidates=5,  # Reduced for prelim
    opt_trials_per_candidate=3,  # Reduced for prelim
    opt_iterations=5,  # Reduced for prelim
    shield_threshold=0.8,
    rl_cache_path="results/cache/prelim_rl_shield_taxinet_agent.pt",
    opt_cache_path="results/cache/prelim_rl_shield_taxinet_opt_realization.json",
    results_path="results/prelim/rl_shield_taxinet_results.json",
    figures_dir="results/prelim/rl_shield_taxinet_figures",
)
