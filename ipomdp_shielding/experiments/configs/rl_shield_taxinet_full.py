"""Full RL shielding experiment configuration for TaxiNet."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    seed=42,
    num_trials=30,
    trial_length=20,
    rl_episodes=1000,
    rl_episode_length=20,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    rl_cache_path="/tmp/full_rl_shield_taxinet_agent.pt",
    opt_cache_path="/tmp/full_rl_shield_taxinet_opt_realization.json",
    results_path="./data/full/rl_shield_taxinet_results.json",
    figures_dir="./data/full/rl_shield_taxinet_figures",
)
