"""Full RL shielding experiment configuration for Refuel."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_refuel_ipomdp

config = RLShieldExperimentConfig(
    case_study_name="refuel",
    build_ipomdp_fn=build_refuel_ipomdp,
    seed=42,
    num_trials=30,
    trial_length=30,
    rl_episodes=1000,
    rl_episode_length=30,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    rl_cache_path="results/cache/full_rl_shield_refuel_agent.pt",
    opt_cache_path="results/cache/full_rl_shield_refuel_opt_realization.json",
    results_path="results/full/rl_shield_refuel_results.json",
    figures_dir="results/full/rl_shield_refuel_figures",
)
