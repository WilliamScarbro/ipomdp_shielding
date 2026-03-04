"""Full RL shielding experiment configuration for Obstacle."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_obstacle_ipomdp

config = RLShieldExperimentConfig(
    case_study_name="obstacle",
    build_ipomdp_fn=build_obstacle_ipomdp,
    seed=42,
    num_trials=30,
    trial_length=30,
    rl_episodes=1000,
    rl_episode_length=30,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    rl_cache_path="results/cache/full_rl_shield_obstacle_agent.pt",
    opt_cache_path="results/cache/full_rl_shield_obstacle_opt_realization.json",
    results_path="results/full/rl_shield_obstacle_results.json",
    figures_dir="results/full/rl_shield_obstacle_figures",
)
