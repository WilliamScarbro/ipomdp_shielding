"""Preliminary RL shielding experiment configuration for Obstacle."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_obstacle_ipomdp

# N=7: 50 states, 4 actions, 3 observations — very fast to train and evaluate.
config = RLShieldExperimentConfig(
    case_study_name="obstacle",
    build_ipomdp_fn=build_obstacle_ipomdp,
    seed=42,
    num_trials=10,
    trial_length=20,
    rl_episodes=100,
    rl_episode_length=20,
    opt_candidates=5,
    opt_trials_per_candidate=3,
    opt_iterations=5,
    shield_threshold=0.8,
    rl_cache_path="results/cache/prelim_rl_shield_obstacle_agent.pt",
    opt_cache_path="results/cache/prelim_rl_shield_obstacle_opt_realization.json",
    results_path="results/prelim/rl_shield_obstacle_results.json",
    figures_dir="results/prelim/rl_shield_obstacle_figures",
)
