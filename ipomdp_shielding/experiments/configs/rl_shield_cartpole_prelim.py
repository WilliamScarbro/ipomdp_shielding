"""Preliminary RL shielding experiment configuration for CartPole."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.CartPole import build_cartpole_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="cartpole",
    build_ipomdp_fn=build_cartpole_ipomdp,
    seed=42,
    num_trials=10,  # Reduced for prelim
    trial_length=10,  # Reduced for prelim
    rl_episodes=100,  # Reduced for prelim
    rl_episode_length=10,  # Reduced for prelim
    opt_candidates=5,  # Reduced for prelim
    opt_trials_per_candidate=3,  # Reduced for prelim
    opt_iterations=5,  # Reduced for prelim
    shield_threshold=0.8,
    rl_cache_path="/tmp/prelim_rl_shield_cartpole_agent.pt",
    opt_cache_path="/tmp/prelim_rl_shield_cartpole_opt_realization.json",
    results_path="./data/prelim/rl_shield_cartpole_results.json",
    figures_dir="./data/prelim/rl_shield_cartpole_figures",
)
