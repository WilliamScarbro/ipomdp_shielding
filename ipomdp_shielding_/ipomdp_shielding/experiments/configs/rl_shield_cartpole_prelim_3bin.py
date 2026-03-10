"""Preliminary RL shielding experiment for CartPole — 3-bin discretization.

Uses the same 82-state model as the coarse experiment so results are comparable.
"""
from .base_config import RLShieldExperimentConfig
from ...CaseStudies.CartPole import build_cartpole_ipomdp

config = RLShieldExperimentConfig(
    case_study_name="cartpole",
    build_ipomdp_fn=build_cartpole_ipomdp,
    seed=42,
    num_trials=10,
    trial_length=10,
    rl_episodes=100,
    rl_episode_length=10,
    opt_candidates=5,
    opt_trials_per_candidate=3,
    opt_iterations=5,
    shield_threshold=0.8,
    rl_cache_path="results/cache/prelim_rl_shield_cartpole3_agent.pt",
    opt_cache_path="results/cache/prelim_rl_shield_cartpole3_opt_realization.json",
    results_path="results/prelim/rl_shield_cartpole_results.json",
    figures_dir="results/prelim/rl_shield_cartpole_figures",
    ipomdp_kwargs={"num_bins": 3},
)
