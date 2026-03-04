"""Preliminary RL shielding experiment configuration for Refuel."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_refuel_ipomdp

# N=7, ENERGY=6: 344 states, 5 actions — tractable training and evaluation.
config = RLShieldExperimentConfig(
    case_study_name="refuel",
    build_ipomdp_fn=build_refuel_ipomdp,
    seed=42,
    num_trials=10,
    trial_length=20,
    rl_episodes=100,
    rl_episode_length=20,
    opt_candidates=5,
    opt_trials_per_candidate=3,
    opt_iterations=5,
    shield_threshold=0.8,
    rl_cache_path="results/cache/prelim_rl_shield_refuel_agent.pt",
    opt_cache_path="results/cache/prelim_rl_shield_refuel_opt_realization.json",
    results_path="results/prelim/rl_shield_refuel_results.json",
    figures_dir="results/prelim/rl_shield_refuel_figures",
)
