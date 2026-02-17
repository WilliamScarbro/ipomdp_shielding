"""Full RL shielding experiment configuration for CartPole."""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.CartPole import build_cartpole_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="cartpole",
    build_ipomdp_fn=build_cartpole_ipomdp,
    seed=42,
    num_trials=30,
    trial_length=20,
    rl_episodes=1000,
    rl_episode_length=20,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    rl_cache_path="results/cache/full_rl_shield_cartpole_agent.pt",
    opt_cache_path="results/cache/full_rl_shield_cartpole_opt_realization.json",
    results_path="results/full/rl_shield_cartpole_results.json",
    figures_dir="results/full/rl_shield_cartpole_figures",
    # Use coarser discretization: 5 bins per dimension = 625 states (vs 2401)
    ipomdp_kwargs={"num_bins": 5},
)
