"""Preliminary coarseness experiment configuration for CartPole."""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.CartPole import build_cartpole_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy

# 5 bins
config = CoarseExperimentConfig(
    case_study_name="cartpole",
    build_ipomdp_fn=build_cartpole_ipomdp,
    seed=42,
    num_trajectories=10,  # Increased from 3 (smaller state space now)
    trajectory_length=10,  # Increased from 5 (smaller state space now)
    initial_state=(2, 2, 2, 2),  # Center state (x, x_dot, theta, theta_dot) for 5 bins
    initial_action=0,
    sampler_budget=50,  # Increased from 20 (now 625 states instead of 2401)
    sampler_k=5,  # Increased from 3
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="results/prelim/coarse_cartpole_results.json",
    # Use coarser discretization: 5 bins per dimension = 625 states (vs 2401)
    ipomdp_kwargs={"num_bins": 5},
)

# # 4 bins
# config = CoarseExperimentConfig(
#     case_study_name="cartpole",
#     build_ipomdp_fn=build_cartpole_ipomdp,
#     seed=42,
#     num_trajectories=10,  # Increased from 3 (smaller state space now)
#     trajectory_length=10,  # Increased from 5 (smaller state space now)
#     initial_state=(2, 2, 2, 2),  # Center state (x, x_dot, theta, theta_dot) for 5 bins
#     initial_action=0,
#     sampler_budget=50,  # Increased from 20 (now 625 states instead of 2401)
#     sampler_k=5,  # Increased from 3
#     sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
#     sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
#     results_path="results/prelim/coarse_cartpole_results.json",
#     # Use coarser discretization: 5 bins per dimension = 625 states (vs 2401)
#     ipomdp_kwargs={"num_bins": 4},
# )
