"""Preliminary coarseness experiment configuration for CartPole."""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.CartPole import build_cartpole_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy


config = CoarseExperimentConfig(
    case_study_name="cartpole",
    build_ipomdp_fn=build_cartpole_ipomdp,
    seed=42,
    num_trajectories=10,  # Reduced for prelim
    trajectory_length=10,  # Reduced for prelim
    initial_state=0,  # CartPole uses integer state indices
    initial_action=0,
    sampler_budget=100,  # Reduced for prelim
    sampler_k=10,  # Reduced for prelim
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="./data/prelim/coarse_cartpole_results.json",
)
