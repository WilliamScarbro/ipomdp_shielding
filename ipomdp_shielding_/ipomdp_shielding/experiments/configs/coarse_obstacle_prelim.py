"""Preliminary coarseness experiment configuration for Obstacle."""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_obstacle_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy

# Default N=7: 7x7=49 states + FAIL = 50 states, 4 actions, 3 observations.
# Very small state space — LFP completes in seconds.
config = CoarseExperimentConfig(
    case_study_name="obstacle",
    build_ipomdp_fn=build_obstacle_ipomdp,
    seed=42,
    num_trajectories=10,
    trajectory_length=10,
    initial_state=(1, 1),   # Valid non-obstacle starting position
    initial_action="north",
    sampler_budget=50,
    sampler_k=5,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="results/prelim/coarse_obstacle_results.json",
)
