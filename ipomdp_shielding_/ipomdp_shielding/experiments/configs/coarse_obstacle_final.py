"""Final coarseness experiment configuration for Obstacle.

5x more trajectories, 2x longer, 2x sampler budget vs prelim.
Estimated runtime: ~17 min.
"""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_obstacle_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy


config = CoarseExperimentConfig(
    case_study_name="obstacle",
    build_ipomdp_fn=build_obstacle_ipomdp,
    seed=42,
    num_trajectories=50,
    trajectory_length=20,
    initial_state=(1, 1),
    initial_action="north",
    sampler_budget=100,
    sampler_k=10,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="results/final/coarse_obstacle_results.json",
)
