"""Final coarseness experiment configuration for CartPole.

3x more trajectories, 1.5x longer; sampler budget matches threshold-sweep
forward-sampling shield (budget=500, K=100).
Uses 3-bin discretisation (82 states) consistent with prelim.
Estimated runtime: ~12 min.
"""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.CartPole import build_cartpole_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy


config = CoarseExperimentConfig(
    case_study_name="cartpole",
    build_ipomdp_fn=build_cartpole_ipomdp,
    seed=42,
    num_trajectories=30,
    trajectory_length=15,
    initial_state=(1, 1, 1, 1),  # Centre state for 3-bin discretisation
    initial_action=0,
    sampler_budget=500,
    sampler_k=100,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="results/final/coarse_cartpole_results.json",
    ipomdp_kwargs={"num_bins": 3},
)
