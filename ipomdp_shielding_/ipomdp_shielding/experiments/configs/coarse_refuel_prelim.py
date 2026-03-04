"""Preliminary coarseness experiment configuration for Refuel."""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_refuel_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy

# Default N=7, ENERGY=6: 7x7x7=343 states + FAIL = 344 states, 5 actions.
# Medium state space — LFP tractable in ~minutes.
config = CoarseExperimentConfig(
    case_study_name="refuel",
    build_ipomdp_fn=build_refuel_ipomdp,
    seed=42,
    num_trajectories=10,
    trajectory_length=10,
    initial_state=(0, 0, 6),  # Refuel station at (0,0) with full fuel
    initial_action="north",
    sampler_budget=50,
    sampler_k=5,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="results/prelim/coarse_refuel_results.json",
)
