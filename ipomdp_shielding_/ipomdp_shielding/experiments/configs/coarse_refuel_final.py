"""Final coarseness experiment configuration for Refuel.

Same parameters as prelim — LFP LP solve takes ~144 s/step (344 states),
making 10 trajectories × 10 steps already ~4.5 h.  Increasing sampling
further is not feasible within an 8-hour budget.
Estimated runtime: ~4.5 h.
"""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_refuel_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy


config = CoarseExperimentConfig(
    case_study_name="refuel",
    build_ipomdp_fn=build_refuel_ipomdp,
    seed=42,
    num_trajectories=10,
    trajectory_length=10,
    initial_state=(0, 0, 6),
    initial_action="north",
    sampler_budget=50,
    sampler_k=5,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="results/final/coarse_refuel_results.json",
)
