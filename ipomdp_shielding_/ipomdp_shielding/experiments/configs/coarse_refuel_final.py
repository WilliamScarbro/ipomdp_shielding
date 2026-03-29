"""Final coarseness experiment configuration for Refuel.

Sampler budget matches threshold-sweep forward-sampling shield (budget=500, K=100).
LFP LP solve dominates at ~144 s/step (344 states); forward-sampler overhead
is negligible relative to the LP cost.
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
    sampler_budget=500,
    sampler_k=100,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="results/final/coarse_refuel_results.json",
)
