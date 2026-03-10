"""Coarseness experiment config for Refuel v2.

Observation change: hascrash and fuel > 0 removed from obs tuple (8 bits, was 10).
LP solve timing is unchanged (~144 s/step), so trajectory count is kept the same
as the final run (10 traj × 10 steps ≈ 4.5 h).
"""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_refuel_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy


config = CoarseExperimentConfig(
    case_study_name="refuel_v2",
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
    results_path="results/v2/coarse_refuel_v2_results.json",
)
