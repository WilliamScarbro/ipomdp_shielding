"""Full coarseness experiment configuration for TaxiNet."""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.Taxinet import build_taxinet_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy


config = CoarseExperimentConfig(
    case_study_name="taxinet",
    build_ipomdp_fn=build_taxinet_ipomdp,
    seed=42,
    num_trajectories=100,
    trajectory_length=20,
    initial_state=(0, 0),
    initial_action=0,
    sampler_budget=200,
    sampler_k=20,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="results/full/coarse_taxinet_results.json",
)
