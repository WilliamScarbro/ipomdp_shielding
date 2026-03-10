"""Final RL shielding experiment configuration for Refuel.

5x more trials, 1.5x longer vs prelim.  Envelope excluded (LP solve
~144 s/step for 344 states makes it infeasible).  Adversarial perception
optimised against single-belief shield.
Reuses prelim RL agent and single-belief opt-realization caches.
Estimated runtime: ~10 min (18 non-envelope combos, fast non-LP shields).
"""

from .base_config import RLShieldExperimentConfig
from ...CaseStudies.GridWorldBenchmarks import build_refuel_ipomdp


config = RLShieldExperimentConfig(
    case_study_name="refuel",
    build_ipomdp_fn=build_refuel_ipomdp,
    seed=42,
    num_trials=50,
    trial_length=30,
    rl_episodes=500,
    rl_episode_length=30,
    opt_candidates=10,
    opt_trials_per_candidate=5,
    opt_iterations=10,
    shield_threshold=0.8,
    adversarial_opt_targets=["single_belief"],
    # Reuse prelim caches — no retraining needed
    rl_cache_path="results/cache/prelim_rl_shield_refuel_agent.pt",
    opt_cache_path="results/cache/prelim_rl_shield_refuel_sb_opt_realization.json",
    results_path="results/final/rl_shield_refuel_results.json",
    figures_dir="results/final/rl_shield_refuel_figures",
)
