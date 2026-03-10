"""Refuel RL shielding experiment — envelope shield excluded.

Envelope shielding is infeasible for refuel (344 states, ~65s per LP step).
This script runs the remaining 18 combinations (2 perceptions × 3 selectors
× 3 shields: none, observation, single_belief) using the corrected model:
  - obs_noise=0.05, distance-scaled noise
  - Fixed ObservationShield (obs_to_states lookup, not pp_shield[obs])

Results are saved to the same path as rl_shield_refuel_prelim so the summary
charts pick them up.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .configs.rl_shield_refuel_prelim import config as _base_config
from .run_rl_shield_experiment import (
    setup, build_grid, run_experiment,
    print_results_table, save_results, plot_results,
)
from dataclasses import replace
import time

# Optimize adversarial perception against single_belief (envelope is infeasible at 344 states).
# Use a separate cache path so we don't clobber the envelope-based realization if it exists.
_config = replace(
    _base_config,
    adversarial_opt_targets=["single_belief"],
    opt_cache_path="results/cache/prelim_rl_shield_refuel_sb_opt_realization.json",
)


def main():
    # Build IPOMDP
    print(f"Loading {_config.case_study_name.upper()} IPOMDP (envelope excluded)...")
    ipomdp, pp_shield, _, _ = _config.build_ipomdp_fn(**_config.ipomdp_kwargs)
    print(f"  States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    rl_selector, optimized_perceptions, setup_info = setup(ipomdp, pp_shield, _config)

    # Build full grid then drop envelope combinations
    full_grid = build_grid(ipomdp, pp_shield, rl_selector, optimized_perceptions, _config)
    grid = [(p, s, sh, perc, sel, shf)
            for p, s, sh, perc, sel, shf in full_grid if sh != "envelope"]
    print(f"\nExperiment grid: {len(grid)} combinations (envelope excluded)")

    t0 = time.time()
    results, trial_data, intervention_stats = run_experiment(
        ipomdp, pp_shield, grid, _config
    )
    total_time = time.time() - t0

    print_results_table(results, _config)
    save_results(results, _config,
                 setup_info={**setup_info, "intervention_stats": {
                     f"{k[0]}/{k[1]}/{k[2]}": v for k, v in intervention_stats.items()
                 }, "note": "envelope excluded (infeasible); adversarial_opt against single_belief"})

    print("\nGenerating figures...")
    plot_results(trial_data, _config, intervention_stats=intervention_stats)
    print(f"\nTotal time: {total_time:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
