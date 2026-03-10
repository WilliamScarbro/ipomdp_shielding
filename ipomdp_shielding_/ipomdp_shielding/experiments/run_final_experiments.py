"""Master runner for the final IPOMDP shielding experiments (~8 hours).

Runs all experiments sequentially:
  Coarse (LFP bound tightness):
    1. taxinet   ~3 min
    2. cartpole  ~12 min
    3. obstacle  ~17 min
    4. refuel    ~4.5 h  (dominates budget)
  RL Shield (runtime safety):
    5. taxinet   ~20 min
    6. cartpole  ~50-70 min
    7. obstacle  ~50-85 min
    8. refuel    ~10 min  (envelope excluded)
  Summary charts generated at the end.

Usage (from repo root):
    python -m ipomdp_shielding.experiments.run_final_experiments
"""

import os
import sys
import time
import importlib

import numpy as np

from .experiment_io import build_metadata, save_experiment_results

# Coarse experiment internals
from .run_coarse_experiment import (
    create_lfp_propagator,
    create_sampler,
    generate_trajectories,
    evaluate_trajectory,
    aggregate_reports,
    print_report,
    try_plot,
)
from ..Evaluation.coarseness_evaluator import CoarsenessEvaluator

# RL shield experiment
from .run_rl_shield_experiment import (
    setup as rl_setup,
    build_grid,
    run_experiment as rl_run_experiment,
    print_results_table,
    save_results,
    plot_results,
)


# ============================================================
# Coarse experiment runner
# ============================================================

def run_coarse(config):
    """Run a coarseness experiment from a CoarseExperimentConfig."""
    print("\n" + "=" * 70)
    print(f"COARSENESS EXPERIMENT: {config.case_study_name.upper()}")
    print(f"Trajectories: {config.num_trajectories}, Length: {config.trajectory_length}, "
          f"Seed: {config.seed}")
    print(f"Sampler budget: {config.sampler_budget}, K: {config.sampler_k}")
    print("=" * 70)

    os.makedirs(os.path.dirname(config.results_path), exist_ok=True)

    print(f"\nBuilding {config.case_study_name.upper()} IPOMDP...")
    ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(seed=config.seed, **config.ipomdp_kwargs)
    n = len(ipomdp.states)
    print(f"  States: {n}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    print("\nCreating LFP propagator (canonical templates)...")
    lfp = create_lfp_propagator(ipomdp)

    print("Creating ForwardSampledBelief propagator...")
    sampler = create_sampler(ipomdp, config, seed=config.seed)

    evaluator = CoarsenessEvaluator(lfp, sampler, pp_shield)

    print(f"\nGenerating {config.num_trajectories} trajectories "
          f"(length {config.trajectory_length})...")
    library = generate_trajectories(ipomdp, pp_shield, config, seed=config.seed)
    actual_lengths = [s.length for s in library]
    print(f"  Trajectory lengths: min={min(actual_lengths)}, max={max(actual_lengths)}, "
          f"mean={np.mean(actual_lengths):.1f}")

    print("\nEvaluating coarseness...")
    reports = []
    t0 = time.time()

    for i, script in enumerate(library):
        if script.length == 0:
            continue
        report = evaluate_trajectory(evaluator, script)
        reports.append(report)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(library)}] max_gap={report.overall_max_safe_gap:.4f}  "
                  f"mean_gap={report.overall_mean_safe_gap:.4f}  ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\nTotal evaluation time: {total_time:.1f}s")

    summary = aggregate_reports(reports)
    print_report(summary, config.case_study_name)

    metadata = build_metadata(config, extra={"total_time_s": total_time})
    save_experiment_results(config.results_path, summary, metadata)
    print(f"\nResults saved to {config.results_path}")

    try_plot(summary, config)

    print("\n" + "=" * 70)
    print(f"COARSE {config.case_study_name.upper()} COMPLETE")
    print("=" * 70)

    return summary, total_time


# ============================================================
# RL shield experiment runner (with optional shield exclusion)
# ============================================================

def run_rl_shield(config, exclude_shields=None):
    """Run an RL shielding experiment from an RLShieldExperimentConfig.

    Parameters
    ----------
    config : RLShieldExperimentConfig
    exclude_shields : list[str] or None
        If provided, combinations with these shield names are skipped.
        Used to exclude 'envelope' for Refuel (LP infeasible).
    """
    print("\n" + "=" * 70)
    print(f"RL SHIELDING EXPERIMENT: {config.case_study_name.upper()}")
    print(f"Trials: {config.num_trials}, Length: {config.trial_length}, "
          f"Seed: {config.seed}")
    if exclude_shields:
        print(f"Excluded shields: {exclude_shields}")
    print("=" * 70)

    os.makedirs(os.path.dirname(config.results_path), exist_ok=True)

    print(f"\nLoading {config.case_study_name.upper()} IPOMDP...")
    ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(**config.ipomdp_kwargs)
    print(f"  States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    rl_selector, optimized_perceptions, setup_info = rl_setup(ipomdp, pp_shield, config)

    full_grid = build_grid(ipomdp, pp_shield, rl_selector, optimized_perceptions, config)

    if exclude_shields:
        grid = [(p, s, sh, perc, sel, shf)
                for p, s, sh, perc, sel, shf in full_grid
                if sh not in exclude_shields]
        print(f"\nExperiment grid: {len(grid)} combinations "
              f"(envelope excluded)")
    else:
        grid = full_grid
        print(f"\nExperiment grid: {len(grid)} combinations "
              f"(2 perceptions x 3 selectors x 4 shields)")

    t0 = time.time()
    results, trial_data, intervention_stats = rl_run_experiment(
        ipomdp, pp_shield, grid, config
    )
    total_time = time.time() - t0

    print_results_table(results, config)
    note = (f"envelope excluded (LP infeasible for {len(ipomdp.states)} states)"
            if exclude_shields else None)
    extra = {**setup_info, "intervention_stats": {
        f"{k[0]}/{k[1]}/{k[2]}": v for k, v in intervention_stats.items()
    }}
    if note:
        extra["note"] = note
    save_results(results, config, setup_info=extra)

    print("\nGenerating figures...")
    plot_results(trial_data, config, intervention_stats=intervention_stats)

    print(f"\nTotal experiment time: {total_time:.1f}s")
    print("=" * 70)
    print(f"RL SHIELD {config.case_study_name.upper()} COMPLETE")
    print("=" * 70)

    return results, trial_data, total_time


# ============================================================
# Main
# ============================================================

def main():
    overall_start = time.time()

    timings = {}

    def _load_config(module_name):
        mod = importlib.import_module(
            f".configs.{module_name}",
            package="ipomdp_shielding.experiments",
        )
        return mod.config

    # -------------------------------------------------------
    # 1. Coarse experiments
    # -------------------------------------------------------
    print("\n" + "#" * 70)
    print("# COARSE EXPERIMENTS")
    print("#" * 70)

    for cs in ["taxinet", "cartpole", "obstacle", "refuel"]:
        cfg_name = f"coarse_{cs}_final"
        print(f"\n>>> Loading config: {cfg_name}")
        cfg = _load_config(cfg_name)
        try:
            _, t = run_coarse(cfg)
            timings[f"coarse_{cs}"] = t
        except Exception as exc:
            print(f"\n!!! coarse_{cs} FAILED: {exc}")
            import traceback
            traceback.print_exc()
            timings[f"coarse_{cs}"] = None

    # -------------------------------------------------------
    # 2. RL shielding experiments
    # -------------------------------------------------------
    print("\n" + "#" * 70)
    print("# RL SHIELDING EXPERIMENTS")
    print("#" * 70)

    for cs in ["taxinet", "cartpole", "obstacle"]:
        cfg_name = f"rl_shield_{cs}_final"
        print(f"\n>>> Loading config: {cfg_name}")
        cfg = _load_config(cfg_name)
        try:
            _, _, t = run_rl_shield(cfg, exclude_shields=None)
            timings[f"rl_{cs}"] = t
        except Exception as exc:
            print(f"\n!!! rl_shield_{cs} FAILED: {exc}")
            import traceback
            traceback.print_exc()
            timings[f"rl_{cs}"] = None

    # Refuel: exclude envelope (LP infeasible)
    cfg_name = "rl_shield_refuel_final"
    print(f"\n>>> Loading config: {cfg_name}")
    cfg = _load_config(cfg_name)
    try:
        _, _, t = run_rl_shield(cfg, exclude_shields=["envelope"])
        timings["rl_refuel"] = t
    except Exception as exc:
        print(f"\n!!! rl_shield_refuel FAILED: {exc}")
        import traceback
        traceback.print_exc()
        timings["rl_refuel"] = None

    # -------------------------------------------------------
    # 3. Summary charts
    # -------------------------------------------------------
    print("\n" + "#" * 70)
    print("# SUMMARY CHARTS")
    print("#" * 70)

    try:
        from .plot_summary import main as plot_main
        sys.argv = [
            "plot_summary",
            "--mode", "final",
            "--output-dir", "results/final/summary",
        ]
        plot_main()
    except Exception as exc:
        print(f"\n!!! plot_summary FAILED: {exc}")
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------
    # Final timing report
    # -------------------------------------------------------
    overall_elapsed = time.time() - overall_start

    print("\n" + "=" * 70)
    print("FINAL RUN COMPLETE — TIMING SUMMARY")
    print("=" * 70)
    for key, t in timings.items():
        if t is None:
            print(f"  {key:<25} FAILED")
        else:
            mm, ss = divmod(int(t), 60)
            hh, mm = divmod(mm, 60)
            print(f"  {key:<25} {hh:02d}h {mm:02d}m {ss:02d}s  ({t:.0f}s)")

    hh, rem = divmod(int(overall_elapsed), 3600)
    mm, ss = divmod(rem, 60)
    print(f"\n  TOTAL                     {hh:02d}h {mm:02d}m {ss:02d}s  ({overall_elapsed:.0f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
