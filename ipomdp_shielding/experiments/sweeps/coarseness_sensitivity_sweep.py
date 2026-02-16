"""Coarseness sensitivity sweep: budget and template type.

Sweeps sampler_budget and template type to show how coarseness changes
with controllable knobs.

Usage:
    python -m ipomdp_shielding.experiments.sweeps.coarseness_sensitivity_sweep
"""

import os
import sys
import csv
import json
import time

import numpy as np

from ..experiment_io import build_metadata, save_experiment_results
from ..run_coarse_experiment import (
    create_lfp_propagator, create_sampler, generate_trajectories,
    evaluate_trajectory, aggregate_reports,
)
from ...Propagators import (
    LFPPropagator, BeliefPolytope, TemplateFactory, ForwardSampledBelief,
)
from ...Propagators.lfp_propagator import default_solver
from ...Evaluation.coarseness_evaluator import CoarsenessEvaluator


def run_coarseness_sensitivity(config=None):
    """Run coarseness sensitivity sweep over budget and template type.

    Parameters
    ----------
    config : optional config object. If None, uses defaults for TaxiNet.
    """
    from ...CaseStudies.Taxinet import build_taxinet_ipomdp
    from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy

    # Default parameters
    seed = 42
    num_trajectories = 30
    trajectory_length = 15
    initial_state = (0, 0)
    initial_action = 0
    budgets = [50, 100, 200, 500]
    k_values = [5, 10, 20]
    template_types = ["canonical"]  # safe_set_indicators requires avoid_states
    results_dir = "./data/sweep/coarseness_sensitivity"

    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("COARSENESS SENSITIVITY SWEEP")
    print(f"Budgets: {budgets}")
    print(f"K values: {k_values}")
    print(f"Template types: {template_types}")
    print("=" * 70)

    # Build IPOMDP
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp(seed=seed)
    n = len(ipomdp.states)

    # Generate trajectories once (shared across all configs)
    from ..configs.base_config import CoarseExperimentConfig
    from ...MonteCarlo import UniformPerceptionModel

    dummy_config = CoarseExperimentConfig(
        case_study_name="taxinet",
        build_ipomdp_fn=build_taxinet_ipomdp,
        seed=seed,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        initial_state=initial_state,
        initial_action=initial_action,
        sampler_budget=100,
        sampler_k=10,
        sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
        sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
        results_path=os.path.join(results_dir, "dummy.json"),
    )

    print(f"\nGenerating {num_trajectories} shared trajectories...")
    library = generate_trajectories(ipomdp, pp_shield, dummy_config, seed=seed)

    tidy_rows = []
    t0 = time.time()

    for template_type in template_types:
        # Create LFP propagator for this template type
        if template_type == "canonical":
            template = TemplateFactory.canonical(n)
        else:
            template = TemplateFactory.canonical(n)

        for budget in budgets:
            for k in k_values:
                print(f"\n  template={template_type}, budget={budget}, K={k}")

                polytope = BeliefPolytope.uniform_prior(n)
                lfp = LFPPropagator(ipomdp, template, default_solver(), polytope)

                rng = np.random.default_rng(seed)
                sampler = ForwardSampledBelief(
                    ipomdp=ipomdp,
                    budget=budget,
                    K_samples=k,
                    likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
                    pruning_strategy=PruningStrategy.FARTHEST_POINT,
                    rng=rng,
                )

                evaluator = CoarsenessEvaluator(lfp, sampler, pp_shield)

                reports = []
                for script in library:
                    if script.length == 0:
                        continue
                    report = evaluate_trajectory(evaluator, script)
                    reports.append(report)

                if not reports:
                    continue

                summary = aggregate_reports(reports)
                runtime = time.time() - t0

                row = {
                    "template_type": template_type,
                    "budget": budget,
                    "K": k,
                    "overall_max_gap_mean": summary["overall_max_gap"]["mean"],
                    "overall_max_gap_std": summary["overall_max_gap"]["std"],
                    "overall_max_gap_median": summary["overall_max_gap"]["median"],
                    "overall_mean_gap_mean": summary["overall_mean_gap"]["mean"],
                    "overall_mean_gap_std": summary["overall_mean_gap"]["std"],
                    "runtime_s": runtime,
                }
                tidy_rows.append(row)

                print(f"    max_gap: {row['overall_max_gap_mean']:.4f} +/- {row['overall_max_gap_std']:.4f}")

    total_time = time.time() - t0

    # Save tidy CSV
    csv_path = os.path.join(results_dir, "sensitivity_tidy.csv")
    if tidy_rows:
        fieldnames = list(tidy_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tidy_rows)
        print(f"\nTidy CSV saved to {csv_path}")

    # Save JSON summary
    metadata = build_metadata(dummy_config, extra={"total_time_s": total_time})
    save_experiment_results(
        os.path.join(results_dir, "sensitivity_summary.json"),
        {"rows": tidy_rows, "total_time_s": total_time},
        metadata,
        tidy_rows,
    )

    # Plot
    _plot_sensitivity(tidy_rows, results_dir)

    print(f"\nTotal sweep time: {total_time:.1f}s")
    return tidy_rows


def _plot_sensitivity(tidy_rows, results_dir):
    """Plot coarseness vs budget for each K value."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Group by K
    k_values = sorted(set(r["K"] for r in tidy_rows))
    budgets = sorted(set(r["budget"] for r in tidy_rows))

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in k_values:
        subset = [r for r in tidy_rows if r["K"] == k]
        subset.sort(key=lambda r: r["budget"])
        xs = [r["budget"] for r in subset]
        ys = [r["overall_max_gap_mean"] for r in subset]
        errs = [r["overall_max_gap_std"] for r in subset]
        ax.errorbar(xs, ys, yerr=errs, marker="o", label=f"K={k}", capsize=3)

    ax.set_xlabel("Sampler Budget")
    ax.set_ylabel("Max Coarseness Gap (mean +/- std)")
    ax.set_title("Coarseness Sensitivity to Budget and K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    fname = "coarseness_sensitivity.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


if __name__ == "__main__":
    run_coarseness_sensitivity()
