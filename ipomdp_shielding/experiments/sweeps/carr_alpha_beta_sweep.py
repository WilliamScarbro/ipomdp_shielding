"""Carr comparison sweep: alpha (interval width) and beta (lifted threshold).

Shows how:
1. Carr winning region size drops with increased observation uncertainty (alpha)
2. Lifted shield behavior varies smoothly with beta
3. Perfect-observation sanity check validates Carr works without uncertainty

Usage:
    python -m ipomdp_shielding.experiments.sweeps.carr_alpha_beta_sweep
"""

import os
import csv
import json
import time
import random

from ..experiment_io import build_metadata, add_rate_cis, save_experiment_results
from ..carr_comparison_experiment import CarrComparisonExperiment
from ..configs.carr_comparison_config import CarrComparisonConfig


def run_carr_sweep(
    alphas=None,
    betas=None,
    num_trials=50,
    trial_length=30,
    seed=42,
    results_dir="./data/sweep/carr_comparison",
):
    """Run Carr comparison sweep over alpha and beta.

    Parameters
    ----------
    alphas : list of float
        CI significance levels to sweep (wider intervals = more uncertainty).
    betas : list of float
        Lifted shield thresholds to sweep.
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    if betas is None:
        betas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("CARR COMPARISON SWEEP")
    print(f"Alphas: {alphas}")
    print(f"Betas: {betas}")
    print(f"Trials: {num_trials}, Length: {trial_length}")
    print("=" * 70)

    tidy_rows = []
    winning_region_data = []
    t0 = time.time()

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"ALPHA = {alpha}")
        print(f"{'='*60}")

        for beta in betas:
            print(f"\n  Beta = {beta}")

            config = CarrComparisonConfig(
                seed=seed,
                num_trials=num_trials,
                trial_length=trial_length,
                lifted_shield_threshold=beta,
                realization_strategy="midpoint",
                results_path=os.path.join(results_dir, f"carr_a{alpha}_b{beta}.json"),
                ipomdp_kwargs={
                    "confidence_method": "Clopper_Pearson",
                    "alpha": alpha,
                    "train_fraction": 0.8,
                    "error": 0.1,
                    "smoothing": True,
                },
            )

            try:
                experiment = CarrComparisonExperiment(config)
                experiment.run_trials()
                metrics = experiment.compute_metrics()

                # Carr statistics (winning region)
                carr_stats = experiment.carr_shield.get_statistics()

                # Store winning region data
                winning_region_data.append({
                    "alpha": alpha,
                    "beta": beta,
                    "total_supports": carr_stats["total_supports"],
                    "winning_supports": carr_stats["winning_supports"],
                    "losing_supports": carr_stats["losing_supports"],
                    "winning_fraction": (
                        carr_stats["winning_supports"] / carr_stats["total_supports"]
                        if carr_stats["total_supports"] > 0 else 0.0
                    ),
                })

                # Tidy rows for both shields
                for shield_name in ["carr", "lifted"]:
                    m = metrics[shield_name]
                    row = {
                        "alpha": alpha,
                        "beta": beta,
                        "shield": shield_name,
                        "num_trials": m["total_trials"],
                        "fail_rate": m["fail_rate"],
                        "stuck_rate": m["stuck_rate"],
                        "success_rate": m["success_rate"],
                        "avg_trajectory_length": m["avg_trajectory_length"],
                    }
                    add_rate_cis(row, m["total_trials"])
                    tidy_rows.append(row)

                    print(f"    {shield_name}: fail={m['fail_rate']:.1%} "
                          f"stuck={m['stuck_rate']:.1%} success={m['success_rate']:.1%}")

                print(f"    Carr winning region: {carr_stats['winning_supports']}"
                      f"/{carr_stats['total_supports']}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    # Perfect-observation sanity check
    print(f"\n{'='*60}")
    print("SANITY CHECK: Perfect observations (alpha=0.001, very tight intervals)")
    print(f"{'='*60}")

    try:
        sanity_config = CarrComparisonConfig(
            seed=seed,
            num_trials=num_trials,
            trial_length=trial_length,
            lifted_shield_threshold=0.8,
            realization_strategy="midpoint",
            results_path=os.path.join(results_dir, "carr_sanity_perfect_obs.json"),
            ipomdp_kwargs={
                "confidence_method": "Clopper_Pearson",
                "alpha": 0.001,  # Very tight intervals ≈ perfect observation
                "train_fraction": 0.8,
                "error": 0.1,
                "smoothing": True,
            },
        )
        sanity_exp = CarrComparisonExperiment(sanity_config)
        sanity_exp.run_trials()
        sanity_metrics = sanity_exp.compute_metrics()
        sanity_stats = sanity_exp.carr_shield.get_statistics()

        for shield_name in ["carr", "lifted"]:
            m = sanity_metrics[shield_name]
            print(f"  {shield_name}: fail={m['fail_rate']:.1%} "
                  f"stuck={m['stuck_rate']:.1%} success={m['success_rate']:.1%}")
        print(f"  Carr winning region: {sanity_stats['winning_supports']}"
              f"/{sanity_stats['total_supports']}")

        # Add sanity check to tidy rows
        for shield_name in ["carr", "lifted"]:
            m = sanity_metrics[shield_name]
            row = {
                "alpha": 0.001,
                "beta": 0.8,
                "shield": shield_name,
                "num_trials": m["total_trials"],
                "fail_rate": m["fail_rate"],
                "stuck_rate": m["stuck_rate"],
                "success_rate": m["success_rate"],
                "avg_trajectory_length": m["avg_trajectory_length"],
                "note": "sanity_check_perfect_obs",
            }
            add_rate_cis(row, m["total_trials"])
            tidy_rows.append(row)
    except Exception as e:
        print(f"  Sanity check failed: {e}")

    total_time = time.time() - t0

    # Save results
    csv_path = os.path.join(results_dir, "results_tidy.csv")
    if tidy_rows:
        fieldnames = list(tidy_rows[0].keys())
        # Ensure all rows have all keys
        all_keys = set()
        for row in tidy_rows:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tidy_rows)
        print(f"\nTidy CSV saved to {csv_path}")

    # Winning region CSV
    wr_csv_path = os.path.join(results_dir, "winning_region_data.csv")
    if winning_region_data:
        fieldnames = list(winning_region_data[0].keys())
        with open(wr_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(winning_region_data)
        print(f"Winning region CSV saved to {wr_csv_path}")

    # Summary JSON
    metadata = build_metadata(None, extra={
        "alphas": alphas,
        "betas": betas,
        "num_trials": num_trials,
        "trial_length": trial_length,
        "total_time_s": total_time,
    })
    save_experiment_results(
        os.path.join(results_dir, "sweep_summary.json"),
        {
            "tidy_rows": tidy_rows,
            "winning_region_data": winning_region_data,
            "total_time_s": total_time,
        },
        metadata,
        tidy_rows,
    )

    # Plots
    _plot_carr_sweep(tidy_rows, winning_region_data, alphas, betas, results_dir)

    print(f"\nTotal sweep time: {total_time:.1f}s")
    return tidy_rows, winning_region_data


def _plot_carr_sweep(tidy_rows, winning_region_data, alphas, betas, results_dir):
    """Generate Carr comparison sweep plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Winning region size vs alpha (for each beta)
    if winning_region_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for beta in sorted(set(r["beta"] for r in winning_region_data)):
            subset = sorted(
                [r for r in winning_region_data if r["beta"] == beta],
                key=lambda r: r["alpha"]
            )
            xs = [r["alpha"] for r in subset]
            ys = [r["winning_fraction"] for r in subset]
            ax.plot(xs, ys, "o-", label=f"beta={beta}")

        ax.set_xlabel("Alpha (CI significance)")
        ax.set_ylabel("Winning Region Fraction")
        ax.set_title("Carr Winning Region Size vs Observation Uncertainty")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        fname = "carr_winning_region_vs_alpha.png"
        fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # 2. Stuck rate comparison: Carr vs Lifted across alpha
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, shield in enumerate(["carr", "lifted"]):
        ax = axes[idx]
        for beta in sorted(set(r["beta"] for r in tidy_rows if r["shield"] == shield)):
            subset = sorted(
                [r for r in tidy_rows if r["shield"] == shield and r["beta"] == beta
                 and r.get("note") != "sanity_check_perfect_obs"],
                key=lambda r: r["alpha"]
            )
            if not subset:
                continue
            xs = [r["alpha"] for r in subset]
            ys = [r["stuck_rate"] for r in subset]
            ax.plot(xs, ys, "o-", label=f"beta={beta}")

        ax.set_xlabel("Alpha (CI significance)")
        ax.set_ylabel("Stuck Rate")
        ax.set_title(f"{shield.capitalize()} Shield — Stuck Rate vs Alpha")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fname = "carr_vs_lifted_stuck_rate.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

    # 3. Lifted shield: fail vs stuck tradeoff across beta (for each alpha)
    fig, ax = plt.subplots(figsize=(8, 5))
    lifted_rows = [r for r in tidy_rows if r["shield"] == "lifted"
                   and r.get("note") != "sanity_check_perfect_obs"]
    for alpha in sorted(set(r["alpha"] for r in lifted_rows)):
        subset = sorted(
            [r for r in lifted_rows if r["alpha"] == alpha],
            key=lambda r: r["beta"]
        )
        if not subset:
            continue
        xs = [r["fail_rate"] for r in subset]
        ys = [r["stuck_rate"] for r in subset]
        ax.plot(xs, ys, "o-", label=f"alpha={alpha}")
        # Annotate with beta values
        for r in subset:
            ax.annotate(f"{r['beta']:.2f}", (r["fail_rate"], r["stuck_rate"]),
                        fontsize=6, textcoords="offset points", xytext=(3, 3))

    ax.set_xlabel("Fail Rate (safety)")
    ax.set_ylabel("Stuck Rate (liveness)")
    ax.set_title("Lifted Shield: Safety vs Liveness Tradeoff")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fname = "lifted_safety_liveness_tradeoff.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


if __name__ == "__main__":
    run_carr_sweep()
