"""Alpha/Beta sweep for RL shielding: safety vs liveness tradeoff.

Sweeps over alpha (CI significance → interval width) and beta (shield threshold)
to produce heatmaps and Pareto plots showing how these hyperparameters affect
failure probability, stuck probability, and mean steps.

Usage:
    python -m ipomdp_shielding.experiments.sweeps.rl_alpha_beta_sweep [--config CONFIG_MODULE]

Default config: sweeps.rl_alpha_beta_sweep_taxinet
"""

import os
import sys
import csv
import json
import time
import copy
import importlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

import numpy as np

from ..experiment_io import (
    build_metadata, add_rate_cis, save_experiment_results, clopper_pearson_ci
)
from ..run_rl_shield_experiment import (
    setup, build_grid, run_experiment, compute_timestep_outcomes,
    create_envelope_shield_factory, create_single_belief_shield_factory,
    ShieldCompliantSelector, NoShield, ObservationShield,
    create_no_shield_factory, create_observation_shield_factory,
)
from ..configs.base_config import RLShieldExperimentConfig
from ...MonteCarlo import (
    UniformPerceptionModel,
    FixedRealizationPerceptionModel,
    train_optimal_realization,
    RandomActionSelector,
    BeliefSelector,
    NeuralActionSelector,
    RandomInitialState,
    run_monte_carlo_trials,
    compute_safety_metrics,
    AdversarialPerceptionModel,
)


@dataclass
class AlphaBetaSweepConfig:
    """Configuration for alpha/beta sweep experiment."""

    # Case study
    case_study_name: str = "taxinet"
    build_ipomdp_fn: Callable = None

    # Sweep grid
    alphas: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    betas: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    # Per-grid-point evaluation
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    num_trials: int = 30
    trial_length: int = 20

    # RL training (done once per alpha at baseline beta)
    rl_episodes: int = 500
    rl_episode_length: int = 20
    opt_candidates: int = 20
    opt_trials_per_candidate: int = 10
    opt_iterations: int = 10
    baseline_beta: float = 0.8
    # Which shield(s) to optimize the fixed realization against.
    # Default preserves legacy behavior: optimize against envelope only.
    adversarial_opt_targets: List[str] = field(default_factory=lambda: ["envelope"])

    # Shields to evaluate (subset for speed)
    shields: List[str] = field(default_factory=lambda: ["single_belief", "envelope"])
    perceptions: List[str] = field(default_factory=lambda: ["uniform", "adversarial_opt"])

    # Output
    results_dir: str = "./data/sweep/rl_alpha_beta"

    # Extra IPOMDP kwargs (besides alpha, which is swept)
    ipomdp_base_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.build_ipomdp_fn is None:
            from ...CaseStudies.Taxinet import build_taxinet_ipomdp
            self.build_ipomdp_fn = build_taxinet_ipomdp


def run_alpha_beta_sweep(config: AlphaBetaSweepConfig):
    """Run the full alpha/beta sweep.

    Strategy (minimal version):
    - For each alpha: train RL agent + adversarial realization once (at baseline beta)
    - Then evaluate across all beta values

    Returns tidy_rows list and summary dict.
    """
    os.makedirs(config.results_dir, exist_ok=True)
    tidy_rows = []
    summary = {"config": config.__dict__.copy(), "grid_results": {}}
    # Make config serializable
    summary["config"]["build_ipomdp_fn"] = str(config.build_ipomdp_fn)

    total_points = len(config.alphas) * len(config.betas) * len(config.seeds)
    print("=" * 70)
    print(f"ALPHA/BETA SWEEP - {config.case_study_name.upper()}")
    print(f"Alphas: {config.alphas}")
    print(f"Betas: {config.betas}")
    print(f"Seeds: {config.seeds}")
    print(f"Total grid points: {total_points}")
    print("=" * 70)

    t0 = time.time()

    for alpha in config.alphas:
        print(f"\n{'='*70}")
        print(f"ALPHA = {alpha}")
        print(f"{'='*70}")

        # Build IPOMDP for this alpha
        ipomdp_kwargs = {**config.ipomdp_base_kwargs, "alpha": alpha}
        ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(**ipomdp_kwargs)
        all_actions = list(ipomdp.actions)
        pomdp = ipomdp.to_pomdp()

        # Train RL agent + adversarial realization once per alpha
        cache_prefix = os.path.join(config.results_dir, f"cache_alpha{alpha}")
        rl_cache = f"{cache_prefix}_rl_agent.pt"
        opt_cache = f"{cache_prefix}_opt_realization.json"

        # RL agent
        if os.path.exists(rl_cache):
            print(f"  Loading cached RL agent for alpha={alpha}")
            rl_selector = NeuralActionSelector.load(rl_cache, ipomdp)
        else:
            print(f"  Training RL agent for alpha={alpha}...")
            rl_selector = NeuralActionSelector(
                actions=all_actions,
                observations=ipomdp.observations,
                maximize_safety=True,
            )
            rl_selector.train(
                ipomdp=ipomdp,
                perception=AdversarialPerceptionModel(pp_shield),
                num_episodes=config.rl_episodes,
                episode_length=config.rl_episode_length,
                verbose=False,
            )
            rl_selector.save(rl_cache)
        rl_selector.exploration_rate = 0.0

        # Adversarial fixed realizations (optionally per shield target)
        targets = list(dict.fromkeys(config.adversarial_opt_targets))
        opt_perceptions: Dict[str, FixedRealizationPerceptionModel] = {}
        for target in targets:
            target_cache = (
                opt_cache if target == "envelope"
                else opt_cache.replace(".json", f"_{target}.json")
            )
            if os.path.exists(target_cache):
                print(f"  Loading cached adversarial realization ({target}) for alpha={alpha}")
                opt_perceptions[target] = FixedRealizationPerceptionModel.load(target_cache)
                continue

            print(f"  Training adversarial realization ({target}) for alpha={alpha}...")
            if target == "envelope":
                rt_factory = create_envelope_shield_factory(
                    ipomdp, pp_shield, config.baseline_beta
                )
            elif target == "single_belief":
                rt_factory = create_single_belief_shield_factory(
                    pomdp, pp_shield, config.baseline_beta
                )
            else:
                raise ValueError(
                    f"Unknown adversarial_opt_target={target!r}. Supported: 'envelope', 'single_belief'."
                )
            opt_perceptions[target] = train_optimal_realization(
                ipomdp=ipomdp,
                pp_shield=pp_shield,
                rt_shield_factory=rt_factory,
                action_selector=RandomActionSelector(),
                initial_generator=RandomInitialState(),
                num_candidates=config.opt_candidates,
                num_trials_per_candidate=config.opt_trials_per_candidate,
                max_iterations=config.opt_iterations,
                trial_length=config.trial_length,
                save_path=target_cache,
                verbose=False,
            )

        perceptions = {}
        if "uniform" in config.perceptions:
            perceptions["uniform"] = UniformPerceptionModel()
        if "adversarial_opt" in config.perceptions:
            perceptions["adversarial_opt"] = opt_perceptions

        # Now sweep beta
        for beta in config.betas:
            print(f"\n  Beta = {beta}")

            # Build shield factories for this beta
            shield_factories = {}
            if "single_belief" in config.shields:
                shield_factories["single_belief"] = create_single_belief_shield_factory(
                    pomdp, pp_shield, beta
                )
            if "envelope" in config.shields:
                shield_factories["envelope"] = create_envelope_shield_factory(
                    ipomdp, pp_shield, beta
                )

            rl_wrapped = ShieldCompliantSelector(rl_selector, all_actions)

            for seed in config.seeds:
                for p_name, perception in perceptions.items():
                    for sh_name, sh_factory in shield_factories.items():
                        # Select the correct adversarial realization if multiple are available.
                        adv_target = ""
                        perception_model = perception
                        if p_name == "adversarial_opt" and isinstance(perception, dict):
                            if sh_name in perception:
                                perception_model = perception[sh_name]
                                adv_target = sh_name
                            elif "envelope" in perception:
                                perception_model = perception["envelope"]
                                adv_target = "envelope"
                            else:
                                perception_model = next(iter(perception.values()))
                                adv_target = "unknown"
                        trial_results = run_monte_carlo_trials(
                            ipomdp=ipomdp,
                            pp_shield=pp_shield,
                            perception=perception_model,
                            rt_shield_factory=sh_factory,
                            action_selector=rl_wrapped,
                            initial_generator=RandomInitialState(),
                            num_trials=config.num_trials,
                            trial_length=config.trial_length,
                            seed=seed,
                        )
                        metrics = compute_safety_metrics(trial_results)

                        row = {
                            "alpha": alpha,
                            "beta": beta,
                            "seed": seed,
                            "perception": p_name,
                            "adversarial_opt_target": adv_target,
                            "shield": sh_name,
                            "selector": "rl",
                            "fail_rate": metrics.fail_rate,
                            "stuck_rate": metrics.stuck_rate,
                            "safe_rate": metrics.safe_rate,
                            "mean_steps": metrics.mean_steps,
                            "num_trials": metrics.num_trials,
                        }
                        add_rate_cis(row, metrics.num_trials)
                        tidy_rows.append(row)

                        print(f"    {p_name}/{sh_name}/seed={seed}: "
                              f"fail={metrics.fail_rate:.1%} stuck={metrics.stuck_rate:.1%} "
                              f"safe={metrics.safe_rate:.1%}")

    total_time = time.time() - t0
    print(f"\nTotal sweep time: {total_time:.1f}s")

    # Save tidy CSV
    csv_path = os.path.join(config.results_dir, "results_tidy.csv")
    if tidy_rows:
        fieldnames = list(tidy_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tidy_rows)
        print(f"Tidy CSV saved to {csv_path}")

    # Aggregate across seeds for summary
    agg = _aggregate_sweep(tidy_rows)
    summary["aggregated"] = agg
    summary["total_time_s"] = total_time

    # Save summary JSON
    metadata = build_metadata(config, extra={"total_time_s": total_time})
    json_path = os.path.join(config.results_dir, "sweep_summary.json")
    save_experiment_results(json_path, summary, metadata)
    print(f"Summary saved to {json_path}")

    # Generate figures
    _plot_sweep(agg, config)

    return tidy_rows, summary


def _aggregate_sweep(tidy_rows):
    """Aggregate per-seed results into mean +/- std per (alpha, beta, perception, shield)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in tidy_rows:
        key = (row["alpha"], row["beta"], row["perception"], row["shield"])
        groups[key].append(row)

    agg = []
    for (alpha, beta, perc, shield), rows in sorted(groups.items()):
        fail_rates = [r["fail_rate"] for r in rows]
        stuck_rates = [r["stuck_rate"] for r in rows]
        safe_rates = [r["safe_rate"] for r in rows]
        mean_steps = [r["mean_steps"] for r in rows]
        agg.append({
            "alpha": alpha,
            "beta": beta,
            "perception": perc,
            "shield": shield,
            "fail_rate_mean": float(np.mean(fail_rates)),
            "fail_rate_std": float(np.std(fail_rates)),
            "stuck_rate_mean": float(np.mean(stuck_rates)),
            "stuck_rate_std": float(np.std(stuck_rates)),
            "safe_rate_mean": float(np.mean(safe_rates)),
            "safe_rate_std": float(np.std(safe_rates)),
            "mean_steps_mean": float(np.mean(mean_steps)),
            "mean_steps_std": float(np.std(mean_steps)),
            "n_seeds": len(rows),
        })
    return agg


def _plot_sweep(agg, config):
    """Generate heatmaps and Pareto plots from aggregated sweep results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print("matplotlib not available, skipping sweep plots")
        return

    figures_dir = os.path.join(config.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    alphas = sorted(config.alphas)
    betas = sorted(config.betas)

    for perc in config.perceptions:
        for shield in config.shields:
            subset = [r for r in agg if r["perception"] == perc and r["shield"] == shield]
            if not subset:
                continue

            # Build 2D grids
            fail_grid = np.full((len(alphas), len(betas)), np.nan)
            stuck_grid = np.full((len(alphas), len(betas)), np.nan)

            for r in subset:
                ai = alphas.index(r["alpha"])
                bi = betas.index(r["beta"])
                fail_grid[ai, bi] = r["fail_rate_mean"]
                stuck_grid[ai, bi] = r["stuck_rate_mean"]

            for metric, grid, cmap in [
                ("fail_rate", fail_grid, "Reds"),
                ("stuck_rate", stuck_grid, "Blues"),
            ]:
                fig, ax = plt.subplots(figsize=(8, 5))
                im = ax.imshow(grid, aspect="auto", origin="lower",
                               cmap=cmap, vmin=0, vmax=1)
                ax.set_xticks(range(len(betas)))
                ax.set_xticklabels([f"{b:.2f}" for b in betas])
                ax.set_yticks(range(len(alphas)))
                ax.set_yticklabels([f"{a:.2f}" for a in alphas])
                ax.set_xlabel("Beta (shield threshold)")
                ax.set_ylabel("Alpha (CI significance)")
                ax.set_title(f"{metric} — {perc} / {shield}")
                fig.colorbar(im, ax=ax)

                # Annotate cells
                for i in range(len(alphas)):
                    for j in range(len(betas)):
                        val = grid[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                    fontsize=8, color="white" if val > 0.5 else "black")

                fname = f"heatmap_{metric}_{perc}_{shield}.png"
                fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved {fname}")

        # Pareto scatter: fail_rate vs stuck_rate, color by beta, facet by alpha
        fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), squeeze=False)
        for ai, alpha in enumerate(alphas):
            ax = axes[0, ai]
            for shield in config.shields:
                subset = [r for r in agg
                          if r["perception"] == perc and r["shield"] == shield
                          and r["alpha"] == alpha]
                if not subset:
                    continue
                xs = [r["fail_rate_mean"] for r in subset]
                ys = [r["stuck_rate_mean"] for r in subset]
                colors = [r["beta"] for r in subset]
                sc = ax.scatter(xs, ys, c=colors, cmap="viridis",
                                label=shield, marker="o" if shield == "envelope" else "s",
                                edgecolors="black", linewidth=0.5, vmin=min(betas), vmax=max(betas))
            ax.set_xlabel("Fail Rate")
            ax.set_ylabel("Stuck Rate")
            ax.set_title(f"alpha={alpha}")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Pareto: Safety vs Liveness — {perc}", fontsize=12)
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="Beta", shrink=0.8)
        fname = f"pareto_{perc}.png"
        fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    print(f"All sweep figures saved to {figures_dir}")


def main():
    """Entry point for alpha/beta sweep."""
    config_module_name = "sweeps.rl_alpha_beta_sweep_taxinet"
    if len(sys.argv) > 1 and sys.argv[1].startswith("--config="):
        config_module_name = sys.argv[1].split("=", 1)[1]
    elif len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_module_name = sys.argv[2]

    try:
        mod = importlib.import_module(
            f".{config_module_name}", package="ipomdp_shielding.experiments"
        )
        config = mod.config
    except ImportError:
        print(f"Config module '{config_module_name}' not found, using defaults")
        config = AlphaBetaSweepConfig()

    run_alpha_beta_sweep(config)


if __name__ == "__main__":
    main()
