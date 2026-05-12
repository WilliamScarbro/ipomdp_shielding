"""Alpha sweep for RL shielding at fixed shield threshold beta.

A 1D analog of `rl_alpha_beta_sweep.py`: sweeps over the CI significance
`alpha` (which controls Clopper-Pearson interval width) while holding the
shield threshold `beta` fixed. Produces a tidy CSV and a Pareto scatter
(fail vs stuck) with one curve per shield, points labelled by alpha.

Usage:
    python -m ipomdp_shielding.experiments.sweeps.rl_alpha_sweep [--config CONFIG_MODULE]

Default config: sweeps.rl_alpha_sweep_taxinet
"""

import os
import sys
import csv
import time
import importlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable

import numpy as np

from ..experiment_io import build_metadata, add_rate_cis, save_experiment_results
from ..run_rl_shield_experiment import (
    create_envelope_shield_factory, create_single_belief_shield_factory,
    create_forward_sampling_shield_factory,
    ShieldCompliantSelector,
)
from ...MonteCarlo import (
    UniformPerceptionModel,
    FixedRealizationPerceptionModel,
    train_optimal_realization,
    RandomActionSelector,
    NeuralActionSelector,
    RandomInitialState,
    run_monte_carlo_trials,
    compute_safety_metrics,
    AdversarialPerceptionModel,
)


@dataclass
class AlphaSweepConfig:
    """Configuration for alpha sweep experiment at fixed beta."""

    case_study_name: str = "taxinet"
    build_ipomdp_fn: Callable = None

    # Sweep
    alphas: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    beta: float = 0.8

    # Per-grid-point evaluation
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    num_trials: int = 30
    trial_length: int = 20

    # RL training (done once per alpha)
    rl_episodes: int = 500
    rl_episode_length: int = 20
    opt_candidates: int = 20
    opt_trials_per_candidate: int = 10
    opt_iterations: int = 10
    adversarial_opt_targets: List[str] = field(default_factory=lambda: ["envelope"])

    # Shields / perceptions
    shields: List[str] = field(default_factory=lambda: ["single_belief", "envelope"])
    perceptions: List[str] = field(default_factory=lambda: ["uniform", "adversarial_opt"])

    # Forward-sampling shield hyperparameters (only used if "forward_sampling" in shields)
    fs_budget: int = 500
    fs_K_samples: int = 100

    # Output
    results_dir: str = "./data/sweep/rl_alpha"

    # Extra IPOMDP kwargs (besides alpha)
    ipomdp_base_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.build_ipomdp_fn is None:
            from ...CaseStudies.Taxinet import build_taxinet_ipomdp
            self.build_ipomdp_fn = build_taxinet_ipomdp


def run_alpha_sweep(config: AlphaSweepConfig):
    """Run the alpha sweep at fixed beta.

    For each alpha: build IPOMDP, train RL agent and adversarial realization
    once, then evaluate across all seeds / perceptions / shields at the single
    fixed beta.
    """
    os.makedirs(config.results_dir, exist_ok=True)
    tidy_rows = []
    summary = {"config": config.__dict__.copy(), "grid_results": {}}
    summary["config"]["build_ipomdp_fn"] = str(config.build_ipomdp_fn)

    total_points = len(config.alphas) * len(config.seeds)
    print("=" * 70)
    print(f"ALPHA SWEEP - {config.case_study_name.upper()}  (beta = {config.beta})")
    print(f"Alphas: {config.alphas}")
    print(f"Seeds:  {config.seeds}")
    print(f"Total grid points: {total_points}")
    print("=" * 70)

    t0 = time.time()

    for alpha in config.alphas:
        print(f"\n{'='*70}\nALPHA = {alpha}\n{'='*70}")

        ipomdp_kwargs = {**config.ipomdp_base_kwargs, "alpha": alpha}
        ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(**ipomdp_kwargs)
        all_actions = list(ipomdp.actions)
        pomdp = ipomdp.to_pomdp()

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

        # Adversarial realizations (optionally per shield target)
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
                rt_factory = create_envelope_shield_factory(ipomdp, pp_shield, config.beta)
            elif target == "single_belief":
                rt_factory = create_single_belief_shield_factory(pomdp, pp_shield, config.beta)
            else:
                raise ValueError(
                    f"Unknown adversarial_opt_target={target!r}. "
                    "Supported: 'envelope', 'single_belief'."
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

        perceptions: Dict[str, Any] = {}
        if "uniform" in config.perceptions:
            perceptions["uniform"] = UniformPerceptionModel()
        if "adversarial_opt" in config.perceptions:
            perceptions["adversarial_opt"] = opt_perceptions

        # Build shield factories at the fixed beta
        shield_factories = {}
        if "single_belief" in config.shields:
            shield_factories["single_belief"] = create_single_belief_shield_factory(
                pomdp, pp_shield, config.beta
            )
        if "envelope" in config.shields:
            shield_factories["envelope"] = create_envelope_shield_factory(
                ipomdp, pp_shield, config.beta
            )
        if "forward_sampling" in config.shields:
            shield_factories["forward_sampling"] = create_forward_sampling_shield_factory(
                ipomdp, pp_shield, config.beta,
                budget=config.fs_budget, K_samples=config.fs_K_samples,
            )

        rl_wrapped = ShieldCompliantSelector(rl_selector, all_actions)

        for seed in config.seeds:
            for p_name, perception in perceptions.items():
                for sh_name, sh_factory in shield_factories.items():
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
                        "beta": config.beta,
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

    csv_path = os.path.join(config.results_dir, "results_tidy.csv")
    if tidy_rows:
        fieldnames = list(tidy_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tidy_rows)
        print(f"Tidy CSV saved to {csv_path}")

    agg = _aggregate_sweep(tidy_rows)
    summary["aggregated"] = agg
    summary["total_time_s"] = total_time

    metadata = build_metadata(config, extra={"total_time_s": total_time})
    json_path = os.path.join(config.results_dir, "sweep_summary.json")
    save_experiment_results(json_path, summary, metadata)
    print(f"Summary saved to {json_path}")

    _plot_sweep(agg, config)

    return tidy_rows, summary


def _aggregate_sweep(tidy_rows):
    """Aggregate per-seed results into mean +/- std per (alpha, perception, shield)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in tidy_rows:
        key = (row["alpha"], row["perception"], row["shield"])
        groups[key].append(row)

    agg = []
    for (alpha, perc, shield), rows in sorted(groups.items()):
        fail_rates = [r["fail_rate"] for r in rows]
        stuck_rates = [r["stuck_rate"] for r in rows]
        safe_rates = [r["safe_rate"] for r in rows]
        mean_steps = [r["mean_steps"] for r in rows]
        agg.append({
            "alpha": alpha,
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


PERCEPTION_LABELS = {
    "uniform": "Uniform Random",
    "adversarial_opt": "Adversarial Optimized",
}

SHIELD_STYLES = {
    "single_belief":    {"color": "steelblue", "marker": "o", "label": "single_belief"},
    "envelope":         {"color": "seagreen",  "marker": "o", "label": "envelope"},
    "forward_sampling": {"color": "darkorange", "marker": "o", "label": "forward_sampling"},
}


def _plot_sweep(agg, config):
    """Pareto scatter: fail_rate vs stuck_rate, one curve per shield, points = alphas.

    Mirrors `plot_pareto_frontiers.py` style:
    one panel per perception, x = stuck rate, y = fail rate, alpha annotated.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping sweep plots")
        return

    figures_dir = os.path.join(config.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    alphas = sorted(config.alphas)

    perceptions_present = [p for p in config.perceptions
                           if any(r["perception"] == p for r in agg)]
    if not perceptions_present:
        print("  No aggregated rows to plot")
        return

    n = len(perceptions_present)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for col, perc in enumerate(perceptions_present):
        ax = axes[0, col]

        for shield in config.shields:
            subset = [r for r in agg
                      if r["perception"] == perc and r["shield"] == shield]
            if not subset:
                continue
            subset.sort(key=lambda r: r["alpha"])
            xs = [r["stuck_rate_mean"] for r in subset]
            ys = [r["fail_rate_mean"] for r in subset]
            ts = [r["alpha"] for r in subset]
            style = SHIELD_STYLES.get(shield, {"color": None, "marker": "o", "label": shield})
            color = style["color"]
            ax.plot(xs, ys, linestyle="-", marker=style["marker"], color=color,
                    linewidth=1.5, markersize=5, label=style["label"], zorder=3)
            # Annotate every other alpha + endpoints to reduce clutter
            for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
                if i % 2 == 0 or i == len(ts) - 1:
                    ax.annotate(f"{t:.2f}", (x, y),
                                textcoords="offset points", xytext=(4, 3),
                                fontsize=7, color=color)
            # Arrow: low->high alpha
            if len(xs) >= 2:
                ax.annotate(
                    "",
                    xy=(xs[0] + 0.25 * (xs[1] - xs[0]),
                        ys[0] + 0.25 * (ys[1] - ys[0])),
                    xytext=(xs[0], ys[0]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Stuck rate", fontsize=9)
        if col == 0:
            ax.set_ylabel("Fail rate", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(PERCEPTION_LABELS.get(perc, perc), fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Pareto Frontier — {config.case_study_name.upper()} "
        f"(RL selector, beta = {config.beta}, alpha ∈ {alphas[0]}–{alphas[-1]})",
        fontsize=11,
    )
    plt.tight_layout()
    fname = "pareto_alpha.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")
    print(f"All sweep figures saved to {figures_dir}")


def main():
    """Entry point for alpha sweep."""
    config_module_name = "sweeps.rl_alpha_sweep_taxinet"
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
        config = AlphaSweepConfig()

    run_alpha_sweep(config)


if __name__ == "__main__":
    main()
