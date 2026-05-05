"""TaxiNetV2 comparison: conformal prediction shielding vs. iPOMDP shielding.

Runs the RL shield experiment for TaxiNetV2 with an augmented shield grid
that includes ConformalPredictionShield alongside the standard iPOMDP shields
(none, observation, single_belief, envelope).  Both approaches are evaluated
via the same Monte Carlo protocol so results are directly comparable.

After running, generates a comparison figure that overlays the MC results
against the Scarbro PRISM baseline (vendored in results/taxinet_v2/).

Usage:
    python -m ipomdp_shielding.experiments.run_taxinet_v2_comparison [--plot-only]

Options:
    --plot-only    Skip the MC experiment; re-plot from saved results.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .experiment_io import build_metadata, add_rate_cis, save_experiment_results
from .run_rl_shield_experiment import (
    ConformalPredictionShield,
    ShieldCompliantSelector,
    create_no_shield_factory,
    create_observation_shield_factory,
    create_single_belief_shield_factory,
    create_envelope_shield_factory,
    create_conformal_shield_factory,
    run_experiment,
    print_results_table,
    save_results,
    plot_results,
    setup,
    compute_timestep_outcomes,
)
from ..CaseStudies.Taxinet.taxinet import taxinet_cte_states, taxinet_he_states
from ..MonteCarlo import (
    UniformPerceptionModel,
    RandomActionSelector,
    BeliefSelector,
)


# ============================================================
# Paths
# ============================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scarbro_path() -> Path:
    return _project_root() / "results" / "taxinet_v2" / "scarbro_baseline_import.json"


# ============================================================
# Augmented grid
# ============================================================

def build_comparison_grid(ipomdp, pp_shield, rl_selector, optimized_perceptions, config):
    """Build an experiment grid that adds ConformalPredictionShield.

    Extends the standard 2×3×4 grid (perception × selector × shield) with a
    fifth shield type, yielding 2×3×5 = 30 combinations.
    """
    all_actions = list(ipomdp.actions)
    pomdp = ipomdp.to_pomdp()

    uniform_perception = UniformPerceptionModel()

    def perception_for(p_name: str, sh_name: str):
        if p_name == "uniform":
            return uniform_perception
        if p_name != "adversarial_opt":
            raise ValueError(f"Unknown perception regime: {p_name!r}")
        if sh_name in optimized_perceptions:
            return optimized_perceptions[sh_name]
        if "envelope" in optimized_perceptions:
            return optimized_perceptions["envelope"]
        return next(iter(optimized_perceptions.values()))

    selectors = {
        "random": RandomActionSelector(),
        "best": BeliefSelector(mode="best"),
        "rl": ShieldCompliantSelector(rl_selector, all_actions),
    }

    cte_st = taxinet_cte_states()
    he_st = taxinet_he_states()

    shields = {
        "none": create_no_shield_factory(all_actions),
        "observation": create_observation_shield_factory(ipomdp, pp_shield, config.shield_threshold),
        "single_belief": create_single_belief_shield_factory(pomdp, pp_shield, config.shield_threshold),
        "envelope": create_envelope_shield_factory(ipomdp, pp_shield, config.shield_threshold),
        "conformal_prediction": create_conformal_shield_factory(
            pp_shield, all_actions, cte_st, he_st
        ),
    }

    grid = []
    for p_name in ["uniform", "adversarial_opt"]:
        for s_name, selector in selectors.items():
            for sh_name, shield_factory in shields.items():
                grid.append((
                    p_name, s_name, sh_name,
                    perception_for(p_name, sh_name),
                    selector,
                    shield_factory,
                ))

    return grid


# ============================================================
# Scarbro baseline loader
# ============================================================

def load_scarbro_baseline() -> Optional[Dict]:
    path = _scarbro_path()
    if not path.exists():
        print(f"  WARNING: Scarbro baseline not found at {path}")
        return None
    with path.open() as fh:
        return json.load(fh)


def extract_scarbro_tradeoff_points(
    scarbro: Dict,
    confidence_level: str = "0.95",
    default_action_only: bool = True,
) -> List[Dict]:
    """Extract scalar (crash, stuck) comparison points from Scarbro baseline.

    Returns one point per action-filter variant at the given confidence level.
    """
    points = []
    for v in scarbro.get("variants", []):
        m = v["metadata"]
        s = v["summary"]
        if m["confidence_level"] != confidence_level:
            continue
        if default_action_only and not m["default_action"]:
            continue
        crash = (s.get("crash") or {}).get("value")
        stuck = (s.get("stuck_or_default") or {}).get("value")
        if crash is None or stuck is None:
            continue
        points.append({
            "label": m["action_filter_tag"],
            "confidence_level": m["confidence_level"],
            "crash": crash,
            "stuck": stuck,
        })
    return sorted(points, key=lambda p: p["stuck"])


# ============================================================
# Comparison figure
# ============================================================

_SHIELD_ORDER = ["none", "observation", "single_belief", "envelope", "conformal_prediction"]
_SHIELD_COLORS = {
    "none": "gray",
    "observation": "darkorange",
    "single_belief": "steelblue",
    "envelope": "green",
    "conformal_prediction": "crimson",
}
_SHIELD_LABELS = {
    "none": "No Shield",
    "observation": "Obs. Shield",
    "single_belief": "Single-Belief",
    "envelope": "Envelope",
    "conformal_prediction": "Conf. Pred.",
}
_SHIELD_MARKERS = {
    "none": "X",
    "observation": "s",
    "single_belief": "D",
    "envelope": "o",
    "conformal_prediction": "^",
}


def plot_comparison(
    our_results: Dict,
    scarbro: Optional[Dict],
    output_dir: str,
    confidence_level: str = "0.95",
    selector: str = "rl",
):
    """Generate comparison scatter plot: safety vs. conservatism.

    X-axis: stuck/default rate (conservatism).
    Y-axis: fail/crash rate (safety risk).
    Lower-left is better.

    One panel per perception regime.  The Scarbro PRISM points are shown only
    on the uniform-perception panel (PRISM is a worst-case formal bound, not
    tied to a specific perception strategy).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available, skipping comparison plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    scarbro_points = []
    if scarbro is not None:
        scarbro_points = extract_scarbro_tradeoff_points(scarbro, confidence_level)

    for p_name in ["uniform", "adversarial_opt"]:
        fig, ax = plt.subplots(figsize=(7, 6))

        # --- our MC results ---
        for sh_name in _SHIELD_ORDER:
            key = f"{p_name}/{selector}/{sh_name}"
            r = our_results.get(key)
            if r is None:
                continue
            ax.scatter(
                r["stuck_rate"], r["fail_rate"],
                s=140, zorder=5,
                color=_SHIELD_COLORS[sh_name],
                marker=_SHIELD_MARKERS[sh_name],
                label=f"iPOMDP: {_SHIELD_LABELS[sh_name]}",
            )
            # 95 % CI error bars on fail_rate
            if "fail_rate_ci_low" in r:
                ax.errorbar(
                    r["stuck_rate"], r["fail_rate"],
                    yerr=[[r["fail_rate"] - r["fail_rate_ci_low"]],
                          [r["fail_rate_ci_high"] - r["fail_rate"]]],
                    fmt="none",
                    ecolor=_SHIELD_COLORS[sh_name],
                    capsize=3, alpha=0.6,
                )

        # --- Scarbro PRISM baseline (uniform panel only) ---
        if p_name == "uniform" and scarbro_points:
            sx = [p["stuck"] for p in scarbro_points]
            sy = [p["crash"] for p in scarbro_points]
            ax.scatter(sx, sy, s=90, zorder=4, color="purple", marker="P",
                       label="PRISM (Scarbro) conf=0.95")
            ax.plot(sx, sy, color="purple", linestyle="--", alpha=0.35, linewidth=1.2)
            for pt in scarbro_points:
                ax.annotate(
                    pt["label"],
                    (pt["stuck"], pt["crash"]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color="purple",
                )

        p_label = "Uniform Perception" if p_name == "uniform" else "Adversarial Perception"
        ax.set_xlabel("Stuck Rate  (conservatism →)", fontsize=11)
        ax.set_ylabel("Fail/Crash Rate  (← safety)", fontsize=11)
        ax.set_title(
            f"TaxiNetV2  |  conf={confidence_level}  |  {p_label}\n"
            f"Safety–Liveness Tradeoff  (MC, {selector.upper()} selector)",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)

        fname = f"comparison_{p_name}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # --- summary table to stdout ---
    print("\nComparison table  (RL selector, conf=0.95):")
    print(f"  {'Shield':<22} {'Perception':<18} {'Fail%':>7} {'Stuck%':>7} {'Safe%':>7}")
    print("  " + "-" * 65)
    for p_name in ["uniform", "adversarial_opt"]:
        for sh_name in _SHIELD_ORDER:
            key = f"{p_name}/rl/{sh_name}"
            r = our_results.get(key)
            if r is None:
                continue
            print(f"  {sh_name:<22} {p_name:<18} "
                  f"{r['fail_rate']:>6.1%} {r['stuck_rate']:>6.1%} {r['safe_rate']:>6.1%}")
    if scarbro_points:
        print("\n  Scarbro PRISM (conf=0.95, worst-case formal bounds):")
        print(f"  {'Variant':<22} {'crash%':>7} {'stuck%':>7}")
        print("  " + "-" * 40)
        for pt in scarbro_points:
            print(f"  {pt['label']:<22} {pt['crash']:>6.1%} {pt['stuck']:>6.1%}")


# ============================================================
# Main experiment runner
# ============================================================

def run(config, skip_run: bool = False):
    """Run the TaxiNetV2 comparison experiment.

    Parameters
    ----------
    config : RLShieldExperimentConfig
    skip_run : bool
        If True, skip the MC experiment and only re-plot from saved results.
    """
    results_path = config.results_path
    figures_dir = config.figures_dir
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if not skip_run:
        print("=" * 70)
        print(f"TAXINET V2 COMPARISON EXPERIMENT")
        print(f"Trials: {config.num_trials}, Length: {config.trial_length}, Seed: {config.seed}")
        print(f"Shield threshold: {config.shield_threshold}")
        print("=" * 70)

        print(f"\nLoading TaxiNetV2 IPOMDP (conf={config.ipomdp_kwargs.get('confidence_level', '?')})...")
        ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(**config.ipomdp_kwargs)
        print(f"  States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
              f"Observations: {len(ipomdp.observations)}")

        rl_selector, optimized_perceptions, setup_info = setup(ipomdp, pp_shield, config)

        grid = build_comparison_grid(ipomdp, pp_shield, rl_selector, optimized_perceptions, config)
        print(f"\nExperiment grid: {len(grid)} combinations "
              f"(2 perceptions × 3 selectors × 5 shields)")

        t0 = time.time()
        results, trial_data, intervention_stats = run_experiment(ipomdp, pp_shield, grid, config)
        total_time = time.time() - t0

        print_results_table(results, config)

        extra = {
            **setup_info,
            "total_time_s": total_time,
            "intervention_stats": {
                f"{k[0]}/{k[1]}/{k[2]}": v for k, v in intervention_stats.items()
            },
            "note": (
                "Augmented grid: adds ConformalPredictionShield to the standard "
                "4-shield set for direct comparison with Scarbro PRISM baseline."
            ),
        }
        save_results(results, config, setup_info=extra)

        print("\nGenerating per-shield time-series figures...")
        plot_results(trial_data, config, intervention_stats=intervention_stats)

        print(f"\nTotal experiment time: {total_time:.1f}s")

    # Load saved results for comparison plot
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}; cannot plot comparison.")
        return

    with open(results_path) as fh:
        saved = json.load(fh)
    our_results = saved.get("results", {})

    scarbro = load_scarbro_baseline()
    conf_level = config.ipomdp_kwargs.get("confidence_level", "0.95")

    print("\nGenerating comparison figure...")
    plot_comparison(our_results, scarbro, figures_dir, confidence_level=conf_level)

    print(f"\nAll outputs written to: {figures_dir}")
    print("=" * 70)
    print("TAXINET V2 COMPARISON COMPLETE")
    print("=" * 70)


def main():
    import importlib
    skip_run = "--plot-only" in sys.argv

    # Allow overriding the config module via --config <name>.
    # Default: rl_shield_taxinet_v2_comparison (conf=0.95)
    config_name = "rl_shield_taxinet_v2_comparison"
    args = sys.argv[1:]
    if "--config" in args:
        idx = args.index("--config")
        config_name = args[idx + 1]

    config_module = importlib.import_module(
        f".configs.{config_name}",
        package="ipomdp_shielding.experiments",
    )
    config = config_module.config

    run(config, skip_run=skip_run)


if __name__ == "__main__":
    main()
