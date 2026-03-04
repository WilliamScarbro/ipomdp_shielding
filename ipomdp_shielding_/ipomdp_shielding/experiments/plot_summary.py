"""Cross-case-study summary chart generator.

Loads RL shield experiment results from all four case studies (TaxiNet,
CartPole, Obstacle, Refuel) and generates:

1. Per-case-study detailed charts for Obstacle and Refuel (fail/stuck rate
   over time and intervention bar charts) — identical in format to those
   produced by run_rl_shield_experiment, but generated from saved JSON so
   results can be re-plotted without re-running experiments.

2. A cross-case-study summary comparison chart showing envelope-shield
   performance (fail rate and safe rate under RL selector) across all
   case studies for both uniform and adversarial perception.

Usage:
    python -m ipomdp_shielding.experiments.plot_summary [OPTIONS]

Options:
    --results-root DIR   Root directory containing prelim/ and full/
                         subdirectories (default: results)
    --output-dir DIR     Directory for generated figures (default: results/summary)
    --mode {prelim,full} Which results set to load (default: prelim)
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple


# ============================================================
# I/O helpers
# ============================================================

def load_results(path: str) -> Optional[Dict]:
    """Load a JSON results file, returning None if not found."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


def results_path(root: str, mode: str, case_study: str) -> str:
    return os.path.join(root, mode, f"rl_shield_{case_study}_results.json")


# ============================================================
# Data extraction
# ============================================================

CASE_STUDIES = ["taxinet", "cartpole", "obstacle", "refuel"]

SHIELD_ORDER = ["none", "observation", "single_belief", "envelope"]
SHIELD_LABELS = {
    "none": "No Shield",
    "observation": "Obs. Shield",
    "single_belief": "Single-Belief",
    "envelope": "Envelope",
}
SHIELD_COLORS = {
    "none": "gray",
    "observation": "orange",
    "single_belief": "steelblue",
    "envelope": "green",
}
PERCEPTION_LABELS = {
    "uniform": "Uniform",
    "adversarial_opt": "Adversarial Opt.",
}
SELECTOR_STYLES = {
    "rl":     ("-",  2.0, 1.0),
    "best":   ("--", 1.5, 0.8),
    "random": (":",  1.2, 0.5),
}


def get_metric(data: Dict, perception: str, selector: str, shield: str,
               metric: str) -> Optional[float]:
    """Extract a scalar metric from results dict."""
    key = f"{perception}/{selector}/{shield}"
    entry = data.get(key)
    if entry is None:
        return None
    return entry.get(metric)


# ============================================================
# Per-case-study charts (Obstacle / Refuel specific)
# ============================================================

def _plot_per_case_study(data: Dict, case_study: str, output_dir: str):
    """Generate fail/stuck rate-over-time charts for a single case study.

    These are identical in structure to those produced by
    run_rl_shield_experiment.plot_results, but built from the saved JSON
    summary rather than raw trial data (so no per-timestep curves are
    available — only terminal rates are shown as horizontal reference lines).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    for p_name in ["uniform", "adversarial_opt"]:
        for outcome in ["fail", "stuck"]:
            metric_key = f"{outcome}_rate"
            fig, ax = plt.subplots(figsize=(9, 5))
            has_data = False

            for sh_name in SHIELD_ORDER:
                color = SHIELD_COLORS[sh_name]
                for s_name, (ls, lw, alpha) in SELECTOR_STYLES.items():
                    val = get_metric(data, p_name, s_name, sh_name, metric_key)
                    if val is None:
                        continue
                    has_data = True
                    label = f"{s_name.upper()} + {SHIELD_LABELS[sh_name]}"
                    ax.axhline(val, label=label, color=color,
                               linestyle=ls, linewidth=lw, alpha=alpha)

            if not has_data:
                plt.close(fig)
                continue

            ax.set_xlabel("Shield / Selector combination")
            ax.set_ylabel(f"P({outcome})")
            ax.set_title(
                f"{outcome.capitalize()} Rate\n"
                f"Perception: {PERCEPTION_LABELS[p_name]} ({case_study.upper()})"
            )
            ax.legend(loc="best", fontsize=7)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            fname = f"{p_name}_{outcome}.png"
            fpath = os.path.join(output_dir, fname)
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved {fpath}")

        # Intervention rate bar chart (RL selector only)
        fig, ax = plt.subplots(figsize=(7, 4))
        shields_shown = []
        rates = []
        for sh_name in SHIELD_ORDER:
            # Intervention rate is stored under intervention_stats in setup_info;
            # it is not in the per-combination metrics.  Fall back gracefully.
            rate = None
            meta = data.get("_metadata", {})
            ist = meta.get("intervention_stats", {})
            key = f"{p_name}/rl/{sh_name}"
            if key in ist:
                rate = ist[key].get("intervention_rate")
            if rate is None:
                continue
            shields_shown.append(SHIELD_LABELS[sh_name])
            rates.append(rate)

        if shields_shown:
            colors = [SHIELD_COLORS[s] for s in SHIELD_ORDER
                      if SHIELD_LABELS[s] in shields_shown]
            bars = ax.bar(shields_shown, rates, color=colors)
            ax.set_ylabel("Intervention Rate")
            ax.set_title(
                f"RL Shield Intervention Rate\n"
                f"Perception: {PERCEPTION_LABELS[p_name]} ({case_study.upper()})"
            )
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{rate:.1%}", ha="center", fontsize=9)
            fname = f"{p_name}_intervention_rate.png"
            fpath = os.path.join(output_dir, fname)
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            print(f"    Saved {fpath}")
        plt.close(fig)


# ============================================================
# Cross-case-study summary chart
# ============================================================

def _plot_summary(all_data: Dict[str, Dict], output_dir: str):
    """Generate cross-case-study comparison charts.

    Creates a 2×2 grid of grouped bar charts:
      - Rows: uniform perception / adversarial-optimized perception
      - Columns: fail rate / safe rate
    Each chart has one group of bars per shield type, with one bar per
    case study, using the RL selector results.

    Also generates a separate envelope-focus chart showing envelope vs.
    no-shield fail rate across all case studies and perception regimes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    present = [cs for cs in CASE_STUDIES if all_data.get(cs) is not None]
    if not present:
        print("  No results available for summary chart.")
        return

    cs_colors = {
        "taxinet":  "#4C72B0",
        "cartpole": "#DD8452",
        "obstacle": "#55A868",
        "refuel":   "#C44E52",
    }
    cs_labels = {
        "taxinet":  "TaxiNet",
        "cartpole": "CartPole",
        "obstacle": "Obstacle",
        "refuel":   "Refuel",
    }

    perceptions = ["uniform", "adversarial_opt"]
    outcomes = ["fail_rate", "safe_rate"]
    outcome_labels = {"fail_rate": "P(fail)", "safe_rate": "P(safe)"}
    selector = "rl"

    # ---- 2×2 grouped bar chart ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle(
        "Shielding Performance Across Case Studies (RL Selector)",
        fontsize=13, fontweight="bold"
    )

    x = np.arange(len(SHIELD_ORDER))
    n_cs = len(present)
    total_width = 0.7
    bar_width = total_width / n_cs

    for row, p_name in enumerate(perceptions):
        for col, metric in enumerate(outcomes):
            ax = axes[row][col]
            for ci, cs in enumerate(present):
                data = all_data[cs]
                vals = []
                for sh in SHIELD_ORDER:
                    v = get_metric(data, p_name, selector, sh, metric)
                    vals.append(v if v is not None else 0.0)

                offset = (ci - n_cs / 2 + 0.5) * bar_width
                bars = ax.bar(
                    x + offset, vals, bar_width,
                    label=cs_labels[cs],
                    color=cs_colors[cs],
                    alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([SHIELD_LABELS[s] for s in SHIELD_ORDER],
                                fontsize=9)
            ax.set_ylabel(outcome_labels[metric], fontsize=10)
            ax.set_title(
                f"{outcome_labels[metric]} — {PERCEPTION_LABELS[p_name]}",
                fontsize=10
            )
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25, axis="y")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "summary_all_case_studies.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ---- Envelope focus: envelope vs. no-shield comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Envelope Shield vs No Shield — Fail Rate (RL Selector)",
        fontsize=12, fontweight="bold"
    )

    for col, p_name in enumerate(perceptions):
        ax = axes[col]
        x = np.arange(len(present))
        width = 0.35

        no_shield_vals = []
        envelope_vals = []
        for cs in present:
            data = all_data[cs]
            ns = get_metric(data, p_name, selector, "none", "fail_rate")
            env = get_metric(data, p_name, selector, "envelope", "fail_rate")
            no_shield_vals.append(ns if ns is not None else 0.0)
            envelope_vals.append(env if env is not None else 0.0)

        ax.bar(x - width / 2, no_shield_vals, width,
               label="No Shield", color=SHIELD_COLORS["none"], alpha=0.85)
        ax.bar(x + width / 2, envelope_vals, width,
               label="Envelope Shield", color=SHIELD_COLORS["envelope"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([cs_labels[cs] for cs in present], fontsize=10)
        ax.set_ylabel("P(fail)", fontsize=10)
        ax.set_title(f"Perception: {PERCEPTION_LABELS[p_name]}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(fontsize=9)

        # Annotate reduction
        for i, (ns, env) in enumerate(zip(no_shield_vals, envelope_vals)):
            if ns > 0:
                reduction = (ns - env) / ns
                ax.text(i, max(ns, env) + 0.03, f"{reduction:+.0%}",
                        ha="center", fontsize=8, color="darkgreen" if reduction > 0 else "red")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "summary_envelope_vs_none.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ---- Shield progression chart: fail rate by shield type ----
    # For each case study, show how fail rate changes as shield strength increases.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Fail Rate by Shield Strength — RL Selector",
        fontsize=12, fontweight="bold"
    )

    for col, p_name in enumerate(perceptions):
        ax = axes[col]
        for cs in present:
            data = all_data[cs]
            vals = []
            for sh in SHIELD_ORDER:
                v = get_metric(data, p_name, selector, sh, "fail_rate")
                vals.append(v if v is not None else float("nan"))
            ax.plot(
                range(len(SHIELD_ORDER)), vals,
                "o-", label=cs_labels[cs],
                color=cs_colors[cs], linewidth=2, markersize=6
            )

        ax.set_xticks(range(len(SHIELD_ORDER)))
        ax.set_xticklabels([SHIELD_LABELS[s] for s in SHIELD_ORDER], fontsize=9)
        ax.set_ylabel("P(fail)", fontsize=10)
        ax.set_title(f"Perception: {PERCEPTION_LABELS[p_name]}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "summary_shield_progression.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# Main
# ============================================================

def main():
    results_root = "results"
    output_dir = "results/summary"
    mode = "prelim"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--results-root" and i + 1 < len(args):
            results_root = args[i + 1]; i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]; i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]; i += 2
        else:
            i += 1

    print("=" * 70)
    print("SUMMARY CHART GENERATOR")
    print(f"Results root: {results_root}  Mode: {mode}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("ERROR: matplotlib is required. Install it with: pip install matplotlib")
        sys.exit(1)

    # Load results for all case studies
    all_data: Dict[str, Optional[Dict]] = {}
    for cs in CASE_STUDIES:
        path = results_path(results_root, mode, cs)
        print(f"\nLoading {cs}: {path}")
        all_data[cs] = load_results(path)

    # Per-case-study charts for Obstacle and Refuel
    for cs in ["obstacle", "refuel"]:
        data = all_data.get(cs)
        if data is None:
            print(f"\nSkipping per-case-study charts for {cs} (no results)")
            continue
        cs_fig_dir = os.path.join(output_dir, f"rl_shield_{cs}_figures")
        print(f"\nGenerating {cs} charts -> {cs_fig_dir}")
        _plot_per_case_study(data, cs, cs_fig_dir)

    # Cross-case-study summary charts
    print(f"\nGenerating summary charts -> {output_dir}")
    _plot_summary(all_data, output_dir)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
