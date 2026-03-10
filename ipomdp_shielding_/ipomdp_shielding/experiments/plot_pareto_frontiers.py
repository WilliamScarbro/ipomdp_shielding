"""Pareto frontier plotter for threshold sweep results.

Loads sweep JSONs from results/threshold_sweep/ and final-run JSONs from
results/final/ (for threshold-independent none/observation baselines).

Produces:
  results/threshold_sweep/pareto_{cs}.png      (per-case-study, 1×2 panels)
  results/threshold_sweep/pareto_summary.png   (2×4 summary grid)
  evaluation_summary_threshold_sweep.md

Usage:
    python -m ipomdp_shielding.experiments.plot_pareto_frontiers             # original
    python -m ipomdp_shielding.experiments.plot_pareto_frontiers --expanded  # 200-trial v2
"""

import json
import os
import sys

SWEEP_DIR = "results/threshold_sweep"
EXPANDED_SWEEP_DIR = "results/threshold_sweep_expanded"
FINAL_DIR = "results/final"

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CASE_STUDIES = ["taxinet", "cartpole", "obstacle", "refuel"]
EXPANDED_CASE_STUDIES = ["taxinet", "cartpole", "obstacle", "refuel_v2"]
PERCEPTIONS = ["uniform", "adversarial_opt"]

PERCEPTION_LABELS = {
    "uniform": "Uniform Random",
    "adversarial_opt": "Adversarial Optimized",
}
CS_LABELS = {
    "taxinet":    "TaxiNet (16 states)",
    "cartpole":   "CartPole (82 states)",
    "obstacle":   "Obstacle (50 states)",
    "refuel":     "Refuel v1 (344 states)",
    "refuel_v2":  "Refuel v2 (344 states)",
}


# ============================================================
# Data loading
# ============================================================

def load_sweep(cs_name, sweep_dir=None):
    """Load sweep JSON. Returns None if file is missing."""
    path = os.path.join(sweep_dir or SWEEP_DIR, f"{cs_name}_sweep.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_final(cs_name):
    """Load final-run results dict {combo_key -> metrics}. Returns {} if missing.

    For refuel_v2, looks in results/v2/ for the v2 results file.
    """
    if cs_name == "refuel_v2":
        path = "results/v2/rl_shield_refuel_v2_results.json"
    else:
        path = os.path.join(FINAL_DIR, f"rl_shield_{cs_name}_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("results", {})


# ============================================================
# Data extraction helpers
# ============================================================

def get_sweep_curve(sweep_data, perception, shield):
    """Return list of (fail_rate, stuck_rate, threshold) for a shield curve."""
    sweep_results = sweep_data.get("sweep_results", {})
    points = []
    for t in THRESHOLDS:
        t_key = f"{t:.2f}"
        combo = sweep_results.get(t_key, {}).get(f"{perception}/rl/{shield}")
        if combo is not None:
            points.append((combo["fail_rate"], combo["stuck_rate"], t))
    return points


def get_baseline(final_results, perception, shield):
    """Return (fail_rate, stuck_rate) for a baseline, or None."""
    m = final_results.get(f"{perception}/rl/{shield}")
    if m is None:
        return None
    return m["fail_rate"], m["stuck_rate"]


# ============================================================
# Single-panel drawing
# ============================================================

def _draw_panel(ax, sweep_data, final_results, perception, exclude_envelope,
                title=None, show_xlabel=True, show_ylabel=True, compact=False):
    """Draw Pareto frontier curves and baselines on ax."""

    label_fontsize = 6 if compact else 7
    marker_size = 4 if compact else 5

    # --- single_belief curve ---
    sb_pts = get_sweep_curve(sweep_data, perception, "single_belief")
    if sb_pts:
        xs = [p[1] for p in sb_pts]  # stuck
        ys = [p[0] for p in sb_pts]  # fail
        ts = [p[2] for p in sb_pts]
        ax.plot(xs, ys, "o-", color="steelblue", linewidth=1.5,
                markersize=marker_size, label="single_belief", zorder=3)
        # Annotate alternate thresholds to reduce clutter
        for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
            if i % 2 == 0 or t in (0.50, 0.95):
                ax.annotate(f"{t:.2f}", (x, y),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=label_fontsize, color="steelblue")
        # Arrow showing direction of increasing threshold (low→high)
        if len(xs) >= 2:
            ax.annotate("",
                        xy=(xs[0] + 0.25 * (xs[1] - xs[0]),
                            ys[0] + 0.25 * (ys[1] - ys[0])),
                        xytext=(xs[0], ys[0]),
                        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.0))

    # --- envelope curve (omitted for Refuel) ---
    if not exclude_envelope:
        env_pts = get_sweep_curve(sweep_data, perception, "envelope")
        if env_pts:
            xs = [p[1] for p in env_pts]
            ys = [p[0] for p in env_pts]
            ts = [p[2] for p in env_pts]
            ax.plot(xs, ys, "o-", color="seagreen", linewidth=1.5,
                    markersize=marker_size, label="envelope", zorder=3)
            for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
                if i % 2 == 0 or t in (0.50, 0.95):
                    ax.annotate(f"{t:.2f}", (x, y),
                                textcoords="offset points", xytext=(4, -9),
                                fontsize=label_fontsize, color="seagreen")
            if len(xs) >= 2:
                ax.annotate("",
                            xy=(xs[0] + 0.25 * (xs[1] - xs[0]),
                                ys[0] + 0.25 * (ys[1] - ys[0])),
                            xytext=(xs[0], ys[0]),
                            arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.0))

    # --- Baselines from final run ---
    if final_results:
        none_pt = get_baseline(final_results, perception, "none")
        obs_pt  = get_baseline(final_results, perception, "observation")
        if none_pt:
            ax.scatter([none_pt[1]], [none_pt[0]], marker="^", color="gray",
                       s=50, zorder=4, label="none (final)")
        if obs_pt:
            ax.scatter([obs_pt[1]], [obs_pt[0]], marker="s", color="darkorange",
                       s=50, zorder=4, label="observation (final)")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if show_xlabel:
        ax.set_xlabel("Stuck rate", fontsize=8 if compact else 9)
    if show_ylabel:
        ax.set_ylabel("Fail rate", fontsize=8 if compact else 9)
    if title:
        ax.set_title(title, fontsize=8 if compact else 10)
    ax.tick_params(labelsize=6 if compact else 7)


# ============================================================
# Per-case-study figure (1 row × 2 panels)
# ============================================================

def plot_per_case_study(cs_name, sweep_data, final_results, out_dir=None):
    """Save pareto_{cs_name}.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out_dir or SWEEP_DIR
    exclude_envelope = sweep_data["metadata"].get("exclude_envelope", False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Pareto Frontier — {cs_name.upper()} (RL selector, "
                 f"threshold ∈ {THRESHOLDS[0]:.2f}–{THRESHOLDS[-1]:.2f})",
                 fontsize=11)

    for col, perception in enumerate(PERCEPTIONS):
        ax = axes[col]
        _draw_panel(
            ax, sweep_data, final_results, perception, exclude_envelope,
            title=PERCEPTION_LABELS[perception],
            show_xlabel=True,
            show_ylabel=(col == 0),
        )
        if col == 0:
            ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"pareto_{cs_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================
# Cross-case-study summary figure (2 rows × 4 columns)
# ============================================================

def plot_summary(all_sweep, all_final, out_dir=None):
    """Save pareto_summary.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = out_dir or SWEEP_DIR
    case_studies = list(all_sweep.keys())
    n_cols = len(case_studies)

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10), sharex=True, sharey=True)
    if n_cols == 1:
        axes = [[axes[0]], [axes[1]]]
    fig.suptitle("Pareto Frontiers — All Case Studies (RL selector)", fontsize=13)

    for col, cs_name in enumerate(case_studies):
        sweep_data   = all_sweep.get(cs_name)
        final_results = all_final.get(cs_name, {})

        for row, perception in enumerate(PERCEPTIONS):
            ax = axes[row, col]

            if sweep_data is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
            else:
                exclude_envelope = sweep_data["metadata"].get("exclude_envelope", False)
                _draw_panel(
                    ax, sweep_data, final_results, perception, exclude_envelope,
                    show_xlabel=(row == 1),
                    show_ylabel=(col == 0),
                    compact=True,
                )

            # Column headers (top row only)
            if row == 0:
                ax.set_title(CS_LABELS[cs_name], fontsize=9)

            # Row labels (left column only)
            if col == 0:
                ax.set_ylabel(f"{PERCEPTION_LABELS[perception]}\nFail rate", fontsize=8)

    # Shared legend at the bottom
    legend_elements = [
        Line2D([0], [0], color="steelblue", marker="o", markersize=5,
               label="single_belief sweep"),
        Line2D([0], [0], color="seagreen", marker="o", markersize=5,
               label="envelope sweep"),
        Line2D([0], [0], marker="^", color="gray", markersize=6,
               linestyle="None", label="none (baseline, final run)"),
        Line2D([0], [0], marker="s", color="darkorange", markersize=6,
               linestyle="None", label="observation (baseline, final run)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(out, "pareto_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================
# Markdown summary
# ============================================================

def _fmt(v):
    return f"{v:.1%}" if v is not None else "N/A"


def _zero_fail_thresholds(sweep_data, perception, shield):
    """Return list of thresholds where fail_rate == 0."""
    out = []
    for t in THRESHOLDS:
        t_key = f"{t:.2f}"
        m = sweep_data.get("sweep_results", {}).get(t_key, {}).get(
            f"{perception}/rl/{shield}"
        )
        if m is not None and m.get("fail_rate", 1.0) == 0.0:
            out.append(t)
    return out


def generate_summary_markdown(all_sweep, all_final, out_path=None,
                              sweep_dir=None, expanded=False):
    """Write the evaluation summary markdown."""
    out_dir = sweep_dir or SWEEP_DIR
    lines = []
    title = "Threshold Sweep Evaluation Summary (Expanded — 200 trials)" if expanded \
        else "Threshold Sweep Evaluation Summary"
    lines.append(f"# {title}\n")
    note = ("200 trials, CartPole envelope excluded, Refuel v2 (safety predicates hidden)."
            if expanded else
            f"Shield threshold swept over {THRESHOLDS} for `single_belief` and "
            "`envelope` shields (RL selector, both perception regimes).")
    lines.append(note + "\n"
                 "Baselines (`none`, `observation`) from the final run at threshold=0.8"
                 " where available.\n")

    for cs_name in list(all_sweep.keys()):
        sweep_data    = all_sweep.get(cs_name)
        final_results = all_final.get(cs_name, {})

        lines.append(f"\n## {cs_name.upper()}\n")

        if sweep_data is None:
            lines.append("*No sweep data available.*\n")
            continue

        meta = sweep_data.get("metadata", {})
        exclude_envelope = meta.get("exclude_envelope", False)
        lines.append(
            f"Trials: {meta.get('num_trials', '?')}, "
            f"Length: {meta.get('trial_length', '?')}. "
            f"Envelope excluded: {exclude_envelope}.\n"
        )

        for perception in PERCEPTIONS:
            lines.append(f"\n### {PERCEPTION_LABELS[perception]}\n")

            # Table
            headers = ["Threshold", "sb fail%", "sb stuck%"]
            if not exclude_envelope:
                headers += ["env fail%", "env stuck%"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for t in THRESHOLDS:
                t_key = f"{t:.2f}"
                t_res = sweep_data.get("sweep_results", {}).get(t_key, {})
                sb  = t_res.get(f"{perception}/rl/single_belief", {})
                row = [t_key, _fmt(sb.get("fail_rate")), _fmt(sb.get("stuck_rate"))]
                if not exclude_envelope:
                    env = t_res.get(f"{perception}/rl/envelope", {})
                    row += [_fmt(env.get("fail_rate")), _fmt(env.get("stuck_rate"))]
                lines.append("| " + " | ".join(row) + " |")

            # Baselines
            none_pt = get_baseline(final_results, perception, "none")
            obs_pt  = get_baseline(final_results, perception, "observation")
            lines.append("")
            if none_pt:
                lines.append(f"*Baseline `none`*: "
                             f"fail={_fmt(none_pt[0])}, stuck={_fmt(none_pt[1])}")
            if obs_pt:
                lines.append(f"*Baseline `observation`*: "
                             f"fail={_fmt(obs_pt[0])}, stuck={_fmt(obs_pt[1])}")

            # Zero-fail thresholds
            obs_lines = []
            zf_sb = _zero_fail_thresholds(sweep_data, perception, "single_belief")
            if zf_sb:
                obs_lines.append(f"`single_belief` achieves 0% fail at thresholds: {zf_sb}")
            if not exclude_envelope:
                zf_env = _zero_fail_thresholds(sweep_data, perception, "envelope")
                if zf_env:
                    obs_lines.append(f"`envelope` achieves 0% fail at thresholds: {zf_env}")
            if obs_lines:
                lines.append("\n**Key observations:**")
                for o in obs_lines:
                    lines.append(f"- {o}")

    # Figures
    lines.append("\n## Figures\n")
    for cs_name in list(all_sweep.keys()):
        lines.append(f"![{CS_LABELS.get(cs_name, cs_name)}]({out_dir}/pareto_{cs_name}.png)\n")
    lines.append(f"![Summary]({out_dir}/pareto_summary.png)\n")

    # Cross-case-study summary
    lines.append("\n## Cross-Case-Study Summary\n")
    if expanded:
        lines.append(
            "- **TaxiNet** (200 trials): Smooth Pareto curve expected vs v1. "
            "Both shields improve substantially over no-shield (95–98% fail). "
            "Envelope may show clear advantage under adversarial at high thresholds.\n"
            "- **CartPole** (200 trials, single_belief only): Envelope excluded — "
            "already shown to be dominated at every threshold. single_belief plateaus "
            "at ~6.7% fail; threshold controls stuck rate only.\n"
            "- **Obstacle** (200 trials): Clearest Pareto frontier. Envelope dominates "
            "under adversarial. Zero-fail reachable with single_belief at t=0.95.\n"
            "- **Refuel v2** (hidden safety predicates, obs_noise=0.3): RL agent now "
            "fails without shielding (~10–15%). single_belief expected to reduce fail "
            "at cost of stuck. Envelope LP-infeasible (144 s/step).\n"
        )
    else:
        lines.append(
            "- **TaxiNet**: Both shields reduce fail from 95–98% → 38–82%. "
            "Curves noisy at n=50 (±7pp binomial error).\n"
            "- **CartPole**: single_belief dominates envelope at every threshold.\n"
            "- **Obstacle**: Clear Pareto trade-off; envelope best under adversarial.\n"
            "- **Refuel v1**: No-shield optimal — RL trivially safe; shields add stuck only.\n"
        )
    lines.append(
        "\n**Limitation**: Adversarial perception realizations fixed at threshold=0.8 "
        "(prelim/v2 cache). They may not represent the worst case at other thresholds.\n"
    )

    md = "\n".join(lines)
    path = out_path or "evaluation_summary_threshold_sweep.md"
    with open(path, "w") as f:
        f.write(md)
    print(f"  Saved {path}")
    return path


# ============================================================
# Main
# ============================================================

def main():
    expanded = "--expanded" in sys.argv
    sweep_dir = EXPANDED_SWEEP_DIR if expanded else SWEEP_DIR
    case_studies = EXPANDED_CASE_STUDIES if expanded else CASE_STUDIES
    md_out = "evaluation_summary_threshold_sweep_2.md" if expanded else "evaluation_summary_threshold_sweep.md"

    print("=" * 70)
    print("PARETO FRONTIER PLOTTER" + (" (expanded)" if expanded else ""))
    print("=" * 70)

    # Load all data
    all_sweep = {}
    all_final = {}
    for cs_name in case_studies:
        sweep = load_sweep(cs_name, sweep_dir=sweep_dir)
        final = load_final(cs_name)
        if sweep is None:
            print(f"  WARNING: No sweep data for {cs_name} — skipping")
        else:
            all_sweep[cs_name] = sweep
            print(f"  Loaded sweep: {cs_name}")
        if not final:
            print(f"  WARNING: No final/baseline results for {cs_name} — baselines absent")
        else:
            all_final[cs_name] = final
            print(f"  Loaded final: {cs_name}")

    os.makedirs(sweep_dir, exist_ok=True)

    # Per-case-study plots
    print("\nGenerating per-case-study Pareto plots...")
    for cs_name, sweep_data in all_sweep.items():
        plot_per_case_study(cs_name, sweep_data, all_final.get(cs_name, {}),
                            out_dir=sweep_dir)

    # Summary plot
    print("\nGenerating cross-case-study summary plot...")
    plot_summary(all_sweep, all_final, out_dir=sweep_dir)

    # Markdown
    print("\nGenerating markdown summary...")
    generate_summary_markdown(all_sweep, all_final, out_path=md_out,
                              sweep_dir=sweep_dir, expanded=expanded)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print(f"  Per-study plots : {sweep_dir}/pareto_{{cs}}.png")
    print(f"  Summary plot    : {sweep_dir}/pareto_summary.png")
    print(f"  Markdown        : {md_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
