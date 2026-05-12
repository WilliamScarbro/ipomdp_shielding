"""Threshold sweep v3 plotter and summary generator.

Produces evaluation_summary_threshold_sweep_3.md with:
  - Pareto frontier curves (TaxiNet, Obstacle) + Carr point overlay
  - Method comparison tables (CartPole, Refuel v2) using best threshold per method

Reads:
  results/threshold_sweep_expanded/{cs}_sweep.json   (from run_threshold_sweep --expanded)
  results/threshold_sweep_expanded/{cs}_carr_results.json  (from run_carr_all_case_studies)
  results/final/rl_shield_{cs}_results.json           (baselines: none / observation)

Writes:
  results/threshold_sweep_expanded/pareto_v3_{cs}.png   (TaxiNet, Obstacle only)
  results/threshold_sweep_expanded/summary_v3.png        (2-panel summary for Pareto cases)
  results/threshold_sweep_expanded/evaluation_summary_v3.md

Usage:
    python -m ipomdp_shielding.experiments.plot_sweep_v3
"""

import json
import os
import sys

SWEEP_DIR = "results/threshold_sweep_expanded"
FINAL_DIR = "results/final"
MD_OUT = os.path.join(SWEEP_DIR, "evaluation_summary_v3.md")

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Case studies that get Pareto plots vs comparison tables.
PARETO_CASES = ["taxinet", "obstacle"]
TABLE_CASES  = ["cartpole", "refuel_v2"]

CS_LABELS = {
    "taxinet":   "TaxiNet (16 states, 16 obs)",
    "cartpole":  "CartPole (82 states, 82 obs)",
    "obstacle":  "Obstacle (50 states, 3 obs)",
    "refuel_v2": "Refuel v2 (344 states, 29 obs)",
}
PERCEPTION_LABELS = {
    "uniform":        "Uniform",
    "adversarial_opt": "Adversarial",
}


# ============================================================
# Data loading helpers
# ============================================================

def load_sweep(cs_name):
    path = os.path.join(SWEEP_DIR, f"{cs_name}_sweep.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_carr(cs_name):
    path = os.path.join(SWEEP_DIR, f"{cs_name}_carr_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # Return just the results sub-dict for convenience.
    if data.get("status") != "ok":
        return None
    return data.get("results", {})


def load_final(cs_name):
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
    """Return [(fail_rate, stuck_rate, threshold), ...] for given shield curve."""
    sr = sweep_data.get("sweep_results", {})
    points = []
    for t in THRESHOLDS:
        t_key = f"{t:.2f}"
        combo = sr.get(t_key, {}).get(f"{perception}/rl/{shield}")
        if combo is not None:
            points.append((combo["fail_rate"], combo["stuck_rate"], t))
    return points


def get_baseline(final_results, perception, shield):
    """Return (fail_rate, stuck_rate) or None."""
    m = final_results.get(f"{perception}/rl/{shield}")
    if m is None:
        return None
    return m["fail_rate"], m["stuck_rate"]


def get_carr_point(carr_results, perception):
    """Return (fail_rate, stuck_rate) for Carr, or None."""
    if carr_results is None:
        return None
    m = carr_results.get(f"{perception}/rl/carr")
    if m is None:
        return None
    return m["fail_rate"], m["stuck_rate"]


def best_threshold_row(sweep_data, perception, shield):
    """Return the row (threshold, fail, stuck) with the minimum fail_rate.

    Ties broken by minimum stuck_rate.
    Returns None if no data.
    """
    pts = get_sweep_curve(sweep_data, perception, shield)
    if not pts:
        return None
    return min(pts, key=lambda p: (p[0], p[1]))


# ============================================================
# Formatting helpers
# ============================================================

def _fmt(v, pct=True):
    if v is None:
        return "N/A"
    if pct:
        return f"{v:.0%}"
    return f"{v:.2f}"


# ============================================================
# Pareto plots (TaxiNet, Obstacle)
# ============================================================

def _draw_pareto_panel(ax, sweep_data, final_results, carr_results,
                       perception, exclude_envelope,
                       title=None, show_xlabel=True, show_ylabel=True,
                       compact=False):
    """Draw Pareto frontier curves + Carr point + baselines on ax."""
    lfs = 6 if compact else 7
    ms = 4 if compact else 5

    # single_belief sweep curve.
    sb_pts = get_sweep_curve(sweep_data, perception, "single_belief")
    if sb_pts:
        xs = [p[1] for p in sb_pts]
        ys = [p[0] for p in sb_pts]
        ts = [p[2] for p in sb_pts]
        ax.plot(xs, ys, "o-", color="steelblue", lw=1.5,
                markersize=ms, label="single_belief", zorder=3)
        for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
            if i % 2 == 0 or t in (0.50, 0.95):
                ax.annotate(f"{t:.2f}", (x, y),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=lfs, color="steelblue")
        if len(xs) >= 2:
            ax.annotate("",
                        xy=(xs[0] + 0.25*(xs[1]-xs[0]),
                            ys[0] + 0.25*(ys[1]-ys[0])),
                        xytext=(xs[0], ys[0]),
                        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.0))

    # envelope sweep curve (omitted when excluded).
    if not exclude_envelope:
        env_pts = get_sweep_curve(sweep_data, perception, "envelope")
        if env_pts:
            xs = [p[1] for p in env_pts]
            ys = [p[0] for p in env_pts]
            ts = [p[2] for p in env_pts]
            ax.plot(xs, ys, "o-", color="seagreen", lw=1.5,
                    markersize=ms, label="envelope", zorder=3)
            for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
                if i % 2 == 0 or t in (0.50, 0.95):
                    ax.annotate(f"{t:.2f}", (x, y),
                                textcoords="offset points", xytext=(4, -9),
                                fontsize=lfs, color="seagreen")
            if len(xs) >= 2:
                ax.annotate("",
                            xy=(xs[0] + 0.25*(xs[1]-xs[0]),
                                ys[0] + 0.25*(ys[1]-ys[0])),
                            xytext=(xs[0], ys[0]),
                            arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.0))

    # Carr single point.
    carr_pt = get_carr_point(carr_results, perception)
    if carr_pt is not None:
        ax.scatter([carr_pt[1]], [carr_pt[0]],
                   marker="D", color="crimson", s=70, zorder=5,
                   label="Carr (support-based)")
        ax.annotate("Carr", (carr_pt[1], carr_pt[0]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=lfs, color="crimson", fontweight="bold")

    # Baselines from final run.
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


def plot_pareto_per_case_study(cs_name, sweep_data, final_results, carr_results):
    """Save pareto_v3_{cs}.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exclude_envelope = sweep_data["metadata"].get("exclude_envelope", False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Pareto Frontier — {CS_LABELS[cs_name]}\n"
        f"(RL selector, threshold ∈ {THRESHOLDS[0]:.2f}–{THRESHOLDS[-1]:.2f},"
        f" 200 trials; Carr shown as single ◆)",
        fontsize=10
    )

    for col, perception in enumerate(["uniform", "adversarial_opt"]):
        ax = axes[col]
        _draw_pareto_panel(
            ax, sweep_data, final_results, carr_results,
            perception, exclude_envelope,
            title=PERCEPTION_LABELS[perception],
            show_xlabel=True,
            show_ylabel=(col == 0),
        )
        if col == 0:
            ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    os.makedirs(SWEEP_DIR, exist_ok=True)
    path = os.path.join(SWEEP_DIR, f"pareto_v3_{cs_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_pareto_summary(pareto_data):
    """2×2 summary for TaxiNet and Obstacle (uniform | adversarial rows)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cases = list(pareto_data.keys())
    n_cols = len(cases)

    fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 10),
                             sharex=True, sharey=True)
    if n_cols == 1:
        axes = [[axes[0]], [axes[1]]]
    fig.suptitle("Pareto Frontiers — TaxiNet & Obstacle (RL selector, 200 trials)",
                 fontsize=12)

    perceptions = ["uniform", "adversarial_opt"]
    for col, cs_name in enumerate(cases):
        sweep_data, final_results, carr_results = pareto_data[cs_name]
        exclude_envelope = sweep_data["metadata"].get("exclude_envelope", False)

        for row, perception in enumerate(perceptions):
            ax = axes[row][col]
            _draw_pareto_panel(
                ax, sweep_data, final_results, carr_results,
                perception, exclude_envelope,
                show_xlabel=(row == 1),
                show_ylabel=(col == 0),
                compact=True,
            )
            if row == 0:
                ax.set_title(CS_LABELS[cs_name], fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    f"{PERCEPTION_LABELS[perception]}\nFail rate", fontsize=8
                )

    legend_elements = [
        Line2D([0], [0], color="steelblue", marker="o", markersize=5,
               label="single_belief sweep"),
        Line2D([0], [0], color="seagreen", marker="o", markersize=5,
               label="envelope sweep"),
        Line2D([0], [0], marker="D", color="crimson", markersize=7,
               linestyle="None", label="Carr (support-based)"),
        Line2D([0], [0], marker="^", color="gray", markersize=6,
               linestyle="None", label="none (baseline)"),
        Line2D([0], [0], marker="s", color="darkorange", markersize=6,
               linestyle="None", label="observation (baseline)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(SWEEP_DIR, "summary_v3.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================
# Comparison tables (CartPole, Refuel v2)
# ============================================================

def _build_comparison_rows(cs_name, sweep_data, final_results, carr_results,
                            exclude_envelope):
    """Build method comparison rows for table case studies.

    Each row: (method_label, fail_unif, stuck_unif, fail_adv, stuck_adv, note)
    """
    rows = []

    def _get_best(shield):
        """Return (label, fail_u, stuck_u, fail_a, stuck_a, note)."""
        pts_u = get_sweep_curve(sweep_data, "uniform", shield)
        pts_a = get_sweep_curve(sweep_data, "adversarial_opt", shield)
        if not pts_u:
            return None
        best_u = min(pts_u, key=lambda p: (p[0], p[1]))
        best_a = min(pts_a, key=lambda p: (p[0], p[1])) if pts_a else None
        t_u = best_u[2]
        t_a = best_a[2] if best_a else None
        note = f"t={t_u:.2f}"
        if t_a is not None and t_a != t_u:
            note += f" (unif) / t={t_a:.2f} (adv)"
        return (
            shield,
            best_u[0], best_u[1],
            best_a[0] if best_a else None,
            best_a[1] if best_a else None,
            note,
        )

    # IPOMDP shields from sweep.
    sb_row = _get_best("single_belief")
    if sb_row:
        rows.append(sb_row)

    if not exclude_envelope:
        env_row = _get_best("envelope")
        if env_row:
            rows.append(env_row)

    # Carr.
    if carr_results is not None:
        cu = get_carr_point(carr_results, "uniform")
        ca = get_carr_point(carr_results, "adversarial_opt")
        rows.append((
            "carr",
            cu[0] if cu else None, cu[1] if cu else None,
            ca[0] if ca else None, ca[1] if ca else None,
            "no threshold",
        ))

    # Baselines (threshold-independent, from final run).
    for shield_name in ("none", "observation"):
        nu = get_baseline(final_results, "uniform", shield_name)
        na = get_baseline(final_results, "adversarial_opt", shield_name)
        rows.append((
            shield_name,
            nu[0] if nu else None, nu[1] if nu else None,
            na[0] if na else None, na[1] if na else None,
            "baseline (final run, t=0.8)",
        ))

    return rows


# ============================================================
# Markdown generation
# ============================================================

def generate_markdown(all_data, carr_availability, out_path=MD_OUT):
    """Write results/threshold_sweep_expanded/evaluation_summary_v3.md."""
    lines = []

    lines.append("# Threshold Sweep Evaluation Summary — v3\n")
    lines.append(
        "**Source data**: 200-trial expanded sweep (this directory).\n"
        "**Carr shield**: support-based (Carr et al.) — no threshold parameter;\n"
        "  built from `SupportMDPBuilder` using midpoint-realization POMDP from each IPOMDP.\n"
        "\n"
        "**Presentation strategy**:\n"
        "- **TaxiNet, Obstacle** — Pareto frontier plots (fail rate vs stuck rate across\n"
        "  thresholds) with Carr shown as a single point (◆).\n"
        "- **CartPole, Refuel v2** — Comparison tables using the *best* threshold per\n"
        "  method (minimising fail rate, then stuck rate) alongside Carr and baselines.\n"
    )

    # ── TaxiNet ────────────────────────────────────────────────────────────────
    lines.append("\n---\n\n## TaxiNet (16 states, 16 obs)\n")
    lines.append("**200 trials × 20 steps.**\n")
    lines.append(
        "![TaxiNet Pareto v3](pareto_v3_taxinet.png)\n"
    )
    cs = "taxinet"
    sd = all_data[cs]["sweep"]
    fr = all_data[cs]["final"]
    cr = all_data[cs]["carr"]

    lines.append("### Threshold sweep table (RL selector)\n")
    lines.append("| Threshold | sb fail% (unif) | sb stuck% | env fail% (unif) | env stuck% | sb fail% (adv) | env fail% (adv) |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in THRESHOLDS:
        t_key = f"{t:.2f}"
        tr = sd.get("sweep_results", {}).get(t_key, {})
        sb  = tr.get("uniform/rl/single_belief", {})
        env = tr.get("uniform/rl/envelope", {})
        sba = tr.get("adversarial_opt/rl/single_belief", {})
        enva = tr.get("adversarial_opt/rl/envelope", {})
        lines.append(
            f"| {t_key}"
            f" | {_fmt(sb.get('fail_rate'))}"
            f" | {_fmt(sb.get('stuck_rate'))}"
            f" | {_fmt(env.get('fail_rate'))}"
            f" | {_fmt(env.get('stuck_rate'))}"
            f" | {_fmt(sba.get('fail_rate'))}"
            f" | {_fmt(enva.get('fail_rate'))}"
            f" |"
        )

    # Carr + baselines for TaxiNet
    lines.append("")
    carr_u = get_carr_point(cr, "uniform")
    carr_a = get_carr_point(cr, "adversarial_opt")
    none_u = get_baseline(fr, "uniform", "none")
    none_a = get_baseline(fr, "adversarial_opt", "none")

    if carr_u or carr_a:
        lines.append(
            f"*Carr*: fail={_fmt(carr_u[0] if carr_u else None)} / "
            f"stuck={_fmt(carr_u[1] if carr_u else None)} (uniform); "
            f"fail={_fmt(carr_a[0] if carr_a else None)} / "
            f"stuck={_fmt(carr_a[1] if carr_a else None)} (adversarial)"
        )
    else:
        lines.append("*Carr*: results not available")

    if none_u:
        lines.append(
            f"*Baseline `none`*: fail={_fmt(none_u[0])} (uniform), "
            f"fail={_fmt(none_a[0]) if none_a else 'N/A'} (adversarial)"
        )

    cs_carr_u = get_carr_point(all_data["taxinet"]["carr"], "uniform")
    cs_carr_a = get_carr_point(all_data["taxinet"]["carr"], "adversarial_opt")
    carr_taxinet_note = (
        f"Carr achieves {_fmt(cs_carr_u[0] if cs_carr_u else None)} fail / "
        f"{_fmt(cs_carr_u[1] if cs_carr_u else None)} stuck (uniform) — the "
        f"support-MDP has **0 winning supports**, meaning no support reachable "
        f"from the initial safe-state prior has a guaranteed safe action under "
        f"the midpoint POMDP. Carr therefore blocks all actions from step 0, "
        f"and the observed {_fmt(cs_carr_u[0] if cs_carr_u else None)} fail comes "
        f"entirely from trials that randomly start in the FAIL state before the "
        f"shield is consulted."
        if cs_carr_u and cs_carr_u[1] > 0.9 else
        "Carr results shown on Pareto plot."
    )
    lines.append("\n### Key findings\n")
    lines.append(
        "With 200 trials the monotone trend is clearly visible. `envelope` dominates\n"
        "`single_belief` at every threshold above 0.80. At t=0.95:\n"
        "- `envelope`: 35% fail / 34% stuck (uniform); 34% fail / 36% stuck (adversarial)\n"
        "- `single_belief`: 44% fail / 11% stuck (uniform); 43% fail / 8% stuck (adversarial)\n"
        "\n"
        f"**Carr**: {carr_taxinet_note}\n"
        "Both IPOMDP shields reduce fail from 95–98% (no-shield) to 34–44% at the\n"
        "best threshold. Carr's probability-free conservatism prevents it from competing.\n"
    )

    # ── Obstacle ──────────────────────────────────────────────────────────────
    lines.append("\n---\n\n## Obstacle (50 states, 3 obs)\n")
    lines.append("**200 trials × 25 steps.**\n")
    lines.append(
        "![Obstacle Pareto v3](pareto_v3_obstacle.png)\n"
    )
    cs = "obstacle"
    sd = all_data[cs]["sweep"]
    fr = all_data[cs]["final"]
    cr = all_data[cs]["carr"]

    lines.append("### Threshold sweep table (RL selector)\n")
    lines.append("| Threshold | sb fail% (unif) | sb stuck% | env fail% (unif) | env stuck% | sb fail% (adv) | env fail% (adv) |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in THRESHOLDS:
        t_key = f"{t:.2f}"
        tr = sd.get("sweep_results", {}).get(t_key, {})
        sb  = tr.get("uniform/rl/single_belief", {})
        env = tr.get("uniform/rl/envelope", {})
        sba = tr.get("adversarial_opt/rl/single_belief", {})
        enva = tr.get("adversarial_opt/rl/envelope", {})
        lines.append(
            f"| {t_key}"
            f" | {_fmt(sb.get('fail_rate'))}"
            f" | {_fmt(sb.get('stuck_rate'))}"
            f" | {_fmt(env.get('fail_rate'))}"
            f" | {_fmt(env.get('stuck_rate'))}"
            f" | {_fmt(sba.get('fail_rate'))}"
            f" | {_fmt(enva.get('fail_rate'))}"
            f" |"
        )

    lines.append("")
    carr_u = get_carr_point(cr, "uniform")
    carr_a = get_carr_point(cr, "adversarial_opt")
    none_u = get_baseline(fr, "uniform", "none")
    none_a = get_baseline(fr, "adversarial_opt", "none")

    if carr_u or carr_a:
        lines.append(
            f"*Carr*: fail={_fmt(carr_u[0] if carr_u else None)} / "
            f"stuck={_fmt(carr_u[1] if carr_u else None)} (uniform); "
            f"fail={_fmt(carr_a[0] if carr_a else None)} / "
            f"stuck={_fmt(carr_a[1] if carr_a else None)} (adversarial)"
        )
    else:
        lines.append("*Carr*: results not available")

    if none_u:
        lines.append(
            f"*Baseline `none`*: fail={_fmt(none_u[0])} (uniform), "
            f"fail={_fmt(none_a[0]) if none_a else 'N/A'} (adversarial)"
        )

    obs_carr_u = get_carr_point(all_data["obstacle"]["carr"], "uniform")
    obs_carr_a = get_carr_point(all_data["obstacle"]["carr"], "adversarial_opt")
    lines.append("\n### Key findings\n")
    lines.append(
        "Obstacle shows the sharpest Pareto trade-off: `envelope` Pareto-dominates\n"
        "`single_belief` at every threshold — lower fail at the cost of higher stuck.\n"
        "At t=0.95: envelope 3% fail / 85% stuck (uniform); single_belief 14% / 50%.\n"
        "\n"
        "**Carr** achieves "
        f"{_fmt(obs_carr_u[0] if obs_carr_u else None)} fail / "
        f"{_fmt(obs_carr_u[1] if obs_carr_u else None)} stuck (uniform) and "
        f"{_fmt(obs_carr_a[0] if obs_carr_a else None)} fail / "
        f"{_fmt(obs_carr_a[1] if obs_carr_a else None)} stuck (adversarial). "
        "With only 3 distinct observations the support remains large and the shield is\n"
        "extremely conservative: the 47,531-state support-MDP has 12,167 winning\n"
        "supports but the RL agent still ends up stuck on nearly every trial. Carr\n"
        "achieves the lowest fail rate of any method but at the highest stuck cost —\n"
        "it sits at the far right of the Pareto frontier and is dominated in practice.\n"
    )

    # ── CartPole ───────────────────────────────────────────────────────────────
    lines.append("\n---\n\n## CartPole (82 states, 82 obs)\n")
    lines.append(
        "**200 trials × 15 steps. Envelope excluded (dominated at every threshold).\n"
        "Results presented as a method comparison table (no Pareto structure).**\n"
    )
    cs = "cartpole"
    sd = all_data[cs]["sweep"]
    fr = all_data[cs]["final"]
    cr = all_data[cs]["carr"]
    exclude_env = sd["metadata"].get("exclude_envelope", False)

    rows = _build_comparison_rows(cs, sd, fr, cr, exclude_env)
    lines.append("\n| Method | Best threshold / note | fail% (unif) | stuck% (unif) | fail% (adv) | stuck% (adv) |")
    lines.append("|---|---|---|---|---|---|")
    for (method, fu, su, fa, sa, note) in rows:
        lines.append(
            f"| {method} | {note}"
            f" | {_fmt(fu)} | {_fmt(su)}"
            f" | {_fmt(fa)} | {_fmt(sa)}"
            f" |"
        )

    none_u = get_baseline(fr, "uniform", "none")
    none_a = get_baseline(fr, "adversarial_opt", "none")
    if none_u:
        lines.append(f"\n*No-shield baseline*: fail={_fmt(none_u[0])} (uniform), "
                     f"fail={_fmt(none_a[0]) if none_a else 'N/A'} (adversarial)\n")

    cp_carr_u = get_carr_point(all_data["cartpole"]["carr"], "uniform")
    cp_carr_a = get_carr_point(all_data["cartpole"]["carr"], "adversarial_opt")
    lines.append("### Key findings\n")
    lines.append(
        "`single_belief` is highly effective for CartPole. Optimal t≈0.65–0.75 gives\n"
        "2% fail / 0% stuck — a 6× improvement over no-shield (12% fail) with zero\n"
        "liveness cost. The fail rate does not improve further at higher thresholds;\n"
        "stuck increases from 0% to 6%.\n"
        "\n"
        "**Carr** achieves "
        f"{_fmt(cp_carr_u[0] if cp_carr_u else None)} fail / "
        f"{_fmt(cp_carr_u[1] if cp_carr_u else None)} stuck (uniform) and "
        f"{_fmt(cp_carr_a[0] if cp_carr_a else None)} fail / "
        f"{_fmt(cp_carr_a[1] if cp_carr_a else None)} stuck (adversarial). "
        "With 82 observations that essentially uniquely identify states, the\n"
        "support-MDP has only 4 reachable supports (3 winning) and the shield\n"
        "collapses to near-singleton supports immediately. This makes Carr competitive\n"
        "with `single_belief` at its optimal threshold: both achieve ≤2% fail with\n"
        "low stuck overhead.\n"
    )

    # ── CartPole low-accuracy ──────────────────────────────────────────────────
    if "cartpole_lowacc" in all_data:
        lines.append(
            "\n---\n\n## CartPole — Low-Accuracy Perception (82 states, 82 obs)\n"
        )
        lines.append(
            "**200 trials × 15 steps. Envelope excluded. Perception model: 175 training "
            "episodes (vs 200 for standard CartPole), mean P_mid≈0.373 (vs 0.532), "
            "matching TaxiNet's difficulty level (P_mid≈0.354).**\n"
        )
        cs = "cartpole_lowacc"
        sd = all_data[cs]["sweep"]
        fr = all_data[cs].get("final", {})
        cr = all_data[cs].get("carr")
        lowacc_rows = _build_comparison_rows(cs, sd, fr, cr, exclude_envelope=True)

        lines.append("\n| Method | Best threshold / note | fail% (unif) | stuck% (unif) | fail% (adv) | stuck% (adv) |")
        lines.append("|---|---|---|---|---|---|")
        for (method, fu, su, fa, sa, note) in lowacc_rows:
            lines.append(
                f"| {method} | {note}"
                f" | {_fmt(fu)} | {_fmt(su)}"
                f" | {_fmt(fa)} | {_fmt(sa)}"
                f" |"
            )

        lines.append("\n### Key findings\n")
        lines.append(
            "Lower perception accuracy (P_mid=0.373 vs 0.532 for standard CartPole) raises\n"
            "the failure rate at low thresholds: at t=0.50, adversarial fail rises from ~4%\n"
            "to ~9%, showing the shield has to work harder under noisier observations.\n"
            "At higher thresholds (t=0.85–0.90) the shield recovers to ~1–2% fail / 0% stuck,\n"
            "comparable to the standard model. This confirms that `single_belief` effectively\n"
            "compensates for perception noise at the cost of a higher optimal threshold.\n"
            "Unlike TaxiNet (which still achieves 34–43% fail despite matched P_mid), CartPole's\n"
            "near-Markovian dynamics remain inherently more controllable under partial observability.\n"
        )

    # ── Refuel v2 ─────────────────────────────────────────────────────────────
    lines.append("\n---\n\n## Refuel v2 (344 states, 29 obs)\n")
    lines.append(
        "**200 trials × 30 steps. Envelope excluded (LP ≈ 144 s/step, infeasible).\n"
        "Results presented as a method comparison table (no Pareto structure).**\n"
    )
    cs = "refuel_v2"
    sd = all_data[cs]["sweep"]
    fr = all_data[cs]["final"]
    cr = all_data[cs]["carr"]
    exclude_env = sd["metadata"].get("exclude_envelope", True)

    rows = _build_comparison_rows(cs, sd, fr, cr, exclude_env)
    lines.append("\n| Method | Best threshold / note | fail% (unif) | stuck% (unif) | fail% (adv) | stuck% (adv) |")
    lines.append("|---|---|---|---|---|---|")
    for (method, fu, su, fa, sa, note) in rows:
        lines.append(
            f"| {method} | {note}"
            f" | {_fmt(fu)} | {_fmt(su)}"
            f" | {_fmt(fa)} | {_fmt(sa)}"
            f" |"
        )

    lines.append("\n*No-shield baseline*: fail≈10–15% (estimated from RL training metrics)\n")

    lines.append("### Key findings\n")
    lines.append(
        "Refuel v2 (safety predicates hidden from observation) is genuinely non-trivial.\n"
        "`single_belief` achieves 0% fail at t=0.80 at 73–74% stuck cost. Sweet spot\n"
        "t≈0.50–0.65 gives 2–3% fail / 32–51% stuck — still much better than no-shield.\n"
        "\n"
        "**Carr on Refuel v2**: Infeasible. The support-MDP BFS starting from the\n"
        "291 safe initial states (344 total, 53 avoid) with 29 observations produced\n"
        "hundreds of millions of reachable support sets and was terminated after\n"
        "exceeding the memory budget. The state-space-to-observation ratio (11.9) is\n"
        "too large for support-MDP construction to be tractable. This parallels the\n"
        "envelope shield's infeasibility (LP ≈ 144 s/step) — Refuel v2 is a case where\n"
        "only the `single_belief` IPOMDP shield is computationally viable.\n"
    )

    # ── Cross-case-study summary ───────────────────────────────────────────────
    lines.append("\n---\n\n## Cross-Case-Study Summary\n")
    lines.append(
        "![Pareto summary v3](summary_v3.png)\n"
        "\n"
        "### Best operating points\n"
    )
    lines.append(
        "| Case study | Best IPOMDP shield | Best threshold | Min fail% | Stuck% at that point |"
    )
    lines.append("|---|---|---|---|---|")

    best_points = {
        "taxinet":   ("envelope",       "0.95", "34–35% (both regimes)", "34–36%"),
        "cartpole":  ("single_belief",  "0.65–0.75", "2%", "0%"),
        "obstacle":  ("envelope",       "0.95", "3–5%", "82–85%"),
        "refuel_v2": ("single_belief",  "0.80", "0%", "73–74%"),
    }
    for cs, (shield, t, fail, stuck) in best_points.items():
        lines.append(f"| {CS_LABELS[cs]} | {shield} | {t} | {fail} | {stuck} |")

    lines.append("")
    lines.append("### Carr vs IPOMDP shields\n")
    lines.append(
        "| Case study | Carr fail% (unif) | Carr stuck% (unif) | Carr fail% (adv) | Carr stuck% (adv) | Carr feasible? |"
    )
    lines.append("|---|---|---|---|---|---|")
    for cs_name in ["taxinet", "cartpole", "obstacle", "refuel_v2"]:
        cr = all_data.get(cs_name, {}).get("carr")
        feasible = carr_availability.get(cs_name, "unknown")
        cu = get_carr_point(cr, "uniform") if cr else None
        ca = get_carr_point(cr, "adversarial_opt") if cr else None
        lines.append(
            f"| {CS_LABELS[cs_name]}"
            f" | {_fmt(cu[0] if cu else None)}"
            f" | {_fmt(cu[1] if cu else None)}"
            f" | {_fmt(ca[0] if ca else None)}"
            f" | {_fmt(ca[1] if ca else None)}"
            f" | {feasible}"
            f" |"
        )

    lines.append("\n### Conclusions\n")
    lines.append(
        "1. **Pareto structure** exists for TaxiNet and Obstacle: higher threshold →\n"
        "   lower fail at cost of more stuck. CartPole and Refuel v2 lack this structure\n"
        "   (CartPole fail plateaus at ~1.5–2.5% and stuck rises sharply above t=0.80;\n"
        "   Refuel v2 similarly), making comparison tables more informative.\n"
        "\n"
        "2. **`envelope` vs `single_belief`**: envelope wins on TaxiNet and Obstacle\n"
        "   (especially under adversarial perception). For CartPole, single_belief is\n"
        "   sufficient; envelope only adds stuck. For Refuel v2, only single_belief is\n"
        "   feasible (envelope LP-infeasible at 144 s/step).\n"
        "\n"
        "3. **Carr shield — case-by-case**:\n"
        "   - **TaxiNet**: degenerate — 0 winning supports in the support-MDP means Carr\n"
        "     blocks every action from step 0. The midpoint-realization POMDP has no\n"
        "     state from which safety can be guaranteed under support tracking. The IPOMDP\n"
        "     shields, which track probability mass (not just support), avoid this trap.\n"
        "   - **CartPole**: competitive — with 82 observations that near-uniquely identify\n"
        "     states, supports collapse to singletons and Carr achieves ~1.5% fail / 5%\n"
        "     stuck — matching single_belief at its best threshold.\n"
        "   - **Obstacle**: too conservative — with only 3 observations supports remain\n"
        "     large; Carr achieves 2% fail but 98% stuck, dominated by the envelope at\n"
        "     t=0.95 (3% fail / 85% stuck) and completely impractical.\n"
        "   - **Refuel v2**: infeasible — 344 states × 29 obs causes the support-MDP BFS\n"
        "     to exceed memory limits, mirroring the envelope LP infeasibility.\n"
        "\n"
        "4. **CartPole optimal point** (t≈0.65–0.75, single_belief): ~2.5% fail / 0% stuck\n"
        "   is the best safety-liveness combination across all case studies, with zero\n"
        "   liveness cost. At higher thresholds (t=0.90–0.95), fail drops marginally to\n"
        "   ~1.5% but stuck increases to 4–6%.\n"
        "\n"
        "5. **Refuel v2 validates shielding**: the v2 redesign (hidden safety predicates)\n"
        "   confirms that IPOMDP shielding is essential when safety is not directly\n"
        "   observable. single_belief achieves 0% fail at a manageable stuck cost.\n"
    )

    md = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"  Saved {out_path}")
    return out_path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("SWEEP V3 PLOTTER")
    print("=" * 70)

    # Load all data.
    all_data = {}
    carr_availability = {}

    for cs_name in PARETO_CASES + TABLE_CASES + ["cartpole_lowacc"]:
        sweep = load_sweep(cs_name)
        final = load_final(cs_name)
        carr  = load_carr(cs_name)

        if sweep is None:
            print(f"  WARNING: No sweep data for {cs_name} — skipping")
            continue

        all_data[cs_name] = {"sweep": sweep, "final": final, "carr": carr}

        if carr is not None:
            carr_availability[cs_name] = "yes"
            print(f"  Loaded: {cs_name}  (Carr: yes)")
        else:
            # Check if the file exists but is infeasible.
            path = os.path.join(SWEEP_DIR, f"{cs_name}_carr_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    raw = json.load(f)
                status = raw.get("status", "missing")
                carr_availability[cs_name] = status
                print(f"  Loaded: {cs_name}  (Carr: {status})")
            else:
                carr_availability[cs_name] = "not yet run"
                print(f"  Loaded: {cs_name}  (Carr: not yet run)")

    os.makedirs(SWEEP_DIR, exist_ok=True)

    # Pareto plots for TaxiNet and Obstacle.
    print("\nGenerating Pareto plots (TaxiNet, Obstacle)...")
    pareto_data = {}
    for cs_name in PARETO_CASES:
        if cs_name not in all_data:
            continue
        d = all_data[cs_name]
        plot_pareto_per_case_study(
            cs_name, d["sweep"], d["final"], d["carr"]
        )
        pareto_data[cs_name] = (d["sweep"], d["final"], d["carr"])

    if len(pareto_data) >= 2:
        print("\nGenerating Pareto summary plot...")
        plot_pareto_summary(pareto_data)

    # Markdown summary.
    print("\nGenerating markdown summary...")
    generate_markdown(all_data, carr_availability, out_path=MD_OUT)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print(f"  Pareto plots : {SWEEP_DIR}/pareto_v3_{{taxinet,obstacle}}.png")
    print(f"  Summary plot : {SWEEP_DIR}/summary_v3.png")
    print(f"  Markdown     : {MD_OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
