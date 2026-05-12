"""Generate results/observation_shield_sweep/evaluation_summary.md from observation shield sweep.

Produces Pareto-frontier plots (fail rate vs stuck rate) and a cross-case
comparison table for the ObservationShield across all case studies.

Also loads single_belief results from the expanded sweep for comparison.

Usage:
    python -m ipomdp_shielding.experiments.plot_observation_sweep
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OBS_SWEEP_DIR  = "results/observation_shield_sweep"
SB_SWEEP_DIR   = "results/threshold_sweep_expanded"
MD_OUT         = os.path.join(OBS_SWEEP_DIR, "evaluation_summary.md")

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

CS_LABELS = {
    "taxinet":         "TaxiNet (16 states, 16 obs)",
    "cartpole":        "CartPole std (82 states, P_mid=0.532)",
    "cartpole_lowacc": "CartPole low-acc (82 states, P_mid=0.373)",
    "obstacle":        "Obstacle (50 states, 3 obs)",
    "refuel_v2":       "Refuel v2 (344 states, 29 obs)",
}

# Which single_belief sweep file to use for comparison per case study.
SB_SWEEP_FILES = {
    "taxinet":         "taxinet_sweep.json",
    "cartpole":        "cartpole_sweep.json",
    "cartpole_lowacc": "cartpole_lowacc_sweep.json",
    "obstacle":        "obstacle_sweep.json",
    "refuel_v2":       "refuel_v2_sweep.json",
}

ALL_CASES = ["taxinet", "cartpole", "cartpole_lowacc", "obstacle", "refuel_v2"]


# ============================================================
# Data loading
# ============================================================

def load_obs_sweep(cs_name):
    path = os.path.join(OBS_SWEEP_DIR, f"{cs_name}_obs_sweep.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_sb_sweep(cs_name):
    fname = SB_SWEEP_FILES.get(cs_name)
    if not fname:
        return None
    path = os.path.join(SB_SWEEP_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_curve(sweep_data, perception, shield, key="sweep_results"):
    sr = sweep_data.get(key, {})
    points = []
    for t in THRESHOLDS:
        t_key = f"{t:.2f}"
        combo = sr.get(t_key, {}).get(f"{perception}/rl/{shield}")
        if combo is not None:
            points.append((combo["fail_rate"], combo["stuck_rate"], t))
    return points


def best_point(pts):
    """(fail, stuck, threshold) minimising fail then stuck."""
    if not pts:
        return None
    return min(pts, key=lambda p: (p[0], p[1]))


def _fmt(v):
    if v is None:
        return "N/A"
    return f"{v:.0%}"


# ============================================================
# Per-case Pareto plots
# ============================================================

def _plot_case(ax, cs_name, obs_data, sb_data, perception, title=None):
    """Draw obs vs single_belief curves for one (case, perception) panel."""
    # observation sweep curve
    if obs_data:
        obs_pts = get_curve(obs_data, perception, "observation")
        if obs_pts:
            xs = [p[1] for p in obs_pts]
            ys = [p[0] for p in obs_pts]
            ts = [p[2] for p in obs_pts]
            ax.plot(xs, ys, "s-", color="darkorange", lw=1.5, markersize=5,
                    label="observation", zorder=4)
            for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
                if i % 2 == 0 or t in (0.50, 0.95):
                    ax.annotate(f"{t:.2f}", (x, y),
                                textcoords="offset points", xytext=(4, 3),
                                fontsize=6, color="darkorange")

    # single_belief sweep curve (for reference)
    if sb_data:
        sb_pts = get_curve(sb_data, perception, "single_belief")
        if sb_pts:
            xs = [p[1] for p in sb_pts]
            ys = [p[0] for p in sb_pts]
            ts = [p[2] for p in sb_pts]
            ax.plot(xs, ys, "o--", color="steelblue", lw=1.2, markersize=4,
                    label="single_belief (ref)", zorder=3, alpha=0.7)

    ax.set_xlabel("Stuck rate", fontsize=7)
    ax.set_ylabel("Fail rate", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=8)


def make_pareto_plots(all_data):
    """One 2×5 figure: rows = uniform/adversarial, cols = case studies."""
    perceptions = ["uniform", "adversarial_opt"]
    perc_labels = {"uniform": "Uniform perception", "adversarial_opt": "Adversarial perception"}

    fig, axes = plt.subplots(2, len(ALL_CASES), figsize=(14, 5))

    for row, perc in enumerate(perceptions):
        for col, cs in enumerate(ALL_CASES):
            ax = axes[row][col]
            obs_d = all_data[cs]["obs"]
            sb_d  = all_data[cs]["sb"]
            title = CS_LABELS[cs].split("(")[0].strip() if row == 0 else None
            _plot_case(ax, cs, obs_d, sb_d, perc, title=title)
            if col == 0:
                ax.set_ylabel(f"{perc_labels[perc]}\nFail rate", fontsize=7)
            else:
                ax.set_ylabel("")

    legend_elements = [
        Line2D([0], [0], color="darkorange", marker="s", markersize=5,
               label="observation sweep"),
        Line2D([0], [0], color="steelblue", marker="o", markersize=4,
               linestyle="--", alpha=0.7, label="single_belief (reference)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    path = os.path.join(OBS_SWEEP_DIR, "pareto_obs_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================
# Markdown generation
# ============================================================

def generate_markdown(all_data, pareto_path, out_path=MD_OUT):
    lines = []
    lines.append("# Observation Shield Threshold Sweep — v4\n")
    lines.append(
        "**Shield**: `ObservationShield` — memoryless, single-observation posterior.\n"
        "Computes P(s | obs) using the IPOMDP midpoint observation model with a uniform\n"
        "prior, then allows action a iff P(a safe | obs) ≥ threshold.\n"
        "\n"
        "**Selector**: RL agent (reused from prior runs).\n"
        "**Perceptions**: uniform + adversarial_opt (reused caches; optimised against\n"
        "single_belief or envelope, not specifically against the observation shield).\n"
        "**Trials**: 200 per combination. **Thresholds**: 0.50 – 0.95.\n"
        "\n"
        "Single_belief results are shown alongside as a reference.\n"
    )

    lines.append(f"\n![Pareto frontiers]({os.path.basename(pareto_path)})\n")

    # Per-case-study tables
    for cs in ALL_CASES:
        obs_d = all_data[cs]["obs"]
        sb_d  = all_data[cs]["sb"]
        lines.append(f"\n---\n\n## {CS_LABELS[cs]}\n")

        # Threshold table
        lines.append(
            "| t | obs fail% (unif) | obs stuck% | obs fail% (adv) | obs stuck% "
            "| sb fail% (unif) | sb stuck% | sb fail% (adv) | sb stuck% |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for t in THRESHOLDS:
            t_key = f"{t:.2f}"
            obs_u = obs_d["sweep_results"].get(t_key, {}).get("uniform/rl/observation", {}) if obs_d else {}
            obs_a = obs_d["sweep_results"].get(t_key, {}).get("adversarial_opt/rl/observation", {}) if obs_d else {}
            sb_u  = sb_d["sweep_results"].get(t_key, {}).get("uniform/rl/single_belief", {}) if sb_d else {}
            sb_a  = sb_d["sweep_results"].get(t_key, {}).get("adversarial_opt/rl/single_belief", {}) if sb_d else {}

            lines.append(
                f"| {t_key}"
                f" | {_fmt(obs_u.get('fail_rate'))}"
                f" | {_fmt(obs_u.get('stuck_rate'))}"
                f" | {_fmt(obs_a.get('fail_rate'))}"
                f" | {_fmt(obs_a.get('stuck_rate'))}"
                f" | {_fmt(sb_u.get('fail_rate'))}"
                f" | {_fmt(sb_u.get('stuck_rate'))}"
                f" | {_fmt(sb_a.get('fail_rate'))}"
                f" | {_fmt(sb_a.get('stuck_rate'))}"
                f" |"
            )

        # Best-point summary
        if obs_d:
            obs_pts_u = get_curve(obs_d, "uniform", "observation")
            obs_pts_a = get_curve(obs_d, "adversarial_opt", "observation")
            best_obs_u = best_point(obs_pts_u)
            best_obs_a = best_point(obs_pts_a)
        else:
            best_obs_u = best_obs_a = None

        if sb_d:
            sb_pts_u  = get_curve(sb_d, "uniform", "single_belief")
            sb_pts_a  = get_curve(sb_d, "adversarial_opt", "single_belief")
            best_sb_u = best_point(sb_pts_u)
            best_sb_a = best_point(sb_pts_a)
        else:
            best_sb_u = best_sb_a = None

        lines.append("\n**Best operating points:**\n")
        lines.append("| Method | Best t (unif) | Min fail% (unif) | Stuck% | Best t (adv) | Min fail% (adv) | Stuck% |")
        lines.append("|---|---|---|---|---|---|---|")

        def _best_row(label, bu, ba):
            return (
                f"| {label}"
                f" | {bu[2]:.2f} | {_fmt(bu[0])} | {_fmt(bu[1])}"
                f" | {ba[2]:.2f} | {_fmt(ba[0])} | {_fmt(ba[1])}"
                f" |"
            ) if bu and ba else f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A |"

        lines.append(_best_row("observation", best_obs_u, best_obs_a))
        lines.append(_best_row("single_belief (ref)", best_sb_u, best_sb_a))

    # Cross-case summary table
    lines.append("\n---\n\n## Cross-Case Summary\n")
    lines.append(
        "| Case study | obs best t (unif) | obs fail% (unif) | obs stuck% (unif) "
        "| obs fail% (adv) | obs stuck% (adv) "
        "| sb fail% (unif) | sb fail% (adv) | observation vs single_belief |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for cs in ALL_CASES:
        obs_d = all_data[cs]["obs"]
        sb_d  = all_data[cs]["sb"]

        if obs_d:
            obs_pts_u = get_curve(obs_d, "uniform", "observation")
            obs_pts_a = get_curve(obs_d, "adversarial_opt", "observation")
            bu = best_point(obs_pts_u)
            ba = best_point(obs_pts_a)
        else:
            bu = ba = None

        if sb_d:
            sb_pts_u  = get_curve(sb_d, "uniform", "single_belief")
            sb_pts_a  = get_curve(sb_d, "adversarial_opt", "single_belief")
            bsu = best_point(sb_pts_u)
            bsa = best_point(sb_pts_a)
        else:
            bsu = bsa = None

        # Verdict: compare at 0-stuck operating point (stuck < 5%).
        # Find the best (min fail) obs point with stuck < 5%; compare to sb equiv.
        if obs_d and sb_d:
            obs_pts_u_all = get_curve(obs_d, "uniform", "observation")
            sb_pts_u_all  = get_curve(sb_d,  "uniform", "single_belief")
            low_stuck_obs = [p for p in obs_pts_u_all if p[1] < 0.05]
            low_stuck_sb  = [p for p in sb_pts_u_all  if p[1] < 0.05]
            if low_stuck_obs and low_stuck_sb:
                obs_f0 = min(low_stuck_obs, key=lambda p: p[0])[0]
                sb_f0  = min(low_stuck_sb,  key=lambda p: p[0])[0]
                if obs_f0 <= sb_f0 * 1.10:
                    verdict = "competitive (0% stuck)"
                elif obs_f0 <= sb_f0 * 2.0:
                    verdict = f"worse (obs {obs_f0:.0%} vs sb {sb_f0:.0%} at 0% stuck)"
                else:
                    verdict = f"much worse at 0% stuck"
            elif low_stuck_obs:
                verdict = "obs has 0%-stuck region; sb does not"
            else:
                verdict = "obs requires stuck to achieve low fail"
        else:
            verdict = "N/A"

        lines.append(
            f"| {CS_LABELS[cs]}"
            f" | {f'{bu[2]:.2f}' if bu else 'N/A'}"
            f" | {_fmt(bu[0] if bu else None)}"
            f" | {_fmt(bu[1] if bu else None)}"
            f" | {_fmt(ba[0] if ba else None)}"
            f" | {_fmt(ba[1] if ba else None)}"
            f" | {_fmt(bsu[0] if bsu else None)}"
            f" | {_fmt(bsa[0] if bsa else None)}"
            f" | {verdict}"
            f" |"
        )

    lines.append("\n### Key findings\n")
    lines.append(
        "1. **TaxiNet**: the observation shield is largely ineffective without history.\n"
        "   At any threshold with 0% stuck (t≤0.90), fail is 56–93% — far worse than\n"
        "   `single_belief` (34–44% fail / 0% stuck). Only at t=0.95 does obs achieve\n"
        "   12% fail, but at 88% stuck cost. The 16-obs/16-state structure is not enough:\n"
        "   TaxiNet's unsafe states share observations with safe ones, so a single obs\n"
        "   gives little posterior information without knowing the history of actions.\n"
        "\n"
        "2. **CartPole (standard, P_mid=0.532)**: observation shield is surprisingly\n"
        "   effective — 1.5–2.5% fail / 0% stuck for t=0.50–0.90, matching `single_belief`.\n"
        "   The 82-obs/82-state near-bijective structure means each observation is almost\n"
        "   uniquely informative; history adds nothing. Performance is nearly threshold-\n"
        "   invariant until t=0.95 where 68% stuck suddenly appears.\n"
        "\n"
        "3. **CartPole (low-accuracy, P_mid=0.373)**: shows the effect of noisier\n"
        "   perception. At t<0.70, the observations are too noisy to reliably block\n"
        "   unsafe actions (15–17% fail). A threshold of t=0.70 sharply reduces fail\n"
        "   to ~4%. Best operating point t=0.85–0.90: ~2.5% fail / 0% stuck.\n"
        "\n"
        "4. **Obstacle (3 obs / 50 states)**: observation shield is essentially\n"
        "   non-functional. With only 3 distinct observations covering ~17 states each,\n"
        "   the posterior is nearly uniform and the shield cannot reliably distinguish\n"
        "   safe from unsafe actions. At t≤0.85: ~46–53% fail AND 36–46% stuck — the\n"
        "   worst of both worlds. History (belief tracking) is essential here.\n"
        "\n"
        "5. **Refuel v2 (29 obs / 344 states)**: surprisingly effective at low thresholds.\n"
        "   t=0.50–0.65: 3–4.5% fail / 0% stuck — better than `single_belief` at the\n"
        "   same thresholds (19–32% stuck for sb). The observation shield avoids the\n"
        "   liveness trap at low t because it is memoryless (no belief to get stuck in).\n"
        "   Sweet spot: t=0.75 → 0.5–1% fail / ~60% stuck, comparable to sb at t=0.75.\n"
        "\n"
        "6. **Observation vs stuck**: the observation shield never gets stuck unless\n"
        "   the threshold is so high that the only safe action at a given obs has\n"
        "   P(safe|obs) < t. Unlike `single_belief`, it has no accumulated belief that\n"
        "   can be driven into a stuck corner by adversarial perception.\n"
        "\n"
        "7. **Adversarial robustness note**: adversarial perceptions were optimised\n"
        "   against `single_belief`/`envelope`, not the observation shield. The\n"
        "   adversarial results here are a conservative lower bound on obs shield\n"
        "   robustness — a dedicated adversarial would likely be worse.\n"
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
    print("OBSERVATION SHIELD SWEEP PLOTTER (v4)")
    print("=" * 70)

    all_data = {}
    for cs in ALL_CASES:
        obs_d = load_obs_sweep(cs)
        sb_d  = load_sb_sweep(cs)
        if obs_d is None:
            print(f"  WARNING: No observation sweep data for {cs} — skipping")
            continue
        all_data[cs] = {"obs": obs_d, "sb": sb_d}
        print(f"  Loaded: {cs}  (sb ref: {'yes' if sb_d else 'no'})")

    if not all_data:
        print("No data loaded. Run run_observation_shield_sweep first.")
        return

    os.makedirs(OBS_SWEEP_DIR, exist_ok=True)

    print("\nGenerating Pareto plots...")
    pareto_path = make_pareto_plots(all_data)

    print("\nGenerating markdown summary...")
    generate_markdown(all_data, pareto_path)

    print("\n" + "=" * 70)
    print(f"DONE — {MD_OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
