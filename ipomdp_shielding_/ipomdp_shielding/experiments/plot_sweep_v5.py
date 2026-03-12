"""Generate v5 evaluation summary: Pareto scatter plots + stacked bar charts.

- Pareto scatter (no connecting lines, subset of threshold labels) for case
  studies where data spans both axes: TaxiNet, Obstacle.
- Stacked bar charts (fail% solid + stuck% hatched) at the best threshold per
  shield, for all four case studies.
- Separate uniform / adversarial subplots in every figure.
- Only most-recent variants: CartPole low-accuracy (P_mid=0.373), Refuel v2.

Usage:
    python -m ipomdp_shielding.experiments.plot_sweep_v5
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = Path("results")
DATA     = BASE / "threshold_sweep_expanded"
OBS      = BASE / "observation_shield_sweep"
OUTDIR   = BASE / "sweep_v5"
MD_PATH  = Path("evaluation_summary_threshold_sweep_5.md")

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
LABEL_T    = {0.90, 0.95}          # thresholds annotated on scatter plots

# ── visual style ───────────────────────────────────────────────────────────────
COLORS = {
    "envelope":      "#E65100",   # deep orange
    "single_belief": "#1565C0",   # blue
    "observation":   "#2E7D32",   # dark green
    "carr":          "#6A1B9A",   # purple
}
MARKERS = {
    "envelope":      "s",
    "single_belief": "o",
    "observation":   "^",
    "carr":          "D",
}
SHIELD_LABELS = {
    "envelope":      "Envelope",
    "single_belief": "Single-Belief",
    "observation":   "Observation",
    "carr":          "Carr",
}
# Display order for bar charts (left → right)
BAR_ORDER = ["envelope", "single_belief", "observation", "carr"]

# ── case study configuration ───────────────────────────────────────────────────
CASES = {
    "taxinet": {
        "label": "TaxiNet\n(16 states, 16 obs)",
        "long":  "TaxiNet (16 states, 16 obs)",
        "sweep": DATA / "taxinet_sweep.json",
        "carr":  DATA / "taxinet_carr_results.json",
        "obs":   OBS  / "taxinet_obs_sweep.json",
        "sweep_shields": ["envelope", "single_belief"],
        "pareto": True,
    },
    "obstacle": {
        "label": "Obstacle\n(50 states, 3 obs)",
        "long":  "Obstacle (50 states, 3 obs)",
        "sweep": DATA / "obstacle_sweep.json",
        "carr":  DATA / "obstacle_carr_results.json",
        "obs":   OBS  / "obstacle_obs_sweep.json",
        "sweep_shields": ["envelope", "single_belief"],
        "pareto": True,
    },
    "cartpole_lowacc": {
        "label": "CartPole low-acc\n(82 states, P_mid=0.373)",
        "long":  "CartPole low-acc (82 states, P_mid=0.373)",
        "sweep": DATA / "cartpole_lowacc_sweep.json",
        "carr":  DATA / "cartpole_lowacc_carr_results.json",
        "obs":   OBS  / "cartpole_lowacc_obs_sweep.json",
        "sweep_shields": ["single_belief"],
        "pareto": False,
    },
    "refuel_v2": {
        "label": "Refuel v2\n(344 states, 29 obs)",
        "long":  "Refuel v2 (344 states, 29 obs)",
        "sweep": DATA / "refuel_v2_sweep.json",
        "carr":  None,              # support-MDP BFS infeasible
        "obs":   OBS  / "refuel_v2_obs_sweep.json",
        "sweep_shields": ["single_belief"],
        "pareto": False,
    },
}

# ── data helpers ───────────────────────────────────────────────────────────────

def load_json(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def sweep_points(data: dict, perception: str, shield: str) -> list[tuple]:
    """Return [(fail, stuck, t)] for each threshold present in sweep data."""
    sr = data.get("sweep_results", {})
    out = []
    for t in THRESHOLDS:
        r = sr.get(f"{t:.2f}", {}).get(f"{perception}/rl/{shield}")
        if r:
            out.append((r["fail_rate"], r["stuck_rate"], t))
    return out


def best_point(data: dict, perception: str, shield: str):
    """(t, fail, stuck) that minimises fail then stuck."""
    pts = sweep_points(data, perception, shield)
    if not pts:
        return None
    f, s, t = min(pts, key=lambda p: (p[0], p[1]))
    return (t, f, s)


def carr_result(data, perception: str):
    """(fail, stuck) for Carr shield, or None."""
    if data is None or data.get("status") == "infeasible":
        return None
    r = data.get("results", {}).get(f"{perception}/rl/carr")
    return (r["fail_rate"], r["stuck_rate"]) if r else None


def collect_best(sweep_data, obs_data, carr_data, perception: str,
                 sweep_shields: list[str]) -> dict:
    """Build ordered {shield: {fail, stuck, threshold}} for bar chart."""
    raw = {}
    for s in sweep_shields:
        bp = best_point(sweep_data, perception, s)
        if bp:
            raw[s] = {"fail": bp[1], "stuck": bp[2], "threshold": bp[0]}
    if obs_data:
        pts = sweep_points(obs_data, perception, "observation")
        if pts:
            f, s, t = min(pts, key=lambda p: (p[0], p[1]))
            raw["observation"] = {"fail": f, "stuck": s, "threshold": t}
    cr = carr_result(carr_data, perception)
    if cr:
        raw["carr"] = {"fail": cr[0], "stuck": cr[1], "threshold": None}
    return {s: raw[s] for s in BAR_ORDER if s in raw}


# ── pareto scatter ─────────────────────────────────────────────────────────────

def _draw_pareto(ax, curves: dict, carr_pt, title: str) -> None:
    for shield, pts in curves.items():
        c, m = COLORS[shield], MARKERS[shield]
        for f, s, t in pts:
            ax.scatter(f * 100, s * 100, color=c, marker=m, s=55, zorder=3)
            if t in LABEL_T:
                ax.annotate(f"{t:.2f}", (f * 100, s * 100),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=7, color=c)
    if carr_pt:
        f, s = carr_pt
        ax.scatter(f * 100, s * 100, color=COLORS["carr"],
                   marker="D", s=90, zorder=4)
        ax.annotate("Carr", (f * 100, s * 100),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7.5, color=COLORS["carr"], fontweight="bold")
    ax.set_xlabel("Fail rate (%)")
    ax.set_ylabel("Stuck rate (%)")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)


def make_pareto_figure(cs_name: str, cs_cfg: dict) -> str:
    sweep = load_json(cs_cfg["sweep"])
    obs   = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
    carr  = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.suptitle(f"Pareto Scatter — {cs_cfg['long']}", fontsize=10)

    for ax, perc, plbl in zip(axes,
                               ["uniform", "adversarial_opt"],
                               ["Uniform perception", "Adversarial perception"]):
        curves = {s: sweep_points(sweep, perc, s) for s in cs_cfg["sweep_shields"]}
        if obs:
            curves["observation"] = sweep_points(obs, perc, "observation")
        _draw_pareto(ax, curves, carr_result(carr, perc), plbl)

    legend_shields = (
        [s for s in BAR_ORDER
         if s in cs_cfg["sweep_shields"] or s == "observation" or
         (s == "carr" and carr is not None)]
    )
    handles = [
        plt.scatter([], [], color=COLORS[s], marker=MARKERS[s],
                    s=55, label=SHIELD_LABELS[s])
        for s in legend_shields
    ]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = OUTDIR / f"pareto_v5_{cs_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


# ── bar chart ──────────────────────────────────────────────────────────────────

def _draw_bars(ax, best_dict: dict, title: str, ylim: float = 115) -> None:
    shields = list(best_dict.keys())
    fails   = [best_dict[s]["fail"]  * 100 for s in shields]
    stucks  = [best_dict[s]["stuck"] * 100 for s in shields]
    colors  = [COLORS.get(s, "#888") for s in shields]
    x, w    = np.arange(len(shields)), 0.52

    ax.bar(x, fails,  w, color=colors, alpha=0.90, zorder=3)
    ax.bar(x, stucks, w, bottom=fails, color=colors, alpha=0.32,
           hatch="///", edgecolor="white", zorder=3)

    for i, s in enumerate(shields):
        total = fails[i] + stucks[i]
        t_val = best_dict[s].get("threshold")
        lbl   = f"t={t_val:.2f}" if t_val is not None else "—"
        ax.text(x[i], total + ylim * 0.015, lbl,
                ha="center", va="bottom", fontsize=7.5)
        if fails[i] >= 6:
            ax.text(x[i], fails[i] / 2, f"{fails[i]:.0f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SHIELD_LABELS.get(s, s) for s in shields], fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)


def make_bar_figure(cs_name: str, cs_cfg: dict) -> str:
    sweep = load_json(cs_cfg["sweep"])
    obs   = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
    carr  = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None

    # Compute a shared y-limit scaled to the actual data range
    all_totals = []
    for perc in ["uniform", "adversarial_opt"]:
        best = collect_best(sweep, obs, carr, perc, cs_cfg["sweep_shields"])
        for d in best.values():
            all_totals.append((d["fail"] + d["stuck"]) * 100)
    max_total = max(all_totals) if all_totals else 10
    ylim = max(max_total * 1.18, 12)   # 18% headroom for labels; min 12

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), sharey=True)
    fig.suptitle(f"Best Operating Point — {cs_cfg['long']}", fontsize=10)

    for ax, perc, plbl in zip(axes,
                               ["uniform", "adversarial_opt"],
                               ["Uniform perception", "Adversarial perception"]):
        best = collect_best(sweep, obs, carr, perc, cs_cfg["sweep_shields"])
        _draw_bars(ax, best, plbl, ylim=ylim)

    fp = mpatches.Patch(facecolor="gray", alpha=0.90, label="Fail %")
    sp = mpatches.Patch(facecolor="gray", alpha=0.32, hatch="///",
                        edgecolor="white", label="Stuck %")
    # Place legend where bars are shortest to avoid overlap
    legend_loc = "upper left" if max_total > 80 else "upper right"
    axes[1].legend(handles=[fp, sp], fontsize=8, loc=legend_loc)

    plt.tight_layout()
    out = OUTDIR / f"barchart_v5_{cs_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


# ── combined overview (2 rows × 4 cols) ───────────────────────────────────────

def make_summary_figure() -> str:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Best Operating Point — All Case Studies", fontsize=12, y=1.01)

    for col, (cs_name, cs_cfg) in enumerate(CASES.items()):
        sweep = load_json(cs_cfg["sweep"])
        obs   = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
        carr  = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None

        all_totals = []
        for perc in ["uniform", "adversarial_opt"]:
            best_tmp = collect_best(sweep, obs, carr, perc, cs_cfg["sweep_shields"])
            for d in best_tmp.values():
                all_totals.append((d["fail"] + d["stuck"]) * 100)
        col_ylim = max(max(all_totals) * 1.18, 12)

        for row, (perc, plbl) in enumerate(
            zip(["uniform", "adversarial_opt"], ["Uniform", "Adversarial"])
        ):
            ax = axes[row][col]
            best = collect_best(sweep, obs, carr, perc, cs_cfg["sweep_shields"])
            _draw_bars(ax, best, f"{cs_cfg['label']}\n({plbl})", ylim=col_ylim)
            if col > 0:
                ax.set_ylabel("")

    fp = mpatches.Patch(facecolor="gray", alpha=0.90, label="Fail %")
    sp = mpatches.Patch(facecolor="gray", alpha=0.32, hatch="///",
                        edgecolor="white", label="Stuck %")
    fig.legend(handles=[fp, sp], fontsize=9,
               loc="upper right", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    out = OUTDIR / "summary_v5_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


# ── markdown helpers ───────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _best_row(sweep_data, obs_data, carr_data, cs_cfg, perception) -> dict:
    """Return {shield: (t, fail, stuck)} for the markdown table."""
    row = {}
    for s in cs_cfg["sweep_shields"]:
        bp = best_point(sweep_data, perception, s)
        if bp:
            row[s] = bp
    if obs_data:
        pts = sweep_points(obs_data, perception, "observation")
        if pts:
            f, s, t = min(pts, key=lambda p: (p[0], p[1]))
            row["observation"] = (t, f, s)
    cr = carr_result(carr_data, perception)
    if cr:
        row["carr"] = (None, cr[0], cr[1])
    return row


def generate_markdown(figures: dict) -> str:
    lines: list[str] = []

    lines += [
        "# Evaluation Summary — v5",
        "",
        "**Shields compared**: Envelope, Single-Belief, Observation, Carr  ",
        "*(where feasible — see per-case notes)*",
        "",
        "**Case studies** (most-recent variants only):",
        "TaxiNet (16 states, 16 obs) · Obstacle (50 states, 3 obs) · "
        "CartPole low-acc (82 states, P_mid=0.373) · Refuel v2 (344 states, 29 obs)",
        "",
        "**Trials**: 200 per combination. Bar charts show the best operating",
        "threshold (min fail%, then min stuck%) for each shield.",
        "Pareto plots shown only where data varies meaningfully on both axes.",
        "",
        "---",
        "",
        "## Cross-Case Summary",
        "",
        f"![Overview bar charts]({figures['summary']})",
        "",
    ]

    # cross-case best-points table
    lines += [
        "### Best operating points",
        "",
        "| Case study | Shield | t | Fail% (unif) | Stuck% (unif) | Fail% (adv) | Stuck% (adv) |",
        "|---|---|---|---|---|---|---|",
    ]
    for cs_name, cs_cfg in CASES.items():
        sweep = load_json(cs_cfg["sweep"])
        obs   = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
        carr  = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None
        ru    = _best_row(sweep, obs, carr, cs_cfg, "uniform")
        ra    = _best_row(sweep, obs, carr, cs_cfg, "adversarial_opt")
        all_s = [s for s in BAR_ORDER if s in ru or s in ra]
        for i, s in enumerate(all_s):
            cs_col = cs_cfg["long"] if i == 0 else ""
            t_u  = f"{ru[s][0]:.2f}" if s in ru and ru[s][0] else "—"
            fu   = _pct(ru[s][1]) if s in ru else "N/A"
            su   = _pct(ru[s][2]) if s in ru else "N/A"
            fa   = _pct(ra[s][1]) if s in ra else "N/A"
            sa   = _pct(ra[s][2]) if s in ra else "N/A"
            lines.append(f"| {cs_col} | {SHIELD_LABELS[s]} | {t_u} | {fu} | {su} | {fa} | {sa} |")
    lines.append("")

    lines += [
        "### Key cross-case findings",
        "",
        "1. **Envelope dominates Single-Belief** (TaxiNet, Obstacle) — lower fail at every"
        " threshold, at comparable or slightly higher stuck cost. Infeasible for CartPole"
        " lowacc and Refuel v2.",
        "",
        "2. **Observation shield is bimodal**: near-optimal for CartPole (2.5% fail / 0%"
        " stuck — matches Single-Belief) and offers a unique 0%-stuck operating region for"
        " Refuel v2 (3% fail / 0% stuck at t=0.65, vs Single-Belief's minimum 38% stuck)."
        " Degrades badly for TaxiNet (>55% fail at 0% stuck) and collapses to"
        " Carr-equivalent behaviour for Obstacle (1.5% fail / 98.5% stuck).",
        "",
        "3. **Carr achieves the lowest raw fail rate** on TaxiNet (7.5%) and Obstacle"
        " (1.5%), but always at near-total stuck cost (92–99%). Competitive only for"
        " CartPole (1.5% fail / 5.5% stuck) where 82 near-unique observations make"
        " Carr non-conservative. Infeasible for Refuel v2.",
        "",
        "4. **Observation informativeness governs shield effectiveness**: CartPole"
        " (82 obs ≈ 82 states) → all memoryless methods near-optimal; Obstacle"
        " (3 obs / 50 states) → both Observation and Carr degenerate; TaxiNet"
        " (16 obs / 16 states, poor posterior) → history essential; Refuel v2"
        " (29 obs / 344 states) → intermediate, with memoryless advantage at low t.",
        "",
        "5. **CartPole lowacc is the only case with zero liveness cost** across all"
        " threshold-based shields — both Single-Belief and Observation achieve ≤2.5% fail"
        " / 0% stuck at their best threshold.",
        "",
        "---",
        "",
    ]

    # ── per-case sections ──────────────────────────────────────────────────────
    for cs_name, cs_cfg in CASES.items():
        sweep = load_json(cs_cfg["sweep"])
        obs   = load_json(cs_cfg["obs"]) if cs_cfg["obs"] else None
        carr  = load_json(cs_cfg["carr"]) if cs_cfg["carr"] else None

        lines += [f"## {cs_cfg['long']}", ""]

        if cs_cfg["pareto"]:
            lines += [
                f"![Pareto scatter]({figures['pareto'][cs_name]})",
                "",
                "> *Each point is one threshold setting. Only t=0.90 and t=0.95 are"
                " labelled. No lines connect points.*",
                "",
            ]

        lines += [
            f"![Bar chart — best threshold per shield]({figures['bar'][cs_name]})",
            "",
        ]

        # per-case table
        lines += [
            "### Best operating points",
            "",
            "| Shield | t (unif) | Fail% (unif) | Stuck% (unif) | t (adv) | Fail% (adv) | Stuck% (adv) |",
            "|---|---|---|---|---|---|---|",
        ]
        ru = _best_row(sweep, obs, carr, cs_cfg, "uniform")
        ra = _best_row(sweep, obs, carr, cs_cfg, "adversarial_opt")
        for s in BAR_ORDER:
            if s not in ru and s not in ra:
                continue
            tu  = f"{ru[s][0]:.2f}" if s in ru and ru[s][0] else "—"
            fu  = _pct(ru[s][1]) if s in ru else "N/A"
            su  = _pct(ru[s][2]) if s in ru else "N/A"
            ta  = f"{ra[s][0]:.2f}" if s in ra and ra[s][0] else "—"
            fa  = _pct(ra[s][1]) if s in ra else "N/A"
            sa  = _pct(ra[s][2]) if s in ra else "N/A"
            lines.append(
                f"| {SHIELD_LABELS[s]} | {tu} | {fu} | {su} | {ta} | {fa} | {sa} |"
            )
        lines.append("")

        # per-case findings
        if cs_name == "taxinet":
            lines += [
                "### Key findings",
                "",
                "- **Envelope** offers the best safety-liveness trade-off: 35% fail / 34%"
                " stuck (uniform) at t=0.95.",
                "- **Single-Belief** is the most liveness-friendly: 44% fail / 11% stuck —"
                " useful when being stuck matters more than the remaining fail rate.",
                "- **Observation** achieves lower fail (12.5%) only at t=0.95, carrying"
                " 87.5% stuck — a poor trade given Envelope's 35% fail / 34% stuck at the"
                " same threshold.",
                "- **Carr** reaches the lowest raw fail (7.5%) but blocks ≥92% of episodes"
                " from step 0. The midpoint POMDP has no winning support, so Carr is"
                " degenerate here.",
                "- History is essential for TaxiNet: unsafe and safe states share"
                " observations, so a single obs provides an uninformative posterior.",
                "",
                "---",
                "",
            ]
        elif cs_name == "obstacle":
            lines += [
                "### Key findings",
                "",
                "- **Envelope** Pareto-dominates Single-Belief at every threshold;"
                " 3% fail / 85% stuck at t=0.95 is the best achievable safety-liveness"
                " point for any threshold-based shield.",
                "- **Observation** and **Carr** converge to the same extreme corner:"
                " ~1.5% fail / 98.5% stuck — nearly indistinguishable. With only 3"
                " distinct observations, the observation posterior is near-uniform and"
                " the observation shield behaves like a support-based (Carr-style) shield.",
                "- **Single-Belief** at t=0.95 (14% fail / 50% stuck) offers the best"
                " liveness of any safe-ish operating point.",
                "- The 3-observation structure makes all memoryless methods highly"
                " conservative; belief tracking (Envelope) is essential.",
                "",
                "---",
                "",
            ]
        elif cs_name == "cartpole_lowacc":
            lines += [
                "### Key findings",
                "",
                "- All three feasible shields achieve **0% stuck** at their best threshold"
                " — CartPole lowacc is the only case with zero liveness cost.",
                "- **Single-Belief** is marginally best: 1% fail / 0% stuck at t=0.85.",
                "- **Observation** matches Single-Belief's liveness (0% stuck) at 2.5%"
                " fail. The near-bijective 82-obs/82-state structure means a single"
                " observation is nearly as informative as the full belief.",
                "- **Carr** is competitive (1.5% fail / 0% stuck uniform); with 82"
                " near-unique observations the lowacc support-MDP has only 2 reachable"
                " supports (1 winning), so Carr adds essentially no conservatism.",
                "- Envelope not available (LP infeasible at ~1.9 s/step for 200-trial sweep).",
                "",
                "---",
                "",
            ]
        elif cs_name == "refuel_v2":
            lines += [
                "### Key findings",
                "",
                "- **Single-Belief** achieves 0% fail at t=0.90 (79% stuck uniform;"
                " 85% stuck adversarial) — the lowest stuck rate among 0%-fail operating"
                " points.",
                "- **Observation** also achieves 0% fail but with 99% stuck at t=0.90.",
                "  Its unique advantage: at t=0.65, it gives **3–4.5% fail / 0% stuck**"
                " — the only 0%-stuck operating point available for Refuel v2. Single-Belief"
                " has ≥38% stuck at every threshold.",
                "- Carr and Envelope are both infeasible (support-MDP BFS and LP exceed"
                " memory / time budgets at 344 states × 29 obs).",
                "- Refuel v2 confirms that IPOMDP shielding is essential when safety"
                " predicates are hidden from the observation.",
                "",
                "---",
                "",
            ]

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    figures: dict = {"pareto": {}, "bar": {}}

    for cs_name, cs_cfg in CASES.items():
        print(f"\n── {cs_cfg['long']} ──")

        if cs_cfg["pareto"]:
            path = make_pareto_figure(cs_name, cs_cfg)
            figures["pareto"][cs_name] = path
            print(f"  Pareto → {path}")

        path = make_bar_figure(cs_name, cs_cfg)
        figures["bar"][cs_name] = path
        print(f"  Bar    → {path}")

    figures["summary"] = make_summary_figure()
    print(f"\n  Summary → {figures['summary']}")

    md = generate_markdown(figures)
    MD_PATH.write_text(md)
    print(f"\n  Markdown → {MD_PATH}")


if __name__ == "__main__":
    main()
