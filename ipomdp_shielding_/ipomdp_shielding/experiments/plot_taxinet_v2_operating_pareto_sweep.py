"""Plot and summarize TaxiNetV2 operating sweep results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


POINT_METHODS = ("single_belief", "envelope", "forward_sampling")
POINT_LABELS = {
    "single_belief": "Single-Belief",
    "envelope": "Envelope",
    "forward_sampling": "Forward-Sampling",
}
POINT_COLORS = {
    "single_belief": "#1565C0",
    "envelope": "#E65100",
    "forward_sampling": "#00838F",
}
POINT_MARKERS = {
    "single_belief": "o",
    "envelope": "s",
    "forward_sampling": "P",
}
CONF_COLORS = {
    "0.95": "#8E24AA",
    "0.99": "#5E35B1",
    "0.995": "#283593",
}
CONF_MARKERS = {
    "0.95": "^",
    "0.99": "D",
    "0.995": "X",
}
CONFORMAL_COLOR = "#5E35B1"
BEST_BAR_ORDER = ("single_belief", "envelope", "forward_sampling", "conformal")
BEST_BAR_LABELS = {
    "single_belief": "Single-Belief",
    "envelope": "Envelope",
    "forward_sampling": "Forward-Sampling",
    "conformal": "Conformal",
}
BEST_BAR_COLORS = {
    "single_belief": POINT_COLORS["single_belief"],
    "envelope": POINT_COLORS["envelope"],
    "forward_sampling": POINT_COLORS["forward_sampling"],
    "conformal": CONFORMAL_COLOR,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/taxinet_v2/operating_pareto_sweep/results.json"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/taxinet_v2/operating_pareto_sweep/figures"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("results/taxinet_v2/operating_pareto_sweep/evaluation_summary.md"),
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _curve(results: dict, perception: str, method: str) -> List[Tuple[float, dict]]:
    points = [
        (float(beta_key), metrics)
        for beta_key, metrics in results["point_sweep"][perception][method].items()
    ]
    return sorted(points, key=lambda item: item[0])


def _conf_curve(results: dict, perception: str, confidence_level: str) -> List[Tuple[float, dict]]:
    points = [
        (float(filter_key), metrics)
        for filter_key, metrics in results["conformal_sweep"][perception][confidence_level].items()
    ]
    return sorted(points, key=lambda item: item[0])


def make_pareto_figure(results: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, perception in zip(axes, ("uniform", "adversarial_opt")):
        for method in POINT_METHODS:
            curve = _curve(results, perception, method)
            xs = [metrics["stuck_rate"] * 100 for _, metrics in curve]
            ys = [metrics["fail_rate"] * 100 for _, metrics in curve]
            ax.scatter(
                xs,
                ys,
                color=POINT_COLORS[method],
                marker=POINT_MARKERS[method],
                s=65,
                label=f"{POINT_LABELS[method]} (beta)",
            )
        for confidence_level in results["metadata"]["confidence_levels"]:
            curve = _conf_curve(results, perception, confidence_level)
            xs = [metrics["stuck_rate"] * 100 for _, metrics in curve]
            ys = [metrics["fail_rate"] * 100 for _, metrics in curve]
            ax.scatter(
                xs,
                ys,
                color=CONF_COLORS[confidence_level],
                marker=CONF_MARKERS[confidence_level],
                s=70,
                label=f"Conformal conf={confidence_level}",
            )
        ax.set_title("Uniform perception" if perception == "uniform" else "Shared adversarial perception")
        ax.set_xlabel("Stuck rate (%)")
        ax.set_ylabel("Fail rate (%)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, framealpha=0.95)
    fig.suptitle("TaxiNetV2 RL Operating Sweep: beta for point shields, conf/af for conformal", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _best_score(metrics: dict) -> Tuple[float, float, float]:
    return (-metrics["safe_rate"], metrics["fail_rate"], metrics["stuck_rate"])


def _best_point_method(results: dict, perception: str, method: str) -> Tuple[str, dict]:
    rows = [
        (f"beta={beta:.2f}", metrics)
        for beta, metrics in _curve(results, perception, method)
    ]
    return min(rows, key=lambda item: _best_score(item[1]))


def _best_conformal(results: dict, perception: str) -> Tuple[str, dict]:
    rows = []
    for confidence_level in results["metadata"]["confidence_levels"]:
        for action_filter, metrics in _conf_curve(results, perception, confidence_level):
            rows.append((f"conf={confidence_level}, af={action_filter:.1f}", metrics))
    return min(rows, key=lambda item: _best_score(item[1]))


def _best_bar_rows(results: dict, perception: str) -> Dict[str, dict]:
    best_rows: Dict[str, dict] = {}
    for method in POINT_METHODS:
        setting, metrics = _best_point_method(results, perception, method)
        best_rows[method] = {"setting": setting, **metrics}
    setting, metrics = _best_conformal(results, perception)
    best_rows["conformal"] = {"setting": setting, **metrics}
    return best_rows


def _draw_best_bars(ax, best_rows: Dict[str, dict], title: str) -> None:
    methods = [method for method in BEST_BAR_ORDER if method in best_rows]
    fails = [best_rows[method]["fail_rate"] * 100 for method in methods]
    stucks = [best_rows[method]["stuck_rate"] * 100 for method in methods]
    safes = [best_rows[method]["safe_rate"] * 100 for method in methods]
    colors = [BEST_BAR_COLORS[method] for method in methods]
    x = np.arange(len(methods))
    width = 0.58

    ax.bar(x, fails, width, color=colors, alpha=0.9, zorder=3)
    ax.bar(
        x,
        stucks,
        width,
        bottom=fails,
        color=colors,
        alpha=0.34,
        hatch="///",
        edgecolor="white",
        zorder=3,
    )
    ax.bar(x, safes, width, bottom=np.array(fails) + np.array(stucks), color=colors, alpha=0.14, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([BEST_BAR_LABELS[method] for method in methods], fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)


def make_best_bar_figure(results: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    fig.suptitle("Best Swept Operating Point per Method", fontsize=11)

    for ax, perception in zip(axes, ("uniform", "adversarial_opt")):
        rows = _best_bar_rows(results, perception)
        title = "Uniform perception" if perception == "uniform" else "Shared adversarial perception"
        _draw_best_bars(ax, rows, title)

    fail_patch = mpatches.Patch(facecolor="gray", alpha=0.9, label="Fail")
    stuck_patch = mpatches.Patch(facecolor="gray", alpha=0.34, hatch="///", edgecolor="white", label="Stuck")
    safe_patch = mpatches.Patch(facecolor="gray", alpha=0.14, label="Safe")
    axes[1].legend(handles=[fail_patch, stuck_patch, safe_patch], fontsize=8, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _best_entry(rows: Iterable[Tuple[str, dict]], key: str) -> Tuple[str, dict]:
    return min(rows, key=lambda item: item[1][key])


def _iter_perception_rows(results: dict, perception: str) -> Iterable[Tuple[str, dict]]:
    for method in POINT_METHODS:
        for beta, metrics in _curve(results, perception, method):
            yield (f"{POINT_LABELS[method]} beta={beta:.2f}", metrics)
    for confidence_level in results["metadata"]["confidence_levels"]:
        for action_filter, metrics in _conf_curve(results, perception, confidence_level):
            yield (f"Conformal conf={confidence_level}, af={action_filter:.1f}", metrics)


def _table_lines(results: dict, perception: str) -> List[str]:
    lines = [
        "| Operating point | Fail | Stuck | Safe | Intervention |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metrics in _iter_perception_rows(results, perception):
        lines.append(
            f"| {label} | {metrics['fail_rate']:.1%} | {metrics['stuck_rate']:.1%} | "
            f"{metrics['safe_rate']:.1%} | {metrics['intervention_rate']:.1%} |"
        )
    return lines


def write_summary(results: dict, figure_path: Path, bar_figure_path: Path, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    meta = results["metadata"]
    uniform_best_fail = _best_entry(_iter_perception_rows(results, "uniform"), "fail_rate")
    adv_best_fail = _best_entry(_iter_perception_rows(results, "adversarial_opt"), "fail_rate")
    uniform_best_stuck = _best_entry(_iter_perception_rows(results, "uniform"), "stuck_rate")
    adv_best_stuck = _best_entry(_iter_perception_rows(results, "adversarial_opt"), "stuck_rate")
    uniform_best_by_method = _best_bar_rows(results, "uniform")
    adv_best_by_method = _best_bar_rows(results, "adversarial_opt")
    lines: List[str] = [
        "# TaxiNetV2 Operating Pareto Sweep",
        "",
        "## Setup",
        "",
        f"- Selector: `{meta['selector']}` only.",
        f"- Shared RL controller cache: `{meta['setup']['rl_cache_path']}`.",
        f"- Shared adversarial realization cache: `{meta['setup']['opt_cache_paths_by_target']['envelope']}`.",
        f"- Beta grid for point shields: `{meta['beta_values']}`.",
        f"- Conformal confidence levels: `{meta['confidence_levels']}`.",
        f"- Conformal action-filter grid: `{meta['action_filter_values']}`.",
        f"- Trials per point: `{meta['run_config']['num_trials']}` with horizon `{meta['run_config']['trial_length']}` and seed `{meta['run_config']['seed']}`.",
        "",
        "## High-Level Findings",
        "",
        f"- Uniform lowest fail: `{uniform_best_fail[0]}` at `{uniform_best_fail[1]['fail_rate']:.1%}` fail and `{uniform_best_fail[1]['stuck_rate']:.1%}` stuck.",
        f"- Adversarial lowest fail: `{adv_best_fail[0]}` at `{adv_best_fail[1]['fail_rate']:.1%}` fail and `{adv_best_fail[1]['stuck_rate']:.1%}` stuck.",
        f"- Uniform lowest stuck: `{uniform_best_stuck[0]}` at `{uniform_best_stuck[1]['stuck_rate']:.1%}` stuck and `{uniform_best_stuck[1]['fail_rate']:.1%}` fail.",
        f"- Adversarial lowest stuck: `{adv_best_stuck[0]}` at `{adv_best_stuck[1]['stuck_rate']:.1%}` stuck and `{adv_best_stuck[1]['fail_rate']:.1%}` fail.",
        "- Only the conformal shield moves with `conf` and `af`; `single_belief`, `envelope`, and `forward_sampling` move only with `beta`.",
        "- The RL controller and adversarial realization are held fixed across the whole sweep, so the plotted differences are shield-operating-point differences rather than controller/retraining differences.",
        "- In the stacked-bar figure, each method keeps only its highest-safe operating point, tie-broken by lower fail and then lower stuck.",
        "",
        "## Best Point per Method",
        "",
        "### Uniform perception",
        "",
        "| Method | Setting | Fail | Stuck | Safe |",
        "|---|---|---:|---:|---:|",
        *[
            f"| {BEST_BAR_LABELS[method]} | `{uniform_best_by_method[method]['setting']}` | "
            f"{uniform_best_by_method[method]['fail_rate']:.1%} | "
            f"{uniform_best_by_method[method]['stuck_rate']:.1%} | "
            f"{uniform_best_by_method[method]['safe_rate']:.1%} |"
            for method in BEST_BAR_ORDER
        ],
        "",
        "### Shared adversarial perception",
        "",
        "| Method | Setting | Fail | Stuck | Safe |",
        "|---|---|---:|---:|---:|",
        *[
            f"| {BEST_BAR_LABELS[method]} | `{adv_best_by_method[method]['setting']}` | "
            f"{adv_best_by_method[method]['fail_rate']:.1%} | "
            f"{adv_best_by_method[method]['stuck_rate']:.1%} | "
            f"{adv_best_by_method[method]['safe_rate']:.1%} |"
            for method in BEST_BAR_ORDER
        ],
        "",
        "## Uniform RL Results",
        "",
        *_table_lines(results, "uniform"),
        "",
        "## Adversarial RL Results",
        "",
        *_table_lines(results, "adversarial_opt"),
        "",
        "## Figure",
        "",
        f"- Pareto scatter: ![]({figure_path.relative_to(summary_path.parent)})",
        f"- Best-point bars: ![]({bar_figure_path.relative_to(summary_path.parent)})",
    ]
    summary_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    results = _load(args.results_json)
    figure_path = make_pareto_figure(results, args.figures_dir / "pareto_scatter.png")
    bar_figure_path = make_best_bar_figure(results, args.figures_dir / "best_method_bars.png")
    write_summary(results, figure_path, bar_figure_path, args.summary_md)
    print(f"Wrote figure to {figure_path}")
    print(f"Wrote figure to {bar_figure_path}")
    print(f"Wrote summary to {args.summary_md}")


if __name__ == "__main__":
    main()
