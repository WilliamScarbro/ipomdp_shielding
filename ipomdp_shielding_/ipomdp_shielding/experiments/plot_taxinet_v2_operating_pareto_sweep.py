"""Plot and summarize TaxiNetV2 operating sweep results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
            ax.plot(
                xs,
                ys,
                color=POINT_COLORS[method],
                marker=POINT_MARKERS[method],
                linewidth=1.8,
                label=f"{POINT_LABELS[method]} (beta)",
            )
            for beta, metrics in curve:
                ax.annotate(
                    f"β={beta:.2f}",
                    (metrics["stuck_rate"] * 100, metrics["fail_rate"] * 100),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=6,
                    color=POINT_COLORS[method],
                )
        for confidence_level in results["metadata"]["confidence_levels"]:
            curve = _conf_curve(results, perception, confidence_level)
            xs = [metrics["stuck_rate"] * 100 for _, metrics in curve]
            ys = [metrics["fail_rate"] * 100 for _, metrics in curve]
            ax.plot(
                xs,
                ys,
                color=CONF_COLORS[confidence_level],
                marker=CONF_MARKERS[confidence_level],
                linewidth=1.6,
                linestyle="--",
                label=f"Conformal conf={confidence_level}",
            )
            for action_filter, metrics in curve:
                ax.annotate(
                    f"af={action_filter:.1f}",
                    (metrics["stuck_rate"] * 100, metrics["fail_rate"] * 100),
                    textcoords="offset points",
                    xytext=(4, -10),
                    fontsize=6,
                    color=CONF_COLORS[confidence_level],
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


def write_summary(results: dict, figure_path: Path, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    meta = results["metadata"]
    uniform_best_fail = _best_entry(_iter_perception_rows(results, "uniform"), "fail_rate")
    adv_best_fail = _best_entry(_iter_perception_rows(results, "adversarial_opt"), "fail_rate")
    uniform_best_stuck = _best_entry(_iter_perception_rows(results, "uniform"), "stuck_rate")
    adv_best_stuck = _best_entry(_iter_perception_rows(results, "adversarial_opt"), "stuck_rate")
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
    ]
    summary_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    results = _load(args.results_json)
    figure_path = make_pareto_figure(results, args.figures_dir / "pareto_scatter.png")
    write_summary(results, figure_path, args.summary_md)
    print(f"Wrote figure to {figure_path}")
    print(f"Wrote summary to {args.summary_md}")


if __name__ == "__main__":
    main()
