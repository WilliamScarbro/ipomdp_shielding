"""Generate a compact evaluation summary with figures for TaxiNetV2 low-accuracy runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results/taxinet_v2/conformal_rl_sweep")
HEADLINE_JSON = RESULTS_DIR / "headline_1000_lowacc87_93.json"
BETA_SWEEP_JSON = RESULTS_DIR / "beta_sweep_lowacc87_93.json"
OUT_DIR = RESULTS_DIR / "figures_lowacc87_93"
SUMMARY_MD = RESULTS_DIR / "evaluation_summary_lowacc87_93.md"
LOWACC_METRICS_JSON = Path("results/cache/taxinet_v2_lowacc_axis_noise_047_053/metrics.json")
SCARBRO_JSON = Path("ipomdp_shielding_/results/taxinet_v2/scarbro_baseline_import.json")

COLORS = {
    "single_belief": "#1565C0",
    "envelope": "#E65100",
    "forward_sampling": "#00838F",
    "cp_control_conformal": "#6A1B9A",
}


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _ordered_headline_rows(headline: dict) -> List[Tuple[str, dict]]:
    rows = [
        ("single_belief", headline["point_baselines"]["single_belief"]),
        ("envelope", headline["point_baselines"]["envelope"]),
        ("forward_sampling", headline["point_baselines"]["forward_sampling"]),
    ]
    for confidence in ["0.95", "0.99", "0.995"]:
        rows.append((f"conformal {confidence}", headline["conformal_results"][confidence]["cp_control_conformal"]))
    return rows


def make_headline_bars(headline: dict) -> Path:
    rows = _ordered_headline_rows(headline)
    labels = [name for name, _ in rows]
    fail = np.array([metrics["fail_rate"] * 100 for _, metrics in rows])
    stuck = np.array([metrics["stuck_rate"] * 100 for _, metrics in rows])
    safe = np.array([metrics["safe_rate"] * 100 for _, metrics in rows])
    colors = [COLORS["cp_control_conformal"] if name.startswith("conformal") else COLORS[name] for name in labels]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, fail, color=colors, label="Fail")
    ax.bar(x, stuck, bottom=fail, color=colors, alpha=0.35, hatch="///", edgecolor="white", label="Stuck")
    ax.bar(x, safe, bottom=fail + stuck, color=colors, alpha=0.12, label="Safe")
    ax.set_ylabel("Rate (%)")
    ax.set_title("TaxiNetV2 Low-Accuracy Headline Results")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "headline_rates.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_conformal_size_plot(headline: dict) -> Path:
    confidences = ["0.95", "0.99", "0.995"]
    sizes = [
        headline["conformal_results"][confidence]["cp_control_conformal"]["mean_conformal_cartesian_size"]
        for confidence in confidences
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(confidences, sizes, color=COLORS["cp_control_conformal"])
    ax.set_ylabel("Mean Cartesian conformal set size")
    ax.set_xlabel("Confidence level")
    ax.set_title("Conformal Set Growth at Lower Base Accuracy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "conformal_set_sizes.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _curve_from_beta_sweep(beta_sweep: dict, method: str, confidence_level: str = "") -> List[Tuple[float, float, float]]:
    points = []
    for beta_key, result in sorted(beta_sweep["grid_results"].items(), key=lambda item: float(item[0])):
        beta = result["beta"]
        if method == "cp_control_conformal":
            metrics = result["conformal_results"][confidence_level]["cp_control_conformal"]
        else:
            metrics = result["point_baselines"][method]
        points.append((beta, metrics["fail_rate"], metrics["stuck_rate"]))
    return points


def make_beta_pareto(beta_sweep: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    panels = [
        ("Point shields", [("single_belief", ""), ("envelope", ""), ("forward_sampling", "")]),
        ("Conformal shield", [("cp_control_conformal", "0.95"), ("cp_control_conformal", "0.99"), ("cp_control_conformal", "0.995")]),
    ]
    for ax, (title, series) in zip(axes, panels):
        for method, confidence in series:
            curve = _curve_from_beta_sweep(beta_sweep, method, confidence)
            xs = [stuck * 100 for _, _, stuck in curve]
            ys = [fail * 100 for _, fail, _ in curve]
            thresholds = [beta for beta, _, _ in curve]
            label = method if method != "cp_control_conformal" else f"conformal {confidence}"
            color = COLORS["cp_control_conformal"] if method == "cp_control_conformal" else COLORS[method]
            ax.plot(xs, ys, marker="o", color=color, label=label)
            for x, y, threshold in zip(xs, ys, thresholds):
                ax.annotate(f"{threshold:.2f}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6)
        ax.set_title(title)
        ax.set_xlabel("Stuck rate (%)")
        ax.set_ylabel("Fail rate (%)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("TaxiNetV2 Sweep: beta for point shields, action_filter for conformal")
    fig.tight_layout()

    out = OUT_DIR / "beta_actionfilter_pareto.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _scarbro_rows(scarbro: dict) -> List[Tuple[str, float, float]]:
    rows = []
    for confidence in ["0.95", "0.99", "0.995"]:
        match = None
        for variant in scarbro.get("variants", []):
            metadata = variant.get("metadata", {})
            if (
                metadata.get("confidence_level") == confidence
                and metadata.get("action_filter_tag") == "af7"
                and metadata.get("default_action") is True
            ):
                match = variant
                break
        if match is None:
            raise KeyError(f"Missing Scarbro af7/default-action variant for confidence={confidence}")
        summary = match["summary"]
        rows.append(
            (
                confidence,
                summary["crash"]["value"],
                summary["stuck_or_default"]["value"],
            )
        )
    return rows


def make_scarbro_compare(headline: dict, scarbro: dict) -> Path:
    confidences = ["0.95", "0.99", "0.995"]
    ours_fail = [
        headline["conformal_results"][confidence]["cp_control_conformal"]["fail_rate"] * 100
        for confidence in confidences
    ]
    ours_stuck = [
        headline["conformal_results"][confidence]["cp_control_conformal"]["stuck_rate"] * 100
        for confidence in confidences
    ]
    scarbro_rows = _scarbro_rows(scarbro)
    scarbro_fail = [row[1] * 100 for row in scarbro_rows]
    scarbro_stuck = [row[2] * 100 for row in scarbro_rows]

    x = np.arange(len(confidences))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - 1.5 * width, ours_fail, width, color="#8E24AA", label="Our fail")
    ax.bar(x - 0.5 * width, scarbro_fail, width, color="#CE93D8", label="Scarbro crash")
    ax.bar(x + 0.5 * width, ours_stuck, width, color="#3949AB", label="Our stuck")
    ax.bar(x + 1.5 * width, scarbro_stuck, width, color="#9FA8DA", label="Scarbro default")
    ax.set_xticks(x)
    ax.set_xticklabels(confidences)
    ax.set_xlabel("Confidence level")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Conformal Low-Accuracy Run vs Scarbro Import (af7)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = OUT_DIR / "scarbro_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary(
    headline: dict,
    beta_sweep: dict,
    lowacc_metrics: dict,
    scarbro: dict,
    figures: Dict[str, Path],
) -> None:
    accuracy = lowacc_metrics["test_metrics"]
    rows = _ordered_headline_rows(headline)
    beta_080 = beta_sweep["grid_results"]["0.80"]
    beta_095 = beta_sweep["grid_results"]["0.95"]
    scarbro_rows = _scarbro_rows(scarbro)

    lines: List[str] = []
    lines.append("# TaxiNetV2 Low-Accuracy Evaluation Summary\n")
    lines.append("## Setup\n")
    point_realization = headline.get("metadata", {}).get("point_realization", "unknown")
    lines.append(
        f"- Base checkpoint: `results/cache/taxinet_v2_lowacc_axis_noise_047_053/best_in_band_model.pth`\n"
        f"- Test accuracy: CTE `{accuracy['cte_accuracy']:.2%}`, HE `{accuracy['he_accuracy']:.2%}`, joint `{accuracy['joint_accuracy']:.2%}`\n"
        f"- Point estimate realization: `{point_realization}` modular realization over the TaxiNetV2 point-estimate IPOMDP\n"
        f"- Headline run: `1000` trials, horizon `30`, initial state `safe`, point-shield beta `0.8`, conformal `action_filter=0.7`\n"
        f"- Beta sweep: `{beta_sweep['metadata']['args']['num_trials']}` trials per operating point, beta/action-filter grid `{beta_sweep['metadata']['args']['threshold_values']}`\n"
    )

    lines.append("## Headline Results\n")
    lines.append("| Method | Fail | Stuck | Safe | Intervention |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, metrics in rows:
        lines.append(
            f"| {name} | {metrics['fail_rate']:.1%} | {metrics['stuck_rate']:.1%} | "
            f"{metrics['safe_rate']:.1%} | {metrics.get('intervention_rate', 0.0):.1%} |"
        )

    lines.append("\n## Observations\n")
    lines.append(
        f"- Lowering the base model accuracy to roughly `87/93` per axis moves conformal behavior into the intended regime: mean Cartesian set size rises from `{headline['conformal_results']['0.95']['cp_control_conformal']['mean_conformal_cartesian_size']:.2f}` at `0.95` to `{headline['conformal_results']['0.995']['cp_control_conformal']['mean_conformal_cartesian_size']:.2f}` at `0.995`."
    )
    lines.append(
        f"- At the headline operating point, `0.99` and `0.995` conformal control are extremely conservative: fail falls to `{headline['conformal_results']['0.99']['cp_control_conformal']['fail_rate']:.1%}` and `{headline['conformal_results']['0.995']['cp_control_conformal']['fail_rate']:.1%}`, but stuck rises to `{headline['conformal_results']['0.99']['cp_control_conformal']['stuck_rate']:.1%}` and `{headline['conformal_results']['0.995']['cp_control_conformal']['stuck_rate']:.1%}`."
    )
    lines.append(
        f"- The new sweep shows the usual beta tradeoff for point shields. At `beta=0.80`, envelope reaches fail `{beta_080['point_baselines']['envelope']['fail_rate']:.1%}` / stuck `{beta_080['point_baselines']['envelope']['stuck_rate']:.1%}`. By `beta=0.95`, it shifts to fail `{beta_095['point_baselines']['envelope']['fail_rate']:.1%}` / stuck `{beta_095['point_baselines']['envelope']['stuck_rate']:.1%}`."
    )
    lines.append(
        "- Sweeping conformal `action_filter` over the same numeric range gives a comparable Pareto curve, but the confidence level dominates behavior more strongly than the filter does. `0.95` remains usable; `0.99+` quickly collapses into near-total stuck behavior."
    )

    lines.append("\n## Scarbro Comparison\n")
    lines.append("| Confidence | Scarbro crash | Scarbro default | Our fail | Our stuck |")
    lines.append("|---|---:|---:|---:|---:|")
    for confidence, scarbro_fail, scarbro_stuck in scarbro_rows:
        ours = headline["conformal_results"][confidence]["cp_control_conformal"]
        lines.append(
            f"| {confidence} | {scarbro_fail:.1%} | {scarbro_stuck:.1%} | {ours['fail_rate']:.1%} | {ours['stuck_rate']:.1%} |"
        )
    lines.append(
        "\nThese are still not apples-to-apples with Scarbro et al.: their imported numbers are PRISM properties over a different controller/default-action semantics, while these are Monte Carlo RL-selector evaluations using the local paired-event artifact model."
    )

    lines.append("\n## Figures\n")
    for label, path in figures.items():
        lines.append(f"- {label}: ![]({path.relative_to(RESULTS_DIR)})")

    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headline = _load_json(HEADLINE_JSON)
    beta_sweep = _load_json(BETA_SWEEP_JSON)
    lowacc_metrics = _load_json(LOWACC_METRICS_JSON)
    scarbro = _load_json(SCARBRO_JSON)

    figures = {
        "Headline bars": make_headline_bars(headline),
        "Conformal set sizes": make_conformal_size_plot(headline),
        "Beta/action-filter Pareto": make_beta_pareto(beta_sweep),
        "Scarbro comparison": make_scarbro_compare(headline, scarbro),
    }
    write_summary(headline, beta_sweep, lowacc_metrics, scarbro, figures)
    print(f"Wrote summary markdown to {SUMMARY_MD}")
    for label, path in figures.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
