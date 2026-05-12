"""Threshold/action-filter sweep for TaxiNetV2 conformal RL evaluation.

For point-IPOMDP shields, we sweep the real-time shield threshold `beta`.
For the cp-control conformal shield, we sweep `action_filter` over the same
numeric grid so the curves can be compared on a common operating-axis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .run_taxinet_v2_conformal_rl_sweep import CONFORMAL_METHOD, POINT_METHODS, run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=500)
    parser.add_argument("--trial-length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-method", default="Clopper_Pearson")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--confidence-levels",
        nargs="+",
        default=["0.95", "0.99", "0.995"],
    )
    parser.add_argument(
        "--threshold-values",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    )
    parser.add_argument(
        "--action-filter-values",
        nargs="+",
        type=float,
        default=None,
        help="If omitted, reuse threshold-values.",
    )
    parser.add_argument("--action-success", type=float, default=0.9)
    parser.add_argument("--forward-budget", type=int, default=500)
    parser.add_argument("--forward-k-samples", type=int, default=100)
    parser.add_argument(
        "--point-realization",
        choices=["uniform", "adversarial"],
        default="uniform",
    )
    parser.add_argument("--rl-episodes", type=int, default=500)
    parser.add_argument("--rl-episode-length", type=int, default=None)
    parser.add_argument(
        "--rl-cache-path",
        type=Path,
        default=Path("results/cache/rl_taxinet_v2_point_modular_uniform_h30_paper_model_agent.pt"),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("/home/dev/cp-control/train/models/best_model.pth"),
    )
    parser.add_argument("--measured-cte-accuracy", type=float, default=None)
    parser.add_argument("--measured-he-accuracy", type=float, default=None)
    parser.add_argument("--measured-joint-accuracy", type=float, default=None)
    parser.add_argument(
        "--conformal-mode",
        default="shared-event/axis-paired",
    )
    parser.add_argument(
        "--initial",
        choices=["safe", "boundary"],
        default="safe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/taxinet_v2/conformal_rl_sweep/beta_sweep_paper_model.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("results/taxinet_v2/conformal_rl_sweep/beta_sweep_paper_model.csv"),
    )
    args = parser.parse_args()
    if args.rl_episode_length is None:
        args.rl_episode_length = args.trial_length
    if args.action_filter_values is None:
        args.action_filter_values = list(args.threshold_values)
    if len(args.threshold_values) != len(args.action_filter_values):
        raise ValueError("threshold-values and action-filter-values must have the same length")
    return args


def _result_row(
    *,
    beta: float,
    action_filter: float,
    method: str,
    confidence_level: str,
    metrics: Dict[str, float],
) -> dict:
    return {
        "beta": beta,
        "action_filter": action_filter,
        "method": method,
        "confidence_level": confidence_level,
        "fail_rate": metrics.get("fail_rate", 0.0),
        "stuck_rate": metrics.get("stuck_rate", 0.0),
        "safe_rate": metrics.get("safe_rate", 0.0),
        "mean_steps": metrics.get("mean_steps", 0.0),
        "mean_stuck_count": metrics.get("mean_stuck_count", 0.0),
        "intervention_rate": metrics.get("intervention_rate", 0.0),
        "mean_conformal_cartesian_size": metrics.get("mean_conformal_cartesian_size", 0.0),
    }


def _write_rows(rows: Iterable[dict], path: Path) -> None:
    fieldnames = [
        "beta",
        "action_filter",
        "method",
        "confidence_level",
        "fail_rate",
        "stuck_rate",
        "safe_rate",
        "mean_steps",
        "mean_stuck_count",
        "intervention_rate",
        "mean_conformal_cartesian_size",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _single_run_args(args: argparse.Namespace, beta: float, action_filter: float) -> argparse.Namespace:
    return argparse.Namespace(
        num_trials=args.num_trials,
        trial_length=args.trial_length,
        seed=args.seed,
        confidence_method=args.confidence_method,
        alpha=args.alpha,
        confidence_levels=list(args.confidence_levels),
        shield_threshold=beta,
        action_filter=action_filter,
        action_success=args.action_success,
        forward_budget=args.forward_budget,
        forward_k_samples=args.forward_k_samples,
        point_realization=args.point_realization,
        rl_episodes=args.rl_episodes,
        rl_episode_length=args.rl_episode_length,
        rl_cache_path=args.rl_cache_path,
        base_checkpoint=args.base_checkpoint,
        measured_cte_accuracy=args.measured_cte_accuracy,
        measured_he_accuracy=args.measured_he_accuracy,
        measured_joint_accuracy=args.measured_joint_accuracy,
        conformal_mode=args.conformal_mode,
        store_trajectories=False,
        initial=args.initial,
        output=Path("__unused__.json"),
        csv_output=Path("__unused__.csv"),
    )


def run_beta_sweep(args: argparse.Namespace) -> dict:
    rows: List[dict] = []
    grid_results: Dict[str, dict] = {}

    for beta, action_filter in zip(args.threshold_values, args.action_filter_values):
        print(f"Running beta={beta:.2f} action_filter={action_filter:.2f}")
        run_args = _single_run_args(args, beta, action_filter)
        result = run_sweep(run_args)
        beta_key = f"{beta:.2f}"
        grid_results[beta_key] = {
            "beta": beta,
            "action_filter": action_filter,
            "point_baselines": result["point_baselines"],
            "conformal_results": result["conformal_results"],
        }

        for method in POINT_METHODS:
            metrics = result["point_baselines"][method]
            rows.append(
                _result_row(
                    beta=beta,
                    action_filter=action_filter,
                    method=method,
                    confidence_level="",
                    metrics=metrics,
                )
            )
        for confidence_level, methods in result["conformal_results"].items():
            metrics = methods[CONFORMAL_METHOD]
            rows.append(
                _result_row(
                    beta=beta,
                    action_filter=action_filter,
                    method=CONFORMAL_METHOD,
                    confidence_level=confidence_level,
                    metrics=metrics,
                )
            )

        payload = {
            "metadata": {
                "case_study": "taxinet_v2_beta_sweep",
                "args": {
                    "num_trials": args.num_trials,
                    "trial_length": args.trial_length,
                    "seed": args.seed,
                    "confidence_method": args.confidence_method,
                    "alpha": args.alpha,
                    "confidence_levels": list(args.confidence_levels),
                    "threshold_values": list(args.threshold_values),
                    "action_filter_values": list(args.action_filter_values),
                    "action_success": args.action_success,
                    "forward_budget": args.forward_budget,
                    "forward_k_samples": args.forward_k_samples,
                    "point_realization": args.point_realization,
                    "rl_episodes": args.rl_episodes,
                    "rl_episode_length": args.rl_episode_length,
                    "rl_cache_path": str(args.rl_cache_path),
                    "base_checkpoint": str(args.base_checkpoint),
                    "measured_cte_accuracy": args.measured_cte_accuracy,
                    "measured_he_accuracy": args.measured_he_accuracy,
                    "measured_joint_accuracy": args.measured_joint_accuracy,
                    "conformal_mode": args.conformal_mode,
                    "initial": args.initial,
                    "output": str(args.output),
                    "csv_output": str(args.csv_output),
                },
            },
            "grid_results": grid_results,
        }
        _write_rows(rows, args.csv_output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(payload, handle, indent=2)

    _write_rows(rows, args.csv_output)
    payload = {
        "metadata": {
            "case_study": "taxinet_v2_beta_sweep",
            "args": {
                "num_trials": args.num_trials,
                "trial_length": args.trial_length,
                "seed": args.seed,
                "confidence_method": args.confidence_method,
                "alpha": args.alpha,
                "confidence_levels": list(args.confidence_levels),
                "threshold_values": list(args.threshold_values),
                "action_filter_values": list(args.action_filter_values),
                "action_success": args.action_success,
                "forward_budget": args.forward_budget,
                "forward_k_samples": args.forward_k_samples,
                "point_realization": args.point_realization,
                "rl_episodes": args.rl_episodes,
                "rl_episode_length": args.rl_episode_length,
                "rl_cache_path": str(args.rl_cache_path),
                "base_checkpoint": str(args.base_checkpoint),
                "measured_cte_accuracy": args.measured_cte_accuracy,
                "measured_he_accuracy": args.measured_he_accuracy,
                "measured_joint_accuracy": args.measured_joint_accuracy,
                "conformal_mode": args.conformal_mode,
                "initial": args.initial,
                "output": str(args.output),
                "csv_output": str(args.csv_output),
            },
        },
        "grid_results": grid_results,
    }
    return payload


def main() -> None:
    args = parse_args()
    result = run_beta_sweep(args)
    print(f"Saved TaxiNetV2 beta sweep JSON to {args.output}")
    print(f"Saved tidy CSV to {args.csv_output}")
    for beta_key, beta_result in result["grid_results"].items():
        print(
            f"beta={beta_key} action_filter={beta_result['action_filter']:.2f} "
            f"single_belief fail={beta_result['point_baselines']['single_belief']['fail_rate']:.1%} "
            f"stuck={beta_result['point_baselines']['single_belief']['stuck_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
