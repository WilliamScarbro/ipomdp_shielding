"""Gate and, if eligible, rerun the TaxiNetV2 conformal RL paper-model path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipomdp_shielding.experiments.run_taxinet_v2_beta_sweep import run_beta_sweep
from ipomdp_shielding.experiments.run_taxinet_v2_conformal_rl_sweep import run_sweep
from recreate_taxinet_v2_perception_artifacts import (
    axis_accuracy_metrics,
    ensure_cp_control_imports,
    load_model,
    make_loader,
    run_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-control-root", type=Path, default=Path("/home/dev/cp-control"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/dev/cp-control/train/models/best_model.pth"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/home/dev/cp-control/data/taxinet"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/dev/ipomdp_shielding/results/taxinet_v2/conformal_rl_sweep"),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(
            "/home/dev/ipomdp_shielding/ipomdp_shielding_/ipomdp_shielding/CaseStudies/TaxiNetV2/artifacts"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headline-trials", type=int, default=1000)
    parser.add_argument("--sweep-trials", type=int, default=100)
    parser.add_argument("--trial-length", type=int, default=30)
    parser.add_argument("--shield-threshold", type=float, default=0.8)
    parser.add_argument("--action-filter", type=float, default=0.7)
    parser.add_argument(
        "--threshold-values",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    )
    parser.add_argument(
        "--confidence-levels",
        nargs="+",
        default=["0.95", "0.99", "0.995"],
    )
    parser.add_argument(
        "--rl-cache-path",
        type=Path,
        default=Path("/home/dev/ipomdp_shielding/results/cache/rl_taxinet_v2_point_modular_uniform_h30_paper_model_agent.pt"),
    )
    parser.add_argument("--conformal-mode", default="shared-event/axis-paired")
    return parser.parse_args()


def evaluate_test_accuracy(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset_cls, model_cls = ensure_cp_control_imports(args.cp_control_root)
    model = load_model(model_cls, args.model_path, device)
    train_models = args.cp_control_root / "train" / "models"
    split_paths = {
        split: train_models / f"{split}_indices.pt"
        for split in ("train", "val", "cal", "test")
    }
    _, _, test_loader, test_indices = make_loader(
        dataset_cls,
        args.data_dir,
        split_paths["test"],
        args.batch_size,
        args.num_workers,
    )
    test_cte, test_he = run_inference(model, test_loader, device)
    metrics = axis_accuracy_metrics(test_cte, test_he)
    return {
        "checkpoint_path": str(args.model_path),
        "split_paths": {name: str(path) for name, path in split_paths.items()},
        "split_evaluated": "test",
        "num_test_samples": len(test_indices),
        **metrics,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def write_abort_note(path: Path, payload: dict) -> None:
    note = (
        "# TaxiNetV2 Paper-Model Gate Rejected\n\n"
        f"- Checkpoint: `{payload['checkpoint_path']}`\n"
        f"- Measured test accuracy: CTE `{payload['cte_accuracy']:.2%}`, "
        f"HE `{payload['he_accuracy']:.2%}`, joint `{payload['joint_accuracy']:.2%}`\n"
        "- Gate rule: abort if either axis exceeds `93%` test accuracy\n"
        "- Decision: rejected for lack of conformal headroom; no perception artifacts or experiment outputs were regenerated.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note)


def run_subprocess(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd="/home/dev/ipomdp_shielding", check=True)


def main() -> None:
    args = parse_args()
    preflight = evaluate_test_accuracy(args)
    preflight_path = args.results_dir / "paper_model_preflight.json"
    write_json(preflight_path, preflight)

    if preflight["cte_accuracy"] > 0.93 or preflight["he_accuracy"] > 0.93:
        write_abort_note(args.results_dir / "paper_model_decision.md", preflight)
        print(f"Gate rejected. Wrote {args.results_dir / 'paper_model_decision.md'}")
        return

    run_subprocess(
        [
            sys.executable,
            str(ROOT / "scripts" / "recreate_taxinet_v2_perception_artifacts.py"),
            "--cp-control-root",
            str(args.cp_control_root),
            "--model-path",
            str(args.model_path),
            "--data-dir",
            str(args.data_dir),
            "--output-root",
            str(args.artifacts_root),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
    )

    common = dict(
        trial_length=args.trial_length,
        seed=args.seed,
        confidence_method="Clopper_Pearson",
        alpha=0.05,
        confidence_levels=list(args.confidence_levels),
        action_success=0.9,
        forward_budget=500,
        forward_k_samples=100,
        point_realization="uniform",
        rl_episodes=500,
        rl_episode_length=args.trial_length,
        rl_cache_path=args.rl_cache_path,
        base_checkpoint=args.model_path,
        measured_cte_accuracy=preflight["cte_accuracy"],
        measured_he_accuracy=preflight["he_accuracy"],
        measured_joint_accuracy=preflight["joint_accuracy"],
        conformal_mode=args.conformal_mode,
        initial="safe",
    )

    headline_args = argparse.Namespace(
        num_trials=args.headline_trials,
        shield_threshold=args.shield_threshold,
        action_filter=args.action_filter,
        store_trajectories=False,
        output=args.results_dir / "headline_1000_paper_model.json",
        csv_output=args.results_dir / "headline_1000_paper_model.csv",
        **common,
    )
    headline = run_sweep(headline_args)
    write_json(headline_args.output, headline)

    beta_args = argparse.Namespace(
        num_trials=args.sweep_trials,
        threshold_values=list(args.threshold_values),
        action_filter_values=list(args.threshold_values),
        output=args.results_dir / "beta_sweep_paper_model.json",
        csv_output=args.results_dir / "beta_sweep_paper_model.csv",
        **common,
    )
    beta_sweep = run_beta_sweep(beta_args)
    write_json(beta_args.output, beta_sweep)

    run_subprocess(
        [
            sys.executable,
            str(ROOT / "ipomdp_shielding" / "experiments" / "plot_taxinet_v2_lowacc_summary.py"),
            "--headline-json",
            str(headline_args.output),
            "--beta-sweep-json",
            str(beta_args.output),
            "--metrics-json",
            str(preflight_path),
            "--out-dir",
            str(args.results_dir / "figures_paper_model"),
            "--summary-md",
            str(args.results_dir / "evaluation_summary_paper_model.md"),
            "--title-prefix",
            "TaxiNetV2 Paper-Model",
        ]
    )

    print(f"Completed paper-model rerun in {args.results_dir}")


if __name__ == "__main__":
    main()
