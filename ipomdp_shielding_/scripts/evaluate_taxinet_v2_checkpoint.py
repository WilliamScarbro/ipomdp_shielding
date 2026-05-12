"""Evaluate a TaxiNetV2 checkpoint on the committed cp-control split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/dev/cp-control/data/taxinet"),
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "cal", "test"],
        default="test",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset_cls, model_cls = ensure_cp_control_imports(args.cp_control_root)
    model = load_model(model_cls, args.model_path, device)

    indices_path = args.cp_control_root / "train" / "models" / f"{args.split}_indices.pt"
    _, _, loader, indices = make_loader(
        dataset_cls,
        args.data_dir,
        indices_path,
        args.batch_size,
        args.num_workers,
    )
    cte, he = run_inference(model, loader, device)
    metrics = axis_accuracy_metrics(cte, he)
    payload = {
        "model_path": str(args.model_path),
        "split": args.split,
        "num_samples": len(indices),
        **metrics,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
