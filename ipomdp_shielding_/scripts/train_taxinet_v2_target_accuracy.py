"""Train a lower-accuracy TaxiNetV2 checkpoint on the fixed committed split.

This keeps the train/val/cal/test partition aligned with the perception
artifacts used by the experiments and stops once both axis accuracies are in a
requested band on the fixed test set.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset


CP_CONTROL_TRAIN = Path("/home/dev/cp-control/train")
if str(CP_CONTROL_TRAIN) not in sys.path:
    sys.path.insert(0, str(CP_CONTROL_TRAIN))

from train import RobotNavigationDataset, RobotNavigationModel  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/dev/cp-control/data/taxinet"),
    )
    parser.add_argument(
        "--indices-dir",
        type=Path,
        default=Path("/home/dev/cp-control/train/models"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cache/taxinet_v2_lowacc"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every-batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-subset-size", type=int, default=600)
    parser.add_argument("--target-low", type=float, default=0.88)
    parser.add_argument("--target-high", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--init-model-path", type=Path, default=None)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--cte-label-noise", type=float, default=None)
    parser.add_argument("--he-label-noise", type=float, default=None)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_indices(indices_dir: Path, name: str) -> list[int]:
    return [int(i) for i in torch.load(indices_dir / name, map_location="cpu", weights_only=False)]


def build_subset(dataset, indices: list[int], limit: int | None = None, seed: int = 0):
    chosen = list(indices)
    if limit is not None and limit < len(chosen):
        rng = random.Random(seed)
        rng.shuffle(chosen)
        chosen = chosen[:limit]
    return Subset(dataset, chosen), chosen


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    cte_ok = 0
    he_ok = 0
    joint_ok = 0
    total = 0
    with torch.no_grad():
        for images, cte_t, he_t in loader:
            images = images.to(device)
            cte_t = cte_t.to(device)
            he_t = he_t.to(device)
            cte_o, he_o = model(images)
            cte_p = cte_o.argmax(1)
            he_p = he_o.argmax(1)
            cte_match = cte_p.eq(cte_t)
            he_match = he_p.eq(he_t)
            cte_ok += int(cte_match.sum())
            he_ok += int(he_match.sum())
            joint_ok += int((cte_match & he_match).sum())
            total += len(images)
    return {
        "cte_accuracy": cte_ok / total,
        "he_accuracy": he_ok / total,
        "joint_accuracy": joint_ok / total,
    }


def accuracy_distance(metrics: dict[str, float], target: float = 0.90) -> float:
    return max(
        abs(metrics["cte_accuracy"] - target),
        abs(metrics["he_accuracy"] - target),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cte_label_noise = args.label_noise if args.cte_label_noise is None else args.cte_label_noise
    he_label_noise = args.label_noise if args.he_label_noise is None else args.he_label_noise

    dataset = RobotNavigationDataset(str(args.data_dir), transform=None)
    train_idx = load_indices(args.indices_dir, "train_indices.pt")
    val_idx = load_indices(args.indices_dir, "val_indices.pt")
    test_idx = load_indices(args.indices_dir, "test_indices.pt")

    train_subset, used_train_idx = build_subset(
        dataset,
        train_idx,
        limit=args.train_subset_size,
        seed=args.seed,
    )
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RobotNavigationModel(pretrained=args.pretrained)
    if args.init_model_path is not None:
        state = torch.load(args.init_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_in_band_model.pth"
    latest_path = args.output_dir / "latest_model.pth"
    metrics_path = args.output_dir / "metrics.json"

    best_distance = float("inf")
    best_payload: dict[str, object] | None = None

    def record_candidate(epoch: int, batch_step: int | None = None) -> bool:
        nonlocal best_distance, best_payload
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        distance = accuracy_distance(test_metrics)
        payload = {
            "epoch": epoch,
            "batch_step": batch_step,
            "train_subset_size": len(used_train_idx),
            "train_indices_sample": used_train_idx[:20],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "device": str(device),
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "init_model_path": str(args.init_model_path) if args.init_model_path else None,
            "label_noise": args.label_noise,
            "cte_label_noise": cte_label_noise,
            "he_label_noise": he_label_noise,
        }
        location = f"epoch={epoch}"
        if batch_step is not None:
            location += f" batch={batch_step}"
        print(
            f"{location} val_cte={val_metrics['cte_accuracy']:.4f} val_he={val_metrics['he_accuracy']:.4f} "
            f"test_cte={test_metrics['cte_accuracy']:.4f} test_he={test_metrics['he_accuracy']:.4f} "
            f"test_joint={test_metrics['joint_accuracy']:.4f}",
            flush=True,
        )
        torch.save(model.state_dict(), latest_path)
        if distance < best_distance:
            best_distance = distance
            best_payload = payload
            torch.save(model.state_dict(), best_path)
        return (
            args.target_low <= test_metrics["cte_accuracy"] <= args.target_high
            and args.target_low <= test_metrics["he_accuracy"] <= args.target_high
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_step, (images, cte_t, he_t) in enumerate(train_loader, start=1):
            images = images.to(device)
            cte_t = cte_t.to(device)
            he_t = he_t.to(device)
            if cte_label_noise > 0.0 or he_label_noise > 0.0:
                cte_mask = torch.rand_like(cte_t, dtype=torch.float32) < cte_label_noise
                he_mask = torch.rand_like(he_t, dtype=torch.float32) < he_label_noise
                if cte_mask.any():
                    cte_t = cte_t.clone()
                    cte_t[cte_mask] = torch.randint(0, 5, (int(cte_mask.sum()),), device=device)
                if he_mask.any():
                    he_t = he_t.clone()
                    he_t[he_mask] = torch.randint(0, 3, (int(he_mask.sum()),), device=device)
            optimizer.zero_grad()
            cte_o, he_o = model(images)
            loss = criterion(cte_o, cte_t) + criterion(he_o, he_t)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_count += len(images)
            if args.eval_every_batches and batch_step % args.eval_every_batches == 0:
                if record_candidate(epoch, batch_step=batch_step):
                    import json
                    metrics_path.write_text(json.dumps(best_payload, indent=2))
                    print(f"saved_model={best_path}")
                    print(f"saved_metrics={metrics_path}")
                    return
        print(
            f"epoch={epoch} train_loss={total_loss/max(total_count,1):.4f}",
            flush=True,
        )
        if record_candidate(epoch):
            break

    if best_payload is None:
        raise RuntimeError("Training finished without producing any checkpoint.")

    import json

    metrics_path.write_text(json.dumps(best_payload, indent=2))
    print(f"saved_model={best_path}")
    print(f"saved_metrics={metrics_path}")


if __name__ == "__main__":
    main()
