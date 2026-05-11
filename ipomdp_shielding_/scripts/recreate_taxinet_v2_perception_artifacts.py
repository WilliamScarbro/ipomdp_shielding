"""Recreate TaxiNetV2 perception artifacts from the cp-control DNN.

This script uses the trained cp-control TaxiNet model and committed split
indices to produce both point-estimate and conformal-set perception artifacts.
The point-estimate artifacts are the base perception model for IPOMDP shields.
The conformal-set artifacts are for the cp-control conformal shielding method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


CTE_INDEX_TO_SIGNED = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
HE_INDEX_TO_SIGNED = {0: -1, 1: 0, 2: 1}
SIGNED_CTE_STATES = [-2, -1, 0, 1, 2]
SIGNED_HE_STATES = [-1, 0, 1]

CONFIDENCE_TO_ALPHA = {
    "95": 0.05,
    "99": 0.01,
    "995": 0.005,
}


@dataclass(frozen=True)
class AxisOutputs:
    logits: np.ndarray
    probs: np.ndarray
    preds: np.ndarray
    targets: np.ndarray


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_cp_control_imports(cp_control_root: Path):
    train_dir = cp_control_root / "train"
    sys.path.insert(0, str(train_dir))
    from train import RobotNavigationDataset, RobotNavigationModel

    return RobotNavigationDataset, RobotNavigationModel


def load_model(model_cls, model_path: Path, device: torch.device):
    model = model_cls(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_loader(dataset_cls, data_dir: Path, indices_path: Path, batch_size: int, num_workers: int):
    dataset = dataset_cls(str(data_dir), transform=None)
    indices = torch.load(indices_path, map_location="cpu", weights_only=False)
    subset = Subset(dataset, [int(i) for i in indices])
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return dataset, subset, loader, [int(i) for i in indices]


def run_inference(model, loader: DataLoader, device: torch.device) -> tuple[AxisOutputs, AxisOutputs]:
    cte_logits = []
    he_logits = []
    cte_probs = []
    he_probs = []
    cte_preds = []
    he_preds = []
    cte_targets = []
    he_targets = []

    with torch.no_grad():
        for images, cte_target, he_target in loader:
            images = images.to(device)
            cte_output, he_output = model(images)
            cte_prob = F.softmax(cte_output, dim=1)
            he_prob = F.softmax(he_output, dim=1)
            cte_pred = torch.argmax(cte_output, dim=1)
            he_pred = torch.argmax(he_output, dim=1)

            cte_logits.append(cte_output.cpu().numpy())
            he_logits.append(he_output.cpu().numpy())
            cte_probs.append(cte_prob.cpu().numpy())
            he_probs.append(he_prob.cpu().numpy())
            cte_preds.append(cte_pred.cpu().numpy())
            he_preds.append(he_pred.cpu().numpy())
            cte_targets.append(cte_target.cpu().numpy())
            he_targets.append(he_target.cpu().numpy())

    return (
        AxisOutputs(
            logits=np.concatenate(cte_logits, axis=0),
            probs=np.concatenate(cte_probs, axis=0),
            preds=np.concatenate(cte_preds, axis=0).astype(int),
            targets=np.concatenate(cte_targets, axis=0).astype(int),
        ),
        AxisOutputs(
            logits=np.concatenate(he_logits, axis=0),
            probs=np.concatenate(he_probs, axis=0),
            preds=np.concatenate(he_preds, axis=0).astype(int),
            targets=np.concatenate(he_targets, axis=0).astype(int),
        ),
    )


def adjusted_qhat(probs: np.ndarray, targets: np.ndarray, alpha: float) -> float:
    scores = 1.0 - probs[np.arange(len(targets)), targets]
    q_level = math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores)
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def prediction_sets(probs: np.ndarray, qhat: float) -> tuple[np.ndarray, int]:
    """Return split-conformal prediction sets with a top-1 non-empty repair.

    The standard score threshold can return an empty set when every class has
    probability below ``1 - qhat``. Empty observations are not meaningful for
    the cp-control TaxiNet shield, so we conservatively add the model's argmax
    only for those rows. This preserves same-model artifacts and can only
    increase empirical coverage relative to the raw threshold sets.
    """
    predsets = (probs >= (1.0 - qhat)).astype(int)
    empty_rows = np.where(predsets.sum(axis=1) == 0)[0]
    if len(empty_rows):
        predsets[empty_rows, np.argmax(probs[empty_rows], axis=1)] = 1
    return predsets, int(len(empty_rows))


def prediction_set_summary(predsets: np.ndarray, targets: np.ndarray) -> dict:
    sizes = predsets.sum(axis=1)
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    return {
        "empty_count": int(np.sum(sizes == 0)),
        "size_counts": {str(int(size)): int(count) for size, count in zip(unique_sizes, counts)},
        "mean_size": float(np.mean(sizes)),
        "empirical_coverage": float(predsets[np.arange(len(targets)), targets].mean()),
    }


def validate_prediction_sets(axis_name: str, predsets: np.ndarray, preds: np.ndarray) -> dict:
    sizes = predsets.sum(axis=1)
    empty_rows = np.where(sizes == 0)[0]
    argmax_missing_rows = np.where(predsets[np.arange(len(preds)), preds] == 0)[0]
    if len(empty_rows) or len(argmax_missing_rows):
        examples = {
            "empty_rows": [int(i) for i in empty_rows[:10]],
            "argmax_missing_rows": [int(i) for i in argmax_missing_rows[:10]],
        }
        raise ValueError(
            f"Invalid {axis_name} conformal prediction sets after top-1 repair: "
            f"{len(empty_rows)} empty rows, {len(argmax_missing_rows)} rows missing argmax; "
            f"examples={examples}"
        )
    return {
        "empty_set_violations": int(len(empty_rows)),
        "argmax_not_in_set_violations": int(len(argmax_missing_rows)),
    }


def write_axis_set_csv(path: Path, axis_name: str, targets: np.ndarray, predsets: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([axis_name] + [f"{axis_name}{i}" for i in range(predsets.shape[1])])
        for target, row in zip(targets, predsets):
            writer.writerow([int(target)] + [int(x) for x in row])


def write_axis_point_csv(path: Path, axis_name: str, targets: np.ndarray, preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([axis_name, f"{axis_name}_pred"])
        for target, pred in zip(targets, preds):
            writer.writerow([int(target), int(pred)])


def write_axis_paired_conformal_csv(
    path: Path,
    axis_name: str,
    raw_indices: list[int],
    targets: np.ndarray,
    preds: np.ndarray,
    predsets: np.ndarray,
    mapping: dict[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "sample_id",
        "raw_index",
        f"true_{axis_name}_idx",
        f"true_{axis_name}",
        f"pred_{axis_name}_idx",
        f"pred_{axis_name}",
    ]
    header += [f"{axis_name}{i}" for i in range(predsets.shape[1])]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, raw_index in enumerate(raw_indices):
            writer.writerow(
                [
                    i,
                    raw_index,
                    int(targets[i]),
                    mapping[int(targets[i])],
                    int(preds[i]),
                    mapping[int(preds[i])],
                ]
                + [int(x) for x in predsets[i]]
            )


def write_totals(path: Path, targets: np.ndarray, preds: np.ndarray, signed_values: list[int], mapping: dict[int, int]) -> None:
    counts = {(true, pred): 0 for true in signed_values for pred in signed_values}
    for target, pred in zip(targets, preds):
        counts[(mapping[int(target)], mapping[int(pred)])] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for true in signed_values:
            for pred in signed_values:
                handle.write(f"{counts[(true, pred)]}\n")


def write_point_detail(path: Path, raw_indices: list[int], cte: AxisOutputs, he: AxisOutputs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "sample_id",
        "raw_index",
        "true_cte_idx",
        "true_he_idx",
        "true_cte",
        "true_he",
        "pred_cte_idx",
        "pred_he_idx",
        "pred_cte",
        "pred_he",
    ]
    header += [f"cte_logit_{i}" for i in range(5)]
    header += [f"he_logit_{i}" for i in range(3)]
    header += [f"cte_prob_{i}" for i in range(5)]
    header += [f"he_prob_{i}" for i in range(3)]
    header += ["correct_cte", "correct_he"]

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, raw_index in enumerate(raw_indices):
            row = [
                i,
                raw_index,
                int(cte.targets[i]),
                int(he.targets[i]),
                CTE_INDEX_TO_SIGNED[int(cte.targets[i])],
                HE_INDEX_TO_SIGNED[int(he.targets[i])],
                int(cte.preds[i]),
                int(he.preds[i]),
                CTE_INDEX_TO_SIGNED[int(cte.preds[i])],
                HE_INDEX_TO_SIGNED[int(he.preds[i])],
            ]
            row += [float(x) for x in cte.logits[i]]
            row += [float(x) for x in he.logits[i]]
            row += [float(x) for x in cte.probs[i]]
            row += [float(x) for x in he.probs[i]]
            row += [int(cte.targets[i] == cte.preds[i]), int(he.targets[i] == he.preds[i])]
            writer.writerow(row)


def write_paired_conformal_csv(
    path: Path,
    raw_indices: list[int],
    cte_targets: np.ndarray,
    he_targets: np.ndarray,
    cte_sets: np.ndarray,
    he_sets: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_id", "raw_index", "true_cte_idx", "true_he_idx", "true_cte", "true_he"]
    header += [f"cte{i}" for i in range(5)] + [f"he{i}" for i in range(3)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, raw_index in enumerate(raw_indices):
            writer.writerow(
                [
                    i,
                    raw_index,
                    int(cte_targets[i]),
                    int(he_targets[i]),
                    CTE_INDEX_TO_SIGNED[int(cte_targets[i])],
                    HE_INDEX_TO_SIGNED[int(he_targets[i])],
                ]
                + [int(x) for x in cte_sets[i]]
                + [int(x) for x in he_sets[i]]
            )


def sample_manifest(data_dir: Path) -> dict:
    files = sorted(data_dir.glob("data_*.npz"))
    entries = []
    total = 0
    for path in files:
        with np.load(path, mmap_mode="r") as data:
            images_shape = list(data["images"].shape)
            labels_shape = list(data["labels"].shape)
            total += images_shape[0]
        entries.append(
            {
                "file": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "images_shape": images_shape,
                "labels_shape": labels_shape,
            }
        )
    return {"num_files": len(files), "total_samples": total, "files": entries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-control-root", type=Path, default=Path("/home/dev/cp-control"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to TaxiNet checkpoint. Defaults to cp-control train/models/best_model.pth",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/dev/cp-control/data/taxinet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/dev/ipomdp_shielding/ipomdp_shielding_/ipomdp_shielding/CaseStudies/TaxiNetV2/artifacts"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    dataset_cls, model_cls = ensure_cp_control_imports(args.cp_control_root)
    model_path = args.model_path or (args.cp_control_root / "train" / "models" / "best_model.pth")
    model = load_model(model_cls, model_path, device)

    train_models = args.cp_control_root / "train" / "models"
    _, _, cal_loader, cal_indices = make_loader(
        dataset_cls,
        args.data_dir,
        train_models / "cal_indices.pt",
        args.batch_size,
        args.num_workers,
    )
    _, _, test_loader, test_indices = make_loader(
        dataset_cls,
        args.data_dir,
        train_models / "test_indices.pt",
        args.batch_size,
        args.num_workers,
    )

    print("Running calibration inference...")
    cal_cte, cal_he = run_inference(model, cal_loader, device)
    print("Running test inference...")
    test_cte, test_he = run_inference(model, test_loader, device)

    compiler_dir = args.output_root / "compiler" / "lib" / "acc90"
    perception_dir = args.output_root / "perception"

    write_axis_point_csv(compiler_dir / "real_cte_single_pred_acc90.csv", "cte", test_cte.targets, test_cte.preds)
    write_axis_point_csv(compiler_dir / "real_he_single_pred_acc90.csv", "he", test_he.targets, test_he.preds)
    write_totals(compiler_dir / "cte_single_totals", test_cte.targets, test_cte.preds, SIGNED_CTE_STATES, CTE_INDEX_TO_SIGNED)
    write_totals(compiler_dir / "he_single_totals", test_he.targets, test_he.preds, SIGNED_HE_STATES, HE_INDEX_TO_SIGNED)
    write_point_detail(perception_dir / "taxinet_point_estimates.csv", test_indices, test_cte, test_he)

    qhats = {}
    conformal_summaries = {}
    paired_axis_validation = {}
    for suffix, alpha in CONFIDENCE_TO_ALPHA.items():
        cte_qhat = adjusted_qhat(cal_cte.probs, cal_cte.targets, alpha)
        he_qhat = adjusted_qhat(cal_he.probs, cal_he.targets, alpha)
        qhats[f"conf{suffix}"] = {"alpha": alpha, "cte_qhat": cte_qhat, "he_qhat": he_qhat}

        cte_sets, cte_repaired = prediction_sets(test_cte.probs, cte_qhat)
        he_sets, he_repaired = prediction_sets(test_he.probs, he_qhat)
        cte_validation = validate_prediction_sets("cte", cte_sets, test_cte.preds)
        he_validation = validate_prediction_sets("he", he_sets, test_he.preds)
        cte_summary = prediction_set_summary(cte_sets, test_cte.targets)
        he_summary = prediction_set_summary(he_sets, test_he.targets)
        paired_axis_validation[f"conf{suffix}"] = {
            "cte": {
                **cte_validation,
                "empirical_coverage": cte_summary["empirical_coverage"],
                "mean_set_size": cte_summary["mean_size"],
            },
            "he": {
                **he_validation,
                "empirical_coverage": he_summary["empirical_coverage"],
                "mean_set_size": he_summary["mean_size"],
            },
        }
        conformal_summaries[f"conf{suffix}"] = {
            "cte": cte_summary,
            "he": he_summary,
            "top1_nonempty_repairs": {"cte": cte_repaired, "he": he_repaired},
            "validation": {"cte": cte_validation, "he": he_validation},
        }
        write_axis_set_csv(
            compiler_dir / f"real_cte_pred_acc90_conf{suffix}.csv",
            "cte",
            test_cte.targets,
            cte_sets,
        )
        write_axis_set_csv(
            compiler_dir / f"real_he_pred_acc90_conf{suffix}.csv",
            "he",
            test_he.targets,
            he_sets,
        )
        write_axis_paired_conformal_csv(
            perception_dir / f"taxinet_cte_paired_conformal_conf{suffix}.csv",
            "cte",
            test_indices,
            test_cte.targets,
            test_cte.preds,
            cte_sets,
            CTE_INDEX_TO_SIGNED,
        )
        write_axis_paired_conformal_csv(
            perception_dir / f"taxinet_he_paired_conformal_conf{suffix}.csv",
            "he",
            test_indices,
            test_he.targets,
            test_he.preds,
            he_sets,
            HE_INDEX_TO_SIGNED,
        )
        write_paired_conformal_csv(
            perception_dir / f"taxinet_conformal_observations_conf{suffix}.csv",
            test_indices,
            test_cte.targets,
            test_he.targets,
            cte_sets,
            he_sets,
        )

    manifest = {
        "source": {
            "cp_control_root": str(args.cp_control_root),
            "data_dir": str(args.data_dir),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "cal_indices_path": str(train_models / "cal_indices.pt"),
            "test_indices_path": str(train_models / "test_indices.pt"),
            "cal_indices_sha256": sha256_file(train_models / "cal_indices.pt"),
            "test_indices_sha256": sha256_file(train_models / "test_indices.pt"),
        },
        "data": sample_manifest(args.data_dir),
        "class_mapping": {
            "cte_index_to_signed": CTE_INDEX_TO_SIGNED,
            "he_index_to_signed": HE_INDEX_TO_SIGNED,
        },
        "split_sizes": {
            "cal": len(cal_indices),
            "test": len(test_indices),
        },
        "qhat": qhats,
        "conformal_artifact_source": "locally_regenerated_from_calibration_split_with_top1_nonempty_repair",
        "paired_axis_validation": paired_axis_validation,
        "conformal_summaries": conformal_summaries,
        "outputs": {
            "compiler_dir": str(compiler_dir),
            "perception_dir": str(perception_dir),
        },
    }
    manifest_path = perception_dir / "taxinet_v2_perception_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
