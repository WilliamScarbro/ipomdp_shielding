"""Data loading utilities for the Scarbro-backed TaxiNetV2 case study."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_ALPHA_LEVEL = "0.95"

SIGNED_CTE_STATES = [-2, -1, 0, 1, 2]
SIGNED_HE_STATES = [-1, 0, 1]

_CTE_INDEX_TO_SIGNED = {idx: val for idx, val in enumerate(SIGNED_CTE_STATES)}
_HE_INDEX_TO_SIGNED = {idx: val for idx, val in enumerate(SIGNED_HE_STATES)}
_ALPHA_TO_SUFFIX = {
    "0.95": "95",
    "0.99": "99",
    "0.995": "995",
}


ConformalAxisObservation = Tuple[int, ...]
ConformalObservation = Tuple[ConformalAxisObservation, ConformalAxisObservation]
SignedTaxiState = Tuple[int, int]


@dataclass(frozen=True)
class TaxiNetV2Metadata:
    """Resolved artifact paths and descriptive metadata."""

    scarbro_root: str
    compiler_artifact_dir: str
    train_artifact_dir: str
    confidence_level: str
    cte_csv: str
    he_csv: str


def _paper_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _scarbro_root() -> Path:
    root = _paper_root() / "scarbro_et_al" / "cp-control"
    if not root.exists():
        raise FileNotFoundError(f"Scarbro artifact not found at {root}")
    return root


def _compiler_artifact_dir() -> Path:
    return _scarbro_root() / "compiler" / "lib" / "acc90"


def _train_artifact_dir() -> Path:
    return _scarbro_root() / "train" / "models"


def _alpha_suffix(confidence_level: str) -> str:
    if confidence_level not in _ALPHA_TO_SUFFIX:
        supported = ", ".join(sorted(_ALPHA_TO_SUFFIX))
        raise ValueError(
            f"Unsupported confidence_level={confidence_level!r}. Supported: {supported}"
        )
    return _ALPHA_TO_SUFFIX[confidence_level]


def _csv_paths(confidence_level: str) -> Tuple[Path, Path]:
    suffix = _alpha_suffix(confidence_level)
    root = _compiler_artifact_dir()
    cte_path = root / f"real_cte_pred_acc90_conf{suffix}.csv"
    he_path = root / f"real_he_pred_acc90_conf{suffix}.csv"
    if not cte_path.exists() or not he_path.exists():
        raise FileNotFoundError(
            f"Missing Scarbro conformal CSVs for confidence_level={confidence_level!r}: "
            f"{cte_path}, {he_path}"
        )
    return cte_path, he_path


def get_taxinet_v2_metadata(confidence_level: str = DEFAULT_ALPHA_LEVEL) -> TaxiNetV2Metadata:
    """Return resolved artifact metadata for TaxiNetV2."""
    cte_path, he_path = _csv_paths(confidence_level)
    return TaxiNetV2Metadata(
        scarbro_root=str(_scarbro_root()),
        compiler_artifact_dir=str(_compiler_artifact_dir()),
        train_artifact_dir=str(_train_artifact_dir()),
        confidence_level=confidence_level,
        cte_csv=str(cte_path),
        he_csv=str(he_path),
    )


def _read_csv_rows(path: Path) -> List[List[int]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        del header
        return [[int(cell) for cell in row] for row in reader if row]


def _decode_axis_row(
    row: Sequence[int],
    mapping: Dict[int, int],
) -> Tuple[int, Tuple[int, ...]]:
    true_state = mapping[row[0]]
    predicted = tuple(mapping[idx] for idx, bit in enumerate(row[1:]) if bit)
    return true_state, predicted


def _project_axis_prediction(true_state: int, predicted: Sequence[int]) -> int:
    if not predicted:
        return true_state
    if true_state in predicted:
        return true_state
    return min(predicted, key=lambda cand: (abs(cand - true_state), cand))


@lru_cache(maxsize=None)
def _load_observation_rows(confidence_level: str) -> List[Tuple[SignedTaxiState, ConformalObservation]]:
    cte_path, he_path = _csv_paths(confidence_level)
    cte_rows = _read_csv_rows(cte_path)
    he_rows = _read_csv_rows(he_path)

    if len(cte_rows) != len(he_rows):
        raise ValueError(
            f"Scarbro CSV row mismatch for confidence_level={confidence_level!r}: "
            f"{len(cte_rows)} CTE rows vs {len(he_rows)} HE rows"
        )

    paired_rows: List[Tuple[SignedTaxiState, ConformalObservation]] = []
    for cte_row, he_row in zip(cte_rows, he_rows):
        true_cte, cte_obs = _decode_axis_row(cte_row, _CTE_INDEX_TO_SIGNED)
        true_he, he_obs = _decode_axis_row(he_row, _HE_INDEX_TO_SIGNED)
        paired_rows.append(((true_cte, true_he), (cte_obs, he_obs)))

    return paired_rows


def get_taxinet_v2_observation_data(
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
) -> List[Tuple[SignedTaxiState, ConformalObservation]]:
    """Return empirical Scarbro observation samples for interval learning."""
    return list(_load_observation_rows(confidence_level))


def get_taxinet_v2_projected_test_models(
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Project conformal sets to concrete observations for legacy perceptor APIs.

    This is a compatibility adapter only. The primary TaxiNetV2 IPOMDP uses the
    conformal-set observations directly.
    """
    cte_model = {state: [] for state in SIGNED_CTE_STATES}
    he_model = {state: [] for state in SIGNED_HE_STATES}

    for (true_cte, true_he), (cte_obs, he_obs) in _load_observation_rows(confidence_level):
        cte_model[true_cte].append(_project_axis_prediction(true_cte, cte_obs))
        he_model[true_he].append(_project_axis_prediction(true_he, he_obs))

    return cte_model, he_model


def get_scarbro_split_indices(split: str) -> List[int]:
    """Load Scarbro train/cal/val/test split indices from the artifact."""
    path = _train_artifact_dir() / f"{split}_indices.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Scarbro split file: {path}")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Loading Scarbro split indices requires torch to be installed."
        ) from exc

    indices = torch.load(path, map_location="cpu")
    return [int(idx) for idx in indices]
