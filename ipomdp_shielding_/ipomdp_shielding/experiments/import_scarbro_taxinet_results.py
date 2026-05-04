"""Import precomputed Scarbro TaxiNet PRISM outputs into our results tree.

This is a reproducible ingestion wrapper around the Scarbro artifact's
`case_studies/taxinet/results/prism_output/*.txt` files. It does not rerun
PRISM; it normalizes the exported result tables into JSON for downstream
comparison scripts and plots.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


RAW_PROPERTIES = {
    "crash_or_default_tradeoff": 'multi(Pmax=? [ F pc=-1 ], R{"defaultControllerUsed"}max=? [ C ]):',
    "crash_or_stuck_tradeoff": "multi(Pmax=? [ F pc=-1 ], Pmax=? [ F pc=-2 ]):",
    "crash": "Pmax=? [ (F pc=-1)&(G pc!=-2) ]:",
    "crash_given_not_stuck": "Pmax=? [ (F pc=-1) ]/Pmax=? [ (G pc!=-2) ]:",
    "crash_and_not_stuck_given_not_stuck": "Pmax=? [ (F pc=-1)&(G pc!=-2) ]/Pmax=? [ (G pc!=-2) ]:",
    "stuck_or_default": "Pmax=? [ F pc=-2 ]:",
    "success": "Pmin=? [ F k=N ]:",
    "success_without_default_or_stuck": "Pmin=? [ (F k=N)&(G pc>0) ]:",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _paper_root() -> Path:
    return _project_root().parents[1]


def _scarbro_results_dir() -> Path:
    return _paper_root() / "scarbro_et_al" / "cp-control" / "case_studies" / "taxinet" / "results" / "prism_output"


def _output_path() -> Path:
    return _project_root() / "results" / "taxinet_v2" / "scarbro_baseline_import.json"


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith("[("):
        values = []
        inner = raw[2:-2]
        for pair in inner.split("), ("):
            x_str, y_str = pair.split(",")
            values.append([float(x_str), float(y_str)])
        return values
    return float(raw)


def _load_result_blocks(path: Path) -> Dict[str, List[Tuple[int, object]]]:
    lines = [line.rstrip("\n") for line in path.read_text().splitlines()]
    blocks: Dict[str, List[Tuple[int, object]]] = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "N\tResult":
            index += 1
            continue

        prop_name = line
        index += 1
        if index < len(lines) and lines[index].strip() == "N\tResult":
            index += 1

        rows: List[Tuple[int, object]] = []
        while index < len(lines):
            current = lines[index].strip()
            if not current:
                index += 1
                break
            if "\t" not in current:
                break
            horizon_str, value_str = current.split("\t", 1)
            rows.append((int(horizon_str), _parse_value(value_str)))
            index += 1

        blocks[prop_name] = rows

    return blocks


def _variant_metadata(path: Path) -> Dict[str, object]:
    stem = path.name.replace("_results.txt", "")
    parts = stem.split("_")
    conf_tag = parts[0]
    af_tag = parts[1]
    return {
        "source_file": path.name,
        "confidence_level": f"0.{conf_tag[4:]}",
        "action_filter_tag": af_tag,
        "default_action": "_def_act" in stem,
    }


def _final_value(rows: List[Tuple[int, object]]):
    if not rows:
        return None
    horizon, value = rows[-1]
    return {"horizon": horizon, "value": value}


def import_results() -> Dict[str, object]:
    source_dir = _scarbro_results_dir()
    variants = []

    paths = sorted(source_dir.glob("conf*_results.txt"))
    paths = [
        path for path in paths
        if "prop_no_violation" not in path.name
    ]

    for path in paths:
        blocks = _load_result_blocks(path)
        metadata = _variant_metadata(path)

        summary = {}
        for key, prop in RAW_PROPERTIES.items():
            if prop in blocks:
                summary[key] = _final_value(blocks[prop])

        variants.append(
            {
                "metadata": metadata,
                "summary": summary,
                "raw_properties": {
                    prop: [{"horizon": n, "value": value} for n, value in rows]
                    for prop, rows in blocks.items()
                },
            }
        )

    out = {
        "metadata": {
            "source_dir": str(source_dir),
            "num_variants": len(variants),
            "note": (
                "This file imports precomputed Scarbro PRISM outputs. "
                "It does not regenerate the baseline."
            ),
        },
        "variants": variants,
    }
    return out


def main():
    output = import_results()
    out_path = _output_path()
    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(output, handle, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
