"""Generate v6 evaluation summary: same as v5 but with updated forward-sampling
results (budget=500, K_samples=100) and corresponding timing data.

Usage:
    python -m ipomdp_shielding.experiments.plot_sweep_v6
"""

from __future__ import annotations
from pathlib import Path

import ipomdp_shielding.experiments.plot_sweep_v5 as _v5

# Override version — OUTDIR and figure filenames derive from this
_v5.VERSION = "v6"
_v5.OUTDIR  = Path("results/sweep_v6")
_v5.MD_PATH = _v5.OUTDIR / "evaluation_summary.md"

_original_generate_markdown = _v5.generate_markdown

def _generate_markdown_v6(figures: dict) -> str:
    md = _original_generate_markdown(figures)
    md = md.replace("# Evaluation Summary — v5", "# Evaluation Summary — v6")
    return md

_v5.generate_markdown = _generate_markdown_v6


def main() -> None:
    _v5.main()


if __name__ == "__main__":
    main()
