"""Run Carr support-based shielding on CartPole low-accuracy IPOMDP.

Uses the lowacc confusion matrices (artifacts_lowacc/) so the POMDP midpoint
observation model matches the one used by the threshold-sweep and observation-
shield experiments for this case study.

Saves results to:
    results/threshold_sweep_expanded/cartpole_lowacc_carr_results.json

Usage:
    python -m ipomdp_shielding.experiments.run_carr_cartpole_lowacc
"""

import json
import os

from .run_carr_all_case_studies import run_carr_for_case_study

OUTPUT_DIR = "results/threshold_sweep_expanded"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cartpole_lowacc_carr_results.json")

PARAMS = {
    "num_trials": 200,
    "trial_length": 15,
    "config_name": "rl_shield_cartpole_lowacc",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = run_carr_for_case_study("cartpole_lowacc", PARAMS)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n>>> Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
