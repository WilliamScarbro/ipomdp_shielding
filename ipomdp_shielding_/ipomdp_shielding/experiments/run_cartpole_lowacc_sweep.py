"""Run threshold sweep for low-accuracy CartPole only.

Trains a fresh RL agent and adversarial realization for the low-accuracy
CartPole perception model (P_mid≈0.373, matching TaxiNet difficulty), then
runs the full threshold sweep (200 trials × 9 thresholds × single_belief).

Saves results to results/threshold_sweep_expanded/cartpole_lowacc_sweep.json.

Usage:
    python -m ipomdp_shielding.experiments.run_cartpole_lowacc_sweep
"""

import os
import sys
import time

from .run_threshold_sweep import (
    run_sweep_for_case_study,
    save_sweep,
    EXPANDED_OUTPUT_DIR,
)

PARAMS = {
    "num_trials": 200,
    "trial_length": 15,
    "exclude_envelope": True,
    "config_name": "rl_shield_cartpole_lowacc",
}


def main():
    print("=" * 70)
    print("LOW-ACCURACY CARTPOLE THRESHOLD SWEEP")
    print("  Perception: artifacts_lowacc (175ep, 10epochs, P_mid≈0.373)")
    print("  Trials: 200, Length: 15, Thresholds: 0.50–0.95")
    print("  Shields: single_belief (envelope excluded)")
    print("=" * 70)

    t0 = time.time()
    os.makedirs(EXPANDED_OUTPUT_DIR, exist_ok=True)

    sweep_results, setup_info, base_config = run_sweep_for_case_study(
        "cartpole_lowacc", PARAMS
    )
    path = save_sweep(
        "cartpole_lowacc", sweep_results, base_config, PARAMS, setup_info,
        output_dir=EXPANDED_OUTPUT_DIR,
    )

    elapsed = time.time() - t0
    hh, rem = divmod(int(elapsed), 3600)
    mm, ss = divmod(rem, 60)
    print(f"\nDone in {hh:02d}h {mm:02d}m {ss:02d}s")
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
