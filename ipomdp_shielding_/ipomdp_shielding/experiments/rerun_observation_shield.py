"""Re-run observation shield baseline for CartPole and TaxiNet.

Patches only the `*/rl/observation` entries in the final results JSONs,
then calls plot_sweep_v3 to regenerate figures and the markdown summary.

Usage:
    python -m ipomdp_shielding.experiments.rerun_observation_shield
"""

import json
import os
import time

from ..MonteCarlo import (
    UniformPerceptionModel,
    FixedRealizationPerceptionModel,
    NeuralActionSelector,
    RandomInitialState,
    run_monte_carlo_trials,
    compute_safety_metrics,
)
from .run_rl_shield_experiment import (
    ObservationShield,
    create_observation_shield_factory,
    ShieldCompliantSelector,
)

NUM_TRIALS = 200
SEED       = 42

CASES = [
    {
        "name": "taxinet",
        "config_module": "ipomdp_shielding.experiments.configs.rl_shield_taxinet_final",
        "results_path": "results/final/rl_shield_taxinet_results.json",
    },
    {
        "name": "cartpole",
        "config_module": "ipomdp_shielding.experiments.configs.rl_shield_cartpole_final",
        "results_path": "results/final/rl_shield_cartpole_results.json",
    },
]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_for_case(case):
    import importlib
    mod = importlib.import_module(case["config_module"])
    config = mod.config

    print(f"\n=== {case['name'].upper()} ===")
    ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(**config.ipomdp_kwargs)
    print(f"  States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
          f"Obs: {len(ipomdp.observations)}")

    # Load cached RL selector
    rl_selector = NeuralActionSelector.load(config.rl_cache_path, ipomdp)
    all_actions = list(ipomdp.actions)
    rl_wrapped  = ShieldCompliantSelector(rl_selector, all_actions)

    # Build observation shield factory (new probability-based)
    obs_factory = create_observation_shield_factory(
        ipomdp, pp_shield, threshold=config.shield_threshold
    )

    # Uniform perception
    uniform_perception = UniformPerceptionModel()

    # Adversarial perception — load cached opt realization (trained vs envelope)
    adv_path = config.opt_cache_path  # "envelope" target → base path
    if os.path.exists(adv_path):
        adversarial_perception = FixedRealizationPerceptionModel.load(adv_path)
        print(f"  Loaded adversarial realization from {adv_path}")
    else:
        print(f"  WARNING: no adversarial cache at {adv_path}, using uniform for adversarial")
        adversarial_perception = uniform_perception

    results_new = {}
    for p_name, perception in [("uniform", uniform_perception),
                                ("adversarial_opt", adversarial_perception)]:
        key = f"{p_name}/rl/observation"
        print(f"  Running {key} ({NUM_TRIALS} trials)...", end=" ", flush=True)
        t0 = time.time()

        trial_results = run_monte_carlo_trials(
            ipomdp=ipomdp,
            pp_shield=pp_shield,
            perception=perception,
            rt_shield_factory=obs_factory,
            action_selector=rl_wrapped,
            initial_generator=RandomInitialState(),
            num_trials=NUM_TRIALS,
            trial_length=config.trial_length,
            seed=SEED,
        )
        metrics = compute_safety_metrics(trial_results)
        elapsed = time.time() - t0

        results_new[key] = {
            "fail_rate":   round(metrics.fail_rate,   4),
            "stuck_rate":  round(metrics.stuck_rate,  4),
            "safe_rate":   round(metrics.safe_rate,   4),
            "mean_steps":  round(metrics.mean_steps,  2),
            "num_trials":  NUM_TRIALS,
        }
        print(f"fail={metrics.fail_rate:.0%}, stuck={metrics.stuck_rate:.0%}  ({elapsed:.0f}s)")

    # Patch results JSON
    data = _load_json(case["results_path"])
    for k, v in results_new.items():
        data["results"][k] = v
    _save_json(case["results_path"], data)
    print(f"  Patched {case['results_path']}")
    return results_new


def main():
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)

    for case in CASES:
        run_for_case(case)

    print("\nRegenerating plots and markdown...")
    from . import plot_sweep_v3
    plot_sweep_v3.main()
    print("Done.")


if __name__ == "__main__":
    main()
