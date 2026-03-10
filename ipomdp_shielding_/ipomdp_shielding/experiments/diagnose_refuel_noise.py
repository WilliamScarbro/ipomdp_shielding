"""Diagnostic: sweep obs_noise for Refuel to find the stuck threshold.

Motivation
----------
The prelim RL shielding experiment shows refuel is 100% stuck with
observation and envelope shields, and 40-70% stuck with single_belief.

Two suspected causes:
  1. ObservationShield bug: pp_shield is keyed by states (ax, ay, fuel)
     but shield looks up pp_shield[obs] where obs is a 10-tuple.
     These never match → always empty → stuck at step 0.
  2. obs_noise=0.1 makes single_belief too conservative.

This script tests cause (2) by sweeping obs_noise = {0.0, 0.01, 0.05, 0.1}
and reporting stuck/fail/safe rates for:
  - No shield (baseline: shows whether refuel itself is solvable)
  - SingleBelief shield (threshold 0.8)
  - ObservationShield (to confirm cause 1)

Skips: envelope (too slow), RL agent (not needed for diagnosis), adversarial
perception (adds confound), opt realization (unnecessary).

Runtime target: < 5 minutes.
"""

import sys
import time

from ..CaseStudies.GridWorldBenchmarks import build_refuel_ipomdp
from ..MonteCarlo import (
    UniformPerceptionModel,
    RandomActionSelector,
    RandomInitialState,
    run_monte_carlo_trials,
    compute_safety_metrics,
)
from .run_rl_shield_experiment import (
    NoShield,
    ObservationShield,
    SingleBeliefShield,
)


OBS_NOISE_VALUES = [0.0, 0.01, 0.05, 0.1]
NUM_TRIALS = 10
TRIAL_LENGTH = 20
SEED = 42
THRESHOLD = 0.8

SHIELD_NAMES = ["none", "observation", "single_belief"]


def build_shields(ipomdp, pp_shield):
    pomdp = ipomdp.to_pomdp()
    all_actions = list(ipomdp.actions)
    obs_to_states = {}
    for s in ipomdp.states:
        for obs in ipomdp.observations:
            if ipomdp.P_lower[s].get(obs, 0.0) > 0:
                obs_to_states.setdefault(obs, []).append(s)
    return {
        "none": lambda: NoShield(all_actions),
        "observation": lambda: ObservationShield(pp_shield, obs_to_states, all_actions),
        "single_belief": lambda: SingleBeliefShield(pomdp, pp_shield, THRESHOLD),
    }


def run_noise_sweep():
    print("=" * 70)
    print("REFUEL NOISE DIAGNOSTIC")
    print(f"Sweeping obs_noise = {OBS_NOISE_VALUES}")
    print(f"Trials: {NUM_TRIALS}, Length: {TRIAL_LENGTH}, Seed: {SEED}")
    print(f"Shields: {SHIELD_NAMES}")
    print(f"Perception: uniform only  |  Selector: random only")
    print("=" * 70)

    rows = []

    for obs_noise in OBS_NOISE_VALUES:
        print(f"\nBuilding Refuel IPOMDP (obs_noise={obs_noise})...")
        ipomdp, pp_shield, _, _ = build_refuel_ipomdp(obs_noise=obs_noise)
        print(f"  States: {len(ipomdp.states)}, Observations: {len(ipomdp.observations)}")

        shields = build_shields(ipomdp, pp_shield)
        perception = UniformPerceptionModel()
        selector = RandomActionSelector()
        init_gen = RandomInitialState()

        for sh_name in SHIELD_NAMES:
            t0 = time.time()
            trial_results = run_monte_carlo_trials(
                ipomdp=ipomdp,
                pp_shield=pp_shield,
                perception=perception,
                rt_shield_factory=shields[sh_name],
                action_selector=selector,
                initial_generator=init_gen,
                num_trials=NUM_TRIALS,
                trial_length=TRIAL_LENGTH,
                seed=SEED,
            )
            m = compute_safety_metrics(trial_results)
            elapsed = time.time() - t0

            rows.append((obs_noise, sh_name, m, elapsed))
            print(f"  noise={obs_noise:.2f}  shield={sh_name:<14} "
                  f"fail={m.fail_rate:.0%}  stuck={m.stuck_rate:.0%}  "
                  f"safe={m.safe_rate:.0%}  steps={m.mean_steps:.1f}  ({elapsed:.1f}s)")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'obs_noise':>10}  {'shield':<14}  {'fail%':>6}  {'stuck%':>6}  {'safe%':>6}  {'steps':>6}"
    print(header)
    print("-" * 70)
    for obs_noise, sh_name, m, _ in rows:
        print(f"{obs_noise:>10.3f}  {sh_name:<14}  "
              f"{m.fail_rate:>5.0%}  {m.stuck_rate:>6.0%}  "
              f"{m.safe_rate:>5.0%}  {m.mean_steps:>6.1f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("\nSingleBelief shield safe% by obs_noise:")
    for obs_noise, sh_name, m, _ in rows:
        if sh_name == "single_belief":
            bar = "#" * int(m.safe_rate * 30)
            print(f"  noise={obs_noise:.3f}: {m.safe_rate:5.0%} safe, "
                  f"{m.stuck_rate:5.0%} stuck  {bar}")

    print("\nNo-shield safe% by obs_noise (verifies IPOMDP solvability):")
    for obs_noise, sh_name, m, _ in rows:
        if sh_name == "none":
            print(f"  noise={obs_noise:.3f}: {m.safe_rate:5.0%} safe, "
                  f"{m.fail_rate:5.0%} fail")

    print("\nObservationShield stuck% by obs_noise (expected: always 100% - bug):")
    for obs_noise, sh_name, m, _ in rows:
        if sh_name == "observation":
            print(f"  noise={obs_noise:.3f}: stuck={m.stuck_rate:.0%}  "
                  f"(mean_steps={m.mean_steps:.1f})")


def main():
    t_total = time.time()
    run_noise_sweep()
    print(f"\nTotal time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
