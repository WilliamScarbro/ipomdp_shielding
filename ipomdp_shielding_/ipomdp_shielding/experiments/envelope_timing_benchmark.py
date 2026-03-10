"""Envelope shield timing benchmark across all four case studies.

Runs a fixed number of trials with the envelope shield only, recording
wall-clock time per LP step.  Used to characterise the feasibility
boundary as a function of state-space size.

Output: results/prelim/envelope_timing.json
"""
import json, os, time

from ..CaseStudies.Taxinet import build_taxinet_ipomdp
from ..CaseStudies.CartPole import build_cartpole_ipomdp
from ..CaseStudies.GridWorldBenchmarks import build_obstacle_ipomdp, build_refuel_ipomdp
from ..MonteCarlo import (
    UniformPerceptionModel, RandomActionSelector,
    RandomInitialState, run_monte_carlo_trials,
)
from .run_rl_shield_experiment import create_envelope_shield_factory

NUM_TRIALS = 5
TRIAL_LENGTH = 10
SEED = 42

CASE_STUDIES = [
    ("taxinet",  build_taxinet_ipomdp,  {}),
    ("obstacle", build_obstacle_ipomdp, {}),
    ("cartpole", build_cartpole_ipomdp, {"num_bins": 3}),
    ("refuel",   build_refuel_ipomdp,   {}),
]

def run():
    results = []
    os.makedirs("results/prelim", exist_ok=True)

    for name, build_fn, kwargs in CASE_STUDIES:
        print(f"\n{'='*60}")
        print(f"ENVELOPE TIMING: {name.upper()}")
        ipomdp, pp_shield, _, _ = build_fn(**kwargs)
        n_states = len(ipomdp.states)
        n_obs    = len(ipomdp.observations)
        print(f"  States={n_states}  Obs={n_obs}")

        factory   = create_envelope_shield_factory(ipomdp, pp_shield, threshold=0.8)
        perception = UniformPerceptionModel()
        selector   = RandomActionSelector()
        init_gen   = RandomInitialState()

        t0 = time.time()
        trial_results = run_monte_carlo_trials(
            ipomdp=ipomdp, pp_shield=pp_shield,
            perception=perception, rt_shield_factory=factory,
            action_selector=selector, initial_generator=init_gen,
            num_trials=NUM_TRIALS, trial_length=TRIAL_LENGTH, seed=SEED,
        )
        elapsed = time.time() - t0

        total_steps = sum(r.steps_completed for r in trial_results)
        stuck = sum(1 for r in trial_results if r.outcome == "stuck")
        safe  = sum(1 for r in trial_results if r.outcome == "safe")
        fail  = sum(1 for r in trial_results if r.outcome == "fail")
        time_per_step = elapsed / max(total_steps, 1)

        print(f"  Elapsed: {elapsed:.1f}s  Steps: {total_steps}  "
              f"fail={fail/NUM_TRIALS:.0%} stuck={stuck/NUM_TRIALS:.0%} "
              f"safe={safe/NUM_TRIALS:.0%}")
        print(f"  Time per step: {time_per_step:.2f}s")

        results.append({
            "case_study": name,
            "n_states": n_states,
            "n_observations": n_obs,
            "num_trials": NUM_TRIALS,
            "trial_length": TRIAL_LENGTH,
            "total_steps": total_steps,
            "elapsed_s": round(elapsed, 2),
            "time_per_step_s": round(time_per_step, 3),
            "fail_rate": fail / NUM_TRIALS,
            "stuck_rate": stuck / NUM_TRIALS,
            "safe_rate": safe / NUM_TRIALS,
        })

    out_path = "results/prelim/envelope_timing.json"
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    print("\n" + "="*65)
    print(f"{'Case study':<12} {'States':>8} {'Obs':>6} {'t/step (s)':>12} {'safe%':>7}")
    print("-"*65)
    for r in results:
        print(f"{r['case_study']:<12} {r['n_states']:>8} {r['n_observations']:>6} "
              f"{r['time_per_step_s']:>12.3f} {r['safe_rate']:>6.0%}")

if __name__ == "__main__":
    run()
