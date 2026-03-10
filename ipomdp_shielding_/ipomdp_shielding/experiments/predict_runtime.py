"""Predict experiment runtime from calibrated per-step envelope timings.

Timings measured from the threshold sweep run (2026-03-09) on this machine.
The LP solve (envelope shield) dominates; single_belief/none/observation are
negligible in comparison.

Calibrated values
-----------------
  TaxiNet  (16 states, 16 obs): 0.047 s/step
    → 47s / (50 trials × 20 steps), avg over 18 envelope combos
  CartPole (82 states, 82 obs): 1.458 s/step
    → 328s / (15 trials × 15 steps), avg over 18 envelope combos
  Obstacle (50 states,  3 obs): 0.278 s/step
    → 174s / (25 trials × 25 steps), avg over first 4 envelope combos
  Refuel  (344 states, 43 obs): ~144 s/step  (from benchmark; not run in sweep)

Usage
-----
As a library:
    from ipomdp_shielding.experiments.predict_runtime import predict_runtime
    p = predict_runtime("cartpole", num_trials=200, trial_length=15,
                        shields=["single_belief", "envelope"],
                        selectors=["rl"], num_thresholds=9)
    print(p["grand_total_hms"])

As a script:
    python -m ipomdp_shielding.experiments.predict_runtime
"""

# ---------------------------------------------------------------------------
# Calibrated constants
# ---------------------------------------------------------------------------

# Measured envelope LP solve time per trial-step (seconds).
# Multiply by (num_trials * trial_length) to get seconds per combo.
ENVELOPE_S_PER_STEP: dict[str, float] = {
    "taxinet":  0.047,   # 47s / (50 × 20) — measured, 18 combo avg
    "cartpole": 1.458,   # 328s / (15 × 15) — measured, 18 combo avg
    "obstacle": 0.278,   # 174s / (25 × 25) — measured, 4 combo avg
    "refuel":   144.0,   # benchmark (LP infeasible at this speed)
}

# Single-belief belief propagation is negligible by comparison.
FAST_SHIELD_S_PER_STEP: float = 0.0001

# One-time overhead per case study: load IPOMDP + RL agent + opt realizations.
SETUP_OVERHEAD_S: float = 15.0


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_runtime(
    case_study: str,
    num_trials: int,
    trial_length: int,
    shields: list[str] | None = None,
    perceptions: list[str] | None = None,
    selectors: list[str] | None = None,
    num_thresholds: int = 1,
) -> dict:
    """Predict wall-clock runtime for one case study.

    Parameters
    ----------
    case_study : str
        One of 'taxinet', 'cartpole', 'obstacle', 'refuel'.
    num_trials : int
        Number of Monte Carlo trials per combination.
    trial_length : int
        Steps per trial.
    shields : list[str], optional
        Shields to include. Default: ['none','observation','single_belief','envelope'].
    perceptions : list[str], optional
        Perception regimes. Default: ['uniform', 'adversarial_opt'].
    selectors : list[str], optional
        Action selectors. Default: ['random', 'best', 'rl'].
    num_thresholds : int
        Number of threshold values to sweep (default 1 = single experiment).

    Returns
    -------
    dict with keys:
        case_study, num_trials, trial_length, num_thresholds,
        shields, perceptions, selectors,
        steps_per_combo,
        breakdown_s_per_threshold  : {shield -> seconds per threshold},
        per_threshold_s            : total seconds per threshold (all shields),
        setup_s                    : one-time setup overhead,
        total_s                    : setup + per_threshold_s * num_thresholds,
        total_min, total_h,
        grand_total_hms            : "HHh MMm SSs" string,
    """
    if shields is None:
        shields = ["none", "observation", "single_belief", "envelope"]
    if perceptions is None:
        perceptions = ["uniform", "adversarial_opt"]
    if selectors is None:
        selectors = ["random", "best", "rl"]

    cs = case_study.lower()
    if cs not in ENVELOPE_S_PER_STEP:
        known = list(ENVELOPE_S_PER_STEP)
        raise ValueError(f"Unknown case study {case_study!r}. Known: {known}")

    combos_per_shield = len(perceptions) * len(selectors)
    steps_per_combo = num_trials * trial_length

    breakdown: dict[str, float] = {}
    per_threshold_s = 0.0

    for shield in shields:
        if shield == "envelope":
            s_per_step = ENVELOPE_S_PER_STEP[cs]
        else:
            s_per_step = FAST_SHIELD_S_PER_STEP

        t = s_per_step * steps_per_combo * combos_per_shield
        breakdown[shield] = t
        per_threshold_s += t

    total_s = SETUP_OVERHEAD_S + per_threshold_s * num_thresholds

    return {
        "case_study": cs,
        "num_trials": num_trials,
        "trial_length": trial_length,
        "num_thresholds": num_thresholds,
        "shields": list(shields),
        "perceptions": list(perceptions),
        "selectors": list(selectors),
        "steps_per_combo": steps_per_combo,
        "breakdown_s_per_threshold": breakdown,
        "per_threshold_s": per_threshold_s,
        "setup_s": SETUP_OVERHEAD_S,
        "total_s": total_s,
        "total_min": total_s / 60,
        "total_h": total_s / 3600,
        "grand_total_hms": _fmt_hms(total_s),
    }


def predict_sweep_runtime(
    sweep_params: dict,
    num_thresholds: int = 9,
) -> dict:
    """Predict total runtime for a multi-case-study threshold sweep.

    Parameters
    ----------
    sweep_params : dict
        Maps case_study_name -> dict with keys:
          num_trials, trial_length, and optionally:
          exclude_envelope (bool), shields, perceptions, selectors.
    num_thresholds : int
        Number of threshold values in the sweep.

    Returns
    -------
    dict with per_case_study predictions and grand total.
    """
    per_cs: dict[str, dict] = {}
    grand_total_s = 0.0

    for cs_name, params in sweep_params.items():
        # Build shield list from params
        shields = params.get("shields")
        if shields is None:
            if params.get("exclude_envelope", False):
                shields = ["none", "observation", "single_belief"]
            else:
                shields = ["none", "observation", "single_belief", "envelope"]

        pred = predict_runtime(
            case_study=cs_name,
            num_trials=params["num_trials"],
            trial_length=params["trial_length"],
            shields=shields,
            perceptions=params.get("perceptions"),
            selectors=params.get("selectors"),
            num_thresholds=num_thresholds,
        )
        per_cs[cs_name] = pred
        grand_total_s += pred["total_s"]

    return {
        "per_case_study": per_cs,
        "num_thresholds": num_thresholds,
        "grand_total_s": grand_total_s,
        "grand_total_min": grand_total_s / 60,
        "grand_total_h": grand_total_s / 3600,
        "grand_total_hms": _fmt_hms(grand_total_s),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_hms(seconds: float) -> str:
    hh, rem = divmod(int(seconds), 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}h {mm:02d}m {ss:02d}s"


def print_prediction(pred: dict, indent: str = "") -> None:
    """Pretty-print a single-case-study prediction."""
    cs = pred["case_study"].upper()
    n, l, t = pred["num_trials"], pred["trial_length"], pred["num_thresholds"]
    print(f"{indent}{cs}  —  {n} trials × {l} steps × {t} threshold(s)")
    for shield, ts in pred["breakdown_s_per_threshold"].items():
        total_for_shield = ts * t
        if total_for_shield >= 1.0:
            bar = "█" * min(40, int(total_for_shield / pred["total_s"] * 40))
            print(f"{indent}  {shield:<16}  {_fmt_hms(total_for_shield):>14}  {bar}")
    print(f"{indent}  {'(setup)':<16}  {_fmt_hms(pred['setup_s']):>14}")
    print(f"{indent}  {'TOTAL':<16}  {_fmt_hms(pred['total_s']):>14}")


def print_sweep_prediction(sweep_pred: dict, title: str = "") -> None:
    """Pretty-print a full sweep prediction."""
    if title:
        print(f"\n{'=' * 60}")
        print(title)
        print("=" * 60)
    for cs_name, pred in sweep_pred["per_case_study"].items():
        print_prediction(pred)
        print()
    print(f"  Grand total:  {sweep_pred['grand_total_hms']}")


# ---------------------------------------------------------------------------
# Main: show current sweep and 200-trial scenarios
# ---------------------------------------------------------------------------

def main() -> None:
    from .run_threshold_sweep import SWEEP_PARAMS, THRESHOLDS

    n_thresholds = len(THRESHOLDS)

    # -----------------------------------------------------------------------
    # 1. Current sweep (as actually run: rl selector only, threshold-sensitive
    #    shields only — none/observation are excluded from the sweep grid)
    # -----------------------------------------------------------------------
    sweep_current = {
        cs: {
            "num_trials": p["num_trials"],
            "trial_length": p["trial_length"],
            "shields": (
                ["single_belief"]
                if p["exclude_envelope"]
                else ["single_belief", "envelope"]
            ),
            "selectors": ["rl"],
        }
        for cs, p in SWEEP_PARAMS.items()
    }
    print_sweep_prediction(
        predict_sweep_runtime(sweep_current, num_thresholds=n_thresholds),
        title="CURRENT SWEEP (as run: rl only, single_belief + envelope)",
    )

    # -----------------------------------------------------------------------
    # 2. 200-trial sweep — same filtering, scaled up
    # -----------------------------------------------------------------------
    sweep_200_all = {
        cs: {**params, "num_trials": 200}
        for cs, params in sweep_current.items()
    }
    print_sweep_prediction(
        predict_sweep_runtime(sweep_200_all, num_thresholds=n_thresholds),
        title="200-TRIAL SWEEP (same structure)",
    )

    # -----------------------------------------------------------------------
    # 3. 200-trial sweep — CartPole envelope excluded
    #    Justified: envelope never beats single_belief for CartPole (fixed
    #    13.3% stuck at every threshold, same fail rate).
    # -----------------------------------------------------------------------
    sweep_200_no_cp_env = {
        cs: {
            **params,
            "shields": ["single_belief"] if cs == "cartpole" else params["shields"],
        }
        for cs, params in sweep_200_all.items()
    }
    print_sweep_prediction(
        predict_sweep_runtime(sweep_200_no_cp_env, num_thresholds=n_thresholds),
        title="200-TRIAL SWEEP (CartPole envelope excluded — recommended)",
    )


if __name__ == "__main__":
    main()
