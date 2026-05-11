"""Fair MC comparison for cp-control and TaxiNetV2 IPOMDP shields.

The comparison uses one cp-control TaxiNet base perception model:
- envelope, forward_sampling, and single_belief use point-estimate confusion;
- cp_control_conformal uses conformal-set observations and the cp-control
  set-intersection controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

from ..CaseStudies.TaxiNetV2 import (
    build_taxinet_v2_conformal_ipomdp,
    build_taxinet_v2_single_estimate_ipomdp,
    get_taxinet_v2_conditional_conformal_axis_data,
    get_taxinet_v2_metadata,
)
from ..Evaluation.conformal_set_shield import ConformalSetIntersectionShield
from ..MonteCarlo import (
    AdversarialPerceptionModel,
    BoundaryInitialState,
    ModularConditionalConformalTaxiNetPerception,
    PerceptionModel,
    RandomActionSelector,
    SafeInitialState,
    UniformPerceptionModel,
    compute_safety_metrics,
    run_monte_carlo_trials,
)
from .run_rl_shield_experiment import (
    create_envelope_shield_factory,
    create_forward_sampling_shield_factory,
    create_single_belief_shield_factory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--trial-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-method", default="Clopper_Pearson")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--confidence-level", default="0.95")
    parser.add_argument("--shield-threshold", type=float, default=0.8)
    parser.add_argument("--action-filter", type=float, default=0.7)
    parser.add_argument("--action-success", type=float, default=0.9)
    parser.add_argument("--forward-budget", type=int, default=500)
    parser.add_argument("--forward-k-samples", type=int, default=100)
    parser.add_argument(
        "--point-realization",
        choices=["uniform", "adversarial"],
        default="uniform",
    )
    parser.add_argument(
        "--initial",
        choices=["safe", "boundary"],
        default="safe",
        help="Initial state sampling regime. Both avoid starting in FAIL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/taxinet_v2/fair_mc_comparison/results.json"),
    )
    return parser.parse_args()


def _metrics_to_dict(metrics) -> dict:
    return {
        "num_trials": metrics.num_trials,
        "fail_rate": metrics.fail_rate,
        "stuck_rate": metrics.stuck_rate,
        "safe_rate": metrics.safe_rate,
        "mean_steps": metrics.mean_steps,
        "mean_stuck_count": metrics.mean_stuck_count,
    }


def _initial_generator(name: str):
    if name == "safe":
        return SafeInitialState()
    if name == "boundary":
        return BoundaryInitialState()
    raise ValueError(f"Unsupported initial generator: {name}")


def _build_point_perception(
    strategy: str,
    pp_shield: Dict,
) -> PerceptionModel:
    if strategy == "uniform":
        return UniformPerceptionModel()
    if strategy == "adversarial":
        return AdversarialPerceptionModel(pp_shield)
    raise ValueError(f"Unsupported TaxiNetV2 point realization strategy: {strategy}")


def run_fair_mc(args: argparse.Namespace) -> dict:
    point_ipomdp, pp_shield, _test_cte, _test_he = build_taxinet_v2_single_estimate_ipomdp(
        confidence_method=args.confidence_method,
        alpha=args.alpha,
        action_success=args.action_success,
        smoothing=True,
    )
    conformal_ipomdp, _conformal_pp, _projected_cte, _projected_he = build_taxinet_v2_conformal_ipomdp(
        confidence_method=args.confidence_method,
        alpha=args.alpha,
        confidence_level=args.confidence_level,
        action_success=args.action_success,
        smoothing=True,
    )

    conditional_cte_sets, conditional_he_sets = get_taxinet_v2_conditional_conformal_axis_data(
        confidence_level=args.confidence_level
    )

    point_perception = _build_point_perception(args.point_realization, pp_shield)
    conformal_perception = ModularConditionalConformalTaxiNetPerception(
        point_perception,
        point_ipomdp,
        conditional_cte_sets,
        conditional_he_sets,
    )
    action_selector = RandomActionSelector()
    initial_generator = _initial_generator(args.initial)

    point_methods: Dict[str, Callable] = {
        "single_belief": create_single_belief_shield_factory(
            point_ipomdp.to_pomdp(), pp_shield, threshold=args.shield_threshold
        ),
        "envelope": create_envelope_shield_factory(
            point_ipomdp, pp_shield, threshold=args.shield_threshold
        ),
        "forward_sampling": create_forward_sampling_shield_factory(
            point_ipomdp,
            pp_shield,
            threshold=args.shield_threshold,
            budget=args.forward_budget,
            K_samples=args.forward_k_samples,
        ),
    }

    def conformal_factory():
        return ConformalSetIntersectionShield.from_tempest_csv(
            action_filter=args.action_filter,
            all_actions=point_ipomdp.actions,
        )

    all_results = {}
    all_trials = {}

    for offset, (method, factory) in enumerate(point_methods.items()):
        trials = run_monte_carlo_trials(
            ipomdp=point_ipomdp,
            pp_shield=pp_shield,
            perception=point_perception,
            rt_shield_factory=factory,
            action_selector=action_selector,
            initial_generator=initial_generator,
            num_trials=args.num_trials,
            trial_length=args.trial_length,
            seed=args.seed + offset,
        )
        all_results[method] = _metrics_to_dict(compute_safety_metrics(trials))
        all_trials[method] = [trial.__dict__ for trial in trials]

    trials = run_monte_carlo_trials(
        ipomdp=conformal_ipomdp,
        pp_shield=pp_shield,
        perception=conformal_perception,
        rt_shield_factory=conformal_factory,
        action_selector=action_selector,
        initial_generator=initial_generator,
        num_trials=args.num_trials,
        trial_length=args.trial_length,
        seed=args.seed + len(point_methods),
    )
    all_results["cp_control_conformal"] = _metrics_to_dict(compute_safety_metrics(trials))
    all_trials["cp_control_conformal"] = [trial.__dict__ for trial in trials]

    metadata = {
        "case_study": "taxinet_v2_fair_mc",
        "base_perception_model": "cp-control train/models/best_model.pth",
        "point_methods_use": (
            "point-estimate cp-control TaxiNet DNN events sampled from the "
            f"point-estimate IPOMDP using modular realization strategy "
            f"{args.point_realization!r}"
        ),
        "conformal_method_uses": (
            "two-stage cp-control TaxiNet confusion data: estimate sampled from the "
            f"point-estimate IPOMDP using modular realization strategy "
            f"{args.point_realization!r}, set from conditional (state, estimate) artifacts"
        ),
        "point_realization": args.point_realization,
        "dynamics": "cp-control stochastic action perturbation",
        "action_selector": "RandomActionSelector",
        "initial": args.initial,
        "args": vars(args) | {"output": str(args.output)},
        "taxinet_v2_metadata": get_taxinet_v2_metadata(args.confidence_level).__dict__,
    }

    return {"metadata": metadata, "results": all_results, "trials": all_trials}


def main() -> None:
    args = parse_args()
    result = run_fair_mc(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved fair MC comparison to {args.output}")
    for method, metrics in result["results"].items():
        print(
            f"{method:<22} fail={metrics['fail_rate']:.1%} "
            f"stuck={metrics['stuck_rate']:.1%} safe={metrics['safe_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
