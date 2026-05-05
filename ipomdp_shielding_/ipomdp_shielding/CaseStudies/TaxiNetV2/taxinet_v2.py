"""TaxiNetV2: TaxiNet dynamics with vendored conformal-set observations."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Hashable, List, Optional, Tuple

from statsmodels.stats.proportion import proportion_confint

from ...Models import IPOMDP, IMDP
from ...Models.Confidence import ConfidenceInterval
from ..Taxinet.taxinet import (
    FAIL,
    taxinet_actions,
    taxinet_dynamics_prob,
    taxinet_next_state,
    taxinet_safe,
    taxinet_states,
)
from .data_loader import (
    DEFAULT_ALPHA_LEVEL,
    ConformalObservation,
    SignedTaxiState,
    get_taxinet_v2_observation_data,
    get_taxinet_v2_projected_test_models,
)


BENCHMARK_SPEC = (
    "TaxiNetV2 keeps the current TaxiNet dynamics, actions, and safety objective, "
    "but replaces the aggregate perception artifact with vendored conformal "
    "prediction-set observations on the same signed TaxiNet state space."
)


def _fill_observation_coverage(
    imdp: IMDP,
    states: List[SignedTaxiState],
    observations: List[ConformalObservation],
    data: List[Tuple[SignedTaxiState, ConformalObservation]],
    action: str,
    alpha: float,
) -> None:
    """Add conservative zero-count upper bounds for unseen state/observation pairs."""
    totals = Counter(true_state for true_state, _ in data)

    for true_state in states:
        key = (true_state, action)
        n = totals.get(true_state, 0)
        if n == 0:
            imdp.P_lower[key] = {obs: 0.0 for obs in observations}
            imdp.P_upper[key] = {obs: 1.0 for obs in observations}
            continue

        if key not in imdp.P_lower:
            imdp.P_lower[key] = {}
            imdp.P_upper[key] = {}

        _, zero_upper = proportion_confint(0, n, alpha=alpha, method="beta")
        for obs in observations:
            if obs not in imdp.P_lower[key]:
                imdp.P_lower[key][obs] = 0.0
                imdp.P_upper[key][obs] = zero_upper


def taxinet_v2_perception(
    confidence_method: str,
    alpha: float,
    observation_data: List[Tuple[SignedTaxiState, ConformalObservation]],
    smoothing: bool = True,
) -> Tuple[IMDP, List[ConformalObservation]]:
    """Build an IMDP over TaxiNetV2 conformal-set observations."""
    states = taxinet_states(with_fail=True)
    observations = sorted({obs for _state, obs in observation_data})
    perceive_action = "PERC"

    obs_ci = ConfidenceInterval(observation_data)
    obs_imdp = obs_ci.produce_imdp(states[:-1], perceive_action, confidence_method, alpha)

    if smoothing:
        _fill_observation_coverage(
            obs_imdp,
            states[:-1],
            observations,
            observation_data,
            perceive_action,
            alpha,
        )

    obs_imdp.actions[FAIL] = [perceive_action]
    obs_imdp.P_lower[(FAIL, perceive_action)] = {FAIL: 1.0}
    obs_imdp.P_upper[(FAIL, perceive_action)] = {FAIL: 1.0}

    return obs_imdp, observations + [FAIL]


def build_taxinet_v2_ipomdp(
    confidence_method: str = "Clopper_Pearson",
    alpha: float = 0.05,
    confidence_level: str = DEFAULT_ALPHA_LEVEL,
    error: float = 0.1,
    smoothing: bool = True,
    seed: Optional[int] = None,
) -> Tuple[IPOMDP, Dict[Hashable, set], Dict[int, List[int]], Dict[int, List[int]]]:
    """Build TaxiNetV2 from vendored conformal-set observation samples.

    The returned test models are projected concrete observations for compatibility
    with older helper code. The IPOMDP itself uses conformal-set observations.
    """
    del seed

    observation_data = get_taxinet_v2_observation_data(confidence_level=confidence_level)
    perception_imdp, observations = taxinet_v2_perception(
        confidence_method=confidence_method,
        alpha=alpha,
        observation_data=observation_data,
        smoothing=smoothing,
    )

    states = taxinet_states(with_fail=True)
    perc_lower = {
        state: dict(perception_imdp.P_lower[(state, "PERC")])
        for state in states
    }
    perc_upper = {
        state: dict(perception_imdp.P_upper[(state, "PERC")])
        for state in states
    }

    dyn_mdp = taxinet_dynamics_prob(error=error)
    actions = [-1, 0, 1]
    ipomdp = IPOMDP(states, observations, actions, dyn_mdp.P, perc_lower, perc_upper)

    dyn_shield = {
        state: {action for action in actions if taxinet_safe(taxinet_next_state(state, action))}
        for state in states
    }

    test_cte_model, test_he_model = get_taxinet_v2_projected_test_models(
        confidence_level=confidence_level
    )
    return ipomdp, dyn_shield, test_cte_model, test_he_model
