"""Debug why Lifted shield is failing with perfect observations."""

import random
from ipomdp_shielding.Models.pomdp import POMDP
from ipomdp_shielding.Models.ipomdp import IPOMDP
from ipomdp_shielding.CaseStudies.Taxinet.taxinet import (
    taxinet_states, taxinet_dynamics_prob, taxinet_actions,
    taxinet_next_state, taxinet_safe
)
from ipomdp_shielding.Evaluation.runtime_shield import RuntimeImpShield
from ipomdp_shielding.Propagators.lfp_propagator import LFPPropagator, TemplateFactory, default_solver
from ipomdp_shielding.Propagators.belief_polytope import BeliefPolytope

# Build models
states = taxinet_states(with_fail=True)
actions_dict = taxinet_actions()
all_actions = list(set(a for acts in actions_dict.values() for a in acts))
mdp = taxinet_dynamics_prob(error=0.0)  # Deterministic

# Perfect observations
P = {s: {obs: 1.0 if obs == s else 0.0 for obs in states} for s in states}

pomdp = POMDP(states=states, observations=states, actions=all_actions, T=mdp.P, P=P)

# Convert to IPOMDP
P_lower = {s: dict(P[s]) for s in states}
P_upper = {s: dict(P[s]) for s in states}
ipomdp = IPOMDP(states, states, all_actions, pomdp.T, P_lower, P_upper)

# Build Lifted shield
avoid_states = frozenset(["FAIL"])
safe_states = frozenset(s for s in states if s not in avoid_states)

n_states = len(states)
state_to_idx = {s: i for i, s in enumerate(states)}
safe_indices = [state_to_idx[s] for s in safe_states]
template = TemplateFactory.safe_set_indicators(n_states, {"safe": safe_indices})
initial_polytope = BeliefPolytope.uniform_prior(n_states)
lfp_propagator = LFPPropagator(ipomdp, template, default_solver(), initial_polytope)

pp_shield = {s: set(all_actions) if s in safe_states else set() for s in states}
lifted_shield = RuntimeImpShield(pp_shield, lfp_propagator, 0.9, default_action=None)

# Run a single trajectory with detailed logging
initial_state = (0, 0)
state = initial_state
lifted_shield.initialize(state)

print(f"Initial state: {state}")
print(f"Safe? {taxinet_safe(state)}")

for t in range(10):
    print(f"\n--- Timestep {t} ---")
    print(f"State: {state}")

    if t == 0:
        allowed_actions = list(lifted_shield.actions)
    else:
        obs = state  # Perfect observation
        print(f"Evidence: obs={obs}, action={action}")
        allowed_actions = lifted_shield.next_actions((obs, action))

    print(f"Allowed actions: {allowed_actions}")

    # Get action probabilities
    action_probs = lifted_shield.get_action_probs()
    for a, allowed_prob, disallowed_prob in action_probs:
        print(f"  Action {a:2d}: allowed_prob={allowed_prob:.3f}, disallowed_prob={disallowed_prob:.3f}")

    if not allowed_actions:
        print("STUCK!")
        break

    # Select action
    action = random.choice(allowed_actions)
    print(f"Selected action: {action}")

    # Take action
    next_state = taxinet_next_state(state, action)
    print(f"Next state: {next_state}, safe: {taxinet_safe(next_state)}")

    if not taxinet_safe(next_state):
        print("FAILED!")
        break

    state = next_state
