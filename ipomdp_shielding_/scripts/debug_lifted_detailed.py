"""Debug Lifted shield behavior with point mass and perfect observations."""

import random
import numpy as np
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

# Build correct perfect-perception shield
avoid_states = frozenset(["FAIL"])
safe_states = frozenset(s for s in states if s not in avoid_states)

pp_shield = {}
for s in states:
    if s in avoid_states:
        pp_shield[s] = set()
    else:
        safe_actions = set()
        for a in all_actions:
            trans = pomdp.T.get((s, a), {})
            all_successors_safe = all(
                next_s not in avoid_states
                for next_s, prob in trans.items()
                if prob > 0
            )
            if all_successors_safe:
                safe_actions.add(a)
        pp_shield[s] = safe_actions

print(f"Perfect-perception shield for state (0,0): {pp_shield[(0,0)]}")

# Build Lifted shield with point mass initial belief
n_states = len(states)
state_to_idx = {s: i for i, s in enumerate(states)}
safe_indices = [state_to_idx[s] for s in safe_states]
template = TemplateFactory.safe_set_indicators(n_states, {"safe": safe_indices})

initial_state = (0, 0)
initial_idx = state_to_idx[initial_state]

# Point mass at initial state
epsilon = 1e-6
A_list = []
d_list = []
for i in range(n_states):
    if i == initial_idx:
        A_list.append(np.array([1.0 if j == i else 0.0 for j in range(n_states)]))
        d_list.append(1.0 + epsilon)
        A_list.append(np.array([-1.0 if j == i else 0.0 for j in range(n_states)]))
        d_list.append(-(1.0 - epsilon))
    else:
        A_list.append(np.array([1.0 if j == i else 0.0 for j in range(n_states)]))
        d_list.append(epsilon)
        A_list.append(np.array([-1.0 if j == i else 0.0 for j in range(n_states)]))
        d_list.append(epsilon)

initial_polytope = BeliefPolytope(n=n_states, A=np.array(A_list), d=np.array(d_list))

lfp_propagator = LFPPropagator(ipomdp, template, default_solver(), initial_polytope)

lifted_shield = RuntimeImpShield(pp_shield, lfp_propagator, 0.9, default_action=None)

# Run trajectory with detailed logging
state = initial_state
lifted_shield.initialize(state)

print(f"\nInitial state: {state}")

for t in range(5):
    print(f"\n--- Timestep {t} ---")
    print(f"State: {state}")

    if t == 0:
        # First timestep - get allowed actions before propagation
        print("First timestep - checking action probabilities before propagation")
        action_probs = lifted_shield.get_action_probs()
        for a, allowed_prob, disallowed_prob in action_probs:
            pp_safe = a in pp_shield.get(state, set())
            print(f"  Action {a:2d}: allowed={allowed_prob:.4f}, disallowed={disallowed_prob:.4f}, pp_safe={pp_safe}")

        allowed_actions = list(lifted_shield.actions)
    else:
        obs = state
        print(f"Evidence: obs={obs}, action={action}")
        allowed_actions = lifted_shield.next_actions((obs, action))

        print("After propagation:")
        action_probs = lifted_shield.get_action_probs()
        for a, allowed_prob, disallowed_prob in action_probs:
            pp_safe = a in pp_shield.get(state, set())
            print(f"  Action {a:2d}: allowed={allowed_prob:.4f}, disallowed={disallowed_prob:.4f}, pp_safe={pp_safe}")

    print(f"Allowed actions: {allowed_actions}")

    if not allowed_actions:
        print("STUCK!")
        break

    # Select action from allowed actions
    action = random.choice(allowed_actions)
    print(f"Selected action: {action}")

    # Take action
    next_state = taxinet_next_state(state, action)
    print(f"Next state: {next_state}, safe: {taxinet_safe(next_state)}")

    if not taxinet_safe(next_state):
        print("FAILED!")
        break

    state = next_state

# Additional debugging - check template bounds
print("\n\n=== DEBUGGING BELIEF POLYTOPE ===")
print(f"Template: {template.names}")
print(f"Belief polytope after initialization:")
polytope = lfp_propagator.belief

# Try to compute bounds manually
safe_template_idx = 0  # First template is P(safe)
v = template.V[safe_template_idx]
max_val = polytope.maximize_linear(v)
min_val = polytope.maximize_linear(-v)
print(f"P(safe) bounds: [{-min_val:.4f}, {max_val:.4f}]")

# After first propagation
print("\nPropagating with action=0, obs=(0,0)...")
success = lfp_propagator.propagate(0, (0,0))
print(f"Propagation success: {success}")

polytope = lfp_propagator.belief
max_val = polytope.maximize_linear(v)
min_val = polytope.maximize_linear(-v)
print(f"P(safe) bounds after propagation: [{-min_val:.4f}, {max_val:.4f}]")

# Check if we can maximize belief at state (0,0)
belief_vec = np.zeros(n_states)
belief_vec[state_to_idx[(0,0)]] = 1.0
max_at_00 = polytope.maximize_linear(belief_vec)
min_at_00 = polytope.maximize_linear(-belief_vec)
print(f"b[(0,0)] bounds: [{-min_at_00:.4f}, {max_at_00:.4f}]")
