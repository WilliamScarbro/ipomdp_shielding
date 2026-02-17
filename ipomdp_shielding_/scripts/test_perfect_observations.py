"""
Test Carr shield with perfect observations (no observation uncertainty).

This tests whether the issue is observation uncertainty vs dynamics uncertainty.
"""

from ipomdp_shielding.Models.pomdp import POMDP
from ipomdp_shielding.CaseStudies.Taxinet.taxinet import taxinet_states, taxinet_dynamics_prob, taxinet_actions
from ipomdp_shielding.Evaluation.carr_shield import CarrShield

# Build TaxiNet with perfect observations
print("Building TaxiNet with perfect observations...")

states = taxinet_states(with_fail=True)
actions_dict = taxinet_actions()
all_actions = list(set(a for acts in actions_dict.values() for a in acts))

# Use TaxiNet dynamics (with error)
mdp = taxinet_dynamics_prob(error=0.01)

# Perfect observations: P(o=s | s) = 1.0
P = {}
for s in states:
    P[s] = {obs: 1.0 if obs == s else 0.0 for obs in states}

pomdp = POMDP(
    states=states,
    observations=states,  # Observations are just states
    actions=all_actions,
    T=mdp.P,
    P=P
)

print(f"Number of states: {len(states)}")
print(f"Number of actions: {len(all_actions)}")

# Test with different initial supports
avoid_states = frozenset(["FAIL"])

initial_supports = [
    ("Single state (0,0)", frozenset([(0, 0)])),
    ("All safe states", frozenset(s for s in states if s not in avoid_states)),
]

for name, initial_support in initial_supports:
    print("\n" + "="*60)
    print(f"Testing: {name}")
    print(f"Initial support size: {len(initial_support)}")
    print("="*60)

    # Build Carr shield
    shield = CarrShield(pomdp, avoid_states, initial_support)
    stats = shield.get_statistics()

    print(f"Total supports: {stats['total_supports']}")
    print(f"Winning supports: {stats['winning_supports']}")
    print(f"Losing supports: {stats['losing_supports']}")

    if stats['winning_supports'] > 0:
        print(f"✓ Non-empty winning region found!")
        print(f"  Winning rate: {stats['winning_supports'] / stats['total_supports']:.1%}")
    else:
        print(f"✗ Winning region is empty")
