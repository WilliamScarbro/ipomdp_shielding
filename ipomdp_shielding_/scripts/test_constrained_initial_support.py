"""
Test Carr shield with constrained initial support.

Instead of starting with all safe states, start with just the initial state.
"""

from ipomdp_shielding.CaseStudies.Taxinet.taxinet import build_taxinet_ipomdp
from ipomdp_shielding.CaseStudies.Taxinet.taxinet_pomdp_adapter import (
    convert_taxinet_to_pomdp,
    get_taxinet_avoid_states,
)
from ipomdp_shielding.Evaluation.carr_shield import CarrShield

# Build TaxiNet
print("Building TaxiNet IPOMDP...")
ipomdp, _, _, _ = build_taxinet_ipomdp(seed=42, error=0.01)

print(f"Number of states: {len(ipomdp.states)}")

# Convert to POMDP
print("Converting to POMDP...")
pomdp = convert_taxinet_to_pomdp(ipomdp, realization="midpoint", seed=42)

# Get avoid states
avoid_states = get_taxinet_avoid_states()
print(f"Avoid states: {avoid_states}")

# Test different initial supports
initial_supports = [
    ("Single state (0,0)", frozenset([(0, 0)])),
    ("Single state (0,0) and neighbors", frozenset([(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)])),
    ("All safe states", frozenset(s for s in ipomdp.states if s not in avoid_states)),
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
    else:
        print(f"✗ Winning region is empty")
