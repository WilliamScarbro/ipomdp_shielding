"""Test script for BeliefPolytope volume computation.

Run from project root:
    python -m ipomdp_shielding.Evaluation.test_volume
"""

import numpy as np
import sys

# Add parent to path for imports
sys.path.insert(0, '/home/scarbro/claude')

from ipomdp_shielding.Propagators.belief_polytope import (
    BeliefPolytope, compute_volume, _enumerate_vertices, _find_interior_point_projected
)


def print_polytope_info(name: str, polytope: BeliefPolytope):
    """Print diagnostic information about a polytope."""
    print(f"\n{'='*60}")
    print(f"Polytope: {name}")
    print(f"{'='*60}")
    print(f"Dimension (n): {polytope.n}")
    print(f"Constraint matrix A shape: {polytope.A.shape}")
    print(f"Constraint vector d shape: {polytope.d.shape}")

    # Print bounds for each coordinate
    print(f"\nCoordinate bounds (via LP):")
    for i in range(polytope.n):
        e_i = np.zeros(polytope.n)
        e_i[i] = 1.0
        try:
            upper = polytope.maximize_linear(e_i)
            lower = -polytope.maximize_linear(-e_i)
            print(f"  b[{i}]: [{lower:.6f}, {upper:.6f}]  (spread: {upper - lower:.6f})")
        except RuntimeError as e:
            print(f"  b[{i}]: LP failed - {e}")

    # Find interior point (in projected (n-1) dimensional space)
    interior_proj = _find_interior_point_projected(polytope)
    if interior_proj is not None:
        # Lift to n dimensions
        interior = np.append(interior_proj, 1.0 - np.sum(interior_proj))
        print(f"\nInterior point (projected): {interior_proj}")
        print(f"Interior point (lifted): {interior}")
        print(f"  Sum: {np.sum(interior):.6f}")
        print(f"  Min coord: {np.min(interior):.6f}")
    else:
        print(f"\nNo interior point found!")

    # Enumerate vertices
    vertices = _enumerate_vertices(polytope)
    if vertices is not None:
        print(f"\nVertices: {len(vertices)} found")
        if len(vertices) <= 10:
            for i, v in enumerate(vertices):
                print(f"  v[{i}]: {v} (sum={np.sum(v):.6f})")
        else:
            print(f"  (showing first 5 and last 5)")
            for i in range(5):
                print(f"  v[{i}]: {vertices[i]} (sum={np.sum(vertices[i]):.6f})")
            print("  ...")
            for i in range(len(vertices)-5, len(vertices)):
                print(f"  v[{i}]: {vertices[i]} (sum={np.sum(vertices[i]):.6f})")
    else:
        print(f"\nVertex enumeration failed!")

    # Compute volume
    vol = compute_volume(polytope)
    print(f"\nVolume (fraction of simplex): {vol:.6f}")
    print(f"Volume (percentage): {vol * 100:.2f}%")


def test_full_simplex(n: int):
    """Test volume of the full probability simplex (should be 1.0)."""
    # No additional constraints beyond b >= 0, sum(b) = 1
    A = np.empty((0, n))
    d = np.empty(0)
    polytope = BeliefPolytope(n=n, A=A, d=d)
    print_polytope_info(f"Full {n}-simplex", polytope)


def test_uniform_prior(n: int):
    """Test volume of uniform prior (point, should be ~0)."""
    polytope = BeliefPolytope.uniform_prior(n)
    print_polytope_info(f"Uniform prior (n={n})", polytope)


def test_half_simplex(n: int):
    """Test volume when first coordinate is constrained to [0, 0.5]."""
    # Constraint: b[0] <= 0.5
    A = np.zeros((1, n))
    A[0, 0] = 1.0
    d = np.array([0.5])
    polytope = BeliefPolytope(n=n, A=A, d=d)
    print_polytope_info(f"Half simplex b[0] <= 0.5 (n={n})", polytope)


def test_box_constraint(n: int, lower: float, upper: float):
    """Test volume with box constraints on all coordinates."""
    # Constraints: lower <= b[i] <= upper for all i
    # b[i] <= upper  -->  b[i] <= upper
    # b[i] >= lower  -->  -b[i] <= -lower
    A_upper = np.eye(n)
    d_upper = upper * np.ones(n)
    A_lower = -np.eye(n)
    d_lower = -lower * np.ones(n)

    A = np.vstack([A_upper, A_lower])
    d = np.concatenate([d_upper, d_lower])

    polytope = BeliefPolytope(n=n, A=A, d=d)
    print_polytope_info(f"Box constraint [{lower}, {upper}] (n={n})", polytope)


def test_shrinking_polytope(n: int):
    """Test how volume changes as we add tighter constraints."""
    print(f"\n{'#'*60}")
    print(f"# Shrinking polytope test (n={n})")
    print(f"{'#'*60}")

    # Start with full simplex
    A = np.empty((0, n))
    d = np.empty(0)

    spreads = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

    for spread in spreads:
        # Constraint: each b[i] in [(1-spread)/n, (1+spread)/n]
        # But we need to be careful the constraints are feasible
        center = 1.0 / n
        half_width = spread / (2 * n)
        lower = max(0, center - half_width)
        upper = min(1, center + half_width)

        A_upper = np.eye(n)
        d_upper = upper * np.ones(n)
        A_lower = -np.eye(n)
        d_lower = -lower * np.ones(n)

        A = np.vstack([A_upper, A_lower])
        d_vec = np.concatenate([d_upper, d_lower])

        polytope = BeliefPolytope(n=n, A=A, d=d_vec)
        vol = compute_volume(polytope)

        # Also compute product of spreads for comparison
        total_spread = 0.0
        spread_product = 1.0
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            try:
                u = polytope.maximize_linear(e_i)
                l = -polytope.maximize_linear(-e_i)
                s = u - l
                total_spread += s
                spread_product *= max(s, 1e-10)
            except RuntimeError:
                pass

        print(f"\nSpread={spread:.1f}: bounds=[{lower:.4f}, {upper:.4f}]")
        print(f"  Volume: {vol:.6f} ({vol*100:.2f}%)")
        print(f"  Sum of spreads: {total_spread:.6f}")
        print(f"  Product of spreads: {spread_product:.10f}")


def test_asymmetric_constraints(n: int):
    """Test with asymmetric constraints - one coordinate more constrained."""
    print(f"\n{'#'*60}")
    print(f"# Asymmetric constraints test (n={n})")
    print(f"{'#'*60}")

    # Constrain b[0] to be in [0.1, 0.3], others free
    A = np.zeros((2, n))
    A[0, 0] = 1.0   # b[0] <= 0.3
    A[1, 0] = -1.0  # -b[0] <= -0.1, i.e., b[0] >= 0.1
    d = np.array([0.3, -0.1])

    polytope = BeliefPolytope(n=n, A=A, d=d)
    print_polytope_info(f"Asymmetric: b[0] in [0.1, 0.3] (n={n})", polytope)


def main():
    print("BeliefPolytope Volume Computation Test")
    print("="*60)

    # Test 1: Full simplex in various dimensions
    print("\n\n" + "#"*60)
    print("# TEST 1: Full simplex (no constraints)")
    print("#"*60)
    for n in [2, 3, 4]:
        test_full_simplex(n)

    # Test 2: Uniform prior (point)
    print("\n\n" + "#"*60)
    print("# TEST 2: Uniform prior (point constraint)")
    print("#"*60)
    for n in [2, 3, 4]:
        test_uniform_prior(n)

    # Test 3: Half simplex
    print("\n\n" + "#"*60)
    print("# TEST 3: Half simplex (b[0] <= 0.5)")
    print("#"*60)
    for n in [2, 3, 4]:
        test_half_simplex(n)

    # Test 4: Box constraints
    print("\n\n" + "#"*60)
    print("# TEST 4: Box constraints")
    print("#"*60)
    test_box_constraint(3, 0.1, 0.5)
    test_box_constraint(3, 0.2, 0.4)

    # Test 5: Shrinking polytope
    test_shrinking_polytope(3)
    test_shrinking_polytope(4)

    # Test 6: Asymmetric constraints
    test_asymmetric_constraints(3)
    test_asymmetric_constraints(4)


if __name__ == "__main__":
    main()
