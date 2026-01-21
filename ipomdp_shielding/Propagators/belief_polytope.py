"""Belief polytope representation and volume computation.

This module provides the BeliefPolytope class for representing convex polytopes
on the probability simplex, along with methods for computing polytope volume.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection


@dataclass
class BeliefPolytope:
    """Represents a belief polytope: { b | A b <= d, b >= 0, 1^T b = 1 }.

    A convex subset of the probability simplex defined by linear inequality
    constraints.

    Attributes
    ----------
    n : int
        Dimension of the belief space (number of states)
    A : np.ndarray
        Constraint matrix (m x n) for inequality constraints A @ b <= d
    d : np.ndarray
        Right-hand side vector (m,) for inequality constraints
    Aeq : Optional[np.ndarray]
        Optional equality constraint matrix (beyond sum-to-one)
    beq : Optional[np.ndarray]
        Optional equality constraint RHS
    """
    n: int
    A: np.ndarray
    d: np.ndarray
    Aeq: Optional[np.ndarray] = None
    beq: Optional[np.ndarray] = None

    def with_added_inequalities(self, A_new: np.ndarray, d_new: np.ndarray) -> BeliefPolytope:
        """Return a new polytope with additional inequality constraints."""
        A2 = np.vstack([self.A, A_new]) if self.A.size else A_new.copy()
        d2 = np.concatenate([self.d, d_new]) if self.d.size else d_new.copy()
        return BeliefPolytope(self.n, A2, d2, self.Aeq, self.beq)

    def as_lp_constraints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (A_ub, b_ub, A_eq, b_eq, lb, ub) suitable for LP."""
        A_ub, b_ub = self.A, self.d

        ones = np.ones((1, self.n))
        if self.Aeq is None:
            A_eq = ones
            b_eq = np.array([1.0])
        else:
            A_eq = np.vstack([self.Aeq, ones])
            b_eq = np.concatenate([self.beq, np.array([1.0])])

        lb = np.zeros(self.n)
        ub = np.ones(self.n)
        return A_ub, b_ub, A_eq, b_eq, lb, ub

    @staticmethod
    def uniform_prior(n: int) -> BeliefPolytope:
        """Returns a BeliefPolytope representing a uniform prior belief."""
        A = np.vstack([np.eye(n), -np.eye(n)])
        d = np.concatenate([(1.0 / n) * np.ones(n), -(1.0 / n) * np.ones(n)])
        return BeliefPolytope(n=n, A=A, d=d, Aeq=None, beq=None)

    def maximize_linear(self, c: np.ndarray) -> float:
        """Solve: max_b c^T b s.t. b in this BeliefPolytope."""
        c = np.asarray(c, dtype=float)
        if c.shape != (self.n,):
            raise ValueError(f"Objective must have shape ({self.n},), got {c.shape}")

        A_ub, b_ub = self.A, self.d

        ones = np.ones((1, self.n))
        if self.Aeq is None:
            A_eq = ones
            b_eq = np.array([1.0])
        else:
            A_eq = np.vstack([self.Aeq, ones])
            b_eq = np.concatenate([self.beq, np.array([1.0])])

        bounds = [(0.0, 1.0) for _ in range(self.n)]

        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")

        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")

        return float(-res.fun)

    def maximum_allowed_prob(self, allowed: List[int]) -> float:
        """Compute max_b sum_{i in allowed} b_i."""
        c = np.zeros(self.n, dtype=float)
        for i in allowed:
            c[i] = 1.0
        return self.maximize_linear(c)

    def minimum_allowed_prob(self, allowed: List[int]) -> float:
        """Compute min_b sum_{i in allowed} b_i."""
        c = np.zeros(self.n, dtype=float)
        for i in allowed:
            if i < 0 or i >= self.n:
                raise ValueError(f"State index out of range: {i}")
            c[i] = 1.0
        return -self.maximize_linear(-c)

    def volume(self) -> float:
        """Compute the volume of this polytope as a fraction of the simplex.

        Returns
        -------
        float
            Volume as a fraction of the full probability simplex (0 to 1).
            Returns 0.0 if volume computation fails.
        """
        return compute_volume(self)


def _simplex_volume(n: int) -> float:
    """Compute the volume of the standard (n-1)-simplex in (n-1) dimensions.

    The standard simplex {x in R^{n-1} | x_i >= 0, sum(x_i) <= 1} has volume 1/(n-1)!
    """
    if n <= 1:
        return 1.0
    return 1.0 / math.factorial(n - 1)


def _project_constraints(polytope: BeliefPolytope) -> Tuple[np.ndarray, np.ndarray]:
    """Project polytope constraints to (n-1) dimensional space.

    Eliminates the last coordinate using b_n = 1 - sum(b_1, ..., b_{n-1}).

    Original constraint: A @ b <= d where b in R^n, sum(b) = 1
    With substitution b_n = 1 - sum(b'), where b' = [b_1, ..., b_{n-1}]:
        A @ [b'; 1-sum(b')] <= d
        A[:, :-1] @ b' + A[:, -1] * (1 - sum(b')) <= d
        A[:, :-1] @ b' + A[:, -1] - A[:, -1] * sum(b') <= d
        (A[:, :-1] - A[:, -1] @ ones^T) @ b' <= d - A[:, -1]

    Returns (A_proj, d_proj) for the projected constraints.
    """
    n = polytope.n
    m = n - 1

    if polytope.A.size == 0:
        return np.empty((0, m)), np.empty(0)

    # A_proj = A[:, :-1] - outer(A[:, -1], ones)
    A_orig = polytope.A
    last_col = A_orig[:, -1].reshape(-1, 1)
    ones_row = np.ones((1, m))
    A_proj = A_orig[:, :-1] - last_col @ ones_row

    # d_proj = d - A[:, -1]
    d_proj = polytope.d - A_orig[:, -1]

    return A_proj, d_proj


def _find_interior_point_projected(polytope: BeliefPolytope) -> Optional[np.ndarray]:
    """Find a strictly interior point in (n-1) dimensional projected space.

    The projected space has variables b' = [b_1, ..., b_{n-1}] with:
    - Projected original constraints
    - b'_i >= 0 for all i
    - sum(b') <= 1  (so that b_n = 1 - sum(b') >= 0)
    """
    n = polytope.n
    m = n - 1  # Projected dimension

    # Get projected constraints
    A_proj, d_proj = _project_constraints(polytope)

    # Variables: [b'_1, ..., b'_{m}, t] where t is the slack for Chebyshev center
    # Maximize t
    c = np.zeros(m + 1)
    c[-1] = -1  # Maximize t (minimize -t)

    # Build inequality constraints with slack
    halfspaces = []
    rhs = []

    # Projected original constraints: A_proj @ b' + t <= d_proj
    if A_proj.size > 0:
        for i in range(A_proj.shape[0]):
            row = np.append(A_proj[i], 1.0)  # Add slack
            halfspaces.append(row)
            rhs.append(d_proj[i])

    # Non-negativity: b'_i >= t, i.e., -b'_i + t <= 0
    for i in range(m):
        row = np.zeros(m + 1)
        row[i] = -1.0
        row[-1] = 1.0
        halfspaces.append(row)
        rhs.append(0.0)

    # b_n >= t: 1 - sum(b') >= t, i.e., sum(b') + t <= 1
    row = np.ones(m + 1)
    halfspaces.append(row)
    rhs.append(1.0)

    A_ub = np.array(halfspaces) if halfspaces else np.empty((0, m + 1))
    b_ub = np.array(rhs) if rhs else np.empty(0)

    # Bounds: b'_i in [0, 1], t >= 0
    bounds = [(0, 1) for _ in range(m)] + [(0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success and res.x[-1] > 1e-9:
        return res.x[:m]
    return None


def _enumerate_vertices(polytope: BeliefPolytope,
                        max_vertices: int = 1000) -> Optional[np.ndarray]:
    """Enumerate vertices of the polytope in projected (n-1) dimensional space.

    Works in the projected space where b_n = 1 - sum(b_1, ..., b_{n-1}).
    Returns vertices in the original n-dimensional space.

    Parameters
    ----------
    polytope : BeliefPolytope
        The polytope to enumerate vertices for
    max_vertices : int
        Maximum number of vertices before warning

    Returns
    -------
    Optional[np.ndarray]
        Array of vertices in n-dimensional space (each row is a vertex),
        or None if enumeration fails
    """
    n = polytope.n
    m = n - 1  # Projected dimension

    if m == 0:
        # 1D case: the only point is [1]
        return np.array([[1.0]])

    # Get projected constraints
    A_proj, d_proj = _project_constraints(polytope)

    # Build halfspace representation in (n-1) dimensions
    # Format: [a_1, ..., a_{m}, -b] for a @ x <= b
    halfspaces = []

    # Projected original constraints
    if A_proj.size > 0:
        for i in range(A_proj.shape[0]):
            halfspaces.append(np.append(A_proj[i], -d_proj[i]))

    # Non-negativity: -b'_i <= 0
    for i in range(m):
        row = np.zeros(m + 1)
        row[i] = -1.0
        halfspaces.append(row)

    # b_n >= 0: 1 - sum(b') >= 0, i.e., sum(b') <= 1
    row = np.zeros(m + 1)
    row[:m] = 1.0
    row[-1] = -1.0
    halfspaces.append(row)

    halfspaces = np.array(halfspaces)

    # Find strictly interior point in projected space
    interior = _find_interior_point_projected(polytope)
    if interior is None:
        return None

    try:
        hs = HalfspaceIntersection(halfspaces, interior)
        vertices_proj = hs.intersections

        if len(vertices_proj) > max_vertices:
            warnings.warn(
                f"Polytope has {len(vertices_proj)} vertices (>{max_vertices}). "
                "Volume computation may be slow or inaccurate.",
                RuntimeWarning
            )

        # Lift vertices back to n dimensions
        vertices = np.zeros((len(vertices_proj), n))
        vertices[:, :-1] = vertices_proj
        vertices[:, -1] = 1.0 - np.sum(vertices_proj, axis=1)

        return vertices
    except Exception:
        return None


def compute_volume(polytope: BeliefPolytope) -> float:
    """Compute polytope volume using vertex enumeration and ConvexHull.

    Projects the polytope onto (n-1) dimensions and computes the volume
    using scipy.spatial.ConvexHull. Returns volume as a fraction of the
    full simplex volume (0 to 1 scale).

    Parameters
    ----------
    polytope : BeliefPolytope
        The belief polytope to compute volume for

    Returns
    -------
    float
        Volume as a fraction of the simplex volume (between 0 and 1).
        Returns 0.0 if volume computation fails.
    """
    n = polytope.n
    if n <= 1:
        return 1.0

    # Special case for n=2: the simplex is a line segment [0,1] in 1D
    # Volume is just the length of the interval
    if n == 2:
        e0 = np.array([1.0, 0.0])
        try:
            upper = polytope.maximize_linear(e0)
            lower = -polytope.maximize_linear(-e0)
            # The simplex in 1D has length 1, so volume fraction = interval length
            return max(0.0, upper - lower)
        except RuntimeError:
            return 0.0

    # Enumerate vertices
    vertices = _enumerate_vertices(polytope)
    if vertices is None or len(vertices) < n:
        warnings.warn(
            "Could not enumerate polytope vertices. Returning 0.",
            RuntimeWarning
        )
        return 0.0

    # Project to (n-1) dimensions by dropping the last coordinate
    # (since sum = 1, last coord is determined by others)
    projected = vertices[:, :-1]

    # Remove duplicate vertices
    projected = np.unique(projected, axis=0)

    if len(projected) < n:
        # Not enough vertices to form a full-dimensional polytope
        return 0.0

    try:
        hull = ConvexHull(projected)
        raw_volume = hull.volume
    except Exception as e:
        warnings.warn(
            f"ConvexHull computation failed: {e}. Returning 0.",
            RuntimeWarning
        )
        return 0.0

    # Normalize by simplex volume
    simplex_vol = _simplex_volume(n)
    return raw_volume / simplex_vol if simplex_vol > 0 else 0.0
