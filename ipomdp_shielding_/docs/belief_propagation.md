# Belief Propagation

This document covers the `Propagators/` package -- the algorithms that track what the agent can infer about its state from observation history under interval uncertainty.

## The Problem

In an IPOMDP, the agent receives observations whose probabilities are only known up to intervals: `Z(o|s) in [P_lower(o|s), P_upper(o|s)]`. After a sequence of actions and observations, the set of beliefs consistent with the history is a **convex set** of probability distributions over states. Propagators compute or approximate this set.

## Base Interface

All propagators extend `IPOMDP_Belief` (`belief_base.py`):

```python
class IPOMDP_Belief:
    def __init__(self, ipomdp): ...
    def restart(self): ...
    def propagate(action, obs) -> bool: ...           # note: (action, obs) order
    def minimum_allowed_probability(allowed) -> float: ...
    def maximum_disallowed_probability(disallowed) -> float: ...
```

- `propagate(action, obs)` updates the internal belief representation and returns `True` on success, `False` on numerical failure.
- `minimum_allowed_probability(allowed)` returns the worst-case (minimum) probability that the true state is in the `allowed` index set, over all beliefs in the current set.
- `maximum_disallowed_probability(disallowed)` returns the worst-case (maximum) probability that the true state is in the `disallowed` index set.

**Note on argument format**: Both probability queries take a `List[int]` of state *indices* (not state labels). The `RuntimeImpShield` builds these index lists via its `inv_shield` maps.

## Propagator Catalog

### LFPPropagator (Over-Approximation)

**File**: `lfp_propagator.py`
**Type**: Sound over-approximation via template-based abstract interpretation
**Representation**: `BeliefPolytope` -- `{b | Ab <= d, b >= 0, sum(b) = 1}`

#### How It Works

Each propagation step:

1. **Build constraint polytope** over the 4n-dimensional space `z = [b, y, w, x]`:
   - `b` (n-dim): prior belief, constrained to current polytope
   - `y` (n-dim): predicted belief `y = T_a^T b` (equality constraints)
   - `w` (n-dim): observation likelihoods, box-constrained to `[P_lower, P_upper]`
   - `x` (n-dim): unnormalized posterior `x = w * y` (McCormick relaxation)

2. **For each template direction** `v_k`:
   - Solve two LFPs: `min/max v_k^T x / sum(x)` using Charnes-Cooper transform
   - This bounds `v_k^T b'` where `b'` is the normalized posterior

3. **Build new polytope** from the computed bounds.

#### Key Dependencies

- `IPOMDP.feasible_unnormalized_posterior_polytope()` -- builds the 4n-dim constraint system
- `IPOMDP._build_T_matrix(action)` -- transition matrix `T[i,j] = P(s_j|s_i,a)`
- `IPOMDP._compute_y_bounds()` -- LP-based bounds on predicted belief (needed for McCormick)
- `solve_lfp_charnes_cooper()` -- Charnes-Cooper transform solver
- `SciPyHiGHSSolver` -- LP backend using scipy/HiGHS

#### Templates

Templates are defined by `Template(V, names)` where `V` is a `(K, n)` matrix. Each row `v_k` defines a linear function `phi_k(b) = v_k^T b` to bound.

`TemplateFactory` provides:
- `canonical(n)` -- identity matrix, bounds each `b[i]` individually
- `safe_set_indicators(n, safe_sets)` -- indicator vectors for action safety queries
- `pca_templates(samples, k)` -- data-driven templates from belief samples
- `hybrid(templates)` -- combine multiple template sets

#### Probability Queries

Delegates to `BeliefPolytope`:
- `minimum_allowed_prob(allowed)` solves `min sum(b[allowed])` via LP
- `maximum_allowed_prob(disallowed)` solves `max sum(b[disallowed])` via LP

**Cost**: One LP per query.

---

### ForwardSampledBelief (Under-Approximation)

**File**: `forward_sampled_belief.py`
**Type**: Sound under-approximation via concrete belief point samples
**Representation**: `np.ndarray` of shape `(N, n)` -- N concrete belief points

#### How It Works

Each propagation step:

1. **Predict**: `predicted = points @ T_a` (vectorized matrix multiply)
2. **Sample** K observation likelihood vectors `w` where `w_j in [L_eff(s_j,o), U_eff(s_j,o)]`
3. **Compute posteriors**: `x = w * predicted`, normalize `b' = x / sum(x)`
4. **Prune** N*K candidates back to budget N

#### Likelihood Sampling Strategies

`LikelihoodSamplingStrategy` enum:

| Strategy | Description |
|---|---|
| `EXTREME_POINTS` | Each `w_j in {L_eff, U_eff}`. Enumerates all `2^n` corners if small, else random binary masks. |
| `UNIFORM_RANDOM` | `w_j ~ Uniform(L_eff, U_eff)` per coordinate. |
| `HYBRID` (default) | 2n coordinate-extremal vectors + all-low + all-high + random fill. |

#### Pruning Strategies

`PruningStrategy` enum:

| Strategy | Description |
|---|---|
| `COORDINATE_EXTREMAL` (default) | Keep min/max per coordinate, random-fill remainder. |
| `FARTHEST_POINT` | Greedy farthest-point sampling for maximal coverage. |
| `RANDOM` | Simple random subsample. |

#### Probability Queries

Direct computation over the point set -- O(N):
- `minimum_allowed_probability(allowed)`: `min over points of sum(b[allowed])`
- `maximum_disallowed_probability(disallowed)`: `max over points of sum(b[disallowed])`

**No LP required.**

#### Construction

```python
from ipomdp_shielding.Propagators import (
    ForwardSampledBelief, LikelihoodSamplingStrategy, PruningStrategy
)

sampler = ForwardSampledBelief(
    ipomdp=ipomdp,
    budget=200,            # Number of belief points to maintain
    K_samples=20,          # Likelihood samples per step
    likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    pruning_strategy=PruningStrategy.COORDINATE_EXTREMAL,
    rng=np.random.default_rng(42),  # For reproducibility
)
```

---

### ExactIHMMBelief

**File**: `exact_hmm.py`
**Type**: Exact belief tracking for standard (non-interval) HMMs
**Use**: Ground-truth comparison when observation probabilities are point-valued.

---

### MinMaxIHMMBelief

**File**: `minmax_hmm.py`
**Type**: Min/max belief bounds under interval uncertainty
**Use**: Conservative bounding without full polytope tracking.

---

### IPOMDP_ApproxBelief

**File**: `approx_belief.py`
**Type**: Lightweight approximate belief
**Use**: Fast but less precise.

## BeliefPolytope

**File**: `belief_polytope.py`

Represents a convex subset of the probability simplex:

```
{ b in R^n | A @ b <= d, b >= 0, sum(b) = 1 }
```

Key methods:
- `uniform_prior(n)` -- creates the singleton `{1/n, ..., 1/n}` polytope
- `maximize_linear(c)` -- solve `max c^T b` over the polytope (LP)
- `minimum_allowed_prob(allowed)` / `maximum_allowed_prob(allowed)` -- probability queries
- `as_lp_constraints()` -- export as `(A_ub, b_ub, A_eq, b_eq, lb, ub)` for LP solvers
- `volume()` -- compute volume via vertex enumeration and ConvexHull

Volume computation pipeline:
1. `_enumerate_vertices()` -- find polytope vertices (with fallbacks for degenerate cases)
2. `_find_affine_basis()` -- SVD to find intrinsic dimension
3. `ConvexHull` in reduced dimension
4. Normalize by simplex volume `sqrt(k+1)/k!`

## Utility Functions

**File**: `utils.py`

### `tightened_likelihood_bounds(ipomdp, s, o_obs) -> (L_eff, U_eff)`

Tightens observation probability bounds using the row-sum constraint `sum_o Z(o|s) = 1`:
- `L_eff = max(L, 1 - sum(U_others))`
- `U_eff = min(U, 1 - sum(L_others))`

Used by both `LFPPropagator` (via `feasible_unnormalized_posterior_polytope`) and `ForwardSampledBelief` (directly).

### `transition_update(ipomdp, b, a) -> b_pred`

Exact belief prediction: `b_pred(s') = sum_s T(s'|s,a) * b(s)`. Dictionary-based (not vectorized). Used by `ExactIHMMBelief`.

## Choosing a Propagator

| Need | Propagator | Why |
|---|---|---|
| Sound runtime shielding | `LFPPropagator` | Over-approximation ensures no unsafe action passes |
| Measuring approximation quality | `ForwardSampledBelief` | Under-approximation gives lower bound on true set |
| Ground-truth comparison | `ExactIHMMBelief` | Exact (when observations are point-valued) |
| Fast prototyping | `ForwardSampledBelief` | No LP, O(N) queries |
| Drop-in for `RuntimeImpShield` | Any `IPOMDP_Belief` subclass | All implement the same interface |
