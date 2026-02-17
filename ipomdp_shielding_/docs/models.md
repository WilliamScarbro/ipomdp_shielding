# Models

This document describes the `Models/` package -- the core mathematical structures used throughout the library.

## Model Hierarchy

```
MDP  ──────>  IMDP
 │              │
 │   add interval transitions
 │              │
 v              v
POMDP ─────>  IPOMDP
       add interval observations
```

All four models share the core concept of states, actions, and transitions. They differ in what is known precisely vs. known only up to intervals, and whether observations are present.

## IPOMDP (Primary Model)

**File**: `Models/ipomdp.py`
**Class**: `IPOMDP` (dataclass)

The **Interval Partially Observable Markov Decision Process** is the main model. Transitions are exact; observation probabilities are interval-bounded.

### Attributes

```python
@dataclass
class IPOMDP:
    states: List[State]                                    # e.g., ['s0', 's1', 's2']
    observations: List[Observation]                        # e.g., ['o0', 'o1']
    actions: List[Action]                                  # e.g., ['left', 'right']
    T: Dict[Tuple[State, Action], Dict[State, float]]     # T[(s,a)][s'] = P(s'|s,a)
    P_lower: Dict[State, Dict[Observation, float]]        # P_lower[s][o] = lower bound
    P_upper: Dict[State, Dict[Observation, float]]        # P_upper[s][o] = upper bound
```

**Constraint**: For each state `s`, `sum_o P_lower[s][o] <= 1 <= sum_o P_upper[s][o]`.

### Key Methods

#### `_build_T_matrix(action) -> np.ndarray`

Returns the `n x n` transition matrix for a given action:
```
T[i, j] = P(s_j | s_i, action)
```
Used by both `LFPPropagator` and `ForwardSampledBelief` for vectorized belief prediction.

#### `_obs_bounds_vectors(obs) -> (w_lo, w_hi)`

Returns vectors of observation probability bounds:
```
w_lo[j] = P_lower[s_j][obs]
w_hi[j] = P_upper[s_j][obs]
```

#### `_compute_y_bounds(prior, action, solver) -> (y_lo, y_hi)`

Computes LP-based bounds on the predicted belief `y = T_a^T @ b` under the prior polytope. Solves 2n LPs (min and max for each state component). Used internally by `feasible_unnormalized_posterior_polytope` for McCormick relaxation.

#### `feasible_unnormalized_posterior_polytope(prior, action, obs, ...) -> (A_ub, b_ub, A_eq, b_eq, lb, ub, slices)`

The main constraint-generation method for LFP propagation. Builds the feasible region in a 4n-dimensional space `z = [b, y, w, x]`:

| Variable | Dim | Meaning | Constraints |
|---|---|---|---|
| `b` | n | Prior belief | In prior polytope |
| `y` | n | Predicted belief | `y = T_a^T @ b` (equality) |
| `w` | n | Observation likelihoods | `w_j in [P_lower[s_j][o], P_upper[s_j][o]]` (box) |
| `x` | n | Unnormalized posterior | `x_j = w_j * y_j` (McCormick relaxation) |

The McCormick envelope linearizes the bilinear product `x = w * y` using 4 inequalities per component, based on the bounds `[y_lo, y_hi]` and `[w_lo, w_hi]`.

Returns the LP constraint system that `LFPPropagator` uses to solve LFPs.

#### `evolve(state, action) -> State`

Samples a next state from the transition distribution. Used in Monte Carlo simulation.

#### `to_pomdp() -> POMDP`

Converts to a standard POMDP by taking the midpoint of observation bounds: `Z(o|s) = (P_lower + P_upper) / 2`. Useful for ground-truth comparison with exact belief tracking.

#### `_state_index() -> Dict[State, int]`

Returns mapping from state labels to integer indices. Used internally for matrix construction.

## POMDP

**File**: `Models/pomdp.py`
**Class**: `POMDP` (dataclass)

Standard Partially Observable MDP with exact (point-valued) observation probabilities.

### Attributes

```python
@dataclass
class POMDP:
    states: List[State]
    observations: List[Observation]
    actions: List[Action]
    T: Dict[Tuple[State, Action], Dict[State, float]]
    P: Dict[State, Dict[Observation, float]]              # P[s][o] = exact P(o|s)
```

### Key Functions

- `expected_perception_from_data(data) -> P` -- estimate observation probabilities from data
- `product_model(pomdp1, pomdp2)` -- construct product POMDP

## MDP

**File**: `Models/mdp.py`
**Class**: `MDP` (dataclass)

Fully observable Markov Decision Process. States, actions, transitions -- no observations.

## IMDP

**File**: `Models/imdp.py`
**Class**: `IMDP` (dataclass)

Interval MDP -- transitions are interval-bounded. No observations.

### Key Functions

- `imdp_from_mdp(mdp, intervals)` -- construct IMDP from MDP with transition intervals
- `product_imdp(imdp1, imdp2)` -- construct product IMDP
- `collapse_imdp(imdp)` -- collapse IMDP to point-valued transitions
- `imdp_interval_width_dist(imdp)` -- distribution of interval widths

## Confidence Intervals

**Directory**: `Models/Confidence/`

Utilities for constructing observation probability intervals from data -- connecting empirical measurements to the IPOMDP model. Used to build `P_lower` and `P_upper` from observation frequency data with statistical guarantees.

## Data Layout Conventions

### State Indexing

States are stored as a list. The integer index of a state is its position in `ipomdp.states`. All internal matrix operations use these indices.

```python
states = ['s0', 's1', 's2']
# s0 -> index 0, s1 -> index 1, s2 -> index 2
```

The `_state_index()` method returns this mapping as a dictionary, but it is also implicitly used by `enumerate(ipomdp.states)`.

### Transition Format

Transitions are stored as a nested dictionary:
```python
T[(state, action)] = {next_state: probability, ...}
```

Missing entries default to 0 probability. Rows should sum to 1.

### Observation Bounds Format

Observation bounds are stored per-state:
```python
P_lower[state] = {observation: lower_bound, ...}
P_upper[state] = {observation: upper_bound, ...}
```

Missing entries default to 0. For each state, `sum(P_lower[s].values()) <= 1 <= sum(P_upper[s].values())`.
