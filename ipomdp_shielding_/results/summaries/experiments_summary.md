# Experiments Summary

This document describes the experimental methodology, case study adaptations, and data-collection conditions for the IPOMDP shielding evaluation. It is intended to support synthesis of an evaluation section for a research paper.

---

## 1. Problem Setting and IPOMDP Framework

All experiments evaluate runtime safety shields for agents operating in **Interval Partially Observable Markov Decision Processes (IPOMDPs)**. An IPOMDP extends a standard POMDP by replacing the point observation model P(o | s) with per-state intervals [P_lower(o | s), P_upper(o | s)], reflecting bounded uncertainty in the perception system. The safety predicate is defined over a subset of states; the agent's goal is to complete tasks while avoiding a designated set of unsafe (FAIL) states.

The IPOMDP framework is motivated by real-world perception systems — such as trained neural networks — where exact likelihoods cannot be known but can be bounded via statistical confidence intervals (e.g., Clopper-Pearson or Goodman) applied to held-out confusion data. The interval structure captures both aleatoric noise in the environment and epistemic uncertainty from finite training data.

---

## 2. Case Studies

Four case studies were used, spanning a range of state-space sizes, observation ambiguity levels, and domain types. Two are adapted from prior work on POMDP shielding (Carr et al., arXiv:2204.00755); two are novel contributions.

### 2.1 TaxiNet (16 states, 16 observations)

**Domain.** TaxiNet models autonomous aircraft taxiing on a runway. The state is a discrete 2D position (cross-track error CTE ∈ {−2,−1,0,1,2}, heading error HE ∈ {−1,0,1}), yielding a 5×3 grid of 15 live states plus a FAIL absorbing state (16 total). Actions are steering offsets ∈ {−1, 0, +1}. The safety predicate is `|CTE| ≤ 2 AND |HE| ≤ 1` (staying on the safe portion of the runway).

**Adaptation to IPOMDP.** Dynamics are built from a probabilistic model with stochastic perturbation (error = 0.1). The perception model is constructed from **real CTE and HE sensor data** collected during autonomous taxiing trials. Independent IMDPs are estimated for each of the two state dimensions using empirical observation frequencies and Clopper-Pearson confidence intervals; these are combined via a product IMDP to form the full observation interval model. Crucially, the 16 observations are one-to-one with states in *label* but not in distribution: safe and unsafe states share overlapping observation support under sensor noise, making single-step posteriors highly uninformative and belief history essential for effective shielding.

**Key parameters.** 16 states, 16 observations, 3 actions, stochastic dynamics (error = 0.1).

---

### 2.2 CartPole — Standard and Low-Accuracy Variants (82 states, 82 observations)

**Domain.** CartPole adapts the OpenAI Gymnasium CartPole-v1 environment. The 4D continuous state (cart position x, cart velocity ẋ, pole angle θ, pole angular velocity θ̇) is discretized into 3 bins per dimension, yielding 3⁴ = 81 live states plus FAIL (82 total). Actions are {left, right}. The safety predicate is the CartPole-v1 termination condition: `|x| ≤ 2.4 m AND |θ| ≤ 0.209 rad`.

**Adaptation to IPOMDP.** Dynamics are built empirically from 200 gymnasium rollouts, producing an MDP over the discretized state space. The perception model reflects **vision-based CNN inference**: a neural network (trained on synthetic images) classifies the continuous state into discrete bin tuples. For each dimension independently, per-class confusion matrices are computed on held-out data, and Clopper-Pearson intervals are applied to each entry; the four factored IMDPs are then combined into a product IMDP. With 3 bins per dimension, there are 82 distinct observation labels, making observations nearly bijective with states.

**Low-accuracy variant.** To create a harder perception problem, the CNN is retrained with reduced data (175 episodes, 10 epochs) to match the mean interval midpoint P_mid ≈ 0.373, close to TaxiNet's P_mid ≈ 0.354. This variant uses the same 82-state, 82-observation structure but with wider observation intervals reflecting greater perceptual uncertainty. Artifacts are stored separately in `artifacts_lowacc/`.

**Key parameters.** 82 states, 82 observations, 2 actions; standard variant: P_mid ≈ 0.85; low-accuracy variant: P_mid ≈ 0.373.

---

### 2.3 Obstacle (50 states, 3 observations)

**Domain.** The Obstacle benchmark (adapted from Carr et al.) places an agent on a 7×7 grid with 5 static obstacles. Valid (non-obstacle) cells total 49, plus FAIL, giving 50 states. Actions are {north, south, east, west}. Dynamics include **slippage**: with probability 0.1 the agent takes an extra step in the chosen direction (clamped at boundaries). Initial position is drawn uniformly from 4 designated start cells. The safety predicate is `NOT at an obstacle cell`.

**Adaptation to IPOMDP.** The original PRISM model uses a deterministic observation function (hascrash, amdone) yielding only 3 distinct observations. This extreme aliasing (50 states → 3 observations) is preserved as a stress test for belief-based methods. To form an IPOMDP, the deterministic observation function is perturbed with `obs_noise = 0.1`, introducing small probability mass on adjacent observations and producing non-trivial P_lower / P_upper intervals. The 3-observation structure means that all memoryless shielding methods are severely aliased; belief history over the 50-dimensional distribution is required to distinguish safe from unsafe positions.

**Key parameters.** 50 states, 3 observations, 4 actions, slippage = 0.1, obs_noise = 0.1.

---

### 2.4 Refuel v2 (344 states, 29 observations)

**Domain.** The Refuel benchmark (adapted from Carr et al.) places an agent on a 7×7 grid with a fuel resource. State is (x, y, fuel) with fuel ∈ {0,…,6}, giving 7×7×7 = 343 live states plus FAIL (344 total). Actions are {north, south, east, west, refuel} (refuel only available at stations, when fuel < max). Dynamics include slippage = 0.3. The safety predicate is `NOT at obstacle AND (fuel > 0 OR at refuel station)`.

**Motivation for redesign (v1 → v2).** The original PRISM observation model included observation bits that directly encoded the safety predicates: bit 5 (`hascrash`) and bit 7 (`fuel > 0`). In preliminary experiments, the RL agent achieved 100% safety **without any shield**, because it could read the safety-critical signals from the observation alone. This defeated the purpose of shielding evaluation. A redesigned observation model (v2) was developed to make the benchmark genuinely challenging.

**Adaptation to IPOMDP (v2).** The observation tuple is reduced from 10 to 8 elements by removing `hascrash` and `fuel > 0`. The remaining bits encode boundary conditions, station availability, and a coarse **fuel meter** (0–3 levels covering fuel ∈ {0,1,2}, {3}, {4}, {5,6} respectively), which obscures exact fuel level. The obstacle cell now shares its observation with 69 regular states, making obstacle detection impossible from a single step. Observation noise is raised from obs_noise = 0.05 to 0.3 using a **distance-scaled Gaussian kernel** on normalized Hamming distance between observation bit-vectors (α = 5), so that structurally similar observations receive larger cross-contamination. This yields 29 unique observations, approximately 12 states per observation. With v2, the unshielded RL agent fails in 10–15% of episodes (vs. 0% in v1), confirming that the safety predicate is genuinely hidden from observation.

**Key parameters.** 344 states, 29 observations, 5 actions, slippage = 0.3, obs_noise = 0.3 (distance-scaled).

---

## 3. Shield Implementations

Four shield strategies are evaluated, all using the **pp_shield** (per-state safe action set) derived from a policy synthesis step on the IPOMDP.

### 3.1 No Shield (Baseline)
Passes all actions through unconditionally. Used to measure unshielded RL performance.

### 3.2 Observation Shield (Memoryless Posterior)
Computes a posterior P(s | o) at each step using the midpoint observation model O_mid = (P_lower + P_upper)/2 and a uniform prior, without any memory of prior observations. Action a is permitted if ∑_{s: a ∈ pp_shield(s)} P(s | o) ≥ t. This is the weakest belief model — equivalent to a single Bayesian update from a flat prior at every step.

### 3.3 Single-Belief Shield (POMDP Belief Filter)
Maintains a standard Bayesian belief over states, propagated via the midpoint transition and observation models. At each step, action a is permitted if ∑_s belief(s) · 1[a ∈ pp_shield(s)] ≥ t. This captures full history under the midpoint model but ignores interval uncertainty.

### 3.4 Envelope Shield (Interval Belief Polytope)
Maintains a **belief polytope** — a convex set of distributions consistent with all IPOMDP observation intervals — propagated via Linear Feasibility Programs (LFPs) at each step. Action a is permitted if min_{b in Polytope} ∑_s b(s) · 1[a ∈ pp_shield(s)] ≥ t. This is the most pessimistic (worst-case) shield, providing formal safety guarantees under interval uncertainty.

Feasibility is limited by LP solve time: approximately 0.095 s/step for TaxiNet, 0.679 s/step for Obstacle, 1.876 s/step for CartPole (marginal), and approximately 144 s/step for Refuel (infeasible for RL sweep timescales). The Envelope shield is excluded from CartPole threshold sweeps and all Refuel experiments.

### 3.5 Carr Shield (Support-Based, Midpoint POMDP)
Implements the shield from Carr et al. (arXiv:2204.00755). The IPOMDP is converted to its midpoint POMDP; an offline BFS constructs a support-MDP over reachable belief supports (minimal sets of states consistent with the observation history). The winning region of the support-MDP is computed, and only actions that keep the current support in the winning region are permitted. This method is memoryless in belief weights but tracks the exact support. It is not threshold-parameterized. Feasibility depends on the number of reachable supports: infeasible for Refuel v2 (ratio 344 states / 29 obs ≈ 11.9 exceeds memory budget), degenerate for TaxiNet (0/7 winning supports — no safe strategy exists in the midpoint POMDP), and feasible for CartPole (4 supports, 3 winning) and Obstacle (47,531 supports, 12,167 winning). A feasibility heuristic skips Carr when n_states > 200 and n_states / n_obs > 5.

---

## 4. Perception Models and Adversarial Evaluation

Each experiment is run under two **perception regimes**:

- **Uniform**: The true observation at each step is drawn uniformly at random from all observations. This is a worst-case perception model that provides no signal about the current state. It tests shields purely in terms of belief propagation under maximum epistemic uncertainty.
- **Adversarial (optimized realization)**: A fixed observation function — drawn from the interval, i.e., satisfying P_lower(o | s) ≤ P(o | s) ≤ P_upper(o | s) for all s, o — is selected to maximize the failure and stuck rates against a target shield. The adversarial realization is found by a CEM-style optimization: 10 candidate realizations are sampled and evaluated over 5 trial rollouts per candidate; the top candidates are combined and the process is iterated for 10 rounds. Adversarial realizations are trained once at threshold 0.8 and reused across all threshold sweep values (retraining per threshold would be computationally prohibitive).

---

## 5. RL Agent and Action Selection

All experiments use an **RL-trained neural action selector** as the underlying policy, supplemented by the shield. The RL agent is a small MLP trained via DQN-style reinforcement learning within the IPOMDP environment. Training parameters:

| Case study | RL episodes | Episode length |
|---|---|---|
| TaxiNet | 500 | 20 steps |
| CartPole (std) | 300 | 15 steps |
| CartPole (lowacc) | 500 | 20 steps |
| Obstacle | 500 | 25 steps |
| Refuel v2 | 500 | 30 steps |

The RL agent's preferred action is passed through the shield; if blocked, the shield falls back to a randomly selected permitted action. RL agents and adversarial realizations are cached and reused across all sweep conditions to ensure that observed differences are attributable to shield behavior rather than agent variability.

Three selector strategies are also included in the full experiment grid for diagnostic purposes: **random** (uniform random from all actions), **best** (heuristic belief-informed selector), and **rl** (the trained agent above). Primary reported results use the **rl selector**.

---

## 6. Evaluation Protocol and Threshold Sweep

### 6.1 Metrics

Each trial is classified into one of three mutually exclusive outcomes:

- **Fail**: the agent reaches the FAIL absorbing state (safety violation).
- **Stuck**: the agent's shield blocks all available actions (liveness violation).
- **Safe**: the trial completes the allotted steps without fail or stuck.

Rates are computed as proportions over all trials: fail_rate + stuck_rate + safe_rate = 1. All reported rates are accompanied by **95% Wilson score confidence intervals**.

### 6.2 Threshold Sweep

Shield behavior is controlled by a threshold parameter t ∈ {0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95} (not applicable to Carr). Higher t imposes stricter belief requirements before an action is permitted, trading liveness (fewer stuck episodes) for safety (fewer fail episodes). The threshold sweep traces the **Pareto frontier** of fail rate vs. stuck rate for each shield.

### 6.3 Sample Sizes and Trial Lengths

The primary evaluation (v5, reported in the paper) uses **200 trials per condition** across all case studies. Trial lengths are set to be comfortably above the minimum steps needed to reach a fail or stuck outcome in typical trajectories:

| Case study | Trials | Trial length |
|---|---|---|
| TaxiNet | 200 | 20 steps |
| CartPole (lowacc) | 200 | 15 steps |
| Obstacle | 200 | 25 steps |
| Refuel v2 | 200 | 30 steps |

Each cell in the experimental grid (perception × shield × threshold) is therefore evaluated over 200 independent rollouts. The full sweep grid per case study comprises 2 perception regimes × 9 thresholds × up to 4 shields = up to 72 conditions per case study, with Carr (unthresholded) adding 2 additional cells.

### 6.4 Best Operating Point

For reporting, the best threshold per shield per case study is selected as the threshold minimizing fail rate, with stuck rate as a tiebreaker (minimize stuck given minimum fail). This reflects a primary safety objective with liveness as a secondary concern.

---

## 7. Statistical Reporting

All rates are computed as sample proportions from N = 200 independent Monte Carlo trials. Confidence intervals use the **Wilson score method**, which provides reliable coverage for both near-zero and near-one proportions. The intervals are stored in result JSON files alongside point estimates and are available for inclusion in figures and tables.

---

## 8. Feasibility Constraints and Exclusions

| Shield | TaxiNet | Obstacle | CartPole (lowacc) | Refuel v2 |
|---|---|---|---|---|
| Observation | ✓ | ✓ | ✓ | ✓ |
| Single-Belief | ✓ | ✓ | ✓ | ✓ |
| Envelope | ✓ | ✓ | ✗ (too slow) | ✗ (too slow) |
| Carr | ✗ (degenerate) | ✓ | ✓ | ✗ (infeasible BFS) |

- **Envelope / CartPole**: LP solve time ≈ 1.876 s/step × 200 trials × 15 steps × 9 thresholds × 2 perceptions ≈ 27 hours; excluded.
- **Envelope / Refuel**: LP solve time ≈ 144 s/step; excluded.
- **Carr / TaxiNet**: The midpoint POMDP has 0 winning supports (no safe strategy exists for any reachable belief support); Carr blocks all actions from the first step, making it degenerate.
- **Carr / Refuel**: BFS over belief supports exceeds memory budget; excluded.

---

## 9. Summary of Cross-Case Findings

The following high-level findings are supported by the experimental data:

1. **Observation informativeness is the primary driver of shield performance.** Cases where observations are nearly bijective with states (CartPole, 82 obs / 82 states) allow memoryless methods (Observation, Carr) to match history-based methods. Cases with severe aliasing (Obstacle, 3 obs / 50 states; Refuel, 29 obs / 344 states) require history for effective liveness.

2. **Envelope shield dominates Single-Belief when feasible.** On TaxiNet and Obstacle, the Envelope achieves lower fail rate at every threshold, at comparable or modestly higher stuck cost. The advantage is largest under adversarial perception.

3. **Carr is degenerate when the midpoint POMDP has no winning strategy** (TaxiNet) and excessively conservative when observations are highly aliased (Obstacle: 98% stuck). It is competitive only when observations near-uniquely identify states (CartPole lowacc: 2% fail / 0% stuck).

4. **The Observation shield's value is case-specific.** It matches Single-Belief for CartPole (near-bijective observations) and provides the *only* zero-stuck operating point for Refuel v2 at low thresholds. It degrades badly for TaxiNet and collapses to Carr-equivalent behavior for Obstacle.

5. **CartPole lowacc is the only case with zero liveness cost.** All threshold-based shields achieve ≤ 2% fail / 0% stuck at their best threshold, because the near-bijective observation structure leaves no ambiguity that would cause stuck.

6. **The Refuel v2 redesign was necessary.** The original benchmark (v1) was trivially safe for an unshielded RL agent (0% fail), because the observation directly encoded the safety-critical state bits. After removing these bits and increasing observation noise, the unshielded agent fails in 10–15% of episodes, making the shield's contribution evaluable.
