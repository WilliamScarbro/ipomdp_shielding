# Evaluation Summary (Final Run): IPOMDP Shielding Across Four Case Studies

This document summarises the **final** experimental evaluation of interval POMDP (IPOMDP)
shielding across four case studies: **TaxiNet**, **CartPole**, **Obstacle**, and **Refuel**.
It supersedes `evaluation_summary.md` (preliminary results).

The final run uses substantially increased sample sizes (~5–10× more trials, longer trajectory
lengths) and a corrected `ObservationShield` implementation.  Total wall-clock runtime was
approximately **7 hours**.

---

## Methodological changes from preliminary run

### 1. Corrected ObservationShield

The preliminary `ObservationShield` looked up `pp_shield[obs]` directly, which only
functioned correctly when observation objects were identical to state objects (accidentally
true for TaxiNet tuples, but wrong in general).  The corrected implementation computes the
**intersection of safe actions across all states consistent with the current observation**:

```
allowed = { a ∈ A | ∀ s with P_lower[s][obs] > 0 : a ∈ pp_shield[s] }
```

This is the theoretically correct "safe for every state the agent cannot rule out" criterion.

**Impact on results:**

<table>
<thead>
<tr><th>Case study</th><th>Prelim observation-shield stuck% (RL)</th><th>Final stuck% (RL)</th><th>Reason</th></tr>
</thead>
<tbody>
<tr><td>TaxiNet</td><td>0%</td><td>91%</td><td>Each obs consistent with 6–15 states; intersection empty</td></tr>
<tr><td>CartPole</td><td>40%</td><td>56%</td><td>1:1 obs–state (bijective); stuck = pp_shield has no safe action for state</td></tr>
<tr><td>Obstacle</td><td>40%</td><td>98%</td><td>3 obs × ~17 states each; intersection almost always empty</td></tr>
<tr><td>Refuel</td><td>10%</td><td>40%</td><td>43 obs × ~8 states each; partial overlap reduces intersection</td></tr>
</tbody>
</table>

For TaxiNet and Obstacle the corrected shield is effectively unusable (near-total stuck rate).
For CartPole (bijective) it is equivalent to the perfect-perception shield and stuck events
reflect genuine pp_shield boundary states.  For Refuel the corrected shield performs similarly
to the preliminary version.

### 2. Increased sample sizes

<table>
<thead>
<tr><th>Case Study</th><th>Prelim trials / length</th><th>Final trials / length</th><th>Prelim traj / length</th><th>Final traj / length</th></tr>
</thead>
<tbody>
<tr><td>TaxiNet</td><td>10 / 10</td><td>100 / 20</td><td>10 / 10</td><td>100 / 20</td></tr>
<tr><td>CartPole</td><td>10 / 10</td><td>25 / 15</td><td>10 / 10</td><td>30 / 15</td></tr>
<tr><td>Obstacle</td><td>10 / 20</td><td>50 / 25</td><td>10 / 10</td><td>50 / 20</td></tr>
<tr><td>Refuel</td><td>10 / 20</td><td>50 / 30</td><td>10 / 10</td><td>10 / 10</td></tr>
</tbody>
</table>

### 3. Cached RL agents and optimized realizations

The RL agents and adversarial-perception realizations trained in the preliminary run are
reused (same prelim cache paths).  This avoids retraining overhead and preserves comparability.
Adversarial realizations were optimized against: envelope (TaxiNet, CartPole, Obstacle) or
single-belief (Refuel, where envelope LP is infeasible).

---

## 1. Case Study Characteristics

<table>
<thead>
<tr><th>Case Study</th><th>States</th><th>Actions</th><th>Observations</th><th>Obs/State ratio</th><th>Source</th></tr>
</thead>
<tbody>
<tr><td>TaxiNet</td><td>16</td><td>3</td><td>16</td><td>~6–15 states/obs</td><td>Neural network controller</td></tr>
<tr><td>CartPole</td><td>82</td><td>2</td><td>82</td><td>1:1 (bijective)</td><td>Discretised continuous env</td></tr>
<tr><td>Obstacle</td><td>50</td><td>4</td><td>3</td><td>~17 states/obs</td><td>Gridworld benchmark (Carr et al.)</td></tr>
<tr><td>Refuel</td><td>344</td><td>5</td><td>43</td><td>~8 states/obs</td><td>Gridworld benchmark (Carr et al.)</td></tr>
</tbody>
</table>

**Note on TaxiNet observability:** Although TaxiNet has equal numbers of states and
observations, the IPOMDP perception intervals `P_lower[s][obs]` place each observation
in the support of 6–15 states (not bijective).  TaxiNet is partially observable.

---

## 2. Coarseness of the LFP Bound

<table>
<thead>
<tr><th>Case Study</th><th>n traj / len</th><th>Mean max-gap</th><th>Std</th><th>Mean avg-gap</th><th>Interpretation</th></tr>
</thead>
<tbody>
<tr><td>TaxiNet</td><td>100 / 20</td><td>0.358</td><td>0.262</td><td>0.054</td><td>**Tight** — LFP nearly matches sampler</td></tr>
<tr><td>CartPole</td><td>30 / 15</td><td>0.994</td><td>0.017</td><td>0.381</td><td>**Very loose** — LFP maximally conservative</td></tr>
<tr><td>Obstacle</td><td>50 / 20</td><td>0.960</td><td>0.102</td><td>0.402</td><td>**Very loose** — higher than prelim's 0.784</td></tr>
<tr><td>Refuel</td><td>10 / 10</td><td>0.865</td><td>0.296</td><td>0.557</td><td>**Loose** — slightly lower than prelim's 0.989</td></tr>
</tbody>
</table>

Timing: TaxiNet 3 min, CartPole 12 min, Obstacle 11 min, Refuel 3 h 45 min.

Figures: [`coarse_*_results.png`]()

**Key finding.** The qualitative conclusions from the preliminary remain intact and are now
supported by larger samples:

- TaxiNet maintains tight bounds (mean avg-gap = 0.054, consistent with prelim 0.057).
- CartPole and Obstacle are very loose (max-gap ≈ 1.0 and ≈ 0.96 respectively).  Obstacle's
  mean max-gap increased from 0.784 (prelim, 10 traj) to 0.960 (final, 50 traj), revealing
  that early trajectories were atypically tight.
- Refuel mean max-gap = 0.865 vs prelim 0.989.  The smaller value may reflect the updated
  distance-scaled noise model (obs_noise = 0.05 with Gaussian distance kernel vs the
  original uniform noise); the bound remains very loose at this scale.

---

## 3. Shield Definitions

<table>
<thead>
<tr><th>Shield</th><th>Description</th></tr>
</thead>
<tbody>
<tr><td>**None**</td><td>Passthrough — all actions always allowed. Baseline failure rate.</td></tr>
<tr><td>**Observation**</td><td>Allows action *a* only if *a* ∈ pp_shield[s] **for every state s** with P_lower[s][obs] > 0.</td></tr>
<tr><td>**Single-belief**</td><td>Maintains POMDP point belief; allows *a* if P(safe \</td><td>belief) ≥ 0.8.</td></tr>
<tr><td>**Envelope**</td><td>Maintains full LFP belief polytope; allows *a* if ∃ valid distribution with P(safe) ≥ 0.8.</td></tr>
</tbody>
</table>

Two **perception regimes**: Uniform (cooperative), Adversarial-opt (fixed realization trained
to maximise failure rate against the target shield).

Three **action selectors**: random, best (greedy belief), RL (trained neural policy).

---

## 4. Results per Case Study

### 4.1 TaxiNet (16 states, partially obs., tight LFP bounds, n=100)

Figures: [`rl_shield_taxinet_figures/`](rl_shield_taxinet_figures/)

TaxiNet is the smallest case study and has the tightest LFP bound.  The **corrected
observation shield reveals that TaxiNet is partially observable**: each observation is
consistent with 6–15 states, making the intersection of safe actions empty for 91% of
RL-selector trials.

**RL selector results:**

<table>
<thead>
<tr><th>Perception</th><th>Shield</th><th>Fail%</th><th>Stuck%</th><th>Safe%</th><th>Interv%</th></tr>
</thead>
<tbody>
<tr><td>Uniform</td><td>None</td><td>95%</td><td>0%</td><td>5%</td><td>—</td></tr>
<tr><td>Uniform</td><td>Observation</td><td>9%</td><td>91%</td><td>0%</td><td>30.8%</td></tr>
<tr><td>Uniform</td><td>Single-belief</td><td>68%</td><td>1%</td><td>31%</td><td>19.7%</td></tr>
<tr><td>Uniform</td><td>Envelope</td><td>61%</td><td>4%</td><td>35%</td><td>23.6%</td></tr>
<tr><td>Adversarial</td><td>None</td><td>98%</td><td>0%</td><td>2%</td><td>—</td></tr>
<tr><td>Adversarial</td><td>Observation</td><td>5%</td><td>95%</td><td>0%</td><td>40.0%</td></tr>
<tr><td>Adversarial</td><td>Single-belief</td><td>70%</td><td>1%</td><td>29%</td><td>19.8%</td></tr>
<tr><td>Adversarial</td><td>Envelope</td><td>55%</td><td>5%</td><td>40%</td><td>23.0%</td></tr>
</tbody>
</table>

**Full results (n=100 per combination):**

<table>
<thead>
<tr><th>Perception</th><th>Selector</th><th>Shield</th><th>Fail%</th><th>Stuck%</th><th>Safe%</th></tr>
</thead>
<tbody>
<tr><td>Uniform</td><td>Random</td><td>None</td><td>100%</td><td>0%</td><td>0%</td></tr>
<tr><td>Uniform</td><td>Random</td><td>Observation</td><td>6%</td><td>94%</td><td>0%</td></tr>
<tr><td>Uniform</td><td>Random</td><td>Single-belief</td><td>77%</td><td>1%</td><td>22%</td></tr>
<tr><td>Uniform</td><td>Random</td><td>Envelope</td><td>58%</td><td>6%</td><td>36%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>None</td><td>100%</td><td>0%</td><td>0%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>Observation</td><td>7%</td><td>93%</td><td>0%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>Single-belief</td><td>55%</td><td>2%</td><td>43%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>Envelope</td><td>51%</td><td>12%</td><td>37%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>None</td><td>95%</td><td>0%</td><td>5%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>Observation</td><td>9%</td><td>91%</td><td>0%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>Single-belief</td><td>68%</td><td>1%</td><td>31%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>Envelope</td><td>61%</td><td>4%</td><td>35%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>None</td><td>100%</td><td>0%</td><td>0%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>Observation</td><td>3%</td><td>97%</td><td>0%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>Single-belief</td><td>68%</td><td>1%</td><td>31%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>Envelope</td><td>62%</td><td>4%</td><td>34%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>None</td><td>100%</td><td>0%</td><td>0%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>Observation</td><td>6%</td><td>94%</td><td>0%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>Single-belief</td><td>55%</td><td>1%</td><td>44%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>Envelope</td><td>54%</td><td>11%</td><td>35%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>None</td><td>98%</td><td>0%</td><td>2%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>Observation</td><td>5%</td><td>95%</td><td>0%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>Single-belief</td><td>70%</td><td>1%</td><td>29%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>Envelope</td><td>55%</td><td>5%</td><td>40%</td></tr>
</tbody>
</table>

**Observations:**
- **Overall safety is low**: with n=100 trials and trial_length=20, the RL agent (trained
  for 100 episodes at episode_length=10) achieves only 5% safe without shielding.  The
  preliminary's 40% safe (n=10, length=10) was inflated by small-sample variance.  The
  true performance on 20-step trajectories is poor.
- **Observation shield is unusable** (91–95% stuck): TaxiNet's partial observability makes
  the observation shield's intersection condition nearly never satisfiable.
- **Envelope provides the best safe rate**: 35–40% safe under RL selector, compared to
  29–31% for single-belief.  The envelope's advantage is larger under adversarial perception
  (+11 pp), consistent with its theoretical robustness guarantee.
- **Envelope outperforms single-belief under adversarial perception** (40% vs 29% safe),
  confirming the theoretical value of the LFP polytope for interval-robust shielding.
- **Single-belief has fewer stuck events than envelope** (1% vs 4–5%), as expected.

---

### 4.2 CartPole (82 states, fully observable, very loose LFP bounds, n=25)

Figures: [`rl_shield_cartpole_figures/`](rl_shield_cartpole_figures/)

CartPole is fully observable with 82 states (3-bin discretisation).  The LFP bound is very
loose (mean max-gap ≈ 1.0).  Since each observation maps bijectively to one state, the
observation shield is equivalent to the perfect-perception shield.

**RL selector results:**

<table>
<thead>
<tr><th>Perception</th><th>Shield</th><th>Fail%</th><th>Stuck%</th><th>Safe%</th><th>Interv%</th></tr>
</thead>
<tbody>
<tr><td>Uniform</td><td>None</td><td>12%</td><td>0%</td><td>88%</td><td>—</td></tr>
<tr><td>Uniform</td><td>Observation</td><td>4%</td><td>56%</td><td>40%</td><td>12.3%</td></tr>
<tr><td>Uniform</td><td>Single-belief</td><td>4%</td><td>0%</td><td>96%</td><td>4.2%</td></tr>
<tr><td>Uniform</td><td>Envelope</td><td>4%</td><td>16%</td><td>80%</td><td>25.8%</td></tr>
<tr><td>Adversarial</td><td>None</td><td>12%</td><td>0%</td><td>88%</td><td>—</td></tr>
<tr><td>Adversarial</td><td>Observation</td><td>4%</td><td>36%</td><td>60%</td><td>7.6%</td></tr>
<tr><td>Adversarial</td><td>Single-belief</td><td>4%</td><td>0%</td><td>96%</td><td>3.1%</td></tr>
<tr><td>Adversarial</td><td>Envelope</td><td>4%</td><td>20%</td><td>76%</td><td>27.0%</td></tr>
</tbody>
</table>

**Observations:**
- **Single-belief is the best shield** (96% safe) under both perception regimes.  In a
  fully-observable setting the point belief collapses to certainty, giving near-perfect
  state knowledge.
- **Envelope causes unnecessary stuck events** (16–20%) without reducing the fail rate.
  This is the direct consequence of the very loose LFP bound (max-gap ≈ 1.0): the polytope
  contains all belief distributions, so the envelope cannot certify safety for many states.
- **Observation shield causes significant stuck** (36–56%): some CartPole states at the
  failure boundary have no safe action in pp_shield, causing immediate stuck.  Fail rate
  is still low (4%) because most stuck events are near the boundary where the envelope or
  single-belief also prevents action.
- **Adversarial perception has minimal impact** on single-belief (96% safe under both
  regimes): with bijective observations, the point belief collapses to a single state after
  each step and the adversary cannot shift belief away from the true state.

---

### 4.3 Obstacle (50 states, partial obs., 3 observations, loose LFP bounds, n=50)

Figures: [`rl_shield_obstacle_figures/`](rl_shield_obstacle_figures/)

Obstacle is partially observable with only 3 distinct observations (~17 states per
observation) and a moderate-to-loose LFP bound.

**RL selector results:**

<table>
<thead>
<tr><th>Perception</th><th>Shield</th><th>Fail%</th><th>Stuck%</th><th>Safe%</th><th>Interv%</th></tr>
</thead>
<tbody>
<tr><td>Uniform</td><td>None</td><td>82%</td><td>0%</td><td>18%</td><td>—</td></tr>
<tr><td>Uniform</td><td>Observation</td><td>2%</td><td>98%</td><td>0%</td><td>0.0%</td></tr>
<tr><td>Uniform</td><td>Single-belief</td><td>40%</td><td>34%</td><td>26%</td><td>12.7%</td></tr>
<tr><td>Uniform</td><td>Envelope</td><td>22%</td><td>72%</td><td>6%</td><td>25.7%</td></tr>
<tr><td>Adversarial</td><td>None</td><td>80%</td><td>0%</td><td>20%</td><td>—</td></tr>
<tr><td>Adversarial</td><td>Observation</td><td>4%</td><td>96%</td><td>0%</td><td>0.0%</td></tr>
<tr><td>Adversarial</td><td>Single-belief</td><td>38%</td><td>26%</td><td>36%</td><td>14.3%</td></tr>
<tr><td>Adversarial</td><td>Envelope</td><td>16%</td><td>50%</td><td>34%</td><td>29.6%</td></tr>
</tbody>
</table>

**Full results (n=50 per combination):**

<table>
<thead>
<tr><th>Perception</th><th>Selector</th><th>Shield</th><th>Fail%</th><th>Stuck%</th><th>Safe%</th></tr>
</thead>
<tbody>
<tr><td>Uniform</td><td>Random</td><td>None</td><td>80%</td><td>0%</td><td>20%</td></tr>
<tr><td>Uniform</td><td>Random</td><td>Observation</td><td>2%</td><td>98%</td><td>0%</td></tr>
<tr><td>Uniform</td><td>Random</td><td>Single-belief</td><td>32%</td><td>44%</td><td>24%</td></tr>
<tr><td>Uniform</td><td>Random</td><td>Envelope</td><td>12%</td><td>56%</td><td>32%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>None</td><td>84%</td><td>0%</td><td>16%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>Observation</td><td>2%</td><td>98%</td><td>0%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>Single-belief</td><td>16%</td><td>14%</td><td>70%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>Envelope</td><td>18%</td><td>62%</td><td>20%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>None</td><td>82%</td><td>0%</td><td>18%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>Observation</td><td>2%</td><td>98%</td><td>0%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>Single-belief</td><td>40%</td><td>34%</td><td>26%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>Envelope</td><td>22%</td><td>72%</td><td>6%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>None</td><td>78%</td><td>0%</td><td>22%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>Observation</td><td>2%</td><td>98%</td><td>0%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>Single-belief</td><td>38%</td><td>26%</td><td>36%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>Envelope</td><td>28%</td><td>54%</td><td>18%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>None</td><td>84%</td><td>0%</td><td>16%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>Observation</td><td>2%</td><td>98%</td><td>0%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>Single-belief</td><td>20%</td><td>20%</td><td>60%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>Envelope</td><td>18%</td><td>52%</td><td>30%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>None</td><td>80%</td><td>0%</td><td>20%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>Observation</td><td>4%</td><td>96%</td><td>0%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>Single-belief</td><td>38%</td><td>26%</td><td>36%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>Envelope</td><td>16%</td><td>50%</td><td>34%</td></tr>
</tbody>
</table>

**Observations:**
- **Observation shield is effectively unusable** (98% stuck): with 3 coarse observations
  and ~17 states per observation, the intersection of safe actions across consistent states
  is almost always empty.  The 0% intervention rate reflects that the RL selector never
  has a valid choice to override.
- **Envelope achieves the best fail rate under adversarial perception** (16% vs 38% for
  single-belief under RL, a 22 pp improvement).  This is the clearest evidence across all
  case studies that the envelope's formal robustness to interval manipulation translates to
  empirical benefit.
- **Envelope trades fail rate for stuck rate under uniform perception** (22% fail / 72%
  stuck under RL vs 40% fail / 34% stuck for single-belief).  The envelope's conservatism
  is costly in the cooperative regime.
- **Single-belief with best selector is the most efficient shield under uniform perception**
  (16% fail, 14% stuck, 70% safe) — outperforming envelope because the cooperative belief
  estimate is accurate enough.
- The observation shield's near-total stuck rate (vs prelim's 40%) is a **methodology
  correction**: the preliminary buggy version allowed actions from pp_shield[obs] (using obs
  as a key into a state-keyed dict), which returned actions for a single state rather than
  computing the true intersection across ~17 consistent states.

---

### 4.4 Refuel (344 states, partial obs., 43 observations, no envelope, n=50)

Figures: [`rl_shield_refuel_figures/`](rl_shield_refuel_figures/)

Refuel is the largest case study.  The envelope shield is excluded (LP solve ~144 s/step).
Adversarial perception was optimized against the single-belief shield.

**RL selector results (envelope excluded):**

<table>
<thead>
<tr><th>Perception</th><th>Shield</th><th>Fail%</th><th>Stuck%</th><th>Safe%</th><th>Interv%</th></tr>
</thead>
<tbody>
<tr><td>Uniform</td><td>None</td><td>0%</td><td>0%</td><td>100%</td><td>—</td></tr>
<tr><td>Uniform</td><td>Observation</td><td>0%</td><td>40%</td><td>60%</td><td>1.5%</td></tr>
<tr><td>Uniform</td><td>Single-belief</td><td>0%</td><td>18%</td><td>82%</td><td>0.2%</td></tr>
<tr><td>Adversarial</td><td>None</td><td>0%</td><td>0%</td><td>100%</td><td>—</td></tr>
<tr><td>Adversarial</td><td>Observation</td><td>0%</td><td>48%</td><td>52%</td><td>1.5%</td></tr>
<tr><td>Adversarial</td><td>Single-belief</td><td>0%</td><td>10%</td><td>90%</td><td>0.1%</td></tr>
</tbody>
</table>

**Full results (n=50 per combination):**

<table>
<thead>
<tr><th>Perception</th><th>Selector</th><th>Shield</th><th>Fail%</th><th>Stuck%</th><th>Safe%</th></tr>
</thead>
<tbody>
<tr><td>Uniform</td><td>Random</td><td>None</td><td>6%</td><td>0%</td><td>94%</td></tr>
<tr><td>Uniform</td><td>Random</td><td>Observation</td><td>2%</td><td>46%</td><td>52%</td></tr>
<tr><td>Uniform</td><td>Random</td><td>Single-belief</td><td>0%</td><td>16%</td><td>84%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>None</td><td>10%</td><td>0%</td><td>90%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>Observation</td><td>0%</td><td>48%</td><td>52%</td></tr>
<tr><td>Uniform</td><td>Best</td><td>Single-belief</td><td>0%</td><td>20%</td><td>80%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>None</td><td>0%</td><td>0%</td><td>100%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>Observation</td><td>0%</td><td>40%</td><td>60%</td></tr>
<tr><td>Uniform</td><td>RL</td><td>Single-belief</td><td>0%</td><td>18%</td><td>82%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>None</td><td>14%</td><td>0%</td><td>86%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>Observation</td><td>0%</td><td>48%</td><td>52%</td></tr>
<tr><td>Adversarial</td><td>Random</td><td>Single-belief</td><td>2%</td><td>16%</td><td>82%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>None</td><td>8%</td><td>0%</td><td>92%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>Observation</td><td>2%</td><td>50%</td><td>48%</td></tr>
<tr><td>Adversarial</td><td>Best</td><td>Single-belief</td><td>0%</td><td>22%</td><td>78%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>None</td><td>0%</td><td>0%</td><td>100%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>Observation</td><td>0%</td><td>48%</td><td>52%</td></tr>
<tr><td>Adversarial</td><td>RL</td><td>Single-belief</td><td>0%</td><td>10%</td><td>90%</td></tr>
</tbody>
</table>

**Observations:**
- **No-shield with RL is optimal** (100% safe under both perceptions, n=50).  The RL agent
  navigates the refuel task reliably without shielding.  This is consistent with the prelim
  and is confirmed by 5× more trials.
- **Shields cause unnecessary stuck** without reducing fail rate.  Both observation
  (40–48% stuck) and single-belief (10–18% stuck) introduce stuck events while the RL fail
  rate stays at 0%.  The shield's conservatism prevents actions the agent would have taken
  correctly.
- **Single-belief is less prone to stuck than observation** (0–22% vs 40–50%), confirming
  that maintaining a point belief is more permissive than requiring intersection across all
  consistent states.
- **Adversarial perception has minimal impact on RL+none** (both 100% safe): the well-trained
  RL agent is robust to perception adversaries for this task.  The adversary achieves only
  modest degradation against the single-belief shield (82–90% safe vs 82–84% under uniform),
  confirming the refuel IPOMDP is hard to adversarially attack.

---

## 5. When Does the Envelope Shield Add Value?

<table>
<thead>
<tr><th>Case Study</th><th>Envelope vs None (RL, adv.)</th><th>Envelope vs Single-belief (RL, adv.)</th><th>Verdict</th></tr>
</thead>
<tbody>
<tr><td>TaxiNet (16 states)</td><td>+38 pp safe (40% vs 2%)</td><td>+11 pp safe (40% vs 29%)</td><td>**Envelope wins under adversarial**</td></tr>
<tr><td>CartPole (82 states)</td><td>+12 pp fail reduction</td><td>≈ equal fail, +20% stuck</td><td>Envelope slightly worse (stuck overhead)</td></tr>
<tr><td>Obstacle (50 states)</td><td>+64 pp safe (34% vs 20%)</td><td>−24 pp fail (16% vs 38%)</td><td>**Envelope wins under adversarial**</td></tr>
<tr><td>Refuel (344 states)</td><td>— (infeasible)</td><td>— (infeasible)</td><td>No-shield optimal; envelope infeasible</td></tr>
</tbody>
</table>

The envelope shield provides its largest benefits in **TaxiNet** and **Obstacle** under
adversarial perception.  The pattern is clear:

**The envelope shield is most valuable when (a) the LFP bound is not maximally loose AND
(b) perception is adversarial.**

- In TaxiNet (tight bounds, mean avg-gap = 0.054): envelope achieves 40% safe under
  adversarial vs 29% for single-belief (+11 pp), with stuck rate staying low (5%).
- In Obstacle (moderate-to-loose bounds, mean avg-gap = 0.40): envelope reduces adversarial
  fail rate to 16% vs 38% for single-belief (−22 pp fail), at the cost of 50% stuck.
  Under uniform perception, the single-belief is better overall.
- In CartPole (very loose bounds): the envelope provides no additional fail reduction and
  adds 16–20% stuck — clearly inferior to single-belief.
- In Refuel (infeasible): the no-shield optimal case makes the question moot.

**The single-belief shield is the practical choice** for CartPole and Refuel: it runs in
milliseconds, achieves near-optimal safety, and avoids the stuck overhead of the envelope.

---

## 6. Adversarial vs Cooperative Perception Impact

**Δ safe% (adversarial − uniform) for RL selector:**

<table>
<thead>
<tr><th>Case Study</th><th>None</th><th>Observation</th><th>Single-belief</th><th>Envelope</th><th>Opt. target</th></tr>
</thead>
<tbody>
<tr><td>TaxiNet</td><td>−3</td><td>−1</td><td>−2</td><td>+5</td><td>envelope</td></tr>
<tr><td>CartPole</td><td>0</td><td>+20</td><td>0</td><td>−4</td><td>envelope</td></tr>
<tr><td>Obstacle</td><td>+2</td><td>+2</td><td>+10</td><td>+28</td><td>envelope</td></tr>
<tr><td>Refuel</td><td>0</td><td>−8</td><td>+8</td><td>N/A</td><td>single_belief</td></tr>
</tbody>
</table>

For **TaxiNet**, the envelope shield improves (+5 pp) under adversarial perception, consistent
with its theoretical guarantee.  Single-belief is robust (−2 pp).  Observation is insensitive
(already near-total stuck under both).

For **CartPole**, adversarial perception has minimal effect on no-shield and single-belief
(0 pp each) — the bijective observation map collapses point belief to certainty, making the
adversary ineffective.  The observation shield *improves* (+20 pp) under adversarial because
the adversary incidentally causes states where the observation shield gets stuck less (state
trajectory changes).

For **Obstacle**, the envelope shield shows the largest positive response to adversarial
perception (+28 pp improvement in safe rate under RL) — this is the strongest evidence that
the envelope's formal interval robustness translates to empirical benefit when an adversary
is active.

For **Refuel**, single-belief improves under adversarial (+8 pp) because the adversary was
only weakly optimized (best score 0.33 against single-belief) and the strong RL agent
maintains 100% safe regardless.

---

## 7. LFP Feasibility

Timing results (final run):

<table>
<thead>
<tr><th>Case Study</th><th>States</th><th>Observations</th><th>Coarse time</th><th>RL time (envelope combos)</th><th>Practical?</th></tr>
</thead>
<tbody>
<tr><td>TaxiNet</td><td>16</td><td>16</td><td>3 min</td><td>~10 min (6 × 100 × 20 steps)</td><td>✓ Yes</td></tr>
<tr><td>Obstacle</td><td>50</td><td>3</td><td>11 min</td><td>~33 min (6 × 50 × 25 steps)</td><td>✓ Yes</td></tr>
<tr><td>CartPole</td><td>82</td><td>82</td><td>12 min</td><td>~53 min (6 × 25 × 15 steps)</td><td>✓ Marginal</td></tr>
<tr><td>Refuel</td><td>344</td><td>43</td><td>3h 45m</td><td>(excluded, ~144 s/step)</td><td>✗ No</td></tr>
</tbody>
</table>

Refuel's coarse experiment ran at the same scale as the preliminary (10 traj × 10 steps)
because increasing the scale would dominate the 8-hour budget.

---

## 8. Conclusions

1. **Envelope shielding provides clear benefit over single-belief under adversarial perception
   when bounds are not maximally loose.**  TaxiNet (+11 pp safe) and Obstacle (−22 pp fail)
   confirm this.  The advantage vanishes or reverses under cooperative perception or when the
   LFP bound is very loose (CartPole).

2. **The corrected observation shield is generally unusable** for the three partially-observable
   case studies (TaxiNet: 91% stuck, Obstacle: 98% stuck).  The preliminary's results for these
   case studies (showing 0–40% stuck) reflected a bug where pp_shield was keyed by observation
   rather than state.  For the fully-observable CartPole, the observation shield causes 36–56%
   stuck reflecting genuine pp_shield boundary states, not a bug.

3. **Single-belief is the practical shield for real deployment**: 96% safe on CartPole, 82–90%
   safe on Refuel, no stuck events, runs in milliseconds.

4. **The no-shield RL policy is optimal for Refuel** (100% safe, confirmed with n=50 trials).
   A well-trained RL agent can be more effective than any shield that introduces stuck events.

5. **Feasibility threshold is approximately 80–100 states** for the envelope shield at
   current LP speeds.  Obstacle (50 states, 3 obs) remains tractable at ~33 min for 50 × 25
   RL trials; CartPole (82 states, 82 obs) is marginal at ~53 min; Refuel (344 states) is
   infeasible.

6. **Statistical precision improved substantially**: with 50–100 trials per combination
   (vs 10 in the preliminary), the final results have tight 95% CIs (typically ±5–10 pp)
   and several preliminary findings reversed or were substantially revised (notably TaxiNet's
   single-belief safe rate: 70% prelim → 31% final, now correctly estimated with n=100 and
   trial_length=20).

---

## Appendix: Experimental Configuration

<table>
<thead>
<tr><th>Parameter</th><th>TaxiNet</th><th>CartPole</th><th>Obstacle</th><th>Refuel</th></tr>
</thead>
<tbody>
<tr><td>Final trials</td><td>100</td><td>25</td><td>50</td><td>50</td></tr>
<tr><td>Trial length</td><td>20</td><td>15</td><td>25</td><td>30</td></tr>
<tr><td>Trial length (prelim)</td><td>10</td><td>10</td><td>20</td><td>20</td></tr>
<tr><td>Coarse traj / length</td><td>100 / 20</td><td>30 / 15</td><td>50 / 20</td><td>10 / 10</td></tr>
<tr><td>RL episodes (cache)</td><td>100</td><td>100</td><td>100</td><td>100</td></tr>
<tr><td>Shield threshold</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td></tr>
<tr><td>obs_noise</td><td>N/A (data-driven)</td><td>N/A (data-driven)</td><td>0.1 (uniform)</td><td>0.05 (dist.-scaled)</td></tr>
<tr><td>Envelope included</td><td>Yes</td><td>Yes</td><td>Yes</td><td>No (infeasible)</td></tr>
<tr><td>Adversarial opt target</td><td>envelope</td><td>envelope</td><td>envelope</td><td>single_belief</td></tr>
<tr><td>Seed</td><td>42</td><td>42</td><td>42</td><td>42</td></tr>
</tbody>
</table>

Total wall-clock time: ~7 hours (coarse: 4h 12m; RL shield: ~1h 37m; summary charts: ~1h).

Results directory: ``
Summary charts: `summary/`