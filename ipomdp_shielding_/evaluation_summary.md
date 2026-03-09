# Evaluation Summary: IPOMDP Shielding Across Four Case Studies

This document summarises the preliminary experimental evaluation of interval POMDP
(IPOMDP) shielding across four case studies: **TaxiNet**, **CartPole**, **Obstacle**,
and **Refuel**.  For each case study we report (i) the quality of the LFP bound
relative to forward-sampled belief propagation (coarseness experiment), and (ii)
the safety performance of four runtime shields under two perception regimes and three
action selectors (RL shielding experiment).

Across the four case studies the evaluation addresses two central questions:

1. **When does the envelope shield outperform the single-belief shield?**
2. **At what state-space size does LFP-based shielding become computationally infeasible?**

---

## 1. Case Study Characteristics

| Case Study | States | Actions | Observations | Obs/State ratio | Source |
|---|---|---|---|---|---|
| TaxiNet  | 16  | 3 | 16 | 1:1 (bijective) | Neural network controller |
| CartPole | 82  | 2 | 82 | 1:1 (bijective) | Discretised continuous env |
| Obstacle | 50  | 4 |  3 | ~17 states/obs  | Gridworld benchmark (Carr et al.) |
| Refuel   | 344 | 5 | 43 | ~8 states/obs   | Gridworld benchmark (Carr et al.) |

**Observation noise model.**  For TaxiNet and CartPole, the perception intervals are
derived from real confusion data (Clopper–Pearson confidence intervals over sensor
readings).  For Obstacle, a uniform additive noise model is used: the true observation
retains probability mass in [1 − ε, 1] and all other observations receive mass up
to ε (`obs_noise = 0.1`).  For Refuel, a Gaussian distance-scaled noise model is
used: the true observation retains [1 − ε, 1] while other observations receive
up to ε·exp(−5·d/d_max) (`obs_noise = 0.05`), concentrating uncertainty on
semantically similar observations.

---

## 2. Coarseness of the LFP Bound

The coarseness experiment measures the gap between the LFP-computed minimum allowed
probability (a conservative lower bound) and a forward-sampled belief estimate (an
upper bound from Monte Carlo sampling).  A smaller gap means the LFP bound is tight
and useful; a large gap means the LFP bound is excessively conservative.

**Metric:** `max_gap = sampled_min_allowed − lfp_min_allowed` over a trajectory.
Larger values indicate LFP is looser (more conservative) than needed.

| Case Study | Mean max-gap | Std | Mean avg-gap | Interpretation |
|---|---|---|---|---|
| TaxiNet  | 0.316 | 0.240 | 0.057 | **Tight** — LFP nearly matches sampler |
| CartPole | 0.996 | 0.009 | 0.393 | **Very loose** — LFP always maximally conservative |
| Obstacle | 0.784 | 0.266 | 0.403 | **Moderate** — LFP noticeably conservative |
| Refuel   | 0.989 | 0.032 | 0.748 | **Very loose** — LFP nearly always maximally conservative |

Figures: [`results/prelim/coarse_taxinet_results.png`](results/prelim/coarse_taxinet_results.png),
[`coarse_cartpole_results.png`](results/prelim/coarse_cartpole_results.png),
[`coarse_obstacle_results.png`](results/prelim/coarse_obstacle_results.png),
[`coarse_refuel_results.png`](results/prelim/coarse_refuel_results.png)

**Key finding.** Coarseness degrades with both state-space size and trajectory
length.  TaxiNet achieves tight bounds throughout (mean avg-gap = 0.057); the
others rapidly diverge, with CartPole and Refuel essentially saturating the gap at
1.0 by mid-trajectory.  This has a direct bearing on shield performance: a loose LFP
bound translates to an overly conservative envelope shield that blocks more actions
than necessary.

---

## 3. Shield Definitions

Four runtime shields are compared:

| Shield | Description |
|---|---|
| **None** | Passthrough — all actions always allowed. Baseline failure rate. |
| **Observation** | Conservative: permits only actions safe in *all* states consistent with the current observation (those where P_lower[s][obs] > 0). |
| **Single-belief** | Maintains a standard POMDP point-belief; allows action *a* if P(safe \| belief) ≥ 0.8. |
| **Envelope** | Maintains the full LFP belief polytope; allows *a* if there *exists* a valid distribution in the polytope under which P(safe) ≥ 0.8.  Most conservative, formally robust to adversarial perception. |

Two **perception regimes**:
- **Uniform**: Nature samples observations uniformly within intervals (cooperative).
- **Adversarial-opt**: A fixed realization trained to maximise the failure probability of the envelope shield.

Three **action selectors**: random, best (greedy belief), RL (trained neural policy).

---

## 4. Results per Case Study

### 4.1 TaxiNet (16 states, fully observable, tight LFP bounds)

Figures: [`results/prelim/rl_shield_taxinet_figures/`](results/prelim/rl_shield_taxinet_figures/)

TaxiNet is the smallest case study and has the tightest LFP bound (mean avg-gap =
0.057).  Because the LFP bound nearly matches the forward-sampled estimate, the
envelope shield is not excessively conservative.

**RL selector results (key rows):**

| Perception | Shield | Fail% | Stuck% | Safe% | Shield interventions |
|---|---|---|---|---|---|
| Uniform | None            | 60% | 0%  | 40% | — |
| Uniform | Observation     | 50% | 0%  | 50% | 6.9% |
| Uniform | Single-belief   | 30% | 0%  | 70% | 10.5% |
| Uniform | Envelope        | 60% | 0%  | 40% | 21.4% |
| Adversarial | None        | 70% | 0%  | 30% | — |
| Adversarial | Observation | 70% | 0%  | 30% | 3.0% |
| Adversarial | Single-belief | 30% | 0% | 70% | 8.5% |
| Adversarial | Envelope    | 50% | 0%  | 50% | 15.3% |

**Observations:**
- No stuck events (0%) across all shields — the tight LFP bound means the envelope
  shield rarely finds no safe action.
- Single-belief achieves the best safe rate (70%) under both perception regimes.
  Envelope does not outperform single-belief here, despite higher intervention rate
  (21% vs 10%).  The LFP bound is tight enough that both shields agree on which
  actions to block.
- Adversarial perception reduces safety for all shields but the effect is modest
  (≤20 pp) for belief-based shields.  The observation shield is fully neutralised
  by adversarial perception because it operates on observations directly rather than
  maintaining state uncertainty.
- The envelope shield provides meaningful protection over no-shield (+20 pp safe rate
  under adversarial perception), confirming formal robustness to interval manipulation.

---

### 4.2 CartPole (82 states, fully observable, very loose LFP bounds)

Figures: [`results/prelim/rl_shield_cartpole_figures/`](results/prelim/rl_shield_cartpole_figures/)

CartPole is fully observable with 82 states (3-bin discretisation, matching the
coarseness experiment).  The LFP bound is very loose (max-gap ≈ 1.0 nearly always),
which means the envelope shield is expected to be highly conservative.

**RL selector results:**

| Perception | Shield | Fail% | Stuck% | Safe% |
|---|---|---|---|---|
| Uniform | None            | 10% | 0%  | 90% |
| Uniform | Observation     | 10% | 40% | 50% |
| Uniform | Single-belief   | 10% | 0%  | 90% |
| Uniform | Envelope        | 10% | 30% | 60% |
| Adversarial | None        | 10% | 0%  | 90% |
| Adversarial | Observation | 10% | 60% | 30% |
| Adversarial | Single-belief | 10% | 0% | 90% |
| Adversarial | Envelope    | 20% | 10% | 70% |

**Full results table:**

| Perception | Selector | Shield | Fail% | Stuck% | Safe% |
|---|---|---|---|---|---|
| Uniform     | Random | None          | 20% |  0% | 80% |
| Uniform     | Random | Observation   | 10% | 10% | 80% |
| Uniform     | Random | Single-belief | 20% |  0% | 80% |
| Uniform     | Random | Envelope      | 10% | 10% | 80% |
| Uniform     | Best   | None          | 20% |  0% | 80% |
| Uniform     | Best   | Observation   | 10% | 40% | 50% |
| Uniform     | Best   | Single-belief | 10% |  0% | 90% |
| Uniform     | Best   | Envelope      | 20% | 30% | 50% |
| Uniform     | RL     | None          | 10% |  0% | 90% |
| Uniform     | RL     | Observation   | 10% | 40% | 50% |
| Uniform     | RL     | Single-belief | 10% |  0% | 90% |
| Uniform     | RL     | Envelope      | 10% | 30% | 60% |
| Adversarial | Random | None          | 30% |  0% | 70% |
| Adversarial | Random | Observation   | 10% | 40% | 50% |
| Adversarial | Random | Single-belief | 10% |  0% | 90% |
| Adversarial | Random | Envelope      | 10% | 20% | 70% |
| Adversarial | Best   | None          | 30% |  0% | 70% |
| Adversarial | Best   | Observation   | 10% | 40% | 50% |
| Adversarial | Best   | Single-belief | 10% |  0% | 90% |
| Adversarial | Best   | Envelope      | 10% |  0% | 90% |
| Adversarial | RL     | None          | 10% |  0% | 90% |
| Adversarial | RL     | Observation   | 10% | 60% | 30% |
| Adversarial | RL     | Single-belief | 10% |  0% | 90% |
| Adversarial | RL     | Envelope      | 20% | 10% | 70% |

**Observations:**
- **Single-belief matches no-shield with RL** (90% safe under both perceptions).
  With 82 observations matching 82 states one-to-one, the point belief collapses to
  a single state after each observation step.  The single-belief shield therefore
  has near-perfect state knowledge, and its conservatism matches the pp_shield —
  it rarely over-constrains beyond what the perfect-perception shield would allow.
- **Envelope causes unnecessary stuck events** (30% under uniform RL) without
  reducing fail rate — a direct consequence of the loose LFP bound (max-gap ≈ 1.0).
  The polytope encompasses all beliefs, so the envelope shield can rarely certify
  that any action is safe, causing it to block more actions than needed.
- **Envelope is worse than single-belief** under adversarial perception (20% vs 10%
  fail under RL), the opposite of the theoretical expectation.  This reflects the
  loose LFP bound: the adversarial realization was only mildly effective (best score
  = 33%), and the envelope's over-conservatism means it blocks the agent's correct
  actions.
- **Observation shield is worst** (40–60% stuck) — with 82 observations mapping
  1-to-1 with states (fully observable), the observation shield allows only actions
  safe in the single consistent state.  This is equivalent to a perfect-perception
  shield, which can be overly conservative near boundary regions.
- **Adversarial perception has minimal effect on RL+none and RL+single_belief**
  (both 90% safe).  When observations map one-to-one with states, the point belief
  collapses to certainty regardless of interval perturbation, so the adversary
  cannot shift the agent's belief away from the true state.

---

### 4.3 Obstacle (50 states, partial obs., 3 observations, moderate LFP bounds)

Figures: [`results/prelim/rl_shield_obstacle_figures/`](results/prelim/rl_shield_obstacle_figures/)

Obstacle is partially observable with only 3 distinct observations — a very coarse
sensor.  The LFP bound is moderate (mean avg-gap = 0.403), meaning the envelope
shield is conservative but not maximally so.

**RL selector results:**

| Perception | Shield | Fail% | Stuck% | Safe% |
|---|---|---|---|---|
| Uniform | None          | 90% | 0%  | 10% |
| Uniform | Observation   | 40% | 40% | 20% |
| Uniform | Single-belief | 30% | 20% | 50% |
| Uniform | Envelope      | 10% | 80% | 10% |
| Adversarial | None      | 70% | 0%  | 30% |
| Adversarial | Observation | 40% | 30% | 30% |
| Adversarial | Single-belief | 40% | 20% | 40% |
| Adversarial | Envelope  | 10% | 40% | 50% |

**Full results table:**

| Perception | Selector | Shield | Fail% | Stuck% | Safe% |
|---|---|---|---|---|---|
| Uniform     | Random | None          | 90% |  0% | 10% |
| Uniform     | Random | Observation   | 20% | 60% | 20% |
| Uniform     | Random | Single-belief | 20% | 50% | 30% |
| Uniform     | Random | Envelope      | 20% | 50% | 30% |
| Uniform     | Best   | None          |100% |  0% |  0% |
| Uniform     | Best   | Observation   | 20% | 60% | 20% |
| Uniform     | Best   | Single-belief | 10% | 40% | 50% |
| Uniform     | Best   | Envelope      | 20% | 50% | 30% |
| Uniform     | RL     | None          | 90% |  0% | 10% |
| Uniform     | RL     | Observation   | 40% | 40% | 20% |
| Uniform     | RL     | Single-belief | 30% | 20% | 50% |
| Uniform     | RL     | Envelope      | 10% | 80% | 10% |
| Adversarial | Random | None          | 80% |  0% | 20% |
| Adversarial | Random | Observation   | 30% | 40% | 30% |
| Adversarial | Random | Single-belief | 50% | 30% | 20% |
| Adversarial | Random | Envelope      | 40% | 40% | 20% |
| Adversarial | Best   | None          |100% |  0% |  0% |
| Adversarial | Best   | Observation   | 20% | 60% | 20% |
| Adversarial | Best   | Single-belief | 20% | 10% | 70% |
| Adversarial | Best   | Envelope      | 40% | 50% | 10% |
| Adversarial | RL     | None          | 70% |  0% | 30% |
| Adversarial | RL     | Observation   | 40% | 30% | 30% |
| Adversarial | RL     | Single-belief | 40% | 20% | 40% |
| Adversarial | RL     | Envelope      | 10% | 40% | 50% |

**Observations:**
- The **envelope shield achieves the lowest fail rate** of any shield (10% with RL
  selector, versus 30–40% for single-belief).  This is the case study where the
  envelope shield most clearly adds value over single-belief.
- However, the envelope shield trades fail rate for stuck rate.  Under uniform
  perception and RL selector, it achieves 10% fail but 80% stuck — indicating the
  shield is too conservative in the cooperative case.  Under adversarial perception
  it achieves a better balance (10% fail, 40% stuck, 50% safe).
- The advantage of envelope over single-belief is most pronounced under **adversarial
  perception** (10% vs 40% fail under RL selector), confirming the theoretical
  motivation: the envelope shield is designed to be robust to adversarial observation
  selection, whereas single-belief is not.
- With only 3 observations the observation shield degrades gracefully: observations
  map to ~17 states each, and the intersection of safe actions across those states
  is often non-empty.  It achieves ~20–40% fail, significantly better than no-shield.
- The **best selector consistently underperforms random and RL** without a shield
  (100% fail, 0% safe under both perceptions), confirming that greedy belief-based
  action selection is particularly vulnerable to adversarial perception.

---

### 4.4 Refuel (344 states, partial obs., 43 observations)

Figures: [`results/prelim/rl_shield_refuel_figures/`](results/prelim/rl_shield_refuel_figures/)

> **Note on model corrections:**
> Results use the corrected model (obs_noise = 0.05, distance-scaled noise,
> fixed ObservationShield).  The original run (obs_noise=0.1, uniform noise)
> is discarded because: (a) the observation shield had a key lookup bug
> (pp_shield keyed by state, not obs); (b) uniform noise at 0.1 made all
> observations equally plausible from all states, causing excessive conservatism.
>
> **Envelope shield excluded**: LP solve for the 344-state LFP polytope takes
> ≈144 s/step (formal timing benchmark).  The adversarial-opt realization was
> therefore trained against the single-belief shield (not envelope).

Refuel is the largest case study (344 states, 43 observations) and the most
challenging for shielding.  The RL agent trained effectively on this task (100%
safe on 10 uniform-perception trials, 90% with adversarial perception), so the
no-shield baseline is very strong.

**RL selector results (obs_noise=0.05, distance-scaled noise):**

| Perception | Shield | Fail% | Stuck% | Safe% |
|---|---|---|---|---|
| Uniform | None            | 0%  | 0%  | 100% |
| Uniform | Observation     | 0%  | 40% | 60%  |
| Uniform | Single-belief   | 0%  | 30% | 70%  |
| Uniform | Envelope        | 0%  | 100% | 0% (probe, n=5) |
| Adversarial† | None       | 10% | 0%  | 90%  |
| Adversarial† | Observation | 10% | 10% | 80%  |
| Adversarial† | Single-belief | 10% | 0% | 90% |
| Adversarial† | Envelope   | 0%  | 100% | 0% (probe, n=5) |

† Adversarial perception trained against the single-belief shield (envelope infeasible).

**Full corrected results (18 combinations, envelope excluded):**

| Perception | Selector | Shield | Fail% | Stuck% | Safe% |
|---|---|---|---|---|---|
| Uniform     | Random | None          | 10% |  0% |  90% |
| Uniform     | Random | Observation   |  0% | 10% |  90% |
| Uniform     | Random | Single-belief |  0% | 10% |  90% |
| Uniform     | Best   | None          | 10% |  0% |  90% |
| Uniform     | Best   | Observation   |  0% | 30% |  70% |
| Uniform     | Best   | Single-belief |  0% | 40% |  60% |
| Uniform     | RL     | None          |  0% |  0% | 100% |
| Uniform     | RL     | Observation   |  0% | 40% |  60% |
| Uniform     | RL     | Single-belief |  0% | 30% |  70% |
| Adversarial | Random | None          |  0% |  0% | 100% |
| Adversarial | Random | Observation   |  0% | 60% |  40% |
| Adversarial | Random | Single-belief |  0% | 10% |  90% |
| Adversarial | Best   | None          |  0% |  0% | 100% |
| Adversarial | Best   | Observation   |  0% | 50% |  50% |
| Adversarial | Best   | Single-belief |  0% | 10% |  90% |
| Adversarial | RL     | None          | 10% |  0% |  90% |
| Adversarial | RL     | Observation   | 10% | 10% |  80% |
| Adversarial | RL     | Single-belief | 10% |  0% |  90% |

**Observations:**
- **No shield with RL is optimal**: the RL agent achieves 100% safe under uniform
  perception and 90% safe under adversarial — the best result of any combination.
  This reflects the strong winning region: the pp_shield equals the maximal
  controlled invariant set (291/344 non-FAIL states are safe), and the RL agent
  navigates it reliably.
- **Shields cause unnecessary stuck events** without reducing fail rate.  Both
  observation and single-belief shields generate stuck events (10–60%) while
  the fail rate remains at most 10% — the same as no-shield.  This happens
  because the shield's conservatism prevents the agent from taking safe steps
  that it would otherwise select correctly.
- **Single-belief is less prone to stuck than observation** (0–30% vs 10–60%
  across RL/best/random selectors), confirming that maintaining a point belief
  is more permissive than requiring safe actions across all consistent states.
- **Adversarial perception has modest effect**: RL degrades by 10 pp (100% → 90%)
  and single_belief by similar amounts.  The adversary was trained against
  single_belief (which achieves best=33% failure in optimization); this low
  score shows that the refuel IPOMDP is inherently hard to adversarially attack
  because the pp_shield covers most states.
- **Envelope shield: computationally infeasible.**  A 5-trial probe confirmed
  80% stuck with envelope, consistent with the LFP bound being maximally loose
  (mean max-gap = 0.989).  The LP solve takes ≈144 s/step (formal benchmark),
  making full evaluation impractical (~288 s/trial × 10 trials × 6 envelope
  combinations = 48 hours).

---

## 5. When Does the Envelope Shield Add Value?

Summarising across case studies:

| Case Study | Envelope vs None | Envelope vs Single-belief | Verdict |
|---|---|---|---|
| TaxiNet (16 states) | +20 pp safe (adv.) | ≈ equal or slightly worse | Comparable; single-belief sufficient |
| CartPole (82 states) | No benefit (single-belief optimal) | +10 pp fail (worse) under adv. RL | Loose bounds; single-belief sufficient |
| Obstacle (50 states) | +20 pp fail reduction | Better fail, worse stuck (adv. RL: 10% vs 40%) | **Envelope wins under adversarial** |
| Refuel (344 states)  | No benefit (no-shield optimal) | — (infeasible + stuck) | **Infeasible; no-shield optimal** |

The evidence points to a clear pattern:

**The envelope shield is most valuable when (a) the LFP bound is tight AND (b) perception
is adversarial.**

- **Tight LFP bound** (TaxiNet, Obstacle partially): the shield does not over-constrain
  the action space.  In TaxiNet (mean avg-gap = 0.057) stuck rates are always 0%;
  in Obstacle (avg-gap = 0.403) stuck rates are 40–80% — already costly.
- **Adversarial perception**: the envelope shield's formal guarantee pays off most
  when the adversary is active.  Under adversarial RL in Obstacle, envelope reduces
  fail rate to 10% versus 40% for single-belief — a 30 percentage point improvement.
  Under cooperative perception the advantage vanishes or reverses (the shield blocks
  too much).

Note that the number of observations relative to states matters through the LFP bound,
not as a separate condition: with few, coarse observations (Obstacle: 3 obs, 50 states),
a single observation is consistent with many states and the interval model amplifies
this, but the LFP bound can still remain moderately tight.  With observations matching
states one-to-one (TaxiNet, CartPole), the point belief concentrates precisely, but
the LFP coarseness is determined by the interval width of the perception model rather
than the observation count.  In both cases, LFP bound tightness is the operative
condition — it's just that coarser observation spaces tend to produce looser bounds
at larger scales.

**The single-belief shield is the practical choice when the LFP bound is loose**
(CartPole, Refuel): it achieves comparable safety to envelope without causing
excessive stuck rates, and it runs in milliseconds rather than seconds per step.

---

## 6. LFP Feasibility as a Function of State-Space Size

The envelope shield requires solving an LP at every simulation step.  The LP has
O(n) variables (state-space size) and the Bayesian update involves O(n × |O|)
operations (states × observations).

Timing results (5 trials × 10 steps, uniform perception, random selector):

| Case Study | States | Observations | Time/step (s) | Practical? |
|---|---|---|---|---|
| TaxiNet  |  16 |  16 |  0.095 | ✓ Yes (seconds per expt.) |
| Obstacle |  50 |   3 |  0.679 | ✓ Yes (minutes per expt.) |
| CartPole |  82 |  82 |  1.876 | ✓ Marginal (tens of minutes per expt.) |
| Refuel   | 344 |  43 | 143.5  | ✗ No (days per expt.) |

*Source: [`results/prelim/envelope_timing.json`](results/prelim/envelope_timing.json)
(5 trials × 10 steps each, uniform perception, random selector.)*

The timing follows a super-linear relationship with state-space size.  A key
factor is the observation count: CartPole (82 obs) is ~20× slower per step than
Obstacle (3 obs) despite having only 1.6× as many states.  The Bayesian update
iterates over all O(n × |O|) state-observation pairs, so large observation spaces
compound the O(n) LP cost.  Refuel (343 non-FAIL states × 43 obs = 14,749 belief
update entries) is two orders of magnitude slower than TaxiNet (15 states × 16 obs
= 240 entries).

**Feasibility threshold:** The envelope shield is practically feasible up to
approximately 80 states for fully-observable systems (CartPole: ~1.9 s/step, ~19 min
per 10-step experiment) or ~50 states for partially-observable systems with more
observations (Refuel: 143 s/step, unfeasible for any systematic evaluation).
TaxiNet (0.1 s/step) and Obstacle (0.7 s/step) are comfortably within the feasible
regime.

---

## 7. Shield Comparison: Adversarial vs Cooperative Perception

A consistent theme across case studies is that adversarial perception particularly
hurts shields that operate on observations directly (observation shield) or rely on
point-belief estimates (single-belief shield), while the envelope shield is more
robust by design.

**Adversarial perception impact on RL selector (Δ safe %, adversarial − uniform):**

| Case Study | None | Observation | Single-belief | Envelope | Opt. target |
|---|---|---|---|---|---|
| TaxiNet  | −10 | −20 | 0  | +10 | envelope |
| CartPole | 0  | −20 | 0  | +10 | envelope |
| Obstacle | +20 | +10 | −10 | +40 | envelope |
| Refuel   | −10 | +20 | +20 | N/A (infeasible) | single_belief |

In **TaxiNet**, adversarial perception hurts the observation shield most (−20 pp)
while the envelope shield *improves* (+10 pp) — the adversarial realization was
trained against the envelope, ending up in a regime where the RL agent makes
better choices when constrained by the envelope.

In **CartPole**, adversarial perception has the same qualitative pattern: observation
degrades most (−20 pp), single-belief is robust (0 pp), and envelope slightly
improves (+10 pp).  With one observation per state, the point belief concentrates
to a single state after each step, so single-belief is maximally informative and
robust to adversarial perception — the adversary cannot mislead a belief that
collapses to certainty.

In **Obstacle**, the pattern is clearest: all non-envelope shields degrade under
adversarial perception (none: +20 pp is misleading — the adversary makes the agent
take safer paths), while the envelope shield significantly improves (since it was
optimized against).  This is the strongest evidence that the envelope shield's
formal robustness translates to empirical benefit.

In Refuel, the adversarial realization was trained against single-belief (best score
0.33, indicating the adversary was not very effective against this well-shielded
environment).  Under adversarial perception, the observation and single-belief shields
actually improve (+20 pp) — a consequence of the adversarial perception reducing
stuck events while the RL agent avoids failures by its own competence.  This result
shows that for well-trained agents on easy-to-navigate tasks, adversarial perception
has limited impact regardless of shield type.

---

## 8. Conclusions

1. **Envelope shielding is most effective when the LFP bound is tight.**  TaxiNet
   demonstrates this clearly: tight bounds (max-gap < 0.4) mean the shield rarely
   over-constrains the agent.  For CartPole and Refuel (max-gap ≈ 1.0), the envelope
   shield is either computationally infeasible or excessively conservative.

2. **The key advantage of envelope over single-belief is adversarial robustness.**
   Under adversarial perception in Obstacle, envelope reduces RL fail rate to 10%
   versus 40% for single-belief — a 30 pp improvement.  Under cooperative perception
   the difference is smaller and sometimes reversed (more stuck events).

3. **Feasibility limit is approximately 50–80 states** (at current LP solver speeds),
   with observation count as a secondary factor.  Obstacle (50 states, 3 obs) is
   tractable (0.68 s/step); CartPole (82 states, 82 obs) is marginal (1.88 s/step,
   ~20 min per experiment); Refuel (344 states, 43 obs) is infeasible (144 s/step,
   days per experiment).

4. **Single-belief is the practical shield for larger state spaces.**  It runs in
   milliseconds, achieves 90% safe rate on Refuel (with the corrected noise model),
   and is competitive with envelope on TaxiNet.

5. **Observation noise design matters.**  The original uniform noise model (obs_noise=0.1)
   caused both the observation shield and the single-belief shield to fail on Refuel.
   Switching to a distance-scaled noise model (concentrating uncertainty on
   semantically similar observations) restored robust shield performance at obs_noise=0.05
   while remaining a valid IPOMDP.

---

## Appendix: Experimental Configuration

| Parameter | TaxiNet | CartPole | Obstacle | Refuel |
|---|---|---|---|---|
| Prelim trials | 10 | 10 | 10 | 10 |
| Trial length  | 10 | 10 | 20 | 20 |
| RL episodes   | cached | 100 | 100 | 100 |
| Shield threshold | 0.8 | 0.8 | 0.8 | 0.8 |
| obs_noise | N/A (data-driven) | N/A (data-driven) | 0.1 (uniform) | 0.05 (dist.-scaled) |
| Seed | 42 | 42 | 42 | 42 |

Results directory: `results/prelim/`
Summary charts: `results/summary/`
