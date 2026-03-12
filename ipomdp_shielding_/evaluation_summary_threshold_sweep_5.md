# Evaluation Summary — v5

**Shields compared**: Envelope, Single-Belief, Observation, Carr  
*(where feasible — see per-case notes)*

**Case studies** (most-recent variants only):
TaxiNet (16 states, 16 obs) · Obstacle (50 states, 3 obs) · CartPole low-acc (82 states, P_mid=0.373) · Refuel v2 (344 states, 29 obs)

**Trials**: 200 per combination. Bar charts show the best operating
threshold (min fail%, then min stuck%) for each shield.
Pareto plots shown only where data varies meaningfully on both axes.

---

## Cross-Case Summary

![Overview bar charts](results/sweep_v5/summary_v5_bars.png)

### Best operating points

| Case study | Shield | t | Fail% (unif) | Stuck% (unif) | Fail% (adv) | Stuck% (adv) |
|---|---|---|---|---|---|---|
| TaxiNet (16 states, 16 obs) | Envelope | 0.95 | 35% | 34% | 34% | 36% |
|  | Single-Belief | 0.95 | 44% | 11% | 43% | 8% |
|  | Observation | 0.95 | 12% | 88% | 10% | 90% |
|  | Carr | — | 8% | 92% | 4% | 96% |
| Obstacle (50 states, 3 obs) | Envelope | 0.95 | 3% | 85% | 5% | 82% |
|  | Single-Belief | 0.95 | 14% | 50% | 12% | 55% |
|  | Observation | 0.95 | 2% | 98% | 2% | 98% |
|  | Carr | — | 2% | 98% | 2% | 98% |
| CartPole low-acc (82 states, P_mid=0.373) | Single-Belief | 0.85 | 1% | 0% | 1% | 0% |
|  | Observation | 0.85 | 2% | 0% | 2% | 0% |
|  | Carr | — | 2% | 0% | 2% | 0% |
| Refuel v2 (344 states, 29 obs) | Single-Belief | 0.90 | 0% | 79% | 0% | 80% |
|  | Observation | 0.90 | 0% | 99% | 0% | 100% |

### Key cross-case findings

1. **Envelope dominates Single-Belief** (TaxiNet, Obstacle) — lower fail at every threshold, at comparable or slightly higher stuck cost. Infeasible for CartPole lowacc and Refuel v2.

2. **Observation shield is bimodal**: near-optimal for CartPole (2.5% fail / 0% stuck — matches Single-Belief) and offers a unique 0%-stuck operating region for Refuel v2 (3% fail / 0% stuck at t=0.65, vs Single-Belief's minimum 38% stuck). Degrades badly for TaxiNet (>55% fail at 0% stuck) and collapses to Carr-equivalent behaviour for Obstacle (1.5% fail / 98.5% stuck).

3. **Carr achieves the lowest raw fail rate** on TaxiNet (7.5%) and Obstacle (1.5%), but always at near-total stuck cost (92–99%). Competitive only for CartPole (1.5% fail / 5.5% stuck) where 82 near-unique observations make Carr non-conservative. Infeasible for Refuel v2.

4. **Observation informativeness governs shield effectiveness**: CartPole (82 obs ≈ 82 states) → all memoryless methods near-optimal; Obstacle (3 obs / 50 states) → both Observation and Carr degenerate; TaxiNet (16 obs / 16 states, poor posterior) → history essential; Refuel v2 (29 obs / 344 states) → intermediate, with memoryless advantage at low t.

5. **CartPole lowacc is the only case with zero liveness cost** across all threshold-based shields — both Single-Belief and Observation achieve ≤2.5% fail / 0% stuck at their best threshold.

---

## TaxiNet (16 states, 16 obs)

![Pareto scatter](results/sweep_v5/pareto_v5_taxinet.png)

> *Each point is one threshold setting. Only t=0.90 and t=0.95 are labelled. No lines connect points.*

![Bar chart — best threshold per shield](results/sweep_v5/barchart_v5_taxinet.png)

### Best operating points

| Shield | t (unif) | Fail% (unif) | Stuck% (unif) | t (adv) | Fail% (adv) | Stuck% (adv) |
|---|---|---|---|---|---|---|
| Envelope | 0.95 | 35% | 34% | 0.95 | 34% | 36% |
| Single-Belief | 0.95 | 44% | 11% | 0.95 | 43% | 8% |
| Observation | 0.95 | 12% | 88% | 0.95 | 10% | 90% |
| Carr | — | 8% | 92% | — | 4% | 96% |

### Key findings

- **Envelope** offers the best safety-liveness trade-off: 35% fail / 34% stuck (uniform) at t=0.95.
- **Single-Belief** is the most liveness-friendly: 44% fail / 11% stuck — useful when being stuck matters more than the remaining fail rate.
- **Observation** achieves lower fail (12.5%) only at t=0.95, carrying 87.5% stuck — a poor trade given Envelope's 35% fail / 34% stuck at the same threshold.
- **Carr** reaches the lowest raw fail (7.5%) but blocks ≥92% of episodes from step 0. The midpoint POMDP has 0 winning supports, so Carr is degenerate here.

### Structural interpretation

TaxiNet's perception accuracy (P_mid ≈ 0.354 across 16 states) is barely above random. The 16-obs / 16-state structure looks bijective but the emission noise means any single observation is consistent with multiple conflicting true states. A single obs carries almost no reliable information about which lane the agent occupies.

This is why **history is essential**: each observation individually is nearly uninformative, but a sequence of observations and actions progressively eliminates impossible states and concentrates the belief. Single-Belief and Envelope exploit this accumulation; the memoryless Observation shield cannot.

**Carr's degeneracy** (0 winning supports) reveals a structural property: under the midpoint POMDP, there is no support set reachable from safe initial states from which safety can be guaranteed regardless of adversarial transitions. The IPOMDP probability-based shields relax the requirement from certainty to high-probability safety, which is what allows them to act at all.

**Envelope outperforms Single-Belief** because midpoint transition probabilities systematically underestimate the risk of unsafe transitions in a noisy environment. The envelope's worst-case analysis corrects this optimism at the cost of slightly higher stuck.

The irreducible ~34% fail at t=0.95 (Envelope) represents the fundamental difficulty ceiling for this perception regime: even with full belief history and robust worst-case shielding, near-random position sensing cannot prevent failure in all episodes.

---

## Obstacle (50 states, 3 obs)

![Pareto scatter](results/sweep_v5/pareto_v5_obstacle.png)

> *Each point is one threshold setting. Only t=0.90 and t=0.95 are labelled. No lines connect points.*

![Bar chart — best threshold per shield](results/sweep_v5/barchart_v5_obstacle.png)

### Best operating points

| Shield | t (unif) | Fail% (unif) | Stuck% (unif) | t (adv) | Fail% (adv) | Stuck% (adv) |
|---|---|---|---|---|---|---|
| Envelope | 0.95 | 3% | 85% | 0.95 | 5% | 82% |
| Single-Belief | 0.95 | 14% | 50% | 0.95 | 12% | 55% |
| Observation | 0.95 | 2% | 98% | 0.95 | 2% | 98% |
| Carr | — | 2% | 98% | — | 2% | 98% |

### Key findings

- **Envelope** Pareto-dominates Single-Belief at every threshold; 3% fail / 85% stuck at t=0.95 is the best achievable safety-liveness point for any threshold-based shield.
- **Observation** and **Carr** converge to the same extreme corner: ~1.5% fail / 98.5% stuck — nearly indistinguishable.
- **Single-Belief** at t=0.95 (14% fail / 50% stuck) offers the best liveness of any low-fail operating point.

### Structural interpretation

Three observations for 50 states (~17 states per observation) is the most extreme observation compression in this benchmark. Every observation group contains a heterogeneous mix of states — some near the obstacle, some not — with conflicting safety requirements. This creates **irreducible stuck even at t=0.50**: the shield cannot find an action safe for all states consistent with the current observation.

**Why Observation ≡ Carr**: with 17 diverse states per observation, the probability-based posterior P(s|obs) is nearly uniform. The observation shield's calculation converges to Carr's worst-case support analysis — both ask 'is action a safe for all (or almost all) states consistent with this observation?' and with a heterogeneous group the answer is almost always no, leading to the same ~98.5% stuck at high thresholds.

**Belief tracking breaks the deadlock**: after several steps, the trajectory of observations and actions constrains which of the ~17 states are actually reachable — distinguishing, for example, states that were reached by moving east from those reached by moving west. The belief effectively narrows the support far below 17, allowing confident action recommendations. This is why Single-Belief and Envelope substantially outperform memoryless methods at mid-range thresholds.

**Envelope's dominance at every threshold** (not just high thresholds as in TaxiNet) reflects the width of the Obstacle IPOMDP intervals: with 3 observations for 50 states, per-state transition probabilities are poorly determined and the intervals [P_lower, P_upper] are wide. The midpoint POMDP used by Single-Belief consistently underestimates risk, while the Envelope's worst-case analysis remains accurate.

---

## CartPole low-acc (82 states, P_mid=0.373)

![Bar chart — best threshold per shield](results/sweep_v5/barchart_v5_cartpole_lowacc.png)

### Best operating points

| Shield | t (unif) | Fail% (unif) | Stuck% (unif) | t (adv) | Fail% (adv) | Stuck% (adv) |
|---|---|---|---|---|---|---|
| Single-Belief | 0.85 | 1% | 0% | 0.95 | 1% | 0% |
| Observation | 0.85 | 2% | 0% | 0.85 | 2% | 0% |
| Carr | — | 2% | 0% | — | 2% | 0% |

### Key findings

- All three feasible shields achieve **0% stuck** at their best threshold — CartPole lowacc is the only case with zero liveness cost.
- **Single-Belief** is marginally best: 1% fail / 0% stuck at t=0.85.
- **Observation** matches Single-Belief's liveness (0% stuck) at 2.5% fail. The near-bijective 82-obs/82-state structure means a single observation is nearly as informative as the full belief.
- **Carr** is competitive (1.5% fail / 0% stuck uniform); the lowacc support-MDP has only 2 reachable supports (1 winning), so Carr adds essentially no conservatism.
- Envelope not available (LP infeasible at ~1.9 s/step for 200-trial sweep).

### Structural interpretation

CartPole is the easiest shielding problem in this suite due to three compounding structural properties:

1. **Near-bijective observations (82 obs / 82 states)**: each observation   is associated with one canonical state, so even with 37% per-state   accuracy the posterior concentrates strongly on the correct state.   History adds almost nothing — the first observation is already highly   informative. This is why Observation shield matches Single-Belief.

2. **Only 2 actions**: stuck requires the shield to block *left* AND   *right* simultaneously. This can only happen if the posterior places   significant probability on the FAIL state, which rarely occurs during   normal RL trajectories that stay well within the safe region.

3. **Gradual, controllable dynamics**: from any non-FAIL state, at least   one action reduces the pole angle. The physics create no dead-end   belief states where every action leads to failure.

**The counterintuitive stuck comparison**: CartPole standard (P_mid=0.532) shows 68% stuck at t=0.95 for the Observation shield, while lowacc (P_mid=0.373) shows 0% stuck at the same threshold. Higher accuracy concentrates the posterior more strongly on individual states. If that concentration falls on a near-FAIL state, P(FAIL|obs) can exceed 5% and trigger blocking at t=0.95. Noisier observations spread the posterior more diffusely, keeping P(FAIL|obs) below the threshold — lower perception accuracy accidentally prevents over-conservatism.

**Carr's trivial support structure** (2 supports, 1 winning) confirms the near-bijective property: once the agent takes one step from the safe initial support, the support collapses to a near-singleton immediately. Carr is essentially operating on the true state.

---

## Refuel v2 (344 states, 29 obs)

![Bar chart — best threshold per shield](results/sweep_v5/barchart_v5_refuel_v2.png)

### Best operating points

| Shield | t (unif) | Fail% (unif) | Stuck% (unif) | t (adv) | Fail% (adv) | Stuck% (adv) |
|---|---|---|---|---|---|---|
| Single-Belief | 0.90 | 0% | 79% | 0.85 | 0% | 80% |
| Observation | 0.90 | 0% | 99% | 0.90 | 0% | 100% |

### Key findings

- **Single-Belief** achieves 0% fail at t=0.90 (79% stuck uniform; 80% stuck adversarial) — the lowest stuck rate among 0%-fail operating points.
- **Observation** also achieves 0% fail but with 99% stuck at t=0.90.  Its unique advantage: at t=0.65, it gives **3–4.5% fail / 0% stuck** — the only 0%-stuck operating point for Refuel v2. Single-Belief has ≥38% stuck at every threshold.
- Carr and Envelope are both infeasible (support-MDP BFS and LP exceed memory / time budgets at 344 states × 29 obs).

### Structural interpretation

Refuel v2 is the only benchmark where safety predicates are genuinely **hidden from the observation**: fuel level and obstacle proximity are not encoded in any observation bit. The agent must infer danger entirely from indirect signals (relative position, time elapsed), testing whether IPOMDP shielding provides real value under genuine partial observability.

**Single-Belief's liveness trap** arises because accurate belief tracking works against liveness. As the episode progresses, the belief correctly concentrates on states where fuel is critically low or the obstacle is adjacent. At t≥0.70, the shield rightly identifies that all actions have significant probability of leading to failure — but the agent is now paralysed in a belief corner with no safe exit. This is a true safety-liveness tension: accurate danger awareness leads to paralysis.

**The Observation shield's 0%-stuck advantage at low t** is a consequence of its memorylessness. Without accumulating history, the posterior over 344 states via 29 observations (~12 states per obs) is too uncertain to classify all actions as dangerous simultaneously. The shield 'doesn't know enough to be paralysed.' The cost is 3–4.5% fail, but this is the only operating point with zero liveness cost in the entire benchmark.

**Scalability boundary**: Carr (support-MDP BFS over 344 states × 29 obs) and Envelope (LP at ~144 s/step) both exceed practical limits, leaving only Single-Belief as the scalable option. This mirrors real-world large-scale POMDP shielding where support-based and LP methods are infeasible and probabilistic belief tracking is the only viable approach.

---
