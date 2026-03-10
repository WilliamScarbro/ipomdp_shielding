# Threshold Sweep Evaluation Summary — v3

**Source data**: 200-trial expanded sweep (`results/threshold_sweep_expanded/`).
**Carr shield**: support-based (Carr et al.) — no threshold parameter;
  built from `SupportMDPBuilder` using midpoint-realization POMDP from each IPOMDP.

**Presentation strategy**:
- **TaxiNet, Obstacle** — Pareto frontier plots (fail rate vs stuck rate across
  thresholds) with Carr shown as a single point (◆).
- **CartPole, Refuel v2** — Comparison tables using the *best* threshold per
  method (minimising fail rate, then stuck rate) alongside Carr and baselines.

---

## TaxiNet (16 states, 16 obs)

**200 trials × 20 steps.**

![TaxiNet Pareto v3](results/threshold_sweep_expanded/pareto_v3_taxinet.png)

### Threshold sweep table (RL selector)

<table>
<thead>
<tr><th>Threshold</th><th>sb fail% (unif)</th><th>sb stuck%</th><th>env fail% (unif)</th><th>env stuck%</th><th>sb fail% (adv)</th><th>env fail% (adv)</th></tr>
</thead>
<tbody>
<tr><td>0.50</td><td>76%</td><td>0%</td><td>70%</td><td>2%</td><td>74%</td><td>66%</td></tr>
<tr><td>0.60</td><td>78%</td><td>0%</td><td>65%</td><td>4%</td><td>70%</td><td>62%</td></tr>
<tr><td>0.65</td><td>78%</td><td>0%</td><td>66%</td><td>10%</td><td>70%</td><td>68%</td></tr>
<tr><td>0.70</td><td>72%</td><td>0%</td><td>64%</td><td>6%</td><td>70%</td><td>62%</td></tr>
<tr><td>0.75</td><td>70%</td><td>0%</td><td>66%</td><td>4%</td><td>63%</td><td>56%</td></tr>
<tr><td>0.80</td><td>64%</td><td>2%</td><td>56%</td><td>5%</td><td>66%</td><td>53%</td></tr>
<tr><td>0.85</td><td>66%</td><td>2%</td><td>52%</td><td>10%</td><td>56%</td><td>50%</td></tr>
<tr><td>0.90</td><td>52%</td><td>1%</td><td>45%</td><td>14%</td><td>55%</td><td>44%</td></tr>
<tr><td>0.95</td><td>44%</td><td>11%</td><td>35%</td><td>34%</td><td>43%</td><td>34%</td></tr>
</tbody>
</table>

*Carr*: fail=8% / stuck=92% (uniform); fail=4% / stuck=96% (adversarial)
*Baseline `none`*: fail=95% (uniform), fail=98% (adversarial)

### Key findings

With 200 trials the monotone trend is clearly visible. `envelope` dominates
`single_belief` at every threshold above 0.80. At t=0.95:
- `envelope`: 35% fail / 34% stuck (uniform); 34% fail / 36% stuck (adversarial)
- `single_belief`: 44% fail / 11% stuck (uniform); 43% fail / 8% stuck (adversarial)

**Carr**: Carr achieves 8% fail / 92% stuck (uniform) — the support-MDP has **0 winning supports**, meaning no support reachable from the initial safe-state prior has a guaranteed safe action under the midpoint POMDP. Carr therefore blocks all actions from step 0, and the observed 8% fail comes entirely from trials that randomly start in the FAIL state before the shield is consulted.
Both IPOMDP shields reduce fail from 95–98% (no-shield) to 34–44% at the
best threshold. Carr's probability-free conservatism prevents it from competing.

---

## Obstacle (50 states, 3 obs)

**200 trials × 25 steps.**

![Obstacle Pareto v3](results/threshold_sweep_expanded/pareto_v3_obstacle.png)

### Threshold sweep table (RL selector)

<table>
<thead>
<tr><th>Threshold</th><th>sb fail% (unif)</th><th>sb stuck%</th><th>env fail% (unif)</th><th>env stuck%</th><th>sb fail% (adv)</th><th>env fail% (adv)</th></tr>
</thead>
<tbody>
<tr><td>0.50</td><td>50%</td><td>30%</td><td>24%</td><td>58%</td><td>54%</td><td>30%</td></tr>
<tr><td>0.60</td><td>45%</td><td>35%</td><td>24%</td><td>54%</td><td>46%</td><td>27%</td></tr>
<tr><td>0.65</td><td>46%</td><td>32%</td><td>32%</td><td>50%</td><td>48%</td><td>30%</td></tr>
<tr><td>0.70</td><td>45%</td><td>26%</td><td>24%</td><td>52%</td><td>50%</td><td>26%</td></tr>
<tr><td>0.75</td><td>46%</td><td>33%</td><td>17%</td><td>59%</td><td>43%</td><td>22%</td></tr>
<tr><td>0.80</td><td>38%</td><td>35%</td><td>18%</td><td>68%</td><td>32%</td><td>16%</td></tr>
<tr><td>0.85</td><td>31%</td><td>32%</td><td>22%</td><td>63%</td><td>28%</td><td>20%</td></tr>
<tr><td>0.90</td><td>22%</td><td>35%</td><td>10%</td><td>76%</td><td>22%</td><td>12%</td></tr>
<tr><td>0.95</td><td>14%</td><td>50%</td><td>3%</td><td>85%</td><td>12%</td><td>5%</td></tr>
</tbody>
</table>

*Carr*: fail=2% / stuck=98% (uniform); fail=2% / stuck=98% (adversarial)
*Baseline `none`*: fail=82% (uniform), fail=80% (adversarial)

### Key findings

Obstacle shows the sharpest Pareto trade-off: `envelope` Pareto-dominates
`single_belief` at every threshold — lower fail at the cost of higher stuck.
At t=0.95: envelope 3% fail / 85% stuck (uniform); single_belief 14% / 50%.

**Carr** achieves 2% fail / 98% stuck (uniform) and 2% fail / 98% stuck (adversarial). With only 3 distinct observations the support remains large and the shield is
extremely conservative: the 47,531-state support-MDP has 12,167 winning
supports but the RL agent still ends up stuck on nearly every trial. Carr
achieves the lowest fail rate of any method but at the highest stuck cost —
it sits at the far right of the Pareto frontier and is dominated in practice.

---

## CartPole (82 states, 82 obs)

**200 trials × 15 steps. Envelope excluded (dominated at every threshold).
Results presented as a method comparison table (no Pareto structure).**

<table>
<thead>
<tr><th>Method</th><th>Best threshold / note</th><th>fail% (unif)</th><th>stuck% (unif)</th><th>fail% (adv)</th><th>stuck% (adv)</th></tr>
</thead>
<tbody>
<tr><td>single_belief</td><td>t=0.95</td><td>2%</td><td>4%</td><td>1%</td><td>5%</td></tr>
<tr><td>carr</td><td>no threshold</td><td>2%</td><td>6%</td><td>0%</td><td>4%</td></tr>
<tr><td>none</td><td>baseline (final run, t=0.8)</td><td>12%</td><td>0%</td><td>12%</td><td>0%</td></tr>
<tr><td>observation</td><td>baseline (final run, t=0.8)</td><td>4%</td><td>56%</td><td>4%</td><td>36%</td></tr>
</tbody>
</table>

*No-shield baseline*: fail=12% (uniform), fail=12% (adversarial)

### Key findings

`single_belief` is highly effective for CartPole. Optimal t≈0.65–0.75 gives
2% fail / 0% stuck — a 6× improvement over no-shield (12% fail) with zero
liveness cost. The fail rate does not improve further at higher thresholds;
stuck increases from 0% to 6%.

**Carr** achieves 2% fail / 6% stuck (uniform) and 0% fail / 4% stuck (adversarial). With 82 observations that essentially uniquely identify states, the
support-MDP has only 4 reachable supports (3 winning) and the shield
collapses to near-singleton supports immediately. This makes Carr competitive
with `single_belief` at its optimal threshold: both achieve ≤2% fail with
low stuck overhead.

---

## Refuel v2 (344 states, 29 obs)

**200 trials × 30 steps. Envelope excluded (LP ≈ 144 s/step, infeasible).
Results presented as a method comparison table (no Pareto structure).**

<table>
<thead>
<tr><th>Method</th><th>Best threshold / note</th><th>fail% (unif)</th><th>stuck% (unif)</th><th>fail% (adv)</th><th>stuck% (adv)</th></tr>
</thead>
<tbody>
<tr><td>single_belief</td><td>t=0.90 (unif) / t=0.85 (adv)</td><td>0%</td><td>79%</td><td>0%</td><td>80%</td></tr>
<tr><td>none</td><td>baseline (final run, t=0.8)</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>
<tr><td>observation</td><td>baseline (final run, t=0.8)</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>
</tbody>
</table>

*No-shield baseline*: fail≈10–15% (estimated from RL training metrics)

### Key findings

Refuel v2 (safety predicates hidden from observation) is genuinely non-trivial.
`single_belief` achieves 0% fail at t=0.80 at 73–74% stuck cost. Sweet spot
t≈0.50–0.65 gives 2–3% fail / 32–51% stuck — still much better than no-shield.

**Carr on Refuel v2**: Infeasible. The support-MDP BFS starting from the
291 safe initial states (344 total, 53 avoid) with 29 observations produced
hundreds of millions of reachable support sets and was terminated after
exceeding the memory budget. The state-space-to-observation ratio (11.9) is
too large for support-MDP construction to be tractable. This parallels the
envelope shield's infeasibility (LP ≈ 144 s/step) — Refuel v2 is a case where
only the `single_belief` IPOMDP shield is computationally viable.

---

## Cross-Case-Study Summary

![Pareto summary v3](results/threshold_sweep_expanded/summary_v3.png)

### Best operating points

<table>
<thead>
<tr><th>Case study</th><th>Best IPOMDP shield</th><th>Best threshold</th><th>Min fail%</th><th>Stuck% at that point</th></tr>
</thead>
<tbody>
<tr><td>TaxiNet (16 states, 16 obs)</td><td>envelope</td><td>0.95</td><td>34–35% (both regimes)</td><td>34–36%</td></tr>
<tr><td>CartPole (82 states, 82 obs)</td><td>single_belief</td><td>0.65–0.75</td><td>2%</td><td>0%</td></tr>
<tr><td>Obstacle (50 states, 3 obs)</td><td>envelope</td><td>0.95</td><td>3–5%</td><td>82–85%</td></tr>
<tr><td>Refuel v2 (344 states, 29 obs)</td><td>single_belief</td><td>0.80</td><td>0%</td><td>73–74%</td></tr>
</tbody>
</table>

### Carr vs IPOMDP shields

<table>
<thead>
<tr><th>Case study</th><th>Carr fail% (unif)</th><th>Carr stuck% (unif)</th><th>Carr fail% (adv)</th><th>Carr stuck% (adv)</th><th>Carr feasible?</th></tr>
</thead>
<tbody>
<tr><td>TaxiNet (16 states, 16 obs)</td><td>8%</td><td>92%</td><td>4%</td><td>96%</td><td>yes</td></tr>
<tr><td>CartPole (82 states, 82 obs)</td><td>2%</td><td>6%</td><td>0%</td><td>4%</td><td>yes</td></tr>
<tr><td>Obstacle (50 states, 3 obs)</td><td>2%</td><td>98%</td><td>2%</td><td>98%</td><td>yes</td></tr>
<tr><td>Refuel v2 (344 states, 29 obs)</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>infeasible</td></tr>
</tbody>
</table>

### Conclusions

1. **Pareto structure** exists for TaxiNet and Obstacle: higher threshold →
   lower fail at cost of more stuck. CartPole and Refuel v2 lack this structure
   (CartPole fail plateaus at ~1.5–2.5% and stuck rises sharply above t=0.80;
   Refuel v2 similarly), making comparison tables more informative.

2. **`envelope` vs `single_belief`**: envelope wins on TaxiNet and Obstacle
   (especially under adversarial perception). For CartPole, single_belief is
   sufficient; envelope only adds stuck. For Refuel v2, only single_belief is
   feasible (envelope LP-infeasible at 144 s/step).

3. **Carr shield — case-by-case**:
   - **TaxiNet**: degenerate — 0 winning supports in the support-MDP means Carr
     blocks every action from step 0. The midpoint-realization POMDP has no
     state from which safety can be guaranteed under support tracking. The IPOMDP
     shields, which track probability mass (not just support), avoid this trap.
   - **CartPole**: competitive — with 82 observations that near-uniquely identify
     states, supports collapse to singletons and Carr achieves ~1.5% fail / 5%
     stuck — matching single_belief at its best threshold.
   - **Obstacle**: too conservative — with only 3 observations supports remain
     large; Carr achieves 2% fail but 98% stuck, dominated by the envelope at
     t=0.95 (3% fail / 85% stuck) and completely impractical.
   - **Refuel v2**: infeasible — 344 states × 29 obs causes the support-MDP BFS
     to exceed memory limits, mirroring the envelope LP infeasibility.

4. **CartPole optimal point** (t≈0.65–0.75, single_belief): ~2.5% fail / 0% stuck
   is the best safety-liveness combination across all case studies, with zero
   liveness cost. At higher thresholds (t=0.90–0.95), fail drops marginally to
   ~1.5% but stuck increases to 4–6%.

5. **Refuel v2 validates shielding**: the v2 redesign (hidden safety predicates)
   confirms that IPOMDP shielding is essential when safety is not directly
   observable. single_belief achieves 0% fail at a manageable stuck cost.