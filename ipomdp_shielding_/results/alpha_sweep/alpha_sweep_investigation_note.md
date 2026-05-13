# Alpha Sweep Investigation Note

Date: 2026-05-13

## Question

The alpha sweep results are concerning because the safety values do not appear
to change much with alpha. The working hypothesis was that the Clopper-Pearson
interval widths might not be changing meaningfully, or that alpha might not be
reaching the shielding computation.

## What I Checked

Alpha is wired through the experiment path:

- `experiments/sweeps/rl_alpha_sweep.py` passes `alpha` into
  `build_taxinet_ipomdp`.
- `CaseStudies/Taxinet/taxinet.py` passes that alpha into the
  Clopper-Pearson interval construction for the HE and CTE perception IMDPs.
- The runtime shield thresholds decisions through the propagated belief via
  `RuntimeImpShield.next_actions`.

The perfect-perception safety shield does not depend on alpha. It is built only
from deterministic state/action safety:

```python
dyn_shield = {
    s: {a for a in actions if taxinet_safe(taxinet_next_state(s, a))}
    for s in states
}
```

So alpha can only affect runtime behavior through observation intervals,
midpoints, interval belief propagation, sampled perception realizations, and
whether the resulting action-safety probabilities cross the beta threshold.

## Interval Width Check

With a fixed train/test split seed, the TaxiNet observation interval widths do
change monotonically as alpha increases:

| alpha | mean width | median width | p90 width | max width |
|---:|---:|---:|---:|---:|
| 0.01 | 0.021342 | 0.005005 | 0.067796 | 0.149177 |
| 0.025 | 0.018571 | 0.004343 | 0.059114 | 0.130422 |
| 0.05 | 0.016255 | 0.003704 | 0.051802 | 0.114579 |
| 0.075 | 0.014784 | 0.003360 | 0.047137 | 0.104451 |
| 0.10 | 0.013677 | 0.003122 | 0.043611 | 0.096787 |
| 0.15 | 0.012005 | 0.002754 | 0.038272 | 0.085167 |
| 0.20 | 0.010724 | 0.002460 | 0.034162 | 0.076213 |
| 0.30 | 0.008744 | 0.002012 | 0.027785 | 0.062302 |

Conclusion: the intervals are changing. The change is large in relative terms,
but the absolute widths are small after product construction.

## Midpoint And Decision Sensitivity

The midpoint observation model barely moves. In the same fixed-split check, the
maximum midpoint probability difference between alpha `0.01` and alpha `0.30`
was about `0.0053`.

A first-step allowed-action sensitivity check showed that these changes mostly
do not cross shield thresholds:

- beta `0.7`: zero allowed-action signature changes across alpha values.
- beta `0.8`: zero allowed-action signature changes across alpha values.
- beta `0.9`: one allowed-action signature change across alpha values.

This explains why safety values look flat: the shield behavior is thresholded,
and most alpha-induced probability shifts are too small to change the allowed
action set.

## Additional Concern

The current alpha sweep is not fully controlled because `build_taxinet_ipomdp`
supports a split seed, but the sweep config does not pass one. Each alpha can
therefore rebuild the IPOMDP from a different random train/test split. That can
hide or confound the already-small alpha effect.

## Current Assessment

The flat safety values do not appear to be caused by alpha being ignored.
Instead, the likely explanation is:

1. Alpha changes interval widths.
2. The resulting absolute changes in observation probabilities are small.
3. The perfect-perception shield is independent of alpha.
4. Runtime decisions only change when action-safety probabilities cross beta.
5. Most cells are threshold-stable, especially at beta `0.7` and `0.8`.
6. The Monte Carlo budget is low: the saved sweep uses 20 trials per seed and
   100 pooled trials per cell, so binomial noise is comparable to the apparent
   alpha effect.
7. Random train/test splits across alpha may further confound the trend.

## Recommended Next Investigation

We need more compute resources to investigate this properly. The next run
should:

1. Fix the TaxiNet train/test split seed across all alpha values.
2. Save interval diagnostics per alpha: mean, median, p90, max width, and
   midpoint deltas.
3. Save shield-decision diagnostics per alpha and beta: per-step action-safety
   probabilities, margins to beta, and allowed-action set changes.
4. Increase Monte Carlo trials per cell substantially, ideally to at least
   1,000 pooled trials per `(alpha, beta, perception, shield)` cell.
5. Consider a denser beta sweep around the observed decision boundary, because
   alpha effects only become visible when beta lies near a crossing point.
6. Re-optimize adversarial realizations at higher fidelity rather than relying
   on low-budget cached adversaries.

Until that controlled higher-budget run is available, the safest statement is:
alpha affects the interval model, but the current experiment is underpowered
and mostly threshold-stable, so the observed safety metrics are not expected to
move much.
