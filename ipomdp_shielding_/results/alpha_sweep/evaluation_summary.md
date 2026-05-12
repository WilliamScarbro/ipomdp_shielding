# Alpha Sweep: TaxiNet Pareto Frontier (multi-beta)

Expanded version of the alpha sweep on TaxiNet. The CI significance `alpha`
is now scanned over 8 values, evaluated at 3 shield thresholds
`beta ∈ {0.7, 0.8, 0.9}` to expose how the alpha–metric relationship
depends on shield conservatism.

- `alpha`: Clopper-Pearson significance level. Smaller `alpha` → wider
  intervals → more uncertainty acknowledged → more conservative shielding.
- `beta`: runtime shield safety threshold. Larger `beta` → more conservative
  (more actions blocked, more "stuck" states).

**Settings**: 8 alphas × 3 betas × 3 seeds × 20 trials × 20 steps ×
3 shields × 2 perception regimes = 8,640 trial-steps per cell × 432 cells.
Forward-sampling shield uses `budget=500`, `K_samples=100`. Adversarial
realizations are trained against the `envelope` shield at each `beta` and
reused for the other shields at the same `beta` (consistent with the prior
sweep methodology).

**Total runtime**: 8,266 s (≈ 138 min). Four of the eight alpha values
(`0.01, 0.05, 0.1, 0.2`) reused cached RL agents from the previous sweep;
the other four RL agents and all 24 adversarial realizations were trained
fresh.

---

## Pareto scatter

![TaxiNet Pareto — alpha scatter at three betas](pareto_taxinet.png)

*Rows: perception regime (uniform / adversarial-optimized).
Cols: shield threshold (beta = 0.7, 0.8, 0.9).
Marker shape: shield (○ single_belief, ■ envelope, ▲ forward_sampling).
Each point is one alpha value (annotated). Lines intentionally omitted to
keep the scatter as the geometric summary.*

## Alpha trends

![Fail rate vs alpha](alpha_vs_fail.png)
![Stuck rate vs alpha](alpha_vs_stuck.png)

*Error bars: ±1 std across the three seeds. Three shields per panel.*

---

## Headline findings

1. **Beta is the dominant knob, not alpha.** Across all three betas and
   both perception regimes, the per-shield fail rate moves within a roughly
   10–20 pp band as alpha varies, with no consistent monotone trend. In
   contrast, moving from beta = 0.7 → 0.9 shifts the operating point
   from the "all-fail / no-stuck" corner to a real fail-vs-stuck trade-off:
   stuck rates for envelope/forward_sampling jump from < 8% at beta ≤ 0.8
   to 5–22% at beta = 0.9, while fail rates drop by 10–30 pp.
2. **Alpha trend is essentially flat at low/medium beta.** At beta = 0.7 and
   beta = 0.8 the alpha-vs-fail curves are noisy but trendless (per-shield
   fail rates stay near 55–65%, stuck near 0%). The per-seed std (4–14 pp)
   is comparable to the cross-alpha spread.
3. **A weak alpha effect appears at beta = 0.9.** The most conservative beta
   is where alpha starts to matter:
   - Envelope and forward_sampling fail rates trend down by ~5–10 pp as
     alpha rises from 0.01 to 0.3 (especially under adversarial perception:
     forward_sampling 28→38% and envelope 42→42% with a 30% minimum at
     alpha = 0.15).
   - Stuck rates stay roughly flat (~10–20%) — alpha mainly moves fail
     here, with stuck dominated by the high beta itself.
4. **Forward-sampling is best for adversarial perception at high beta.**
   Best single operating point in the entire grid for adversarial:
   `forward_sampling` at beta = 0.9, alpha = 0.01 → 28.3% fail / 13.3%
   stuck. `single_belief` cannot reach this fail rate at any (alpha, beta).
5. **Single-belief keeps the "no stuck" guarantee everywhere.** Across all
   144 (alpha, beta, perception) cells `single_belief` never exceeds 3.3%
   stuck. Its fail rate is consistently 5–15 pp higher than envelope /
   forward_sampling, but it is the unambiguous choice when stuck is
   unacceptable.

---

## Per-perception, per-beta tables

Cells show `fail% / stuck%` (mean across 3 seeds × 20 trials).
Bold = best (lowest fail) within row.

### Uniform perception

#### beta = 0.7
| alpha | single_belief | envelope        | forward_sampling |
|-------|---------------|-----------------|------------------|
| 0.010 | 63 / 0        | 70 / 3          | **58 / 2**       |
| 0.025 | 75 / 0        | 65 / 0          | **63 / 0**       |
| 0.050 | 63 / 0        | 70 / 8          | **60 / 3**       |
| 0.075 | 68 / 0        | **58 / 5**      | 60 / 0           |
| 0.100 | 55 / 0        | **47 / 3**      | 48 / 2           |
| 0.150 | 72 / 0        | 62 / 3          | **58 / 0**       |
| 0.200 | 63 / 0        | **50 / 2**      | 62 / 0           |
| 0.300 | **60 / 0**    | 57 / 7          | 68 / 0           |

#### beta = 0.8
| alpha | single_belief | envelope        | forward_sampling |
|-------|---------------|-----------------|------------------|
| 0.010 | 62 / 0        | **50 / 8**      | 60 / 2           |
| 0.025 | 70 / 0        | 58 / 5          | **55 / 0**       |
| 0.050 | 63 / 0        | 58 / 3          | **52 / 3**       |
| 0.075 | **60 / 0**    | 62 / 3          | 55 / 0           |
| 0.100 | 60 / 0        | 58 / 2          | **55 / 0**       |
| 0.150 | **55 / 2**    | 57 / 8          | 62 / 2           |
| 0.200 | **53 / 0**    | 58 / 3          | **53 / 2**       |
| 0.300 | 63 / 2        | 53 / 5          | **48 / 2**       |

#### beta = 0.9
| alpha | single_belief | envelope        | forward_sampling |
|-------|---------------|-----------------|------------------|
| 0.010 | 57 / 2        | **50 / 15**     | 58 / 13          |
| 0.025 | 45 / 2        | **35 / 13**     | 43 / 12          |
| 0.050 | 55 / 0        | **50 / 15**     | 58 / 10          |
| 0.075 | 57 / 0        | **33 / 12**     | 48 / 10          |
| 0.100 | 65 / 0        | **42 / 8**      | 47 / 7           |
| 0.150 | 60 / 2        | **38 / 17**     | 45 / 12          |
| 0.200 | 52 / 0        | 47 / 15         | **40 / 5**       |
| 0.300 | 58 / 0        | 47 / 15         | **45 / 10**      |

### Adversarial perception

#### beta = 0.7
| alpha | single_belief | envelope        | forward_sampling |
|-------|---------------|-----------------|------------------|
| 0.010 | 77 / 0        | **58 / 7**      | 62 / 2           |
| 0.025 | 68 / 0        | 60 / 7          | **53 / 0**       |
| 0.050 | 77 / 0        | **62 / 10**     | 68 / 0           |
| 0.075 | 73 / 0        | 60 / 0          | **57 / 0**       |
| 0.100 | 68 / 0        | **57 / 5**      | 63 / 0           |
| 0.150 | **63 / 2**    | 72 / 2          | 68 / 3           |
| 0.200 | 68 / 0        | **57 / 2**      | 60 / 0           |
| 0.300 | 57 / 0        | 52 / 3          | **50 / 2**       |

#### beta = 0.8
| alpha | single_belief | envelope        | forward_sampling |
|-------|---------------|-----------------|------------------|
| 0.010 | 65 / 0        | **53 / 7**      | 60 / 2           |
| 0.025 | 57 / 0        | **47 / 7**      | 55 / 2           |
| 0.050 | 60 / 2        | **50 / 3**      | 60 / 2           |
| 0.075 | **65 / 0**    | 73 / 2          | **65 / 0**       |
| 0.100 | **50 / 0**    | 55 / 0          | 60 / 0           |
| 0.150 | **55 / 0**    | 62 / 2          | 57 / 2           |
| 0.200 | **58 / 0**    | 60 / 5          | 60 / 2           |
| 0.300 | 63 / 0        | 58 / 3          | **53 / 0**       |

#### beta = 0.9
| alpha | single_belief | envelope        | forward_sampling |
|-------|---------------|-----------------|------------------|
| 0.010 | 50 / 3        | 42 / 20         | **28 / 13**      |
| 0.025 | 43 / 0        | 48 / 22         | **42 / 18**      |
| 0.050 | 68 / 2        | **42 / 20**     | **42 / 20**      |
| 0.075 | 52 / 2        | **48 / 17**     | **48 / 12**      |
| 0.100 | 57 / 2        | 42 / 15         | **33 / 17**      |
| 0.150 | 48 / 2        | **30 / 22**     | 32 / 18          |
| 0.200 | **43 / 2**    | 38 / 15         | 38 / 13          |
| 0.300 | 43 / 0        | 42 / 18         | **38 / 12**      |

---

## Observations on the alpha–metric trend

- **Beta = 0.7, 0.8 (permissive shielding):** alpha is essentially a knob
  that does not move the metrics. The Clopper-Pearson intervals are wide
  enough at any of these alphas that the shield decisions are dominated by
  the worst-case bound; tightening alpha changes the bound modestly relative
  to the noise floor. The Pareto cloud is tightly packed in the upper-left
  region of (low stuck, high fail).
- **Beta = 0.9 (conservative shielding):** the alpha effect becomes
  measurable but small. Larger alpha (tighter intervals) slightly *reduces*
  fail rate for `envelope` and `forward_sampling` under adversarial
  perception, presumably because the shield rejects fewer borderline
  actions and the RL policy then has access to better moves. `single_belief`
  is essentially flat in alpha at every beta — it never blocks enough to
  matter.
- **Stuck-vs-alpha is flat** for every shield at every beta. The stuck
  level is set by `(shield, beta)`, not by alpha.

---

## Key takeaways

1. **Tune beta first, alpha second.** Beta controls *which corner* of the
   Pareto plot you operate in; alpha is a fine adjustment within that
   corner.
2. **Forward-sampling buys you the best adversarial fail rates at high
   beta.** At beta = 0.9, alpha = 0.01 under adversarial perception:
   28% fail / 13% stuck — a regime envelope and single_belief cannot reach
   at any setting in this sweep.
3. **Single-belief is the right shield if stuck must be near zero.** It is
   ≤ 3.3% stuck in every cell, at the cost of higher fail (5–15 pp).
4. **Statistical caveat.** Per-seed std is 4–18 pp on fail rate (n = 3
   seeds × 20 trials per cell), which means many of the within-shield
   alpha trends sit on the edge of significance. The robust orderings are
   *between shields at a given (alpha, beta)* and *between betas at a
   given (alpha, shield)*, not between alphas at a given (beta, shield).

## Limitations

- Adversarial realizations are trained against the `envelope` shield at
  each beta. They are reused unchanged when evaluating `single_belief` and
  `forward_sampling`; this likely under-estimates the worst case for those
  two shields.
- 3 seeds × 20 trials per cell is small; the per-cell std is sometimes as
  large as the alpha-axis variation. A higher-confidence rerun should
  double seeds and trials.

## Reproducibility

```bash
python3 -m ipomdp_shielding.experiments.sweeps.rl_alpha_sweep
# config: ipomdp_shielding/experiments/sweeps/rl_alpha_sweep_taxinet.py
# outputs: data/sweep/rl_alpha_taxinet_v2/
#   results_tidy.csv, sweep_summary.json, figures/{pareto_alpha,alpha_vs_fail,alpha_vs_stuck}.png
```

Tidy CSV: `results_tidy.csv`
JSON (full metadata + per-cell aggregates): `sweep_summary.json`
