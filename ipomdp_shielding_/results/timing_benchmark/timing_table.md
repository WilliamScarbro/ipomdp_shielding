| Shield | TaxiNet | Obstacle | CartPole low-acc | Refuel v2 |
|---|---|---|---|---|
| No Shield | 0.1±0.1 μs (p95=0.1) | 0.1±0.0 μs (p95=0.1) | 0.1±0.0 μs (p95=0.2) | 0.1±0.1 μs (p95=0.2) |
| Observation | 1.6±0.9 μs (p95=1.6) | 17.9±3.0 μs (p95=18.4) | 2.0±3.6 μs (p95=14.6) | 0.1±0.0 ms (p95=0.1) |
| Single-Belief | 9.4±2.3 μs (p95=9.7) | 31.6±4.1 μs (p95=37.1) | 32.0±11.8 μs (p95=69.8) | 0.2±0.0 ms (p95=0.2) |
| Fwd-Sampling | 21.2±0.2 ms (p95=21.4) | 67.0±0.9 ms (p95=68.4) | 7.1±6.2 ms (p95=29.9) | 383.9±31.1 ms (p95=397.5) |
| Envelope | 83.1±4.5 ms (p95=93.6) | 643.1±79.0 ms (p95=744.3) | — | — |
| Carr | 0.9±1.3 μs (p95=0.9) | 7.9±5.0 μs (p95=12.9) | 3.2±10.5 μs (p95=38.7) | — |
*Timing: mean ± std (p95) per step, 300 steps × threshold=0.90, random-walk trajectories. Envelope: 30 steps (LP-based). Units: μs = microseconds, ms = milliseconds. — = infeasible.*