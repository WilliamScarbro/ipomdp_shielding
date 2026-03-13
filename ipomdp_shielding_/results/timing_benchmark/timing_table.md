| Shield | TaxiNet | Obstacle | CartPole low-acc | Refuel v2 |
|---|---|---|---|---|
| No Shield | 0.1±0.1 μs (p95=0.1) | 0.1±0.3 μs (p95=0.1) | 0.1±0.0 μs (p95=0.1) | 0.1±0.0 μs (p95=0.1) |
| Observation | 1.6±0.8 μs (p95=1.7) | 18.1±0.7 μs (p95=18.5) | 2.0±3.5 μs (p95=14.4) | 0.1±0.0 ms (p95=0.1) |
| Single-Belief | 9.6±2.3 μs (p95=9.7) | 31.2±4.0 μs (p95=36.7) | 32.4±11.2 μs (p95=67.3) | 0.2±0.0 ms (p95=0.2) |
| Fwd-Sampling | 0.3±0.0 ms (p95=0.3) | 0.6±0.0 ms (p95=0.6) | 1.3±0.2 ms (p95=1.9) | 5.6±0.2 ms (p95=5.6) |
| Envelope | 83.1±4.5 ms (p95=94.3) | 646.2±78.5 ms (p95=745.4) | — | — |
| Carr | 1.0±1.4 μs (p95=1.1) | 8.1±4.8 μs (p95=13.0) | 3.2±10.3 μs (p95=37.5) | — |
*Timing: mean ± std (p95) per step, 300 steps × threshold=0.90, random-walk trajectories. Envelope: 30 steps (LP-based). Units: μs = microseconds, ms = milliseconds. — = infeasible.*