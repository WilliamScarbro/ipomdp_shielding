# Experiments

This directory contains experimental scripts and examples for the IPOMDP shielding framework.

## Available Experiments

### `optimal_realization_example.py`

**ML-Optimized Fixed Interval Realization**

Demonstrates the complete workflow for training and evaluating optimal fixed interval realizations:

1. Train optimal realization against random action selector
2. Train optimal realization against safest action selector
3. Compare with uniform and adversarial perception models
4. Save/load trained models
5. Visualize results

**Usage:**
```bash
python3 experiments/optimal_realization_example.py
```

**What it demonstrates:**
- Using `train_optimal_realization()` high-level API
- Creating runtime shield factories
- Comparing different perception models
- Save/load functionality for trained models
- Evaluating with `MonteCarloSafetyEvaluator`

**Configuration:**
The example uses a minimal configuration for demonstration (fast runtime):
- 20 candidates per iteration
- 10 trials per candidate
- 20 iterations

For production use, increase these values (see documentation).

**Expected output:**
```
============================================================
TRAINING OPTIMAL REALIZATION VS RANDOM ACTION SELECTOR
============================================================
Parameter shape: (15, 16)
Candidates per iteration: 20
Trials per candidate: 10
Max iterations: 20
============================================================
Iteration 1/20: best=0.1000, mean=0.0500, elite_mean=0.0800, std=0.2850
Iteration 2/20: best=0.1500, mean=0.0700, elite_mean=0.1200, std=0.2708
...
Training complete! Best score: 0.2500

============================================================
COMPARING PERCEPTION MODELS
============================================================
UNIFORM perception:
  Fail rate: 5.00%
  Stuck rate: 10.00%
  Safe rate: 85.00%

ADVERSARIAL perception:
  Fail rate: 15.00%
  Stuck rate: 12.00%
  Safe rate: 73.00%

OPTIMAL_FIXED perception:
  Fail rate: 25.00%
  Stuck rate: 8.00%
  Safe rate: 67.00%
```

## Running Experiments

### Prerequisites

Ensure the IPOMDP shielding framework is installed:
```bash
cd /home/scarbro/claude
pip install -e .
```

### Quick Test

Run with minimal configuration to verify everything works:
```bash
python3 experiments/optimal_realization_example.py
```

### Production Run

For production experiments, edit the configuration in the script:
```python
optimal_perception = train_optimal_realization(
    ...
    num_candidates=50,           # More candidates
    num_trials_per_candidate=20,  # More trials
    max_iterations=100,           # More iterations
    ...
)
```

## Adding New Experiments

To add a new experiment:

1. Create a new Python file in this directory
2. Import necessary components from `ipomdp_shielding`
3. Follow the pattern in `optimal_realization_example.py`:
   - Load IPOMDP model
   - Create runtime shield factory
   - Run experiment
   - Print/save results
4. Add documentation to this README

## Experiment Output

Results are typically saved to `/tmp/` for testing. For production:
- Save trained models: `/path/to/models/model_name.json`
- Save visualizations: `/path/to/images/experiment_name.png`
- Save metrics: `/path/to/results/experiment_name.json`

## Troubleshooting

### Slow Performance

The optimization can be computationally intensive. For faster testing:
- Reduce `num_candidates` (e.g., 10)
- Reduce `num_trials_per_candidate` (e.g., 5)
- Reduce `max_iterations` (e.g., 10)
 
For production accuracy:
- Increase all parameters
- Consider running on a more powerful machine
- Use parallel evaluation (future extension)

### Memory Issues

If you encounter memory issues:
- Set `store_trajectories=False` in evaluation calls
- Reduce `num_trials` in evaluation
- Process results incrementally rather than all at once

## Further Reading

- Main implementation documentation: `ipomdp_shielding/docs/FIXED_REALIZATION_IMPLEMENTATION.md`
