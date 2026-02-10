#!/bin/bash
# Full experiments runner
# Runs 4 experiments: (coarse, rl_shield) x (taxinet, cartpole)

set -e  # Exit on error

echo "========================================"
echo "FULL EXPERIMENTS"
echo "========================================"
echo ""

# Create output directory
mkdir -p ./data/full

# Experiment 1: Coarse + TaxiNet
echo "========================================"
echo "Experiment 1/4: Coarse + TaxiNet"
echo "========================================"
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_full
echo ""

# Experiment 2: Coarse + CartPole
echo "========================================"
echo "Experiment 2/4: Coarse + CartPole"
echo "========================================"
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_cartpole_full
echo ""

# Experiment 3: RL Shield + TaxiNet
echo "========================================"
echo "Experiment 3/4: RL Shield + TaxiNet"
echo "========================================"
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_full
echo ""

# Experiment 4: RL Shield + CartPole
echo "========================================"
echo "Experiment 4/4: RL Shield + CartPole"
echo "========================================"
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_cartpole_full
echo ""

echo "========================================"
echo "ALL FULL EXPERIMENTS COMPLETE"
echo "========================================"
echo "Results saved to ./data/full/"
echo ""
echo "Completed experiments:"
echo "  ✓ Coarse + TaxiNet"
echo "  ✓ Coarse + CartPole (625 states with num_bins=5)"
echo "  ✓ RL Shield + TaxiNet"
echo "  ✓ RL Shield + CartPole (625 states with num_bins=5)"
