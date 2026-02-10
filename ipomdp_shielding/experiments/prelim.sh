#!/bin/bash
# Preliminary experiments runner
# Runs 4 experiments: (coarse, rl_shield) x (taxinet, cartpole)

set -e  # Exit on error

echo "========================================"
echo "PRELIMINARY EXPERIMENTS"
echo "========================================"
echo ""

# Create output directory
mkdir -p ./data/prelim

# Experiment 1: Coarse + TaxiNet
echo "========================================"
echo "Experiment 1/4: Coarse + TaxiNet"
echo "========================================"
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_prelim
echo ""

# Experiment 2: Coarse + CartPole
echo "========================================"
echo "Experiment 2/4: Coarse + CartPole"
echo "========================================"
python -m ipomdp_shielding.experiments.run_coarse_experiment configs.coarse_cartpole_prelim
echo ""

# Experiment 3: RL Shield + TaxiNet
echo "========================================"
echo "Experiment 3/4: RL Shield + TaxiNet"
echo "========================================"
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim
echo ""

# Experiment 4: RL Shield + CartPole
echo "========================================"
echo "Experiment 4/4: RL Shield + CartPole"
echo "========================================"
python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_cartpole_prelim
echo ""

echo "========================================"
echo "ALL PRELIMINARY EXPERIMENTS COMPLETE"
echo "========================================"
echo "Results saved to ./data/prelim/"
