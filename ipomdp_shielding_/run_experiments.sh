#!/usr/bin/env bash
set -euo pipefail

# Run ipomdp_shielding experiments.
# Usage:
#   ./run_experiments.sh prelim          # Run all prelim experiments
#   ./run_experiments.sh full            # Run all full experiments
#   ./run_experiments.sh carr            # Run Carr comparison
#   ./run_experiments.sh artifacts       # Generate paper artifacts from results

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

# Ensure results directories exist
mkdir -p results/{prelim,full,carr_comparison,cache}

run_prelim() {
    echo "=== Running prelim coarseness experiments ==="
#    "$PYTHON" -mipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_prelim
    "$PYTHON" -mipomdp_shielding.experiments.run_coarse_experiment configs.coarse_cartpole_prelim

    echo "=== Running prelim RL shielding experiments ==="
    "$PYTHON" -mipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim
#    "$PYTHON" -mipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_cartpole_prelim
}

run_full() {
    echo "=== Running full coarseness experiments ==="
#    "$PYTHON" -mipomdp_shielding.experiments.run_coarse_experiment configs.coarse_taxinet_full
    "$PYTHON" -mipomdp_shielding.experiments.run_coarse_experiment configs.coarse_cartpole_full

    echo "=== Running full RL shielding experiments ==="
    "$PYTHON" -mipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_full
#    "$PYTHON" -mipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_cartpole_full
}

run_carr() {
    echo "=== Running Carr comparison experiment ==="
    "$PYTHON" -mipomdp_shielding.experiments.run_carr_comparison default
}

run_artifacts() {
    echo "=== Generating paper artifacts ==="
    "$PYTHON" -mipomdp_shielding.experiments.make_paper_artifacts --data-root results --output-dir results
}

case "${1:-help}" in
    prelim)    run_prelim ;;
    full)      run_full ;;
    carr)      run_carr ;;
    artifacts) run_artifacts ;;
    help|*)
        echo "Usage: $0 {prelim|full|carr|artifacts}"
        echo ""
        echo "  prelim     Run all preliminary (quick) experiments"
        echo "  full       Run all full experiments"
        echo "  carr       Run Carr shield comparison"
        echo "  artifacts  Generate paper artifacts from results"
        exit 1
        ;;
esac
