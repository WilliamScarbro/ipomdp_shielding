"""Example: Train neural action selector on TaxiNet IPOMDP.

This script demonstrates:
1. Loading the TaxiNet IPOMDP model
2. Creating a NeuralActionSelector
3. Training with UniformPerceptionModel (cooperative nature)
4. Training with AdversarialPerceptionModel (adversarial nature)
5. Saving trained models
6. Loading and evaluating models
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipomdp_shielding.CaseStudies.Taxinet import build_taxinet_ipomdp
from ipomdp_shielding.MonteCarlo import (
    NeuralActionSelector,
    UniformPerceptionModel,
    AdversarialPerceptionModel,
)


def train_safe_agent():
    """Train neural selector to maximize safety (avoid FAIL)."""
    print("=" * 60)
    print("Training Safe Agent (Maximize Safety)")
    print("=" * 60)

    # Load IPOMDP
    print("\nLoading TaxiNet IPOMDP...")
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp()
    print(f"  States: {len(ipomdp.states)}")
    print(f"  Actions: {list(ipomdp.actions)}")
    print(f"  Observations: {len(ipomdp.observations)}")

    # Create selector
    print("\nCreating NeuralActionSelector...")
    selector = NeuralActionSelector(
        actions=list(ipomdp.actions),
        observations=ipomdp.observations,
        history_window=10,
        maximize_safety=True,
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration=0.01,
        batch_size=64,
        target_update_freq=500,
        replay_capacity=50000,
    )

    # Train (NO SHIELD)
    print("\nTraining neural selector (uniform perception)...")
    print("  Episodes: 1000")
    print("  Episode length: 20")
    print("  Objective: Maximize safety (avoid FAIL)")
    print()

    metrics = selector.train(
        ipomdp=ipomdp,
        perception=UniformPerceptionModel(),
        num_episodes=1000,
        episode_length=20,
        verbose=True
    )

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final safe rate: {metrics['final_safe_rate']:.2%}")
    print(f"Final fail rate: {metrics['final_fail_rate']:.2%}")

    # Save model
    model_path = "models/taxinet_neural_safe.pt"
    os.makedirs("models", exist_ok=True)
    selector.save(model_path)
    print(f"\nModel saved to: {model_path}")

    return selector, metrics


def train_adversarial_agent():
    """Train neural selector to minimize safety (seek FAIL)."""
    print("\n" + "=" * 60)
    print("Training Adversarial Agent (Minimize Safety)")
    print("=" * 60)

    # Load IPOMDP
    print("\nLoading TaxiNet IPOMDP...")
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp()

    # Create selector
    print("\nCreating NeuralActionSelector...")
    selector = NeuralActionSelector(
        actions=list(ipomdp.actions),
        observations=ipomdp.observations,
        history_window=10,
        maximize_safety=False,  # Seek failure
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration=0.01,
        batch_size=64,
        target_update_freq=500,
        replay_capacity=50000,
    )

    # Train with adversarial perception
    print("\nTraining neural selector (adversarial perception)...")
    print("  Episodes: 1000")
    print("  Episode length: 20")
    print("  Objective: Minimize safety (seek FAIL)")
    print()

    metrics = selector.train(
        ipomdp=ipomdp,
        perception=AdversarialPerceptionModel(pp_shield),
        num_episodes=1000,
        episode_length=20,
        verbose=True
    )

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final safe rate: {metrics['final_safe_rate']:.2%}")
    print(f"Final fail rate: {metrics['final_fail_rate']:.2%}")

    # Save model
    model_path = "models/taxinet_neural_adversarial.pt"
    os.makedirs("models", exist_ok=True)
    selector.save(model_path)
    print(f"\nModel saved to: {model_path}")

    return selector, metrics


def load_and_evaluate():
    """Load trained model and evaluate."""
    print("\n" + "=" * 60)
    print("Loading and Evaluating Trained Model")
    print("=" * 60)

    # Load IPOMDP
    ipomdp, pp_shield, _, _ = build_taxinet_ipomdp()

    # Load model
    model_path = "models/taxinet_neural_safe.pt"
    if not os.path.exists(model_path):
        print(f"\nModel not found: {model_path}")
        print("Please train the model first by running this script.")
        return

    print(f"\nLoading model from: {model_path}")
    selector = NeuralActionSelector.load(model_path, ipomdp)

    print(f"  Episodes trained: {selector.episodes_trained}")
    print(f"  Final safe rate: {selector.final_safe_rate:.2%}")
    print(f"  Exploration rate: {selector.exploration_rate:.3f}")

    # Test action selection
    print("\nTesting action selection...")
    test_histories = [
        [],
        [((0, 0), -1)],
        [((0, 0), -1), ((1, 1), 0)],
        [((i, i), 0) for i in range(10)],
    ]

    for i, history in enumerate(test_histories):
        action = selector.select(history, list(ipomdp.actions))
        print(f"  History length {len(history)}: selected action = {action}")


def plot_training_curves(metrics):
    """Plot training curves."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        print("\n" + "=" * 60)
        print("Plotting Training Curves")
        print("=" * 60)

        # Compute rolling average
        window = 50
        rewards = np.array(metrics['episode_rewards'])
        outcomes = metrics['episode_outcomes']

        rolling_rewards = np.convolve(
            rewards,
            np.ones(window) / window,
            mode='valid'
        )

        # Compute rolling safe rate
        safe_rates = []
        for i in range(window, len(outcomes) + 1):
            recent = outcomes[i-window:i]
            safe_rate = sum(1 for o in recent if o == "safe") / len(recent)
            safe_rates.append(safe_rate)

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Rewards
        ax1.plot(rolling_rewards)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward (Rolling Avg)')
        ax1.set_title('Training Rewards')
        ax1.grid(True)

        # Safe rate
        ax2.plot(safe_rates)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Safe Rate')
        ax2.set_title('Safety Performance')
        ax2.set_ylim([0, 1])
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('models/training_curves.png', dpi=150)
        print("\nPlot saved to: models/training_curves.png")

    except ImportError:
        print("\nMatplotlib not available, skipping plots")


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train neural action selector on TaxiNet"
    )
    parser.add_argument(
        '--mode',
        choices=['safe', 'adversarial', 'both', 'load'],
        default='both',
        help='Training mode'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot training curves'
    )

    args = parser.parse_args()

    if args.mode == 'safe':
        selector, metrics = train_safe_agent()
        if args.plot:
            plot_training_curves(metrics)

    elif args.mode == 'adversarial':
        selector, metrics = train_adversarial_agent()
        if args.plot:
            plot_training_curves(metrics)

    elif args.mode == 'both':
        # Train safe agent
        selector1, metrics1 = train_safe_agent()
        if args.plot:
            plot_training_curves(metrics1)

        # Train adversarial agent
        selector2, metrics2 = train_adversarial_agent()

    elif args.mode == 'load':
        load_and_evaluate()


if __name__ == "__main__":
    main()
