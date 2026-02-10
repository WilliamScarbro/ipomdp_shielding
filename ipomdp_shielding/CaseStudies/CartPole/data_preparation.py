"""Data preparation pipeline for CartPole case study.

This module bridges between the existing cartpole training code and the
ipomdp_shielding framework structure. It generates:
1. Trained CartPoleStateNet model
2. Confusion matrices for each state dimension
3. Bin edges for state discretization
4. Empirical dynamics MDP from gymnasium rollouts
"""

import sys
from pathlib import Path
import pickle
from collections import defaultdict
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import gymnasium as gym
from tqdm import trange

# Add cartpole directory to path to import existing modules
CARTPOLE_DIR = Path(__file__).parent.parent.parent.parent / "cartpole"
sys.path.insert(0, str(CARTPOLE_DIR))

from model import (
    collect_cartpole_data,
    make_dataloaders,
    CartPoleStateNet,
    train_state_net,
    collect_predictions,
    make_confusion_matrices,
)

# Import framework components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Models.mdp import MDP


def prepare_perception_data(
    num_episodes: int = 200,
    epochs: int = 20,
    seed: int = 0,
    device: str = "cuda",
    data_dir: Optional[Path] = None,
):
    """Generate and save perception model data.

    Steps:
    1. Collect frame pairs and states from CartPole-v1
    2. Train CartPoleStateNet on collected data
    3. Generate confusion matrices for each dimension
    4. Save confusion matrices, bin edges, and trained model

    Args:
        num_episodes: Number of episodes to collect for training
        epochs: Number of training epochs
        seed: Random seed for reproducibility
        device: Device for training ("cuda" or "cpu")
        data_dir: Directory to save data. If None, uses artifacts/ subdirectory
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "artifacts"
    data_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 1: Collecting CartPole data...")
    print("=" * 60)
    frame_pairs, states = collect_cartpole_data(
        num_episodes=num_episodes,
        seed=seed
    )
    print(f"Collected {len(states)} state observations")

    print("\n" + "=" * 60)
    print("STEP 2: Creating data loaders...")
    print("=" * 60)
    train_loader, val_loader, test_loader = make_dataloaders(
        frame_pairs, states, batch_size=64
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    print("\n" + "=" * 60)
    print("STEP 3: Training CartPoleStateNet...")
    print("=" * 60)
    model = train_state_net(
        train_loader,
        val_loader,
        epochs=epochs,
        device=device
    )

    print("\n" + "=" * 60)
    print("STEP 4: Generating confusion matrices...")
    print("=" * 60)
    y_true, y_pred = collect_predictions(model, test_loader, device=device)
    cms, edges_list = make_confusion_matrices(y_true, y_pred)

    print("\n" + "=" * 60)
    print("STEP 5: Saving data to artifacts/...")
    print("=" * 60)

    # Save bin edges
    edges_array = np.array(edges_list, dtype=object)
    np.save(data_dir / "bin_edges.npy", edges_array, allow_pickle=True)
    print(f"✓ Saved bin_edges.npy")

    # Save confusion matrices for each dimension
    dim_names = ["x", "x_dot", "theta", "theta_dot"]
    for i, dim in enumerate(dim_names):
        np.save(data_dir / f"{dim}_confusion.npy", cms[i])
        print(f"✓ Saved {dim}_confusion.npy (shape: {cms[i].shape})")

    # Save trained model
    torch.save(model.state_dict(), data_dir / "cartpole_state_net.pt")
    print(f"✓ Saved cartpole_state_net.pt")

    print("\n" + "=" * 60)
    print("Perception data preparation complete!")
    print("=" * 60)


def discretize_state(state: np.ndarray, bin_edges: np.ndarray) -> Tuple[int, int, int, int]:
    """Discretize a continuous state into bin indices.

    Args:
        state: Continuous state [x, x_dot, theta, theta_dot]
        bin_edges: Array of shape (4, k+1) containing bin edges for each dimension

    Returns:
        Tuple of 4 bin indices
    """
    bins = []
    for d in range(4):
        edges = bin_edges[d]
        k = len(edges) - 1
        bin_idx = np.clip(np.digitize(state[d], edges) - 1, 0, k - 1)
        bins.append(int(bin_idx))
    return tuple(bins)


def prepare_dynamics_data(
    num_episodes: int = 10000,
    max_steps: int = 200,
    seed: int = 0,
    data_dir: Optional[Path] = None,
):
    """Generate and save empirical dynamics MDP from gymnasium rollouts.

    Collects transition counts by running random policy in CartPole-v1,
    then normalizes to create an MDP.

    Args:
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        data_dir: Directory to save data. If None, uses artifacts/ subdirectory
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "artifacts"
    data_dir.mkdir(exist_ok=True)

    # Load bin edges
    bin_edges = np.load(data_dir / "bin_edges.npy", allow_pickle=True)

    print("=" * 60)
    print("Collecting dynamics data from CartPole-v1...")
    print("=" * 60)

    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)

    # Track transition counts
    transition_counts = defaultdict(lambda: defaultdict(int))
    FAIL = "FAIL"

    for ep in trange(num_episodes, desc="Episodes"):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        state = np.array(env.unwrapped.state, dtype=np.float32)
        s_discrete = discretize_state(state, bin_edges)

        for t in range(max_steps):
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(env.unwrapped.state, dtype=np.float32)
            s_next_discrete = discretize_state(next_state, bin_edges)

            # If episode terminated, transition to FAIL state
            if terminated:
                s_next_discrete = FAIL

            transition_counts[(s_discrete, action)][s_next_discrete] += 1

            if terminated or truncated:
                break

            s_discrete = s_next_discrete
            state = next_state

    env.close()

    print("\n" + "=" * 60)
    print("Building MDP from transition counts...")
    print("=" * 60)

    # Build state space
    num_bins = 7
    states = [
        (x_bin, xdot_bin, theta_bin, thetadot_bin)
        for x_bin in range(num_bins)
        for xdot_bin in range(num_bins)
        for theta_bin in range(num_bins)
        for thetadot_bin in range(num_bins)
    ]
    states.append(FAIL)

    # Build action dictionary
    actions_dict = {s: [0, 1] for s in states if s != FAIL}
    actions_dict[FAIL] = []  # No actions from FAIL state

    # Build transition probabilities
    P = {}

    # Add FAIL self-loop (absorbing state)
    # Note: FAIL has no actions, so no transitions needed in P

    # Normalize transition counts to probabilities
    for (s, a), next_counts in transition_counts.items():
        total = sum(next_counts.values())
        if total > 0:
            P[(s, a)] = {s_next: count / total for s_next, count in next_counts.items()}
        else:
            # Should not happen, but handle gracefully
            P[(s, a)] = {}

    # Fill in missing transitions with uniform distribution over visited states
    # (or could use single self-loop for unobserved state-action pairs)
    num_state_actions = 0
    num_observed = len(P)

    for s in states:
        if s == FAIL:
            continue
        for a in [0, 1]:
            num_state_actions += 1
            if (s, a) not in P:
                # Unobserved state-action: add self-loop (conservative)
                P[(s, a)] = {s: 1.0}

    mdp = MDP(states, actions_dict, P)

    print(f"States: {len(states)}")
    print(f"State-action pairs: {num_state_actions}")
    print(f"Observed transitions: {num_observed} ({100 * num_observed / num_state_actions:.1f}%)")
    print(f"Unobserved transitions: {num_state_actions - num_observed} (filled with self-loops)")

    print("\n" + "=" * 60)
    print("Saving dynamics MDP...")
    print("=" * 60)

    with open(data_dir / "dynamics_mdp.pkl", "wb") as f:
        pickle.dump(mdp, f)
    print(f"✓ Saved dynamics_mdp.pkl")

    print("\n" + "=" * 60)
    print("Dynamics data preparation complete!")
    print("=" * 60)


def prepare_all_data(
    perception_episodes: int = 200,
    dynamics_episodes: int = 10000,
    epochs: int = 20,
    seed: int = 0,
    device: str = "cuda",
):
    """Run complete data preparation pipeline.

    Args:
        perception_episodes: Episodes for perception model training
        dynamics_episodes: Episodes for dynamics collection
        epochs: Training epochs for perception model
        seed: Random seed
        device: Device for training
    """
    print("\n" + "█" * 60)
    print("CartPole Data Preparation Pipeline")
    print("█" * 60 + "\n")

    # Step 1: Perception data
    prepare_perception_data(
        num_episodes=perception_episodes,
        epochs=epochs,
        seed=seed,
        device=device,
    )

    print("\n")

    # Step 2: Dynamics data
    prepare_dynamics_data(
        num_episodes=dynamics_episodes,
        seed=seed + 1,  # Different seed for diversity
    )

    print("\n" + "█" * 60)
    print("All data preparation complete!")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    # Run with default parameters
    prepare_all_data()
