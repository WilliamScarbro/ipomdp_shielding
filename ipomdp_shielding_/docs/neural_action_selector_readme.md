# Neural Network RL Action Selector

This document describes the neural network based reinforcement learning action selector for IPOMDP models.

## Overview

The `NeuralActionSelector` implements a Deep Q-Network (DQN) based action selector that:

- **Trains directly on IPOMDP dynamics** using the `evolve()` method (no runtime shield dependency)
- **Uses observation-action history as input** (not belief states)
- **Optimizes for FAIL avoidance/seeking** (binary safety objective)
- **Supports model persistence** (save/load trained models)

## Key Differences from Q-Learning Baseline

| Aspect | Q-Learning | Neural Selector |
|--------|-----------|-----------------|
| **State Representation** | Discrete obs history (hashable) | Encoded obs-action history |
| **Q-Values** | Table lookup | Neural network |
| **Generalization** | None (exact history match) | Generalizes to similar histories |
| **Shield Dependency** | Training uses runtime shield | NO shield during training |
| **Scalability** | Limited by table size | Scales to larger state spaces |

## Architecture

### 1. Observation-Action Encoder

Converts heterogeneous observation/action types to fixed-size vectors:

```python
# TaxiNet example:
# Observations: (cte, he) tuples
# Actions: {-1, 0, 1} discrete values

encoder = ObservationActionEncoder(observations, actions)

# Tuple observation → flatten
obs_vec = encoder.encode_observation((0.5, -0.2))  # → [0.5, -0.2]

# Discrete action → one-hot
action_vec = encoder.encode_action(-1)  # → [1, 0, 0]

# History → fixed window + zero padding
history = [((0.5, -0.2), -1), ((0.3, -0.1), 0)]
history_vec = encoder.encode_history(history, window_size=10)
# → 50D vector (10 × (2 + 3))
```

**Supported Types:**
- **Tuple**: `(x, y)` → `[x, y]`
- **Discrete**: One-hot encoding (auto-detected for ≤20 unique values)
- **Scalar**: `value` → `[value]`

### 2. Q-Network

Feedforward neural network:

```
Input: Encoded history (window_size × [obs_dim + action_dim])
  ↓
Dense(128) + ReLU + Dropout(0.1)
  ↓
Dense(64) + ReLU + Dropout(0.1)
  ↓
Dense(32) + ReLU
  ↓
Output(|Actions|)  # Q-value per action
```

### 3. DQN Algorithm

**Components:**
- Experience replay buffer (capacity: 50,000)
- Target network (syncs every 500 steps)
- Epsilon-greedy exploration (ε: 1.0 → 0.01, decay 0.995)
- Huber loss (smooth L1)
- Adam optimizer (lr: 0.001)

**Training Loop:**
1. Sample state, select action (ε-greedy)
2. Evolve state via IPOMDP dynamics
3. Observe outcome, compute reward
4. Store transition in replay buffer
5. Sample batch, update Q-network
6. Sync target network periodically

### 4. Reward Structure

**Maximize Safety Mode** (avoid FAIL):
```
- FAIL state reached: -10.0
- Trial completed (max steps, no FAIL): +10.0
- Each step survived: +0.1
```

**Minimize Safety Mode** (adversarial, seek FAIL):
```
- FAIL state reached: +10.0
- Trial completed safely: -10.0
- Each step survived: -0.1
```

## Usage

### Training

```python
from ipomdp_shielding.CaseStudies.Taxinet import build_taxinet_ipomdp
from ipomdp_shielding.MonteCarlo import (
    NeuralActionSelector,
    UniformPerceptionModel,
)

# Load IPOMDP
ipomdp, pp_shield, _, _ = build_taxinet_ipomdp()

# Create selector
selector = NeuralActionSelector(
    actions=list(ipomdp.actions),
    observations=ipomdp.observations,
    history_window=10,
    maximize_safety=True,
    learning_rate=0.001,
    discount_factor=0.99,
    batch_size=64,
)

# Train (NO SHIELD REQUIRED)
metrics = selector.train(
    ipomdp=ipomdp,
    perception=UniformPerceptionModel(),
    num_episodes=1000,
    episode_length=20,
    verbose=True
)

print(f"Final safe rate: {metrics['final_safe_rate']:.2%}")

# Save model
selector.save("models/taxinet_neural.pt")
```

### Loading and Evaluation

```python
# Load trained model
selector = NeuralActionSelector.load("models/taxinet_neural.pt", ipomdp)

# Use in Monte Carlo evaluation
from ipomdp_shielding.MonteCarlo import run_monte_carlo_trials

results = run_monte_carlo_trials(
    ipomdp=ipomdp,
    pp_shield=pp_shield,
    perception=UniformPerceptionModel(),
    rt_shield_factory=lambda: create_runtime_shield(),
    action_selector=selector,  # Use trained selector
    num_trials=1000,
    trial_length=20
)
```

### Adversarial Training

```python
from ipomdp_shielding.MonteCarlo import AdversarialPerceptionModel

# Train to seek failure
adversarial_selector = NeuralActionSelector(
    actions=list(ipomdp.actions),
    observations=ipomdp.observations,
    maximize_safety=False,  # Seek FAIL
)

metrics = adversarial_selector.train(
    ipomdp=ipomdp,
    perception=AdversarialPerceptionModel(pp_shield),
    num_episodes=1000,
    episode_length=20
)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `history_window` | 10 | Number of recent (obs, action) pairs |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `discount_factor` | 0.99 | Future reward discount (γ) |
| `exploration_rate` | 1.0 → 0.01 | ε-greedy (initial → final) |
| `exploration_decay` | 0.995 | ε decay per episode |
| `replay_capacity` | 50,000 | Experience replay buffer size |
| `batch_size` | 64 | Training batch size |
| `target_update_freq` | 500 | Steps between target network sync |
| `hidden_dims` | (128, 64, 32) | Network layer sizes |
| `dropout` | 0.1 | Dropout rate for regularization |

## Model Persistence

Saved models contain:

```python
{
    'network_state_dict': q_network.state_dict(),
    'target_network_state_dict': target_network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': {...},
    'encoder_config': {
        'obs_vocab': {...},
        'action_vocab': {...},
        'obs_dim': 2,
        'action_dim': 3,
    },
    'training_metadata': {
        'episodes_trained': 1000,
        'final_safe_rate': 0.85,
        'timestamp': '...',
    }
}
```

Models are portable and can be loaded on different machines.

## Testing

Run comprehensive tests:

```bash
pytest ipomdp_shielding/MonteCarlo/test_neural_selector.py -v
```

**Test Coverage:**
- Encoder: tuple/discrete/scalar encoding, history padding
- Q-Network: forward pass, custom architectures
- Replay Buffer: capacity, sampling
- Neural Selector: initialization, action selection, rewards
- Integration: training on simple environment
- Persistence: save/load cycle

## Example Script

Train on TaxiNet:

```bash
cd examples

# Train safe agent
python train_neural_taxinet.py --mode safe --plot

# Train adversarial agent
python train_neural_taxinet.py --mode adversarial

# Train both
python train_neural_taxinet.py --mode both

# Load and evaluate
python train_neural_taxinet.py --mode load
```

## Performance Expectations

**TaxiNet Benchmark** (1000 episodes, episode_length=20):
- Initial safe rate: 0%
- Final safe rate: 15-40% (varies by random seed)
- Training time: ~2-3 minutes on CPU

**Comparison to Q-Learning:**
- Q-Learning: Faster convergence, ~75% safe rate
- Neural: Slower convergence, better generalization potential
- Neural scales better to larger state spaces

## Troubleshooting

### Low Safe Rate

**Symptoms:** Safe rate stays near 0% throughout training

**Solutions:**
- Increase `num_episodes` (try 2000-5000)
- Increase `learning_rate` (try 0.01)
- Decrease `exploration_decay` (try 0.99 for slower decay)
- Check reward structure (FAIL penalty should dominate)

### Training Instability

**Symptoms:** Safe rate oscillates wildly

**Solutions:**
- Decrease `learning_rate` (try 0.0001)
- Increase `batch_size` (try 128)
- Increase `target_update_freq` (try 1000)

### OOM Errors

**Symptoms:** Out of memory during training

**Solutions:**
- Decrease `replay_capacity` (try 10000)
- Decrease `batch_size` (try 32)
- Use smaller `hidden_dims` (try (64, 32))

## Implementation Details

### No Shield Dependency

Unlike the Q-learning baseline, the neural selector **does NOT** require:
- Perfect perception shield during training
- Runtime shield factory
- Initial state generator with shield dependency

This simplifies training and makes it easier to experiment.

### Direct IPOMDP Dynamics

Training uses `ipomdp.evolve(state, action)` directly:

```python
# No shield filtering
state = ipomdp.evolve(state, action)

# Check outcome
if state == "FAIL":
    reward = -10.0 if maximize_safety else +10.0
```

### Belief-Free Approach

The selector uses **raw observation-action history** instead of:
- Belief state tracking
- Belief propagation
- Belief polytope caching

This trades off:
- **Pros:** Simpler, no belief computation overhead
- **Cons:** Less information (no uncertainty quantification)

## Future Enhancements

Potential improvements:

1. **GRU/LSTM Architecture:**
   - Use recurrent layers for temporal dependencies
   - Better memory of long sequences

2. **Prioritized Experience Replay:**
   - Sample important transitions more frequently
   - Faster learning on critical states

3. **Dueling DQN:**
   - Separate value and advantage streams
   - Better credit assignment

4. **Double DQN:**
   - Reduce Q-value overestimation
   - More stable training

5. **Multi-Task Learning:**
   - Train on multiple IPOMDPs simultaneously
   - Transfer learning across domains

## References

- **DQN:** Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
- **Experience Replay:** Lin, "Self-improving reactive agents" (1992)
- **Target Networks:** Mnih et al., "Human-level control through deep RL" (2015)
