"""Action selector strategies for Monte Carlo safety evaluation.

This module provides various strategies for selecting actions from the shield's
allowed action set. Strategies range from simple random selection to RL-trained
policies that maximize or minimize safety.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import random
import numpy as np

from ..Models.ipomdp import IPOMDP


# ============================================================
# Action Selector Base Classes
# ============================================================

class ActionSelector(ABC):
    """Base class for action selection strategies.

    Type signature: Callable[[List[Tuple[obs, action]], List[action]], action]

    The history parameter provides full observation-action sequence for RL policies.
    The allowed_actions ensure safety (pre-filtered by shield).
    """

    @abstractmethod
    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select an action from allowed actions given history.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Full observation-action sequence up to current step
        allowed_actions : list
            Actions permitted by the shield
        context : dict, optional
            Additional context for selection. May contain:
            - "rt_shield": RuntimeImpShield instance
            - "history": Reference to history list

        Returns
        -------
        action
            Selected action from allowed_actions
        """
        pass


# ============================================================
# Simple Action Selectors
# ============================================================

class RandomActionSelector(ActionSelector):
    """Baseline random selection from allowed actions."""

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Randomly select from allowed actions."""
        if not allowed_actions:
            raise ValueError("No allowed actions to select from")
        return random.choice(allowed_actions)


class UniformFallbackSelector(ActionSelector):
    """Random selection with fallback to all actions if stuck.

    If no allowed actions are provided (shield stuck), falls back to
    selecting uniformly from all available actions.
    """

    def __init__(self, all_actions: List[Any]):
        """Initialize with complete action set for fallback.

        Parameters
        ----------
        all_actions : list
            Complete set of actions in the IPOMDP
        """
        self.all_actions = all_actions

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select from allowed actions, or all actions if empty."""
        if allowed_actions:
            return random.choice(allowed_actions)
        else:
            return random.choice(self.all_actions)


# ============================================================
# Heuristic-Based Action Selectors
# ============================================================

class SafestActionSelector(ActionSelector):
    """Selects the safest action from allowed actions.

    Uses observation history to estimate current state distribution, then
    selects the action that maximizes expected safety margin in next states.
    """

    def __init__(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]],
        exploration_rate: float = 0.0
    ):
        """Initialize safest action selector.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model (for dynamics)
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions
        exploration_rate : float
            Probability of random action (for exploration)
        """
        self.ipomdp = ipomdp
        self.pp_shield = pp_shield
        self.exploration_rate = exploration_rate
        self._build_obs_to_states_map()

    def _build_obs_to_states_map(self):
        """Build mapping from observations to likely states."""
        self.obs_to_states = {}
        for state in self.ipomdp.states:
            if state == "FAIL":
                continue
            for obs in self.ipomdp.observations:
                # Use upper bound as proxy for likelihood
                prob = self.ipomdp.P_upper[state].get(obs, 0.0)
                if prob > 0:
                    if obs not in self.obs_to_states:
                        self.obs_to_states[obs] = []
                    self.obs_to_states[obs].append((state, prob))

        # Normalize probabilities for each observation
        for obs in self.obs_to_states:
            total = sum(p for _, p in self.obs_to_states[obs])
            if total > 0:
                self.obs_to_states[obs] = [(s, p/total) for s, p in self.obs_to_states[obs]]

    def _estimate_state_dist(self, history: List[Tuple[Any, Any]]) -> Dict[Any, float]:
        """Estimate state distribution from observation history."""
        if not history:
            # Uniform prior
            n = len([s for s in self.ipomdp.states if s != "FAIL"])
            return {s: 1.0/n for s in self.ipomdp.states if s != "FAIL"}

        # Use most recent observation to estimate state
        last_obs = history[-1][0]
        if last_obs in self.obs_to_states:
            return {s: p for s, p in self.obs_to_states[last_obs]}

        # Fallback to uniform
        n = len([s for s in self.ipomdp.states if s != "FAIL"])
        return {s: 1.0/n for s in self.ipomdp.states if s != "FAIL"}

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select the safest action based on estimated state and expected next-state safety."""
        if not allowed_actions:
            raise ValueError("No allowed actions to select from")

        if len(allowed_actions) == 1:
            return allowed_actions[0]

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(allowed_actions)

        # Estimate current state distribution from history
        state_dist = self._estimate_state_dist(history)

        # Score each action by expected safety margin in next states
        action_scores = []
        for action in allowed_actions:
            score = self._compute_safety_score(action, state_dist)
            action_scores.append((action, score))

        # Select action with highest safety score
        action_scores.sort(key=lambda x: x[1], reverse=True)
        return action_scores[0][0]

    def _compute_safety_score(self, action: Any, state_dist: Dict[Any, float]) -> float:
        """Compute safety score for an action given state distribution.

        Score = sum over states s of:
            P(s) * sum over next states s' of:
                P(s'|s, action) * (safety_value(s'))

        safety_value(s') = num_safe_actions in s', or -100 if FAIL
        """
        score = 0.0

        for state, state_prob in state_dist.items():
            if state == "FAIL":
                continue

            # Get transition probabilities for this action
            transitions = self.ipomdp.T.get((state, action), {})

            for next_state, trans_prob in transitions.items():
                if next_state == "FAIL":
                    # Penalty for reaching failure state
                    score -= state_prob * trans_prob * 100.0
                else:
                    # Reward for safe actions available in next state
                    safe_actions = self.pp_shield.get(next_state, set())
                    score += state_prob * trans_prob * len(safe_actions)

        return score


class RiskiestActionSelector(ActionSelector):
    """Selects the riskiest action from allowed actions.

    Uses observation history to estimate current state, then chooses actions
    most likely to lead to failure or getting stuck.
    Used to test worst-case agent behavior.
    """

    def __init__(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]],
        exploration_rate: float = 0.0
    ):
        """Initialize riskiest action selector.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model (for dynamics)
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions
        exploration_rate : float
            Probability of random action (for exploration)
        """
        self.ipomdp = ipomdp
        self.pp_shield = pp_shield
        self.exploration_rate = exploration_rate
        self._build_obs_to_states_map()

    def _build_obs_to_states_map(self):
        """Build mapping from observations to likely states."""
        self.obs_to_states = {}
        for state in self.ipomdp.states:
            if state == "FAIL":
                continue
            for obs in self.ipomdp.observations:
                prob = self.ipomdp.P_upper[state].get(obs, 0.0)
                if prob > 0:
                    if obs not in self.obs_to_states:
                        self.obs_to_states[obs] = []
                    self.obs_to_states[obs].append((state, prob))

        for obs in self.obs_to_states:
            total = sum(p for _, p in self.obs_to_states[obs])
            if total > 0:
                self.obs_to_states[obs] = [(s, p/total) for s, p in self.obs_to_states[obs]]

    def _estimate_state_dist(self, history: List[Tuple[Any, Any]]) -> Dict[Any, float]:
        """Estimate state distribution from observation history."""
        if not history:
            n = len([s for s in self.ipomdp.states if s != "FAIL"])
            return {s: 1.0/n for s in self.ipomdp.states if s != "FAIL"}

        last_obs = history[-1][0]
        if last_obs in self.obs_to_states:
            return {s: p for s, p in self.obs_to_states[last_obs]}

        n = len([s for s in self.ipomdp.states if s != "FAIL"])
        return {s: 1.0/n for s in self.ipomdp.states if s != "FAIL"}

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select the riskiest action that minimizes expected safety."""
        if not allowed_actions:
            raise ValueError("No allowed actions to select from")

        if len(allowed_actions) == 1:
            return allowed_actions[0]

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(allowed_actions)

        # Estimate current state distribution from history
        state_dist = self._estimate_state_dist(history)

        # Score each action by expected risk (inverse of safety)
        action_scores = []
        for action in allowed_actions:
            score = self._compute_risk_score(action, state_dist)
            action_scores.append((action, score))

        # Select action with highest risk score
        action_scores.sort(key=lambda x: x[1], reverse=True)
        return action_scores[0][0]

    def _compute_risk_score(self, action: Any, state_dist: Dict[Any, float]) -> float:
        """Compute risk score for an action given state distribution.

        Score = sum over states s of:
            P(s) * sum over next states s' of:
                P(s'|s, action) * risk(s')

        risk(s') = 100 if FAIL, 50 if stuck, 1/num_safe_actions otherwise
        """
        score = 0.0

        for state, state_prob in state_dist.items():
            if state == "FAIL":
                continue

            transitions = self.ipomdp.T.get((state, action), {})

            for next_state, trans_prob in transitions.items():
                if next_state == "FAIL":
                    # High reward for reaching failure state
                    score += state_prob * trans_prob * 100.0
                else:
                    # Higher score for fewer safe actions
                    safe_actions = self.pp_shield.get(next_state, set())
                    num_safe = len(safe_actions)
                    if num_safe == 0:
                        score += state_prob * trans_prob * 50.0  # Stuck
                    else:
                        score += state_prob * trans_prob * (1.0 / num_safe)

        return score


class BeliefSelector(ActionSelector):
    """Selects actions based on runtime shield belief probabilities.

    Uses RuntimeImpShield.get_action_probs() to access the current belief state
    and select actions based on their safety probability.

    Modes:
    - "best": Select action with highest allowed probability (safety-maximizing)
    - "worst": Select action with lowest allowed probability (stress testing)
    """

    def __init__(self, mode: str = "best", exploration_rate: float = 0.0):
        """Initialize belief-based selector.

        Parameters
        ----------
        mode : str
            Selection mode: "best" or "worst"
        exploration_rate : float
            Probability of random selection (0.0 to 1.0)
        """
        if mode not in ["best", "worst"]:
            raise ValueError(f"mode must be 'best' or 'worst', got {mode}")

        self.mode = mode
        self.exploration_rate = exploration_rate

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select action using shield belief probabilities.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Full observation-action sequence
        allowed_actions : list
            Actions permitted by the shield
        context : dict, optional
            Must contain "rt_shield": RuntimeImpShield instance

        Returns
        -------
        action
            Selected action from allowed_actions
        """
        if not allowed_actions:
            raise ValueError("No allowed actions to select from")

        if len(allowed_actions) == 1:
            return allowed_actions[0]

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(allowed_actions)

        # Get shield from context
        if context is None or "rt_shield" not in context:
            # Fallback to random if no shield available
            return random.choice(allowed_actions)

        rt_shield = context["rt_shield"]

        # Get action probabilities from shield
        action_probs = rt_shield.get_action_probs()

        # Build map: action -> (allowed_prob, disallowed_prob)
        prob_map = {
            action: (allowed_prob, disallowed_prob)
            for action, allowed_prob, disallowed_prob in action_probs
        }

        # Filter to only allowed actions and score them
        action_scores = []
        for action in allowed_actions:
            if action not in prob_map:
                continue

            allowed_prob, _ = prob_map[action]
            action_scores.append((action, allowed_prob))

        if not action_scores:
            return random.choice(allowed_actions)

        # Select based on mode
        if self.mode == "best":
            # Highest allowed probability
            action_scores.sort(key=lambda x: x[1], reverse=True)
        else:  # mode == "worst"
            # Lowest allowed probability
            action_scores.sort(key=lambda x: x[1])

        return action_scores[0][0]


# ============================================================
# RL-Based Action Selectors
# ============================================================

class RLActionSelector(ActionSelector):
    """Base class for RL-based action selectors.

    Provides interface for training and using learned policies.
    """

    @abstractmethod
    def train(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]],
        rt_shield_factory: Callable,
        perception: "PerceptionModel",
        num_episodes: int,
        episode_length: int
    ) -> Dict[str, List[float]]:
        """Train the RL policy.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        pp_shield : dict
            Perfect perception shield
        rt_shield_factory : callable
            Factory for creating runtime shields
        perception : PerceptionModel
            Perception model for training
        num_episodes : int
            Number of training episodes
        episode_length : int
            Maximum steps per episode

        Returns
        -------
        dict
            Training metrics (e.g., rewards per episode)
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset any episode-specific state."""
        pass


class QLearningActionSelector(RLActionSelector):
    """Q-Learning based action selector.

    Learns state-action values to maximize (or minimize) safety.
    Uses observation history as state for the Q-table since true state
    is not observable.
    """

    def __init__(
        self,
        actions: List[Any],
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.01,
        maximize_safety: bool = True,
        history_length: int = 3
    ):
        """Initialize Q-learning selector.

        Parameters
        ----------
        actions : list
            All possible actions in the environment
        learning_rate : float
            Q-value update rate (alpha)
        discount_factor : float
            Future reward discount (gamma)
        exploration_rate : float
            Initial exploration rate (epsilon)
        exploration_decay : float
            Decay factor for exploration rate per episode
        min_exploration : float
            Minimum exploration rate
        maximize_safety : bool
            If True, maximize safety (best agent)
            If False, minimize safety (worst agent)
        history_length : int
            Number of recent (obs, action) pairs to use as state
        """
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.initial_exploration = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.maximize_safety = maximize_safety
        self.history_length = history_length

        # Q-table: maps (state_key, action) -> Q-value
        self.q_table: Dict[Tuple[Any, Any], float] = {}

        # Episode state
        self.last_state_key = None
        self.last_action = None

    def _get_state_key(self, history: List[Tuple[Any, Any]]) -> Tuple:
        """Convert history to hashable state key."""
        # Use last N observations as state representation
        recent = history[-self.history_length:] if history else []
        return tuple(obs for obs, _ in recent)

    def _get_q_value(self, state_key: Any, action: Any) -> float:
        """Get Q-value, defaulting to 0."""
        return self.q_table.get((state_key, action), 0.0)

    def _set_q_value(self, state_key: Any, action: Any, value: float):
        """Set Q-value."""
        self.q_table[(state_key, action)] = value

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select action using epsilon-greedy policy."""
        if not allowed_actions:
            raise ValueError("No allowed actions to select from")

        if len(allowed_actions) == 1:
            action = allowed_actions[0]
        elif random.random() < self.exploration_rate:
            action = random.choice(allowed_actions)
        else:
            # Greedy selection
            state_key = self._get_state_key(history)
            q_values = [(a, self._get_q_value(state_key, a)) for a in allowed_actions]

            if self.maximize_safety:
                # Best agent: maximize Q-values (which reward safety)
                q_values.sort(key=lambda x: x[1], reverse=True)
            else:
                # Worst agent: minimize Q-values
                q_values.sort(key=lambda x: x[1])

            action = q_values[0][0]

        # Store for learning update
        self.last_state_key = self._get_state_key(history)
        self.last_action = action

        return action

    def update(
        self,
        history: List[Tuple[Any, Any]],
        reward: float,
        done: bool,
        allowed_actions: List[Any]
    ):
        """Update Q-value based on reward.

        Parameters
        ----------
        history : list
            Current observation-action history
        reward : float
            Reward received
        done : bool
            Whether episode terminated
        allowed_actions : list
            Actions available in next state
        """
        if self.last_state_key is None or self.last_action is None:
            return

        state_key = self.last_state_key
        action = self.last_action
        next_state_key = self._get_state_key(history)

        # Get current Q-value
        current_q = self._get_q_value(state_key, action)

        if done:
            # Terminal state - no future rewards
            target = reward
        else:
            # Get max future Q-value
            if allowed_actions:
                future_q_values = [self._get_q_value(next_state_key, a)
                                   for a in allowed_actions]
                if self.maximize_safety:
                    max_future_q = max(future_q_values)
                else:
                    max_future_q = min(future_q_values)
            else:
                max_future_q = 0.0

            target = reward + self.discount_factor * max_future_q

        # Q-learning update
        new_q = current_q + self.learning_rate * (target - current_q)
        self._set_q_value(state_key, action, new_q)

    def decay_exploration(self):
        """Decay exploration rate after episode."""
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

    def reset(self):
        """Reset episode state."""
        self.last_state_key = None
        self.last_action = None

    def train(
        self,
        ipomdp: IPOMDP,
        pp_shield: Dict[Any, Set[Any]],
        rt_shield_factory: Callable,
        perception: "PerceptionModel",
        num_episodes: int,
        episode_length: int,
        initial_generator: Optional["InitialStateGenerator"] = None
    ) -> Dict[str, Any]:
        """Train Q-learning policy through simulation.

        Parameters
        ----------
        ipomdp : IPOMDP
            The interval POMDP model
        pp_shield : dict
            Perfect perception shield
        rt_shield_factory : callable
            Factory for creating runtime shields
        perception : PerceptionModel
            Perception model for training
        num_episodes : int
            Number of training episodes
        episode_length : int
            Maximum steps per episode
        initial_generator : InitialStateGenerator, optional
            Strategy for initial states. Defaults to random.

        Returns
        -------
        dict
            Training metrics including episode rewards and safety rates
        """
        # Import here to avoid circular imports
        from .initial_states import RandomInitialState

        if initial_generator is None:
            initial_generator = RandomInitialState()

        # Reset exploration rate for training
        self.exploration_rate = self.initial_exploration

        episode_rewards = []
        episode_outcomes = []

        for episode in range(num_episodes):
            # Generate initial state
            state, action = initial_generator.generate(ipomdp, pp_shield)

            # Create fresh shield
            rt_shield = rt_shield_factory()
            rt_shield.restart()

            self.reset()
            history = []
            total_reward = 0.0
            outcome = "safe"

            perception_context = {"rt_shield": rt_shield, "history": history}

            for step in range(episode_length):
                if state == "FAIL":
                    # Failure reward
                    reward = -1.0 if self.maximize_safety else 1.0
                    self.update(history, reward, done=True, allowed_actions=[])
                    total_reward += reward
                    outcome = "fail"
                    break

                # Get observation
                obs = perception.sample_observation(state, ipomdp, perception_context)
                history.append((obs, action))

                # Get allowed actions
                allowed_actions = rt_shield.next_actions((obs, action))

                if not allowed_actions:
                    # Stuck reward
                    reward = -0.5 if self.maximize_safety else 0.5
                    self.update(history, reward, done=True, allowed_actions=[])
                    total_reward += reward
                    outcome = "stuck"
                    break

                # Step reward (small positive for survival if maximizing safety)
                step_reward = 0.01 if self.maximize_safety else -0.01
                self.update(history, step_reward, done=False, allowed_actions=allowed_actions)
                total_reward += step_reward

                # Build context for action selector
                action_selector_context = {
                    "rt_shield": rt_shield,
                    "history": history
                }

                # Select action
                action = self.select(history, allowed_actions, context=action_selector_context)

                # Evolve state
                state = ipomdp.evolve(state, action)

            # Episode complete
            if outcome == "safe":
                # Completion bonus
                reward = 1.0 if self.maximize_safety else -1.0
                self.update(history, reward, done=True, allowed_actions=[])
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_outcomes.append(outcome)
            self.decay_exploration()

        # Compute metrics
        safe_count = sum(1 for o in episode_outcomes if o == "safe")
        fail_count = sum(1 for o in episode_outcomes if o == "fail")

        return {
            "episode_rewards": episode_rewards,
            "episode_outcomes": episode_outcomes,
            "final_safe_rate": safe_count / num_episodes,
            "final_fail_rate": fail_count / num_episodes,
            "q_table_size": len(self.q_table)
        }


def create_rl_action_selector(
    ipomdp: IPOMDP,
    pp_shield: Dict[Any, Set[Any]],
    rt_shield_factory: Callable,
    perception: "PerceptionModel",
    maximize_safety: bool = True,
    num_episodes: int = 500,
    episode_length: int = 20,
    **kwargs
) -> QLearningActionSelector:
    """Convenience function to create and train an RL action selector.

    Parameters
    ----------
    ipomdp : IPOMDP
        The interval POMDP model
    pp_shield : dict
        Perfect perception shield
    rt_shield_factory : callable
        Factory for creating runtime shields
    perception : PerceptionModel
        Perception model for training
    maximize_safety : bool
        If True, train to maximize safety (best agent)
        If False, train to minimize safety (worst agent)
    num_episodes : int
        Number of training episodes
    episode_length : int
        Maximum steps per episode
    **kwargs
        Additional arguments passed to QLearningActionSelector

    Returns
    -------
    QLearningActionSelector
        Trained selector
    """
    selector = QLearningActionSelector(
        actions=list(ipomdp.actions),
        maximize_safety=maximize_safety,
        **kwargs
    )

    print(f"Training RL selector ({'maximize' if maximize_safety else 'minimize'} safety)...")
    metrics = selector.train(
        ipomdp=ipomdp,
        pp_shield=pp_shield,
        rt_shield_factory=rt_shield_factory,
        perception=perception,
        num_episodes=num_episodes,
        episode_length=episode_length
    )

    print(f"  Episodes: {num_episodes}")
    print(f"  Q-table size: {metrics['q_table_size']}")
    print(f"  Final safe rate: {metrics['final_safe_rate']:.2%}")
    print(f"  Final fail rate: {metrics['final_fail_rate']:.2%}")

    # Set to evaluation mode (low exploration)
    selector.exploration_rate = 0.0

    return selector
