"""Action selector strategies for Monte Carlo safety evaluation.

This module provides various strategies for selecting actions from the shield's
allowed action set. Strategies range from simple random selection to RL-trained
policies that maximize or minimize safety.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import random


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


class BeliefShieldedActionSelector(ActionSelector):
    """Combines an RL policy with belief-envelope safety from the runtime shield.

    Selection logic:
    1. Get the RL selector's preferred action (over all actions)
    2. If that action is in the shield's allowed set, use it
    3. If not, fall back to the allowed action with the highest allowed
       probability from the belief envelope (same metric as BeliefSelector)
    4. If the shield returns no allowed actions, use the RL choice unrestricted
    """

    def __init__(
        self,
        rl_selector: ActionSelector,
        all_actions: List[Any],
        exploration_rate: float = 0.0
    ):
        """Initialize belief-neural action selector.

        Parameters
        ----------
        rl_selector : ActionSelector
            Trained RL-based action selector (e.g. NeuralActionSelector)
        all_actions : list
            Complete set of actions in the IPOMDP
        exploration_rate : float
            Probability of random action selection (0.0 to 1.0)
        """
        self.rl_selector = rl_selector
        self.all_actions = all_actions
        self.exploration_rate = exploration_rate

        # Selection statistics
        self.rl_selections = 0
        self.fallback_selections = 0
        self.unshielded_selections = 0

    def _select_safest_allowed(
        self,
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Select allowed action with highest belief-envelope allowed probability.

        Uses rt_shield.get_action_probs() — the same safety metric as
        BeliefSelector in "best" mode.
        """
        if context is None or "rt_shield" not in context:
            return random.choice(allowed_actions)

        rt_shield = context["rt_shield"]
        action_probs = rt_shield.get_action_probs()

        prob_map = {
            action: allowed_prob
            for action, allowed_prob, _ in action_probs
        }

        action_scores = []
        for action in allowed_actions:
            if action in prob_map:
                action_scores.append((action, prob_map[action]))

        if not action_scores:
            return random.choice(allowed_actions)

        action_scores.sort(key=lambda x: x[1], reverse=True)
        return action_scores[0][0]

    def select(
        self,
        history: List[Tuple[Any, Any]],
        allowed_actions: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Select action using RL policy with belief-envelope safety fallback.

        Parameters
        ----------
        history : list of (observation, action) pairs
            Full observation-action sequence
        allowed_actions : list
            Actions permitted by the shield's belief envelope
        context : dict, optional
            Must contain "rt_shield": RuntimeImpShield instance
            for belief-based fallback

        Returns
        -------
        action
            Selected action
        """
        if not allowed_actions:
            self.unshielded_selections += 1
            return self.rl_selector.select(history, self.all_actions, context)

        if len(allowed_actions) == 1:
            return allowed_actions[0]

        if random.random() < self.exploration_rate:
            return random.choice(allowed_actions)

        # Get RL selector's preferred action over all actions
        rl_action = self.rl_selector.select(history, self.all_actions, context)

        if rl_action in allowed_actions:
            self.rl_selections += 1
            return rl_action

        # RL action not allowed — pick safest allowed by belief probability
        self.fallback_selections += 1
        return self._select_safest_allowed(allowed_actions, context)

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics on action selection behavior."""
        total = self.rl_selections + self.fallback_selections + self.unshielded_selections
        non_unshielded = self.rl_selections + self.fallback_selections
        acceptance_rate = (self.rl_selections / non_unshielded
                         if non_unshielded > 0 else 0.0)

        return {
            'rl_selections': self.rl_selections,
            'fallback_selections': self.fallback_selections,
            'unshielded_selections': self.unshielded_selections,
            'total_selections': total,
            'rl_acceptance_rate': acceptance_rate,
        }

    def reset_stats(self):
        """Reset selection statistics."""
        self.rl_selections = 0
        self.fallback_selections = 0
        self.unshielded_selections = 0


# ============================================================
# RL-Based Action Selector Base Class
# ============================================================

class RLActionSelector(ActionSelector):
    """Base class for RL-based action selectors.

    Provides interface for training and using learned policies.
    """

    @abstractmethod
    def train(
        self,
        ipomdp: "IPOMDP",
        pp_shield: Dict[Any, Set[Any]],
        rt_shield_factory: Callable,
        perception: "PerceptionModel",
        num_episodes: int,
        episode_length: int
    ) -> Dict[str, List[float]]:
        """Train the RL policy."""
        ...

    @abstractmethod
    def reset(self):
        """Reset any episode-specific state."""
        pass

