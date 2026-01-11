"""
Linear Thompson Sampling bandit algorithm for music recommendations.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


class LinearThompsonSampling:
    """
    Linear Thompson Sampling bandit algorithm for personalized music recommendations.

    Uses Bayesian approach to balance exploration and exploitation in music selection
    based on user feedback and context features.
    """

    def __init__(self, n_features: int = 20, alpha: float = 1.0, lambda_reg: float = 1.0):
        """
        Initialize the Linear Thompson Sampling bandit.

        Args:
            n_features: Number of context features
            alpha: Exploration parameter controlling sample variance
            lambda_reg: L2 regularization parameter
        """
        self.n_features = n_features
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # Initialize posterior distribution parameters
        self.B = np.identity(n_features) * lambda_reg  # Precision matrix
        self.mu = np.zeros(n_features)  # Mean vector
        self.f = np.zeros(n_features)  # Sufficient statistics

        # Interaction tracking
        self.n_interactions = 0
        self.total_reward = 0.0

    def sample_theta(self) -> np.ndarray:
        """
        Sample parameters from the posterior distribution.

        Returns:
            Sampled parameter vector
        """
        try:
            B_inv = np.linalg.inv(self.B)
            self.mu = B_inv @ self.f
            theta_sample = np.random.multivariate_normal(self.mu, self.alpha * B_inv)
            return theta_sample
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            return self.mu + np.random.randn(self.n_features) * 0.1

    def predict(self, context: np.ndarray, theta: Optional[np.ndarray] = None) -> float:
        """
        Predict reward for a given context using current parameters.

        Args:
            context: Feature vector for the context
            theta: Optional parameter vector (uses mean if not provided)

        Returns:
            Predicted reward
        """
        if theta is None:
            theta = self.mu
        return np.dot(theta, context)

    def update(self, context: np.ndarray, reward: float, decay_factor: float = 0.98) -> None:
        """
        Update the posterior distribution with new interaction data.

        Args:
            context: Feature vector for the interaction
            reward: Observed reward
            decay_factor: Decay factor for historical data
        """
        # Apply decay to existing information
        self.B *= decay_factor
        self.f *= decay_factor

        # Update with new observation
        self.B += np.outer(context, context)
        self.f += reward * context

        # Track statistics
        self.n_interactions += 1
        self.total_reward += reward

    def get_confidence(self, context: np.ndarray) -> float:
        """
        Get confidence estimate for a prediction.

        Args:
            context: Feature vector

        Returns:
            Confidence score (higher = more confident)
        """
        try:
            B_inv = np.linalg.inv(self.B)
            variance = context.T @ B_inv @ context
            return 1.0 / (1.0 + variance)
        except np.linalg.LinAlgError:
            return 0.1  # Low confidence if matrix is singular

    def select_arm(self, contexts: List[np.ndarray], arm_ids: List[str]) -> tuple[str, float, np.ndarray]:
        """
        Select the best arm from available options using Thompson Sampling.

        Args:
            contexts: List of context vectors for each arm
            arm_ids: List of arm identifiers

        Returns:
            Tuple of (selected_arm_id, expected_reward, sampled_theta)
        """
        if not contexts or not arm_ids:
            return None, 0.0, np.zeros(self.n_features)

        theta = self.sample_theta()
        expected_rewards = []

        for context in contexts:
            reward = self.predict(context, theta)
            expected_rewards.append(reward)

        best_idx = np.argmax(expected_rewards)
        return arm_ids[best_idx], expected_rewards[best_idx], theta

    def get_arm_stats(self, arm_id: str) -> dict:
        """
        Get statistics for a specific arm.

        Args:
            arm_id: Arm identifier

        Returns:
            Dictionary containing arm statistics
        """
        return {
            "interactions": self.n_interactions,
            "total_reward": self.total_reward,
            "average_reward": self.get_average_reward(),
            "confidence": 0.0  # Would need arm-specific tracking
        }

    def get_average_reward(self) -> float:
        """
        Get the average reward across all interactions.

        Returns:
            Average reward per interaction
        """
        if self.n_interactions == 0:
            return 0.0
        return self.total_reward / self.n_interactions

    def reset(self) -> None:
        """Reset the bandit to initial state."""
        self.B = np.identity(self.n_features) * self.lambda_reg
        self.mu = np.zeros(self.n_features)
        self.f = np.zeros(self.n_features)
        self.n_interactions = 0
        self.total_reward = 0.0

    def get_model_parameters(self) -> dict:
        """
        Get current model parameters for saving/serialization.

        Returns:
            Dictionary containing model state
        """
        return {
            "B": self.B.tolist(),
            "mu": self.mu.tolist(),
            "f": self.f.tolist(),
            "n_features": self.n_features,
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
            "n_interactions": self.n_interactions,
            "total_reward": self.total_reward
        }

    def load_model_parameters(self, params: dict) -> None:
        """
        Load model parameters from saved state.

        Args:
            params: Dictionary containing model parameters
        """
        if "B" in params:
            self.B = np.array(params["B"])
        if "mu" in params:
            self.mu = np.array(params["mu"])
        if "f" in params:
            self.f = np.array(params["f"])
        if "n_features" in params:
            self.n_features = params["n_features"]
        if "alpha" in params:
            self.alpha = params["alpha"]
        if "lambda_reg" in params:
            self.lambda_reg = params["lambda_reg"]
        if "n_interactions" in params:
            self.n_interactions = params["n_interactions"]
        if "total_reward" in params:
            self.total_reward = params["total_reward"]