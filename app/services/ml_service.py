import pickle
import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import structlog

from app.core.config import settings
from app.schemas.therapy import TherapyCondition
from app.services.utils import BigFiveValidator

logger = structlog.get_logger(__name__)


class LinearThompsonSampling:
    """
    Linear Thompson Sampling bandit algorithm for music recommendation.
    """

    def __init__(self, feature_dim: int = 20, lambda_: float = 1.0, alpha: float = 1.0):
        self.feature_dim = feature_dim
        self.lambda_ = lambda_  # Regularization parameter
        self.alpha = alpha  # Exploration parameter

        # Initialize parameters for each arm (song)
        self.A = {}  # Feature covariance matrices
        self.b = {}  # Reward vectors
        self.theta = {}  # Model parameters
        self.pulls = {}  # Number of pulls per arm
        self.rewards = {}  # Cumulative rewards per arm

    def add_arm(self, arm_id: str):
        """Add a new arm (song) to the bandit."""
        if arm_id not in self.A:
            self.A[arm_id] = self.lambda_ * np.eye(self.feature_dim)
            self.b[arm_id] = np.zeros(self.feature_dim)
            self.theta[arm_id] = np.zeros(self.feature_dim)
            self.pulls[arm_id] = 0
            self.rewards[arm_id] = 0.0

    def select_arm(self, context: np.ndarray, available_arms: List[str]) -> str:
        """Select an arm using Thompson Sampling."""
        if not available_arms:
            raise ValueError("No available arms")

        # Sample theta for each available arm
        thetas = []
        for arm_id in available_arms:
            self.add_arm(arm_id)

            # Sample from posterior
            A_inv = np.linalg.inv(self.A[arm_id])
            theta_sample = np.random.multivariate_normal(
                self.theta[arm_id],
                self.alpha**2 * A_inv
            )
            thetas.append(theta_sample)

        # Calculate expected rewards
        expected_rewards = [theta.dot(context) for theta in thetas]

        # Select arm with highest expected reward
        best_idx = np.argmax(expected_rewards)
        return available_arms[best_idx]

    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """Update the bandit with observed reward."""
        self.add_arm(arm_id)

        # Update parameters
        self.A[arm_id] += np.outer(context, context)
        self.b[arm_id] += reward * context
        self.theta[arm_id] = np.linalg.solve(self.A[arm_id], self.b[arm_id])

        # Update statistics
        self.pulls[arm_id] += 1
        self.rewards[arm_id] += reward

    def get_confidence(self, arm_id: str) -> float:
        """Get confidence score for an arm."""
        if arm_id not in self.A:
            return 0.0

        A_inv = np.linalg.inv(self.A[arm_id])
        return 1.0 / np.sqrt(np.trace(A_inv))

    def save_model(self, filepath: str):
        """Save the bandit model."""
        model_data = {
            'A': {k: v.tolist() for k, v in self.A.items()},
            'b': {k: v.tolist() for k, v in self.b.items()},
            'theta': {k: v.tolist() for k, v in self.theta.items()},
            'pulls': self.pulls,
            'rewards': self.rewards,
            'feature_dim': self.feature_dim,
            'lambda_': self.lambda_,
            'alpha': self.alpha
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load the bandit model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.A = {k: np.array(v) for k, v in model_data['A'].items()}
            self.b = {k: np.array(v) for k, v in model_data['b'].items()}
            self.theta = {k: np.array(v) for k, v in model_data['theta'].items()}
            self.pulls = model_data['pulls']
            self.rewards = model_data['rewards']
            self.feature_dim = model_data['feature_dim']
            self.lambda_ = model_data['lambda_']
            self.alpha = model_data['alpha']

            logger.info(f"Loaded bandit model with {len(self.A)} arms")
        except Exception as e:
            logger.error(f"Failed to load bandit model: {e}")
            raise


class Big5PersonalityProfiler:
    """
    Big Five personality traits profiler.
    """

    def __init__(self):
        # BFI-2 item allocation (simplified)
        self.openness_items = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
        self.conscientiousness_items = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
        self.extraversion_items = [3, 8, 13, 18, 23, 28, 33, 38, 43, 48]
        self.agreeableness_items = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
        self.neuroticism_items = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    def calculate_scores(self, responses: List[int]) -> Dict[str, float]:
        """
        Calculate Big Five scores from questionnaire responses.
        Responses should be 1-5 Likert scale for BFI-2 (50 items) or 1-7 scale for simplified (10 items).
        """
        if len(responses) == 50:
            # Standard BFI-2 format with 1-5 scale
            return self._calculate_bfi2_scores(responses)
        elif len(responses) == 10:
            # Simplified format with 1-7 scale (2 items per trait)
            return self._calculate_simplified_scores(responses)
        else:
            raise ValueError("Expected 50 responses for BFI-2 or 10 responses for simplified format")

    def _calculate_bfi2_scores(self, responses: List[int]) -> Dict[str, float]:
        """Calculate scores for standard BFI-2 format (50 items, 1-5 scale)."""
        def calculate_trait_score(items):
            trait_scores = []
            for i, item_num in enumerate(items):
                if item_num <= len(responses):
                    response = responses[item_num - 1]
                    # Reverse code for negatively keyed items (simplified)
                    if item_num in [6, 12, 18, 24, 30]:  # Example reverse-coded items
                        response = 6 - response
                    trait_scores.append(response)
            return np.mean(trait_scores) / 5.0  # Normalize to 0-1

    def _calculate_simplified_scores(self, responses: List[int]) -> Dict[str, float]:
        """Calculate scores for simplified format (10 items, 1-7 scale, 2 items per trait)."""
        return BigFiveValidator.calculate_big_five_scores(responses)


class MLService:
    """
    Machine Learning service for music recommendations.
    """

    def __init__(self):
        self.bandits = {}  # Bandits for different conditions
        self.personality_profiler = Big5PersonalityProfiler()
        self.feature_dim = 20
        self.load_models()

    def load_models(self):
        """Load ML models from disk while guaranteeing a fallback bandit."""
        self.bandits = {}

        # Always create a general-purpose bandit to fall back on
        general_bandit = LinearThompsonSampling(self.feature_dim)
        model_path = getattr(settings, 'MODEL_PATH', None)
        if model_path:
            try:
                # Try to load the model file with the new format first
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Check if it's the new format (dict with bandits key)
                if isinstance(model_data, dict) and 'bandits' in model_data:
                    logger.info("Loading model with new format")
                    # Load condition-specific bandits if available
                    for condition_str, bandit_obj in model_data['bandits'].items():
                        try:
                            # Convert condition string to TherapyCondition enum
                            if condition_str == 'dementia':
                                condition = TherapyCondition.DEMENTIA
                            elif condition_str == 'adhd':
                                condition = TherapyCondition.ADHD
                            elif condition_str == 'down_syndrome':
                                condition = TherapyCondition.DOWN_SYNDROME
                            else:
                                continue

                            self.bandits[condition] = bandit_obj
                            logger.info(f"Loaded bandit model for {condition_str}")
                        except Exception as e:
                            logger.warning(f"Failed to load bandit for {condition_str}: {e}")
                else:
                    # Try old format (direct bandit parameters)
                    general_bandit.load_model(model_path)
                    logger.info("Loaded general bandit model with old format")
            except Exception as e:
                logger.warning(f"Failed to load persisted bandit model, starting fresh: {e}")
        else:
            logger.info("MODEL_PATH not configured; starting with a fresh bandit")

        # Ensure we have a general bandit
        if 'general' not in self.bandits:
            self.bandits['general'] = general_bandit

        # Initialize condition-specific bandits if they don't exist
        conditions = [
            TherapyCondition.DEMENTIA,
            TherapyCondition.ADHD,
            TherapyCondition.DOWN_SYNDROME
        ]
        for condition in conditions:
            if condition not in self.bandits:
                self.bandits[condition] = LinearThompsonSampling(self.feature_dim)

    def _get_bandit(self, condition: TherapyCondition) -> LinearThompsonSampling:
        """
        Return the bandit for a condition, falling back to the general model.
        """
        bandit = self.bandits.get(condition)
        if bandit:
            return bandit

        general_bandit = self.bandits.get('general')
        if not general_bandit:
            general_bandit = LinearThompsonSampling(self.feature_dim)
            self.bandits['general'] = general_bandit
            logger.warning("General bandit missing; created a fresh instance")

        return general_bandit

    def extract_context_features(self, patient_info: Dict[str, Any],
                               big_five_scores: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Extract context features for bandit algorithm.
        """
        features = []

        # Age features (normalized)
        age = patient_info.get('age', 50)
        features.append(age / 100.0)
        features.append((age > 65) * 1.0)  # Elderly flag

        # Gender features
        gender = patient_info.get('gender', '').lower()
        features.append((gender == 'male') * 1.0)
        features.append((gender == 'female') * 1.0)

        # Condition features
        condition = patient_info.get('condition', '').lower()
        features.append((condition == 'dementia') * 1.0)
        features.append((condition == 'adhd') * 1.0)
        features.append((condition == 'down_syndrome') * 1.0)

        # Personality features
        if big_five_scores:
            features.extend([
                big_five_scores.get('openness', 0.5),
                big_five_scores.get('conscientiousness', 0.5),
                big_five_scores.get('extraversion', 0.5),
                big_five_scores.get('agreeableness', 0.5),
                big_five_scores.get('neuroticism', 0.5)
            ])
        else:
            features.extend([0.5] * 5)  # Default values

        # Time features
        hour = datetime.now().hour
        features.append(hour / 24.0)  # Time of day
        features.append((hour >= 6 and hour <= 18) * 1.0)  # Daytime flag

        # Cultural context (simplified)
        features.append(0.7)  # Bengali culture weight
        features.append(0.3)  # International culture weight

        # Fill remaining features with defaults
        while len(features) < self.feature_dim:
            features.append(0.0)

        return np.array(features[:self.feature_dim])

    def calculate_reward(self, feedback_type: str) -> float:
        """
        Calculate reward value for feedback type.
        """
        reward_mapping = {
            'like': 1.0,
            'love': 1.0,
            'neutral': 0.0,
            'skip': -0.5,
            'dislike': -1.0,
            'inappropriate': -2.0
        }
        return reward_mapping.get(feedback_type.lower(), 0.0)

    def recommend_songs(self, patient_info: Dict[str, Any],
                       condition: TherapyCondition,
                       available_songs: List[Dict[str, Any]],
                       big_five_scores: Optional[Dict[str, float]] = None,
                       num_recommendations: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Generate song recommendations using bandit algorithm.
        """
        try:
            bandit = self._get_bandit(condition)
            context = self.extract_context_features(patient_info, big_five_scores)

            recommendations = []
            selected_arms = set()

            # Generate recommendations
            for _ in range(min(num_recommendations, len(available_songs))):
                # Get available songs not yet selected
                remaining_songs = [
                    song for song in available_songs
                    if str(song.get('id', '')) not in selected_arms
                ]

                if not remaining_songs:
                    break

                # Select song using bandit
                song_ids = [str(song.get('id', '')) for song in remaining_songs]
                selected_arm = bandit.select_arm(context, song_ids)

                # Find corresponding song
                selected_song = next(
                    (song for song in remaining_songs
                     if str(song.get('id', '')) == selected_arm),
                    remaining_songs[0]
                )

                # Calculate recommendation score
                score = max(0.0, min(1.0, np.random.normal(0.7, 0.2)))

                recommendations.append((selected_song, score))
                selected_arms.add(str(selected_song.get('id', '')))

            # Sort by score
            recommendations.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"Generated {len(recommendations)} recommendations for {condition}")
            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            # Fallback to random recommendations
            return [(song, 0.5) for song in available_songs[:num_recommendations]]

    def update_bandit(self, condition: TherapyCondition, song_id: str,
                     patient_info: Dict[str, Any], feedback_type: str,
                     big_five_scores: Optional[Dict[str, float]] = None):
        """
        Update bandit model with user feedback.
        """
        try:
            bandit = self._get_bandit(condition)
            context = self.extract_context_features(patient_info, big_five_scores)
            reward = self.calculate_reward(feedback_type)

            bandit.update(str(song_id), context, reward)

            logger.info(f"Updated {condition} bandit with reward {reward} for song {song_id}")

            # Save model periodically
            if hasattr(settings, 'MODEL_PATH') and settings.MODEL_PATH:
                bandit.save_model(settings.MODEL_PATH)

        except Exception as e:
            logger.error(f"Failed to update bandit: {e}")

    def calculate_big_five_scores(self, responses: List[int]) -> Dict[str, float]:
        """Calculate Big Five personality scores."""
        return self.personality_profiler.calculate_scores(responses)

    def _calculate_simplified_scores(self, responses: List[int]) -> Dict[str, float]:
        """
        Calculate Big Five scores from simplified 10-item questionnaire.
        Maps each of the 10 responses to the 5 personality traits.
        """
        if len(responses) != 10:
            raise ValueError(f"Simplified Big Five requires exactly 10 responses, got {len(responses)}")

        return BigFiveValidator.calculate_big_five_scores(responses)
