"""
Memory-Efficient ML Service for TheraMuse
Solves memory issues by implementing lazy loading, model sharing, and proper cleanup.
"""

import pickle
import numpy as np
import json
import gc
import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from functools import lru_cache
import structlog
from threading import Lock
from contextlib import contextmanager

from app.core.config import settings
from app.schemas.therapy import TherapyCondition
from app.services.utils import BigFiveValidator

# Try to import psutil, use fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = structlog.get_logger(__name__)


class MemoryEfficientLinearThompsonSampling:
    """
    Memory-efficient version of Linear Thompson Sampling bandit algorithm.
    Implements lazy loading, pruning, and memory cleanup.
    """

    def __init__(self, feature_dim: int = 20, lambda_: float = 1.0, alpha: float = 1.0,
                 max_arms: int = 10000):
        self.feature_dim = feature_dim
        self.lambda_ = lambda_
        self.alpha = alpha
        self.max_arms = max_arms  # Limit arms to prevent memory explosion

        # Use sparse representations where possible
        self.A = {}  # Feature covariance matrices
        self.b = {}  # Reward vectors
        self.theta = {}  # Model parameters
        self.pulls = {}  # Number of pulls per arm
        self.rewards = {}  # Cumulative rewards per arm

        self._last_cleanup = datetime.now()
        self._lock = Lock()  # Thread safety for concurrent updates

    def add_arm(self, arm_id: str):
        """Add a new arm (song) to the bandit with memory limits."""
        with self._lock:
            if arm_id not in self.A and len(self.A) < self.max_arms:
                self.A[arm_id] = self.lambda_ * np.eye(self.feature_dim, dtype=np.float32)
                self.b[arm_id] = np.zeros(self.feature_dim, dtype=np.float32)
                self.theta[arm_id] = np.zeros(self.feature_dim, dtype=np.float32)
                self.pulls[arm_id] = 0
                self.rewards[arm_id] = 0.0
            elif len(self.A) >= self.max_arms:
                logger.warning(f"Maximum arms ({self.max_arms}) reached, skipping new arm: {arm_id}")

    def select_arm(self, context: np.ndarray, available_arms: List[str]) -> str:
        """Select an arm using Thompson Sampling with memory efficiency."""
        if not available_arms:
            raise ValueError("No available arms")

        # Filter to only arms we have loaded
        loaded_arms = [arm for arm in available_arms if arm in self.A]
        if not loaded_arms:
            # Add first available arm if we have capacity
            if len(self.A) < self.max_arms:
                self.add_arm(available_arms[0])
                loaded_arms = [available_arms[0]]
            else:
                # Use random selection if at capacity
                return available_arms[0]

        # Sample theta for each available arm (use efficient sampling)
        thetas = []
        for arm_id in loaded_arms:
            try:
                # More efficient sampling using Cholesky decomposition
                A_inv = np.linalg.inv(self.A[arm_id])
                L = np.linalg.cholesky(self.alpha**2 * A_inv)
                theta_sample = self.theta[arm_id] + L @ np.random.randn(self.feature_dim)
                thetas.append(theta_sample)
            except np.linalg.LinAlgError:
                # Fallback to previous method if Cholesky fails
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
        return loaded_arms[best_idx]

    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """Update the bandit with observed reward and cleanup old data."""
        with self._lock:
            self.add_arm(arm_id)

            if arm_id in self.A:
                # Update parameters
                self.A[arm_id] += np.outer(context, context)
                self.b[arm_id] += reward * context
                self.theta[arm_id] = np.linalg.solve(self.A[arm_id], self.b[arm_id])

                # Update statistics
                self.pulls[arm_id] += 1
                self.rewards[arm_id] += reward

                # Periodic cleanup
                if (datetime.now() - self._last_cleanup).hours >= 1:
                    self._cleanup_old_arms()
                    self._last_cleanup = datetime.now()

    def _cleanup_old_arms(self):
        """Remove arms with very few pulls to free memory."""
        if len(self.A) > self.max_arms * 0.8:  # Cleanup when 80% full
            arms_to_remove = []
            for arm_id, pulls in self.pulls.items():
                if pulls < 3:  # Remove arms with fewer than 3 interactions
                    arms_to_remove.append(arm_id)

            for arm_id in arms_to_remove[:100]:  # Remove max 100 at a time
                del self.A[arm_id]
                del self.b[arm_id]
                del self.theta[arm_id]
                del self.pulls[arm_id]
                del self.rewards[arm_id]

            if arms_to_remove:
                logger.info(f"Cleaned up {len(arms_to_remove)} unused arms")
                gc.collect()  # Force garbage collection

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            rss_mb = memory_info.rss / 1024 / 1024
            vms_mb = memory_info.vms / 1024 / 1024
        else:
            # Fallback estimation
            rss_mb = 100.0  # Conservative estimate
            vms_mb = 100.0

        total_matrices = len(self.A) * 4  # A, b, theta, pulls per arm
        return {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "num_arms": len(self.A),
            "estimated_matrices": total_matrices
        }

    def save_model(self, filepath: str):
        """Save the bandit model efficiently."""
        model_data = {
            'A': {k: v.tolist() for k, v in self.A.items()},
            'b': {k: v.tolist() for k, v in self.b.items()},
            'theta': {k: v.tolist() for k, v in self.theta.items()},
            'pulls': self.pulls,
            'rewards': self.rewards,
            'feature_dim': self.feature_dim,
            'lambda_': self.lambda_,
            'alpha': self.alpha,
            'max_arms': self.max_arms
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, filepath: str):
        """Load the bandit model with memory limits."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            # Limit loaded arms to prevent memory explosion
            self.A = {}
            self.b = {}
            self.theta = {}

            # Sort by pulls and keep only the most used arms
            arms_by_usage = sorted(model_data['pulls'].items(),
                                 key=lambda x: x[1], reverse=True)[:self.max_arms]

            for arm_id, _ in arms_by_usage:
                self.A[arm_id] = np.array(model_data['A'][arm_id], dtype=np.float32)
                self.b[arm_id] = np.array(model_data['b'][arm_id], dtype=np.float32)
                self.theta[arm_id] = np.array(model_data['theta'][arm_id], dtype=np.float32)

            self.pulls = {k: v for k, v in model_data['pulls'].items() if k in self.A}
            self.rewards = {k: v for k, v in model_data['rewards'].items() if k in self.A}

            self.feature_dim = model_data['feature_dim']
            self.lambda_ = model_data['lambda_']
            self.alpha = model_data['alpha']

            logger.info(f"Loaded bandit model with {len(self.A)} arms (limited from {len(model_data['A'])})")
        except Exception as e:
            logger.error(f"Failed to load bandit model: {e}")
            raise


class MemoryEfficientMLService:
    """
    Memory-efficient ML Service implementing lazy loading, model sharing, and cleanup.
    """

    def __init__(self):
        self._bandits = {}  # Lazy-loaded bandits
        self._personality_profiler = None  # Lazy-loaded profiler
        self.feature_dim = 20
        self._lock = Lock()
        self._last_memory_check = datetime.now()

    @contextmanager
    def _monitor_memory(self, operation: str):
        """Monitor memory usage during operations."""
        try:
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            yield

            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - start_memory

            if abs(memory_delta) > 50:  # Log if > 50MB change
                logger.info(f"Memory change during {operation}: {memory_delta:+.1f}MB")

                # Force garbage collection if memory increased significantly
                if memory_delta > 100:
                    gc.collect()

        except Exception as e:
            logger.warning(f"Memory monitoring failed for {operation}: {e}")

    def _get_personality_profiler(self):
        """Lazy load personality profiler."""
        if self._personality_profiler is None:
            with self._lock:
                if self._personality_profiler is None:
                    self._personality_profiler = Big5PersonalityProfiler()
                    logger.info("Personality profiler loaded")
        return self._personality_profiler

    def _get_bandit(self, condition: TherapyCondition) -> MemoryEfficientLinearThompsonSampling:
        """Get or create bandit for condition with lazy loading."""
        if condition not in self._bandits:
            with self._lock:
                if condition not in self._bandits:
                    logger.info(f"Creating new bandit for {condition}")
                    self._bandits[condition] = MemoryEfficientLinearThompsonSampling(
                        feature_dim=self.feature_dim,
                        max_arms=5000  # Limit arms per condition
                    )

                    # Try to load existing model
                    model_path = getattr(settings, 'MODEL_PATH', None)
                    if model_path:
                        try:
                            self._bandits[condition].load_model(model_path)
                            logger.info(f"Loaded existing model for {condition}")
                        except Exception as e:
                            logger.warning(f"Failed to load model for {condition}: {e}")

        return self._bandits[condition]

    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory usage status."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            rss_mb = memory_info.rss / 1024 / 1024
            vms_mb = memory_info.vms / 1024 / 1024
        else:
            rss_mb = 0.0
            vms_mb = 0.0

        bandit_memory = {}
        for condition, bandit in self._bandits.items():
            bandit_memory[str(condition)] = bandit.get_memory_usage()

        return {
            "process_memory_rss_mb": rss_mb,
            "process_memory_vms_mb": vms_mb,
            "bandit_count": len(self._bandits),
            "bandit_memory": bandit_memory,
            "personality_profiler_loaded": self._personality_profiler is not None
        }

    def recommend_songs(self, patient_info: Dict[str, Any],
                       condition: TherapyCondition,
                       available_songs: List[Dict[str, Any]],
                       big_five_scores: Optional[Dict[str, float]] = None,
                       num_recommendations: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Generate recommendations with memory monitoring."""
        with self._monitor_memory("recommend_songs"):
            try:
                bandit = self._get_bandit(condition)
                context = self.extract_context_features(patient_info, big_five_scores)

                recommendations = []
                selected_arms = set()

                # Generate recommendations efficiently
                for _ in range(min(num_recommendations, len(available_songs))):
                    remaining_songs = [
                        song for song in available_songs
                        if str(song.get('id', '')) not in selected_arms
                    ]

                    if not remaining_songs:
                        break

                    song_ids = [str(song.get('id', '')) for song in remaining_songs]
                    selected_arm = bandit.select_arm(context, song_ids)

                    selected_song = next(
                        (song for song in remaining_songs
                         if str(song.get('id', '')) == selected_arm),
                        remaining_songs[0]
                    )

                    score = max(0.0, min(1.0, np.random.normal(0.7, 0.2)))
                    recommendations.append((selected_song, score))
                    selected_arms.add(str(selected_song.get('id', '')))

                recommendations.sort(key=lambda x: x[1], reverse=True)

                # Periodic memory check and cleanup
                if (datetime.now() - self._last_memory_check).minutes >= 5:
                    self._periodic_cleanup()
                    self._last_memory_check = datetime.now()

                return recommendations

            except Exception as e:
                logger.error(f"Failed to generate recommendations: {e}")
                return [(song, 0.5) for song in available_songs[:num_recommendations]]

    def update_bandit(self, condition: TherapyCondition, song_id: str,
                     patient_info: Dict[str, Any], feedback_type: str,
                     big_five_scores: Optional[Dict[str, float]] = None):
        """Update bandit with memory monitoring."""
        with self._monitor_memory("update_bandit"):
            try:
                bandit = self._get_bandit(condition)
                context = self.extract_context_features(patient_info, big_five_scores)
                reward = self.calculate_reward(feedback_type)

                bandit.update(str(song_id), context, reward)

                # Save model less frequently to reduce I/O
                if np.random.random() < 0.1:  # 10% chance to save
                    if hasattr(settings, 'MODEL_PATH') and settings.MODEL_PATH:
                        bandit.save_model(settings.MODEL_PATH)

            except Exception as e:
                logger.error(f"Failed to update bandit: {e}")

    def _periodic_cleanup(self):
        """Perform periodic memory cleanup."""
        try:
            # Force garbage collection
            gc.collect()

            # Log memory status
            memory_status = self.get_memory_status()
            logger.info(f"Memory status: {memory_status['process_memory_rss_mb']:.1f}MB RSS")

            # Cleanup bandits if memory is high
            if memory_status['process_memory_rss_mb'] > 1000:  # > 1GB
                logger.warning("High memory usage detected, forcing bandit cleanup")
                for bandit in self._bandits.values():
                    bandit._cleanup_old_arms()
                gc.collect()

        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")

    def extract_context_features(self, patient_info: Dict[str, Any],
                               big_five_scores: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Extract context features efficiently."""
        features = []

        # Age features (normalized)
        age = patient_info.get('age', 50)
        features.append(age / 100.0)
        features.append((age > 65) * 1.0)

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
            features.extend([0.5] * 5)

        # Time features
        hour = datetime.now().hour
        features.append(hour / 24.0)
        features.append((hour >= 6 and hour <= 18) * 1.0)

        # Cultural context
        features.append(0.7)  # Bengali culture weight
        features.append(0.3)  # International culture weight

        # Fill remaining features with defaults
        while len(features) < self.feature_dim:
            features.append(0.0)

        return np.array(features[:self.feature_dim], dtype=np.float32)

    def calculate_reward(self, feedback_type: str) -> float:
        """Calculate reward value for feedback type."""
        reward_mapping = {
            'like': 1.0,
            'love': 1.0,
            'neutral': 0.0,
            'skip': -0.5,
            'dislike': -1.0,
            'inappropriate': -2.0
        }
        return reward_mapping.get(feedback_type.lower(), 0.0)

    def calculate_big_five_scores(self, responses: List[int]) -> Dict[str, float]:
        """Calculate Big Five personality scores."""
        profiler = self._get_personality_profiler()
        return profiler.calculate_scores(responses)

    def cleanup_resources(self):
        """Clean up all resources and free memory."""
        with self._lock:
            logger.info("Cleaning up ML service resources")

            # Clear bandits
            self._bandits.clear()

            # Clear profiler
            self._personality_profiler = None

            # Force garbage collection
            gc.collect()

            logger.info("ML service resources cleaned up")


# Import the original Big5PersonalityProfiler for compatibility
class Big5PersonalityProfiler:
    """Big Five personality traits profiler (simplified version)."""

    def __init__(self):
        # BFI-2 item allocation (simplified)
        self.openness_items = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
        self.conscientiousness_items = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
        self.extraversion_items = [3, 8, 13, 18, 23, 28, 33, 38, 43, 48]
        self.agreeableness_items = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
        self.neuroticism_items = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    def calculate_scores(self, responses: List[int]) -> Dict[str, float]:
        """Calculate Big Five scores from questionnaire responses."""
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
        trait_responses = {
            'openness': [responses[i-1] for i in self.openness_items if i <= len(responses)],
            'conscientiousness': [responses[i-1] for i in self.conscientiousness_items if i <= len(responses)],
            'extraversion': [responses[i-1] for i in self.extraversion_items if i <= len(responses)],
            'agreeableness': [responses[i-1] for i in self.agreeableness_items if i <= len(responses)],
            'neuroticism': [responses[i-1] for i in self.neuroticism_items if i <= len(responses)]
        }

        scores = {}
        for trait, trait_resp in trait_responses.items():
            if trait_resp:
                avg_score = np.mean(trait_resp)
                normalized_score = (avg_score - 1) / 4.0  # Convert 1-5 to 0-1
                scores[trait] = float(normalized_score)
            else:
                scores[trait] = 0.5

        return scores

    def _calculate_simplified_scores(self, responses: List[int]) -> Dict[str, float]:
        """Calculate scores for simplified format (10 items, 1-7 scale, 2 items per trait)."""
        return BigFiveValidator.calculate_big_five_scores(responses)


# Global instance with lazy initialization
_global_ml_service = None
_ml_service_lock = Lock()

def get_ml_service() -> MemoryEfficientMLService:
    """Get global ML service instance (singleton pattern with lazy initialization)."""
    global _global_ml_service

    if _global_ml_service is None:
        with _ml_service_lock:
            if _global_ml_service is None:
                _global_ml_service = MemoryEfficientMLService()
                logger.info("Global ML service initialized")

    return _global_ml_service

def cleanup_ml_service():
    """Clean up global ML service."""
    global _global_ml_service

    with _ml_service_lock:
        if _global_ml_service is not None:
            _global_ml_service.cleanup_resources()
            _global_ml_service = None
            logger.info("Global ML service cleaned up")
