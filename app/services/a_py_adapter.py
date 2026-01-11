"""
Adapter service that integrates a.py standalone script with the FastAPI backend.

This adapter loads the TheramuseRecommender class from a.py and provides
a bridge between the FastAPI endpoints and the standalone recommendation logic.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


class APyAdapter:
    """
    Adapter for the a.py standalone script.

    Loads and executes the TheramuseRecommender from a.py,
    providing recommendations that can be used by the FastAPI backend.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the a.py adapter.

        Args:
            data_path: Optional path to the music dataset (defaults to settings.MUSIC_CATALOG_PATH)
        """
        self._module = None
        self._recommender = None
        self._script_path = Path(__file__).resolve().parents[2] / "a.py"

        # Use provided path or default from settings
        self.data_path = data_path or settings.MUSIC_CATALOG_PATH

        logger.info(f"Initializing a.py adapter with dataset: {self.data_path}")

    def _load_module(self) -> bool:
        """
        Load the a.py module dynamically.

        Returns:
            True if successful, False otherwise
        """
        if self._module is not None:
            return True

        if not self._script_path.exists():
            logger.error(f"a.py script not found at {self._script_path}")
            return False

        try:
            # Import a.py as a module
            spec = importlib.util.spec_from_file_location("a_py", self._script_path)
            if spec is None or spec.loader is None:
                logger.error("Failed to create import spec for a.py")
                return False

            self._module = importlib.util.module_from_spec(spec)
            sys.modules["a_py"] = self._module

            # Execute the module (this will trigger all imports in a.py)
            spec.loader.exec_module(self._module)

            logger.info("Successfully loaded a.py module")
            return True

        except Exception as e:
            logger.error(f"Failed to load a.py module: {e}", exc_info=True)
            return False

    def _get_recommender(self):
        """
        Get or create the TheramuseRecommender instance from a.py.

        Returns:
            TheramuseRecommender instance or None if failed
        """
        if self._recommender is not None:
            return self._recommender

        if not self._load_module():
            return None

        try:
            # Access the TheramuseRecommender class from a.py
            TheramuseRecommender = self._module.TheramuseRecommender

            # Create instance with the data path
            self._recommender = TheramuseRecommender(self.data_path)

            logger.info("Successfully created TheramuseRecommender instance from a.py")
            return self._recommender

        except Exception as e:
            logger.error(f"Failed to create TheramuseRecommender: {e}", exc_info=True)
            return None

    def generate_recommendations(
        self,
        intake_data: Dict[str, Any],
        n_recommendations: int = 60
    ) -> Optional[Dict[str, Any]]:
        """
        Generate music recommendations using the a.py logic.

        Args:
            intake_data: Patient intake data dictionary
            n_recommendations: Number of recommendations to generate

        Returns:
            Dictionary with recommendations or None if failed
        """
        recommender = self._get_recommender()
        if recommender is None:
            return None

        try:
            # Import the IntakeResponses and TherapeuticCondition from a.py
            IntakeResponses = self._module.IntakeResponses
            TherapeuticCondition = self._module.TherapeuticCondition

            # Convert intake_data to IntakeResponses format
            condition_map = {
                'dementia': TherapeuticCondition.DEMENTIA,
                'adhd': TherapeuticCondition.ADHD,
                'down_syndrome': TherapeuticCondition.DOWN_SYNDROME
            }

            condition_str = intake_data.get('condition', 'dementia').lower()
            condition = condition_map.get(condition_str, TherapeuticCondition.DEMENTIA)

            # Create IntakeResponses object
            intake = IntakeResponses(
                name=intake_data.get('name', 'Unknown'),
                sex=intake_data.get('sex', 'Female'),
                date_of_birth=intake_data.get('dateOfBirth', '1950-01-01'),
                birthplace_city=intake_data.get('birthplaceCity', ''),
                birthplace_country=intake_data.get('birthplaceCountry', ''),
                condition=condition,
                favorite_musician=intake_data.get('favoriteMusician', ''),
                favorite_genres=intake_data.get('favoriteGenres', []),
                preferred_instruments=intake_data.get('instruments', []),
                preferred_languages=intake_data.get('preferredLanguages', []),
                natural_elements=intake_data.get('naturalElements', []),
                difficulty_sleeping=intake_data.get('difficultySleeping', False),
                trouble_remembering=intake_data.get('troubleRemembering', False),
                forgets_everyday_things=intake_data.get('forgetsEverydayThings', False),
                difficulty_recalling_old_memories=intake_data.get('difficultyRecallingOldMemories', False),
                memory_worse_than_year_ago=intake_data.get('memoryWorseThanYearAgo', False),
                visited_mental_health_professional=intake_data.get('visitedMentalHealthProfessional', False),
                big_five_responses=intake_data.get('bigFiveResponses', [4] * 10),
                big_five_scores=intake_data.get('big_five', {})
            )

            # Generate recommendations using the recommender
            recommendations_df = recommender.recommend_for_intake(
                intake,
                n_recommendations=n_recommendations
            )

            if recommendations_df is None or recommendations_df.empty:
                logger.warning("No recommendations generated from a.py")
                return None

            # Convert DataFrame to list of dictionaries
            recommendations = self._format_recommendations(recommendations_df, n_recommendations)

            return {
                'recommendations': recommendations,
                'total_count': len(recommendations),
                'source': 'a_py_theramuse_recommender',
                'condition': condition_str,
                'dataset_path': str(self.data_path)
            }

        except Exception as e:
            logger.error(f"Failed to generate recommendations from a.py: {e}", exc_info=True)
            return None

    def _format_recommendations(
        self,
        df: pd.DataFrame,
        n: int
    ) -> List[Dict[str, Any]]:
        """
        Format DataFrame recommendations into API response format.

        Args:
            df: Recommendations DataFrame
            n: Maximum number of recommendations to return

        Returns:
            List of formatted recommendation dictionaries
        """
        recommendations = []

        # Build column map for flexible column naming
        column_map = {col.strip().lower(): col for col in df.columns}

        def get_value(row, *candidates):
            """Get row value by trying multiple column name candidates"""
            for candidate in candidates:
                key = candidate.strip().lower()
                if key in column_map:
                    return row.get(column_map[key])
            return None

        for idx, row in df.head(n).iterrows():
            song_dict = dict(row)  # Convert Series to dict

            recommendation = {
                'id': idx + 1,
                'song_id': idx + 1,
                'song_title': get_value(row, 'song_name', 'title', 'name') or 'Unknown',
                'artist': get_value(row, 'singer', 'artist') or 'Unknown Artist',
                'genre': get_value(row, 'genre') or '',
                'language': get_value(row, 'language') or '',
                'year': get_value(row, 'released_date', 'year'),
                'tempo': get_value(row, 'tempo'),
                'valence': get_value(row, 'valence'),
                'energy': get_value(row, 'energy', 'arousal'),
                'danceability': get_value(row, 'danceability'),
                'acousticness': get_value(row, 'acousticness'),
                'speechiness': get_value(row, 'speechiness'),
                'loudness': get_value(row, 'loudness'),
                'recommendation_score': 0.8,
                'rank': idx + 1,
                'algorithm_used': 'TheramuseRecommender',
                'youtube_url': get_value(row, 'youtube link', 'youtube_url', 'youtube') or '',
                'spotify_url': get_value(row, 'spotify link', 'spotify_url', 'spotify') or '',
                'category': 'personalized',
                'source': 'a_py'
            }

            # Clean up None values
            recommendation = {k: v for k, v in recommendation.items() if v is not None}
            recommendations.append(recommendation)

        return recommendations

    def get_music_profile(self, condition: str) -> Optional[Dict[str, Any]]:
        """
        Get the music profile for a specific condition from a.py.

        Args:
            condition: Condition name (dementia, adhd, down_syndrome)

        Returns:
            Music profile dictionary or None if failed
        """
        recommender = self._get_recommender()
        if recommender is None:
            return None

        try:
            TherapeuticCondition = self._module.TherapeuticCondition

            condition_map = {
                'dementia': TherapeuticCondition.DEMENTIA,
                'adhd': TherapeuticCondition.ADHD,
                'down_syndrome': TherapeuticCondition.DOWN_SYNDROME
            }

            condition_enum = condition_map.get(condition.lower())
            if condition_enum is None:
                logger.warning(f"Unknown condition: {condition}")
                return None

            profile = recommender.get_profile(condition_enum)

            # Convert MusicProfile dataclass to dict
            return {
                'tempo_range': profile.tempo_range,
                'energy_range': profile.energy_range,
                'valence_range': profile.valence_range,
                'danceability_range': profile.danceability_range,
                'acousticness_range': profile.acousticness_range,
                'speechiness_max': profile.speechiness_max,
                'loudness_range': profile.loudness_range,
                'preferred_genres': profile.preferred_genres,
                'therapeutic_tags': profile.therapeutic_tags
            }

        except Exception as e:
            logger.error(f"Failed to get music profile: {e}", exc_info=True)
            return None

    def is_available(self) -> bool:
        """
        Check if the a.py adapter is available and functional.

        Returns:
            True if adapter can be used, False otherwise
        """
        return self._get_recommender() is not None


# Global singleton instance
_a_py_adapter_instance: Optional[APyAdapter] = None


def get_a_py_adapter() -> Optional[APyAdapter]:
    """
    Get or create the global a.py adapter instance.

    Returns:
        APyAdapter instance or None if initialization failed
    """
    global _a_py_adapter_instance

    if _a_py_adapter_instance is None:
        try:
            _a_py_adapter_instance = APyAdapter()
            if not _a_py_adapter_instance.is_available():
                logger.warning("a.py adapter initialized but recommender not available")
                _a_py_adapter_instance = None
        except Exception as e:
            logger.error(f"Failed to initialize a.py adapter: {e}", exc_info=True)
            _a_py_adapter_instance = None

    return _a_py_adapter_instance
