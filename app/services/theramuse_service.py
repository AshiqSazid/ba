"""
Main TheraMuse service class for therapy recommendations.

Integrates all therapy modules, ML algorithms, and data sources
to provide personalized music therapy recommendations.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from app.services.bandit_algorithm import LinearThompsonSampling
from app.services.bangladeshi_music import BangladeshiGenerationalMatrix, BangladeshiSingerQueryGenerator
from app.services.database_manager import DatabaseManager
from app.services.local_music_dataset import LocalMusicDataset
from app.services.personality_mapping import BigFivePersonalityMapping


class TheraMuseService:
    """
    Main TheraMuse service class that coordinates all therapy recommendations.

    Integrates ML algorithms, personality mapping, music datasets,
    and database management to provide personalized therapy recommendations.
    """

    def __init__(self, model_path: str = "theramuse_model.pkl", database: Optional[str] = None):
        """
        Initialize TheraMuse service.

        Args:
            model_path: Path to trained ML model
            database: Database name override
        """
        print("\n" + "🎵" * 30)
        print("Initializing TheraMuse Service v10.0 - Modular FastAPI Build")
        print("🎵" * 30 + "\n")

        self.model_path = model_path

        # Initialize database manager
        self.db = DatabaseManager(database=database)

        # Initialize core services
        self.local_music_dataset = LocalMusicDataset()
        self.personality_mapping = BigFivePersonalityMapping()
        self.bangladeshi_music = BangladeshiSingerQueryGenerator()
        self.generational_matrix = BangladeshiGenerationalMatrix()

        # Initialize ML bandits for each condition
        self.bandits = {
            "dementia": LinearThompsonSampling(),
            "down_syndrome": LinearThompsonSampling(),
            "adhd": LinearThompsonSampling()
        }

        # Configure exploration parameters
        self.exploration_rate = 0.3
        self.min_exploration_rate = 0.1

        # Load ML model if available
        self._load_model()

        print("✅ TheraMuse Service initialized successfully!\n")

    def _load_model(self) -> None:
        """Load trained ML model if available."""
        try:
            import joblib
            self.ml_model = joblib.load(self.model_path)
            print(f"✅ ML model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"ℹ️  No ML model found at {self.model_path}, using rule-based recommendations")
            self.ml_model = None
        except Exception as e:
            print(f"⚠️  Failed to load ML model: {e}")
            self.ml_model = None

    def _extract_context_features(self, patient_info: Dict, condition: str) -> np.ndarray:
        """
        Extract numerical features from patient information for ML algorithms.

        Args:
            patient_info: Patient demographic and clinical data
            condition: Therapy condition (dementia, down_syndrome, adhd)

        Returns:
            Feature vector for ML algorithms
        """
        features = np.zeros(20)

        # Condition encoding (0-2)
        condition_map = {"dementia": 0, "down_syndrome": 1, "adhd": 2}
        if condition in condition_map:
            features[condition_map[condition]] = 1.0

        # Age normalization
        if "age" in patient_info:
            features[4] = patient_info["age"] / 100.0

        # Dementia-specific features
        if condition == "dementia":
            features[5] = float(patient_info.get("difficulty_sleeping", False))
            features[6] = float(patient_info.get("trouble_remembering", False))
            features[7] = float(patient_info.get("forgets_everyday_things", False))
            features[8] = float(patient_info.get("difficulty_recalling_old_memories", False))
            features[9] = float(patient_info.get("memory_worse_than_year_ago", False))

        # Big Five personality scores (normalized to 0-1)
        if "big5_scores" in patient_info:
            big5 = patient_info["big5_scores"]
            features[10] = big5.get("openness", 4) / 7.0
            features[11] = big5.get("conscientiousness", 4) / 7.0
            features[12] = big5.get("extraversion", 4) / 7.0
            features[13] = big5.get("agreeableness", 4) / 7.0
            features[14] = big5.get("neuroticism", 4) / 7.0

        # Time-based features
        hour = datetime.now().hour
        features[15] = hour / 24.0

        # Preference-based features
        if "instruments" in patient_info:
            features[16] = min(len(patient_info["instruments"]), 5) / 5.0
        if "natural_elements" in patient_info:
            features[17] = min(len(patient_info["natural_elements"]), 5) / 5.0

        return features

    def _big5_scores_changed(self, previous: Optional[Dict[str, Any]],
                           current: Optional[Dict[str, Any]], threshold: float = 0.3) -> bool:
        """
        Check if Big Five personality scores have significantly changed.

        Args:
            previous: Previous personality scores
            current: Current personality scores
            threshold: Change threshold to consider significant

        Returns:
            True if scores changed significantly
        """
        if not current:
            return False
        if not previous:
            return True

        traits = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
        for trait in traits:
            prev_val = float(previous.get(trait) or 0)
            curr_val = float(current.get(trait) or 0)
            if abs(prev_val - curr_val) >= threshold:
                return True
        return False

    def _get_personality_recommendations(self, patient_info: Dict) -> List[Dict]:
        """Get music recommendations based on Big Five personality traits."""
        big5_scores = patient_info.get("big5_scores")
        if not big5_scores:
            return []

        # Get personality-based genre preferences
        personality_genres = self.personality_mapping.get_genres_for_personality(big5_scores)

        # Search local music dataset for matching songs
        recommendations = []
        for genre_info in personality_genres[:5]:  # Top 5 genres
            genre = genre_info["genre"]
            songs = self.local_music_dataset.search_tracks(genre, max_results=3)
            for song in songs:
                song["personality_match"] = {
                    "genre": genre,
                    "trait": genre_info["trait"],
                    "score": genre_info["score"]
                }
                recommendations.append(song)

        return recommendations[:10]  # Return top 10 personality matches

    def _get_nostalgia_recommendations(self, patient_info: Dict) -> List[Dict]:
        """Get nostalgia-based music recommendations."""
        birth_year = patient_info.get("birth_year")
        if not birth_year:
            return []

        # Calculate nostalgia window (ages 10-30 are typically formative)
        start_year, end_year = birth_year + 10, birth_year + 30

        # Get songs from nostalgia window
        nostalgia_songs = self.db.get_songs_for_nostalgia_window(
            city=patient_info.get("birthplace"),
            start_year=start_year,
            end_year=end_year,
            limit=10
        )

        return nostalgia_songs

    def _get_bangladeshi_recommendations(self, patient_info: Dict) -> List[Dict]:
        """Get Bangladeshi music recommendations based on preferences."""
        birthplace = patient_info.get("birthplace")
        favorite_genre = patient_info.get("favorite_genre", "")

        # Get personalized queries
        query_results = self.bangladeshi_music.get_queries(birthplace, favorite_genre)

        # Search for songs based on generated queries
        recommendations = []
        for query in query_results["queries"][:5]:  # Top 5 queries
            songs = self.local_music_dataset.search_tracks(query, max_results=2)
            recommendations.extend(songs)

        return recommendations[:8]  # Return top 8 matches

    def _get_preference_matches(self, patient_info: Dict) -> List[Dict]:
        """Get music based on patient's stated preferences."""
        return self.local_music_dataset.search_by_patient_preferences(patient_info, max_results=8)

    def get_therapy_recommendations(self, patient_info: Dict, condition: str,
                                 patient_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive therapy recommendations for a patient.

        Args:
            patient_info: Patient demographic and preference data
            condition: Therapy condition (dementia, down_syndrome, adhd)
            patient_id: Optional patient ID for tracking

        Returns:
            Comprehensive recommendation results
        """
        session_id = f"therapy_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Process Big Five scores if provided
        big5_scores = patient_info.get("big5_scores")
        previous_big5 = None
        big5_changed = False
        seen_personality_ids: List[str] = []

        if patient_id and big5_scores:
            previous_big5 = self.db.get_latest_big5_scores(patient_id)
            big5_changed = self._big5_scores_changed(previous_big5, big5_scores)
            if big5_changed:
                seen_personality_ids = self.db.get_recent_personality_song_ids(patient_id, limit=50)

        # Save patient data
        if patient_id:
            self.db.save_patient(patient_id, patient_info)
            patient_info["_theramuse_patient_id"] = patient_id

        # Store Big Five metadata
        if big5_scores:
            patient_info["_big5_previous_scores"] = previous_big5
            patient_info["_big5_changed"] = big5_changed
            if big5_changed and seen_personality_ids:
                patient_info["_seen_personality_song_ids"] = seen_personality_ids

        # Get different types of recommendations
        recommendations = {
            "session_id": session_id,
            "patient_id": patient_id,
            "condition": condition,
            "categories": {},
            "total_songs": 0,
            "generated_at": datetime.now().isoformat()
        }

        # 1. Personality-based recommendations
        personality_songs = self._get_personality_recommendations(patient_info)
        if personality_songs:
            recommendations["categories"]["personality_based"] = {
                "songs": personality_songs,
                "count": len(personality_songs),
                "description": "Songs matched to personality traits"
            }
            recommendations["total_songs"] += len(personality_songs)

        # 2. Nostalgia-based recommendations
        nostalgia_songs = self._get_nostalgia_recommendations(patient_info)
        if nostalgia_songs:
            recommendations["categories"]["nostalgia"] = {
                "songs": nostalgia_songs,
                "count": len(nostalgia_songs),
                "description": "Songs from formative years"
            }
            recommendations["total_songs"] += len(nostalgia_songs)

        # 3. Bangladeshi music recommendations
        bangladeshi_songs = self._get_bangladeshi_recommendations(patient_info)
        if bangladeshi_songs:
            recommendations["categories"]["bangladeshi_music"] = {
                "songs": bangladeshi_songs,
                "count": len(bangladeshi_songs),
                "description": "Bangladeshi music based on regional preferences"
            }
            recommendations["total_songs"] += len(bangladeshi_songs)

        # 4. Preference-based recommendations
        preference_songs = self._get_preference_matches(patient_info)
        if preference_songs:
            recommendations["categories"]["preference_matches"] = {
                "songs": preference_songs,
                "count": len(preference_songs),
                "description": "Songs matching stated preferences"
            }
            recommendations["total_songs"] += len(preference_songs)

        # Add bandit statistics
        bandit = self.bandits.get(condition)
        if bandit:
            recommendations["bandit_stats"] = {
                "n_interactions": bandit.n_interactions,
                "avg_reward": bandit.get_average_reward(),
                "exploration_rate": self.exploration_rate
            }

        # Save session and recommendations to database
        if patient_id:
            self.db.save_session(
                session_id, patient_id, condition, "modular_therapy",
                recommendations["total_songs"], self.exploration_rate
            )
            self.db.save_recommendations(session_id, patient_id, recommendations)

            # Save Big Five scores if provided
            if big5_scores:
                self.db.save_big5_scores(session_id, patient_id, big5_scores)

        return recommendations

    def process_feedback(self, patient_id: str, session_id: str, condition: str,
                        song: Dict[str, Any], reward: float,
                        feedback_type: str = "user_rating") -> bool:
        """
        Process user feedback for a song recommendation.

        Args:
            patient_id: Patient identifier
            session_id: Therapy session identifier
            condition: Therapy condition
            song: Song information
            reward: Feedback reward (positive/negative)
            feedback_type: Type of feedback

        Returns:
            True if feedback processed successfully
        """
        try:
            # Extract context features for bandit learning
            patient_info = {"age": 0}  # Default, would be populated from DB
            context_features = self._extract_context_features(patient_info, condition)

            # Save feedback to database
            self.db.save_feedback(
                patient_id, session_id, condition, song, reward,
                feedback_type, context_features.tolist()
            )

            # Update bandit algorithm
            bandit = self.bandits.get(condition)
            if bandit and context_features is not None:
                bandit.update(context_features, reward)

            return True
        except Exception as e:
            print(f"⚠️  Failed to process feedback: {e}")
            return False

    def get_analytics(self) -> Dict[str, Any]:
        """Get usage and performance analytics."""
        return self.db.get_analytics()

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all services."""
        status = {
            "theramuse_service": "healthy",
            "services": {}
        }

        # Check database
        status["services"]["database"] = self.db.get_api_health_status()

        # Check local music dataset
        status["services"]["local_music_dataset"] = self.local_music_dataset.get_api_health_status()

        # Check bandits
        bandit_status = {}
        for condition, bandit in self.bandits.items():
            bandit_status[condition] = {
                "n_interactions": bandit.n_interactions,
                "avg_reward": bandit.get_average_reward(),
                "status": "healthy"
            }
        status["services"]["bandits"] = bandit_status

        return status