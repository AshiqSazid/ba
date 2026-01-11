"""
Music Feature to Cognitive & Lifestyle Indicators Mapping Service

This service maps music features (lyrics and audio characteristics) to
Cognitive & Lifestyle Indicators using weighted heuristics.

Features:
- 12 input features (0.0-1.0 scale): Lyric's Reappraisal, Distracting, Uplifting, Relaxing, Suppressing, Motivational
                               audio's Reappraisal, Distracting, Uplifting, Relaxing, Suppressing, Motivational
- 6 output indicators: Difficulty Sleeping, Trouble Remembering Recent Events, Forgets Everyday Tasks,
                       Difficulty Recalling Older Memories, Memory Worse Than a Year Ago, Visited Mental Health Professional
- Logistic mapping to ensure outputs stay in [0,1] range
- Song suggestions based on cognitive indicators
"""

import math
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)


class MusicFeatureMappingService:
    """
    Service to map music features to Cognitive & Lifestyle Indicators
    and provide song suggestions based on cognitive profiles.
    """

    def __init__(self, music_catalog_path: Optional[str] = None):
        """
        Initialize the mapping service.

        Args:
            music_catalog_path: Optional path to music catalog database
        """
        self.music_catalog_path = music_catalog_path
        self.songs_database = []
        self.catalog_loaded = False

        # Load music catalog if path provided
        if music_catalog_path:
            self._load_music_catalog()

    def _load_music_catalog(self):
        """Load music catalog from the specified path."""
        try:
            import pandas as pd
            df = pd.read_excel(self.music_catalog_path)
            self.songs_database = df.to_dict('records')
            self.catalog_loaded = True
            logger.info(f"Loaded {len(self.songs_database)} songs from music catalog")
        except Exception as e:
            logger.warning(f"Failed to load music catalog from {self.music_catalog_path}: {e}")
            self.catalog_loaded = False

    # Feature names as expected in the input
    FEATURES = [
        "Lyric's Reappraisal", "Lyric's Distracting", "Lyric's Uplifting", "Lyric's Relaxing",
        "Lyric's Suppressing", "Lyric's Motivational", "audio's Reappraisal", "audio's Distracting",
        "audio's Uplifting", "audio's Relaxing", "audio's Suppressing", "audio's Motivational"
    ]

    # Weight matrix for mapping features to cognitive indicators
    WEIGHTS = {
        "Difficulty Sleeping": {
            "intercept": -0.4055,
            "w": {
                "Lyric's Reappraisal": -0.05, "Lyric's Distracting": 0.07, "Lyric's Uplifting": -0.03, "Lyric's Relaxing": -0.12,
                "Lyric's Suppressing": 0.10, "Lyric's Motivational": 0.06, "audio's Reappraisal": -0.03, "audio's Distracting": 0.10,
                "audio's Uplifting": -0.02, "audio's Relaxing": -0.18, "audio's Suppressing": 0.12, "audio's Motivational": 0.10
            }
        },
        "Trouble Remembering Recent Events": {
            "intercept": -0.6190,
            "w": {
                "Lyric's Reappraisal": -0.10, "Lyric's Distracting": 0.10, "Lyric's Uplifting": -0.06, "Lyric's Relaxing": -0.08,
                "Lyric's Suppressing": 0.14, "Lyric's Motivational": -0.03, "audio's Reappraisal": -0.06, "audio's Distracting": 0.10,
                "audio's Uplifting": -0.04, "audio's Relaxing": -0.06, "audio's Suppressing": 0.14, "audio's Motivational": -0.02
            }
        },
        "Forgets Everyday Tasks": {
            "intercept": -0.8473,
            "w": {
                "Lyric's Reappraisal": -0.08, "Lyric's Distracting": 0.14, "Lyric's Uplifting": -0.03, "Lyric's Relaxing": -0.04,
                "Lyric's Suppressing": 0.10, "Lyric's Motivational": -0.05, "audio's Reappraisal": -0.05, "audio's Distracting": 0.16,
                "audio's Uplifting": -0.02, "audio's Relaxing": -0.04, "audio's Suppressing": 0.10, "audio's Motivational": -0.08
            }
        },
        "Difficulty Recalling Older Memories": {
            "intercept": -1.0986,
            "w": {
                "Lyric's Reappraisal": -0.10, "Lyric's Distracting": 0.10, "Lyric's Uplifting": -0.08, "Lyric's Relaxing": -0.06,
                "Lyric's Suppressing": 0.18, "Lyric's Motivational": -0.02, "audio's Reappraisal": -0.06, "audio's Distracting": 0.10,
                "audio's Uplifting": -0.06, "audio's Relaxing": -0.06, "audio's Suppressing": 0.18, "audio's Motivational": -0.02
            }
        },
        "Memory Worse Than a Year Ago": {
            "intercept": -0.8473,
            "w": {
                "Lyric's Reappraisal": -0.08, "Lyric's Distracting": 0.12, "Lyric's Uplifting": -0.06, "Lyric's Relaxing": -0.06,
                "Lyric's Suppressing": 0.16, "Lyric's Motivational": -0.04, "audio's Reappraisal": -0.06, "audio's Distracting": 0.12,
                "audio's Uplifting": -0.04, "audio's Relaxing": -0.06, "audio's Suppressing": 0.16, "audio's Motivational": -0.02
            }
        },
        "Visited Mental Health Professional": {
            "intercept": -1.3863,
            "w": {
                "Lyric's Reappraisal": -0.06, "Lyric's Distracting": 0.12, "Lyric's Uplifting": -0.04, "Lyric's Relaxing": -0.06,
                "Lyric's Suppressing": 0.20, "Lyric's Motivational": -0.02, "audio's Reappraisal": -0.04, "audio's Distracting": 0.12,
                "audio's Uplifting": -0.03, "audio's Relaxing": -0.06, "audio's Suppressing": 0.20, "audio's Motivational": -0.02
            }
        }
    }

    @staticmethod
    def sigmoid(z: float) -> float:
        """Sigmoid function to map values to [0,1] range."""
        return 1.0 / (1.0 + math.exp(-z))

    def map_music_to_indicators(self, features: Dict[str, float], sensitivity: float = 4.0) -> Dict[str, float]:
        """
        Map music features to Cognitive & Lifestyle Indicators.

        Args:
            features: Dictionary with the 12 music feature keys (0.0-1.0 values)
            sensitivity: Sensitivity multiplier for the mapping (default: 4.0)

        Returns:
            Dictionary with 6 cognitive indicator scores in [0,1] range
        """
        results = {}

        for indicator, spec in self.WEIGHTS.items():
            intercept, weights = spec["intercept"], spec["w"]
            z = intercept

            for feature, weight in weights.items():
                # Get feature value, default to 0.5 if missing
                x = features.get(feature, 0.5)
                z += sensitivity * weight * (x - 0.5)

            # Apply sigmoid and clamp to [0,1]
            score = max(0.0, min(1.0, self.sigmoid(z)))
            results[indicator] = score

        return results

    def get_cognitive_level_interpretation(self, score: float) -> str:
        """
        Get interpretation text for cognitive indicator scores.

        Args:
            score: Score value in [0,1] range

        Returns:
            Interpretation level string
        """
        if score >= 0.70:
            return "Strong indication"
        elif score >= 0.40:
            return "Moderate indication"
        else:
            return "Low indication"

    def get_cognitive_profile_summary(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the cognitive profile.

        Args:
            indicators: Dictionary of cognitive indicator scores

        Returns:
            Dictionary containing profile summary and insights
        """
        # Calculate overall cognitive risk score
        overall_score = sum(indicators.values()) / len(indicators)

        # Identify high-risk areas (score >= 0.70)
        high_risk_areas = [indicator for indicator, score in indicators.items() if score >= 0.70]

        # Identify low-risk areas (score <= 0.40)
        low_risk_areas = [indicator for indicator, score in indicators.items() if score <= 0.40]

        # Determine primary concerns (top 3 highest scores)
        sorted_indicators = sorted(indicators.items(), key=lambda x: x[1], reverse=True)
        primary_concerns = sorted_indicators[:3]

        return {
            "overall_cognitive_risk_score": round(overall_score, 3),
            "overall_risk_level": self.get_cognitive_level_interpretation(overall_score),
            "high_risk_areas": high_risk_areas,
            "low_risk_areas": low_risk_areas,
            "primary_concerns": [{"indicator": indicator, "score": round(score, 3), "level": self.get_cognitive_level_interpretation(score)} for indicator, score in primary_concerns],
            "total_indicators_assessed": len(indicators),
            "assessment_timestamp": datetime.now().isoformat()
        }

    def get_song_suggestions_for_cognitive_indicators(
        self,
        indicators: Dict[str, float],
        total_suggestions: int = 20,
        preferred_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get song suggestions based on cognitive indicator profile.

        Args:
            indicators: Dictionary of cognitive indicator scores
            total_suggestions: Number of song suggestions to generate
            preferred_languages: Optional list of preferred languages

        Returns:
            Dictionary containing song suggestions and reasoning
        """
        if not self.catalog_loaded:
            return {
                "recommendations": [],
                "recommendation_metadata": {
                    "total_recommendations": 0,
                    "music_database": "Not loaded",
                    "cognitive_approach": "Therapeutic music selection",
                    "timestamp": datetime.now().isoformat()
                }
            }

        # Determine therapeutic approach based on cognitive profile
        suggestions = []

        # Focus on memory and sleep support for high cognitive risk
        memory_indicators = ["Trouble Remembering Recent Events", "Forgets Everyday Tasks", "Difficulty Recalling Older Memories"]
        sleep_indicator = "Difficulty Sleeping"

        avg_memory_score = sum(indicators.get(ind, 0) for ind in memory_indicators) / len(memory_indicators)
        sleep_score = indicators.get(sleep_indicator, 0)

        # Filter songs by cognitive therapeutic properties
        therapeutic_songs = []

        for song in self.songs_database:
            if len(therapeutic_songs) >= total_suggestions * 2:  # Get more candidates than needed
                break

            # Check language preference
            if preferred_languages:
                song_lang = str(song.get('language', '')).lower()
                if not any(
                    song_lang == pref.lower() or
                    song_lang == pref[:2].lower() or
                    (pref.lower() == 'english' and song_lang == 'en') or
                    (pref.lower() in ['bengali', 'bangla'] and song_lang == 'bn')
                    for pref in preferred_languages
                ):
                    continue

            # Score song based on therapeutic properties for cognitive support
            therapeutic_score = 0.0
            reasons = []

            # Prioritize relaxing and uplifting music for cognitive support
            valence = song.get('Valence', 0.5)
            energy = song.get('energy', 0.5)
            tempo = song.get('tempo', 120)

            # Moderate tempo (60-100 BPM) is better for cognitive engagement
            if 60 <= tempo <= 100:
                therapeutic_score += 0.3
                reasons.append("Moderate tempo for cognitive engagement")

            # Positive valence for mood enhancement
            if valence >= 0.6:
                therapeutic_score += 0.2
                reasons.append("Positive emotional content")

            # Low to moderate energy to avoid overstimulation
            if 0.2 <= energy <= 0.6:
                therapeutic_score += 0.2
                reasons.append("Appropriate energy level")

            # Certain genres are better for cognitive therapy
            genre = str(song.get('genre', '')).lower()
            cognitive_friendly_genres = ['classical', 'jazz', 'folk', 'ambient', 'instrumental']
            if any(cog_genre in genre for cog_genre in cognitive_friendly_genres):
                therapeutic_score += 0.2
                reasons.append("Genre suitable for cognitive therapy")

            # Add mood considerations
            mood = str(song.get('mood', '')).lower()
            cognitive_friendly_moods = ['calm', 'peaceful', 'soothing', 'uplifting', 'gentle']
            if any(cog_mood in mood for cog_mood in cognitive_friendly_moods):
                therapeutic_score += 0.1
                reasons.append("Supportive mood for cognitive function")

            if therapeutic_score > 0.3:  # Minimum threshold
                therapeutic_songs.append({
                    "song": song,
                    "therapeutic_score": therapeutic_score,
                    "therapeutic_reasons": reasons,
                    "cognitive_target": self._determine_cognitive_target(song, indicators)
                })

        # Sort by therapeutic score
        therapeutic_songs.sort(key=lambda x: x["therapeutic_score"], reverse=True)

        # Format top suggestions
        for i, item in enumerate(therapeutic_songs[:total_suggestions], 1):
            song = item["song"]
            suggestions.append({
                "id": i,
                "song_id": song.get('id', i),
                "song_title": song.get('song_name', song.get('title', 'Unknown')),
                "artist": song.get('singer', song.get('artist', 'Unknown')),
                "genre": song.get('genre', ''),
                "language": song.get('language', ''),
                "mood": song.get('mood', ''),
                "released_date": str(song.get('released_date', '')) if song.get('released_date') else '',
                "tempo": song.get('tempo', 0),
                "valence": song.get('Valence', 0.5),
                "energy": song.get('energy', 0.5),
                "therapeutic_score": round(item["therapeutic_score"], 3),
                "cognitive_target": item["cognitive_target"],
                "therapeutic_reasons": item["therapeutic_reasons"],
                "recommendation_rank": i,
                "algorithm_used": "Cognitive_Indicators_Music_Mapping_v1.0",
                "category": "cognitive_indicators",
                "youtube_url": song.get('youtube link', ''),
                "spotify_url": song.get('spotify link', '')
            })

        # Generate cognitive profile summary
        profile_summary = self.get_cognitive_profile_summary(indicators)

        return {
            "recommendations": suggestions,
            "cognitive_profile": profile_summary,
            "therapeutic_approach": {
                "memory_support_level": "High" if avg_memory_score >= 0.70 else "Moderate" if avg_memory_score >= 0.40 else "Low",
                "sleep_support_level": "High" if sleep_score >= 0.70 else "Moderate" if sleep_score >= 0.40 else "Low",
                "overall_strategy": "Integrated cognitive and emotional support through music" if profile_summary["overall_cognitive_risk_score"] >= 0.60 else "General wellness and maintenance"
            },
            "recommendation_metadata": {
                "total_recommendations": len(suggestions),
                "algorithm": "Cognitive_Indicators_Music_Mapping_v1.0",
                "songs_analyzed": len(therapeutic_songs),
                "music_database": "d.xlsx" if self.catalog_loaded else "Not available",
                "cognitive_indicators_considered": list(indicators.keys()),
                "therapeutic_approach": "Evidence-based music selection for cognitive support",
                "timestamp": datetime.now().isoformat()
            }
        }

    def _determine_cognitive_target(self, song: Dict[str, Any], indicators: Dict[str, float]) -> str:
        """
        Determine the primary cognitive target for a song based on its properties.

        Args:
            song: Song dictionary
            indicators: Cognitive indicator scores

        Returns:
            String describing the primary cognitive target
        """
        tempo = song.get('tempo', 120)
        valence = song.get('Valence', 0.5)
        energy = song.get('energy', 0.5)
        mood = str(song.get('mood', '')).lower()

        # Memory support: moderate tempo, positive mood
        memory_indicators = ["Trouble Remembering Recent Events", "Forgets Everyday Tasks", "Difficulty Recalling Older Memories"]
        avg_memory_score = sum(indicators.get(ind, 0) for ind in memory_indicators) / len(memory_indicators)

        # Sleep support: low tempo, calming mood
        sleep_score = indicators.get("Difficulty Sleeping", 0)

        if tempo <= 80 and sleep_score >= 0.5:
            return "Sleep support"
        elif 60 <= tempo <= 100 and valence >= 0.6 and avg_memory_score >= 0.5:
            return "Memory engagement"
        elif "calm" in mood or "peaceful" in mood:
            return "Cognitive relaxation"
        elif valence >= 0.7:
            return "Mood enhancement"
        else:
            return "General cognitive support"

    def process_music_features_and_get_songs(
        self,
        music_features: Dict[str, float],
        total_suggestions: int = 20,
        preferred_languages: Optional[List[str]] = None,
        sensitivity: float = 4.0
    ) -> Dict[str, Any]:
        """
        Complete workflow: Process music features to get cognitive indicators and song suggestions.

        Args:
            music_features: Dictionary with 12 music feature scores (0.0-1.0)
            total_suggestions: Number of song suggestions to generate
            preferred_languages: Optional list of preferred languages
            sensitivity: Sensitivity multiplier for feature mapping

        Returns:
            Dictionary containing cognitive indicators and song suggestions
        """
        # Map features to cognitive indicators
        cognitive_indicators = self.map_music_to_indicators(music_features, sensitivity)

        # Add interpretation levels
        interpreted_indicators = {}
        for indicator, score in cognitive_indicators.items():
            interpreted_indicators[indicator] = {
                "score": round(score, 3),
                "level": self.get_cognitive_level_interpretation(score),
                "interpretation": f"{self.get_cognitive_level_interpretation(score).lower()} of {indicator.lower()}"
            }

        # Get song suggestions based on cognitive profile
        song_suggestions = self.get_song_suggestions_for_cognitive_indicators(
            cognitive_indicators,
            total_suggestions,
            preferred_languages
        )

        return {
            "input_features": {feature: round(music_features.get(feature, 0.5), 3) for feature in self.FEATURES},
            "cognitive_indicators": interpreted_indicators,
            "song_recommendations": song_suggestions,
            "processing_metadata": {
                "algorithm": "Music_Feature_to_Cognitive_Indicators_Mapping_v1.0",
                "sensitivity_used": sensitivity,
                "features_processed": len([f for f in self.FEATURES if f in music_features]),
                "indicators_generated": len(cognitive_indicators),
                "mapping_approach": "Logistic regression with weighted heuristics",
                "timestamp": datetime.now().isoformat()
            }
        }