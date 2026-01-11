"""
Improved Big Five Personality-Based Music Recommendation Service
Maps personality traits to genres and provides personalized song recommendations
with proper normalization, validation, and fallback handling
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import structlog

from .utils import (
    ConditionNormalizer, GenreMapper, BigFiveValidator,
    AudioValidator, FallbackProvider
)

logger = structlog.get_logger(__name__)


class ImprovedBigFiveRecommendationService:
    """
    Improved service for generating Big Five personality-based music recommendations
    with enhanced error handling and validation
    """

    def __init__(self, excel_path: str = "data/d.xlsx"):
        """
        Initialize the improved recommendation service

        Args:
            excel_path: Path to the d.xlsx music database
        """
        self.excel_path = Path(excel_path)
        self.songs_df = None
        self.genre_songs_cache = {}

        # Import Down syndrome mapping service
        try:
            from .down_syndrome_music_mapping import DownSyndromeMusicMapping
            self.down_syndrome_mapping = DownSyndromeMusicMapping()
        except ImportError:
            self.down_syndrome_mapping = None
            logger.warning("Down syndrome mapping service not available")

        # Genre mappings for different conditions
        self._load_genre_mappings()

    def _load_genre_mappings(self):
        """Load condition-specific genre mappings"""

        # Generic Big Five personality to genre mappings (for dementia/default)
        self.personality_genre_mapping = {
            "openness": {
                "genres": [
                    "Rabindra Sangeet", "Classical", "Semi-classical", "bn classical",
                    "Folk", "bn folk", "Folk-rock", "Sufi-Fusion", "Folk/Pop",
                    "Rock", "Art rock", "Alternative rock", "Rock/Pop Ballad",
                    "Soul", "R&B", "Motown", "Funk rock", "Psychedelic soul",
                    "Pop rock", "Soft rock", "Contemporary Folk", "Fusion"
                ],
                "description": "Creative and curious people enjoy complex, artistic music"
            },
            "conscientiousness": {
                "genres": [
                    "Pop", "Soft rock", "Traditional pop", "Ballad",
                    "Country", "Pop-country", "Filmi", "Filmi melodic",
                    "Romantic ballads", "bn romantic", "Filmi romantic",
                    "Adult Contemporary", "Contemporary Pop"
                ],
                "description": "Organized people prefer structured, melodic, clean-sounding music"
            },
            "extraversion": {
                "genres": [
                    "Dance-pop", "Pop-dance", "Eurodance", "Dance",
                    "Hip hop", "Rap", "Latin hip hop", "Hip Hop",
                    "Rock and roll", "Pop rock", "Britpop",
                    "Funk rock", "Soul-Pop", "Indi-pop", "bn pop", "Bangla Pop",
                    "Club Music", "EDM", "Latin Pop", "Reggae"
                ],
                "description": "Energetic social people enjoy upbeat, rhythmic, danceable music"
            },
            "agreeableness": {
                "genres": [
                    "Rabindra Sangeet", "Folk", "bn folk-pop", "Contemporary folk-pop",
                    "Romantic", "Filmi romantic", "Pop ballads",
                    "Soul", "R&B", "Pop-soul", "Soul, R&B, Pop",
                    "Soft rock", "Pop Ballads", "Devotional"
                ],
                "description": "Warm, cooperative people enjoy soothing, melodic, emotional music"
            },
            "neuroticism": {
                "genres": [
                    "Rock", "Grunge", "Alternative rock", "Post-grunge",
                    "Pop ballad", "Sad pop", "Emotional pop",
                    "RnB", "R&B emotional", "Soul emotional",
                    "Filmi emotional", "Soft rock", "Rock ballad",
                    "Indie", "Indie pop"
                ],
                "description": "Emotionally intense people use music for emotional expression and catharsis"
            }
        }

        # Down syndrome-specific genre mappings (developmentally appropriate)
        self.down_syndrome_genre_mapping = {
            "openness": {
                "genres": [
                    "Classical", "Instrumental", "Folk", "Soft World Music",
                    "Movie Soundtracks", "Children's Classical"
                ],
                "description": "Simple, melodic structure for emotional exploration"
            },
            "conscientiousness": {
                "genres": [
                    "Children's Songs", "Nursery Rhymes", "Repetitive Pop",
                    "Simple Religious Music", "Educational Songs"
                ],
                "description": "Predictable patterns for structured learning"
            },
            "extraversion": {
                "genres": [
                    "Pop", "Dance", "Upbeat Rock", "Group Songs",
                    "Karaoke-style Music", "Children's Pop"
                ],
                "description": "Social music for engagement and movement"
            },
            "agreeableness": {
                "genres": [
                    "Soft Rock", "Folk", "Acoustic Pop", "Religious/Spiritual",
                    "Love Songs", "Children's Love Songs"
                ],
                "description": "Warm, gentle music for cooperative interaction"
            },
            "neuroticism": {
                "genres": [
                    "Calm Instrumental", "Lullabies", "Slow Melodic Pop",
                    "Nature-sound Music", "Children's Bedtime Music"
                ],
                "description": "Soothing music for emotional regulation"
            }
        }

    def load_music_database(self) -> bool:
        """Load music database from Excel file with error handling"""
        try:
            if not self.excel_path.exists():
                logger.error(f"Music database not found at {self.excel_path}")
                return False

            self.songs_df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded {len(self.songs_df)} songs from music database")

            # Validate required columns exist
            required_columns = ['song_name', 'singer', 'genre']
            missing_columns = [col for col in required_columns if col not in self.songs_df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")

            return True
        except Exception as e:
            logger.error(f"Failed to load music database: {e}")
            return False

    def _safe_float(self, value, default=0.5) -> float:
        """Safely convert value to float, handling non-numeric values."""
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_songs_by_personality_trait(
        self,
        trait: str,
        limit: int = 10,
        condition: str = 'DEMENTIA'
    ) -> List[Dict[str, Any]]:
        """
        Get songs that match a specific Big Five personality trait with improved matching

        Args:
            trait: Big Five trait name ('openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism')
            limit: Maximum number of songs to return
            condition: Condition for genre mapping

        Returns:
            List of songs matching personality trait
        """
        if self.songs_df is None:
            if not self.load_music_database():
                return []

        # Normalize inputs
        trait = trait.lower().strip()
        condition = ConditionNormalizer.normalize_condition(condition)

        # Select appropriate mapping
        if condition == 'DOWN_SYNDROME':
            mapping = self.down_syndrome_genre_mapping
        else:
            mapping = self.personality_genre_mapping

        if trait not in mapping:
            logger.error(f"Unknown personality trait: {trait}")
            return []

        trait_config = mapping[trait]
        target_genres = trait_config["genres"]

        matching_songs = []

        for _, song in self.songs_df.iterrows():
            song_genre = str(song.get('genre', '')).lower().strip()
            normalized_genre = GenreMapper.normalize_genre_name(song.get('genre', ''))

            # Check for matches with improved genre handling
            found_match = False
            matched_genre = None

            for target_genre in target_genres:
                target_lower = target_genre.lower()

                # Direct match
                if target_lower in song_genre or song_genre in target_lower:
                    found_match = True
                    matched_genre = target_genre
                    break
                # Match with normalized genre
                elif target_lower in normalized_genre.lower() or normalized_genre.lower() in target_lower:
                    found_match = True
                    matched_genre = target_genre
                    break
                # For Down syndrome, check mapped database genres
                elif condition == 'DOWN_SYNDROME':
                    db_genres = GenreMapper.map_down_syndrome_genre_to_database(target_genre)
                    for db_genre in db_genres:
                        if db_genre.lower() in song_genre or song_genre in db_genre.lower():
                            found_match = True
                            matched_genre = db_genre
                            break
                    if found_match:
                        break

            if found_match:
                # Validate audio features
                song_features = {
                    'tempo': self._safe_float(song.get('tempo', 120)),
                    'valence': self._safe_float(song.get('Valence', 0.5)),
                    'energy': self._safe_float(song.get('energy', 0.5)),
                    'genre': matched_genre
                }

                is_valid, reason, normalized_features = AudioValidator.validate_audio_features(
                    song_features, condition
                )

                if is_valid:
                    song_dict = song.to_dict()
                    song_dict.update({
                        'personality_trait': trait,
                        'trait_description': trait_config['description'],
                        'match_reason': f"Genre match: {matched_genre}",
                        'confidence': 1.0 if target_lower == song_genre else 0.8,
                        'condition': condition
                    })
                    matching_songs.append(song_dict)

        # Sort by confidence and additional factors
        matching_songs.sort(key=lambda x: (
            x['confidence'],
            normalized_features['valence'] if trait in ['agreeableness', 'conscientiousness'] else 0.0,
            normalized_features['energy'] if trait == 'extraversion' else 0.0,
            normalized_features.get('acousticness', 0.5) if trait == 'openness' else 0.0
        ), reverse=True)

        return matching_songs[:limit]

    def get_big_five_recommendations(
        self,
        big5_responses: List[int],
        total_recommendations: int = 20,
        preferred_languages: List[str] = None,
        condition: str = "dementia"
    ) -> Dict[str, Any]:
        """
        Generate Big Five personality-based song recommendations with improved handling

        Args:
            big5_responses: List of 10 responses on 1-7 scale (BFI-2-X format)
            total_recommendations: Total number of songs to recommend
            preferred_languages: List of preferred language codes
            condition: Condition type

        Returns:
            Dictionary containing recommendations and personality analysis
        """
        try:
            # Normalize inputs
            condition = ConditionNormalizer.normalize_condition(condition)
            preferred_languages = ConditionNormalizer.normalize_languages(preferred_languages)

            # Load music database
            if self.songs_df is None:
                if not self.load_music_database():
                    logger.warning("Music database not available, using fallback recommendations")
                    return self._get_fallback_recommendations(big5_responses, total_recommendations, condition)

            # Calculate Big Five scores using improved validator
            big5_scores = BigFiveValidator.calculate_big_five_scores(big5_responses)

            # Sort traits by score to get dominant personality traits
            sorted_traits = sorted(big5_scores.items(), key=lambda x: x[1], reverse=True)
            dominant_traits = [trait[0] for trait in sorted_traits[:3]]  # Top 3 traits

            logger.info(f"Dominant personality traits: {dominant_traits} for condition: {condition}")

            # Get recommendations for each dominant trait
            recommendations = []
            songs_per_trait = max(3, total_recommendations // len(dominant_traits))

            for i, trait in enumerate(dominant_traits):
                trait_songs = self.get_songs_by_personality_trait(trait, limit=songs_per_trait, condition=condition)

                for song in trait_songs:
                    # Add trait information
                    trait_score_1_7 = max(1, min(7, round(big5_scores[trait] * 6 + 1)))
                    song.update({
                        'trait_score_1_7': trait_score_1_7,
                        'trait_score_0_1': big5_scores[trait],
                        'trait_rank': i + 1,
                        'trait_level': 'High' if big5_scores[trait] >= 0.67 else 'Medium' if big5_scores[trait] >= 0.33 else 'Low'
                    })

                recommendations.extend(trait_songs)

            # Remove duplicates while preserving order
            seen_songs = set()
            unique_recommendations = []
            for song in recommendations:
                song_key = (song.get('song_name', ''), song.get('singer', ''))
                if song_key not in seen_songs:
                    seen_songs.add(song_key)
                    unique_recommendations.append(song)

            # Apply language filtering if preferred languages are specified
            if preferred_languages:
                unique_recommendations = self._apply_language_filter(
                    unique_recommendations, preferred_languages
                )

            # If no recommendations found, use fallback
            if not unique_recommendations:
                logger.warning("No matching songs found, using fallback recommendations")
                return self._get_fallback_recommendations(big5_responses, total_recommendations, condition)

            # Limit to requested number
            final_recommendations = unique_recommendations[:total_recommendations]

            # Generate personality summary
            personality_summary = self._generate_personality_summary(big5_scores, dominant_traits)

            return {
                "personality_scores": {
                    trait: {
                        "score_0_1": round(score, 3),
                        "score_1_7": max(1, min(7, round(score * 6 + 1))),
                        "level": "High" if score >= 0.67 else "Medium" if score >= 0.33 else "Low"
                    }
                    for trait, score in big5_scores.items()
                },
                "dominant_traits": dominant_traits,
                "personality_summary": personality_summary,
                "recommendations": final_recommendations,
                "recommendation_count": len(final_recommendations),
                "condition": condition,
                "algorithm_metadata": {
                    "algorithm": f"Big_Five_Personality_Genre_Mapping_{condition}_v2.0",
                    "total_songs_analyzed": len(self.songs_df) if self.songs_df is not None else 0,
                    "traits_considered": len(dominant_traits),
                    "genre_database": "d.xlsx",
                    "condition_specific": condition,
                    "psychological_basis": f"BFI-2-X with {condition}-specific genre-preference research"
                }
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_fallback_recommendations(big5_responses, total_recommendations, condition)

    def _apply_language_filter(
        self,
        recommendations: List[Dict[str, Any]],
        preferred_languages: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply language filtering to recommendations"""
        if not preferred_languages:
            return recommendations

        before_filter = recommendations
        filtered_recommendations = []

        logger.info(f"Applying language filter: {preferred_languages}")

        for song in recommendations:
            song_language = str(song.get('language', '')).lower().strip()

            # Normalize song language
            song_lang_code = ConditionNormalizer.normalize_languages([song_language])
            if song_lang_code and song_lang_code[0] in preferred_languages:
                filtered_recommendations.append(song)

        if filtered_recommendations:
            logger.info(f"Language filtering: {len(filtered_recommendations)}/{len(before_filter)} songs kept")
            return filtered_recommendations
        else:
            logger.warning(f"No songs match language preferences {preferred_languages}, using all {len(before_filter)} songs as fallback")
            return before_filter

    def _generate_personality_summary(self, scores: Dict[str, float], dominant_traits: List[str]) -> str:
        """Generate a human-readable personality summary"""
        trait_descriptions = {
            'openness': 'Creative and curious',
            'conscientiousness': 'Organized and responsible',
            'extraversion': 'Energetic and social',
            'agreeableness': 'Warm and cooperative',
            'neuroticism': 'Emotionally stable'  # Reverse scored
        }

        music_preferences = {
            'openness': 'complex and artistic music like classical, folk, and alternative rock',
            'conscientiousness': 'structured and melodic music like pop, soft rock, and traditional ballads',
            'extraversion': 'upbeat and rhythmic music like dance-pop, rock, and hip hop',
            'agreeableness': 'soothing and emotional music like folk, soul, and romantic ballads',
            'neuroticism': 'emotionally expressive music for catharsis and mood regulation'
        }

        if len(dominant_traits) >= 2:
            primary = trait_descriptions.get(dominant_traits[0], 'Balanced')
            secondary = trait_descriptions.get(dominant_traits[1], 'balanced')
            primary_music = music_preferences.get(dominant_traits[0], 'diverse music')
            secondary_music = music_preferences.get(dominant_traits[1], 'various genres')

            return f"{primary} with {secondary.lower()} tendencies. This person enjoys {primary_music} along with {secondary_music}."
        elif len(dominant_traits) == 1:
            primary = trait_descriptions.get(dominant_traits[0], 'Balanced')
            primary_music = music_preferences.get(dominant_traits[0], 'a variety of music')
            return f"{primary} personality. This person prefers {primary_music}."
        else:
            return "Balanced personality across all traits. This person enjoys diverse music genres."

    def _get_fallback_recommendations(
        self,
        big5_responses: List[int],
        total_recommendations: int,
        condition: str
    ) -> Dict[str, Any]:
        """Get fallback recommendations when main recommendation fails"""
        try:
            big5_scores = BigFiveValidator.calculate_big_five_scores(big5_responses)
            fallback_songs = FallbackProvider.get_fallback_songs(condition, total_recommendations)

            # Add personality information to fallback songs
            for i, song in enumerate(fallback_songs):
                trait = song.get('trait', 'openness')
                score = big5_scores.get(trait, 0.5)
                song.update({
                    'trait_score_1_7': max(1, min(7, round(score * 6 + 1))),
                    'trait_score_0_1': score,
                    'trait_level': 'High' if score >= 0.67 else 'Medium' if score >= 0.33 else 'Low',
                    'confidence': 0.5,  # Lower confidence for fallback
                    'match_reason': 'Fallback recommendation for therapeutic appropriateness',
                    'condition': condition
                })

            return {
                "personality_scores": {
                    trait: {
                        "score_0_1": round(score, 3),
                        "score_1_7": max(1, min(7, round(score * 6 + 1))),
                        "level": "High" if score >= 0.67 else "Medium" if score >= 0.33 else "Low"
                    }
                    for trait, score in big5_scores.items()
                },
                "dominant_traits": sorted(big5_scores.items(), key=lambda x: x[1], reverse=True)[:3],
                "personality_summary": self._generate_personality_summary(
                    big5_scores,
                    [trait for trait, score in sorted(big5_scores.items(), key=lambda x: x[1], reverse=True)[:3]]
                ),
                "recommendations": fallback_songs,
                "recommendation_count": len(fallback_songs),
                "condition": condition,
                "algorithm_metadata": {
                    "algorithm": f"Big_Five_Personality_Fallback_{condition}_v2.0",
                    "fallback_used": True,
                    "condition_specific": condition,
                    "therapeutic_approach": "Research-based fallback recommendations for therapeutic appropriateness"
                }
            }

        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            # Return minimal fallback
            return {
                "error": "Unable to generate recommendations",
                "condition": condition,
                "fallback_used": True,
                "recommendations": [],
                "recommendation_count": 0
            }


def main():
    """Example usage of the improved Big Five recommendation service"""
    service = ImprovedBigFiveRecommendationService("data/d.xlsx")

    # Example Big Five responses (1-7 scale for 10 questions)
    example_responses = [5, 6, 4, 5, 3, 4, 2, 3, 6, 5]

    print("🎵 Improved Big Five Personality Music Recommendation Service")
    print("=" * 60)

    # Test with different conditions
    conditions = ["dementia", "down_syndrome", "unknown", "Down Syndrome"]

    for condition in conditions:
        print(f"\n🧠 Testing with condition: {condition}")
        recommendations = service.get_big_five_recommendations(
            example_responses,
            total_recommendations=5,
            preferred_languages=["English", "Bengali"],
            condition=condition
        )

        if 'error' in recommendations:
            print(f"   ❌ Error: {recommendations['error']}")
        else:
            print(f"   ✅ Condition detected: {recommendations['condition']}")
            print(f"   📊 Recommendations: {len(recommendations['recommendations'])} songs")
            print(f"   🎯 Algorithm: {recommendations['algorithm_metadata']['algorithm']}")

            for i, song in enumerate(recommendations['recommendations'][:2], 1):
                print(f"      {i}. {song.get('song_name', 'Unknown')} by {song.get('singer', 'Unknown')}")
                print(f"         Genre: {song.get('genre', 'Unknown')}, Trait: {song.get('personality_trait', 'Unknown')}")


if __name__ == "__main__":
    main()