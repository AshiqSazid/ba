"""
Big Five Personality-Based Music Recommendation Service
Maps personality traits to genres and provides personalized song recommendations
"""

import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
import structlog

from app.services.utils import ConditionNormalizer

logger = structlog.get_logger(__name__)


class BigFiveRecommendationService:
    """
    Service for generating Big Five personality-based music recommendations
    from the d.xlsx music database
    """

    def __init__(self, excel_path: str = "data/d.xlsx"):
        """
        Initialize the recommendation service

        Args:
            excel_path: Path to the d.xlsx music database
        """
        self.excel_path = Path(excel_path)
        self.songs_df = None
        self.genre_songs_cache = {}

        # Import Down syndrome mapping service
        try:
            from app.services.down_syndrome_music_mapping import DownSyndromeMusicMapping
            self.down_syndrome_mapping = DownSyndromeMusicMapping()
        except ImportError:
            self.down_syndrome_mapping = None
            logger.warning("Down syndrome mapping service not available")

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

        # Dementia-specific genre mappings (complex, memory-triggering)
        self.dementia_genre_mapping = {
            "openness": {
                "genres": [
                    "Rabindra Sangeet", "Classical", "Semi-classical", "bn classical",
                    "Progressive Rock", "Art rock", "Jazz", "Fusion"
                ],
                "description": "Complex melodies for cognitive stimulation"
            },
            "conscientiousness": {
                "genres": [
                    "Baroque", "Orchestral", "Traditional Pop", "Filmi melodic",
                    "bn romantic", "Romantic ballads"
                ],
                "description": "Structured, familiar patterns for comfort"
            },
            "extraversion": {
                "genres": [
                    "Rock and roll", "Funk", "Soul", "Pop rock",
                    "Bangla Pop", "Filmi up-tempo"
                ],
                "description": "Energetic nostalgic music from youth"
            },
            "agreeableness": {
                "genres": [
                    "Devotional", "Romantic Ballads", "Folk", "bn folk",
                    "Soul", "R&B"
                ],
                "description": "Emotionally warm, familiar melodies"
            },
            "neuroticism": {
                "genres": [
                    "Ambient", "Classical", "Soft Rock", "Filmi emotional",
                    "Jazz Ballads"
                ],
                "description": "Calming, emotionally safe music"
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

    def _safe_float(self, value, default=0.5) -> float:
        """Safely convert value to float, handling non-numeric values."""
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def load_music_database(self) -> bool:
        """Load music database from Excel file"""
        try:
            self.songs_df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded {len(self.songs_df)} songs from music database")
            return True
        except Exception as e:
            logger.error(f"Failed to load music database: {e}")
            return False

    def _normalize_genre_name(self, genre: str) -> str:
        """Normalize genre name for better matching"""
        if pd.isna(genre):
            return ""

        genre_lower = genre.lower().strip()

        # Common normalizations
        normalizations = {
            "bn film song": "filmi",
            "bn modern/filmi": "filmi",
            "bn folk": "folk",
            "bn pop": "pop",
            "bn rock": "rock",
            "bn classical": "classical",
            "r&b": "r&b",
            "r&b / pop": "r&b",
            "soul, r&b, pop": "r&b",
            "rock and roll": "rock and roll",
            "rock and roll": "rock and roll"
        }

        for key, value in normalizations.items():
            if key in genre_lower:
                return value

        return genre_lower

    def get_songs_by_personality_trait(self, trait: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get songs that match a specific Big Five personality trait

        Args:
            trait: Big Five trait name ('openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism')
            limit: Maximum number of songs to return

        Returns:
            List of songs matching personality trait
        """
        if self.songs_df is None:
            if not self.load_music_database():
                return []

        if trait not in self.personality_genre_mapping:
            logger.error(f"Unknown personality trait: {trait}")
            return []

        trait_config = self.personality_genre_mapping[trait]
        target_genres = trait_config["genres"]

        matching_songs = []

        for _, song in self.songs_df.iterrows():
            song_genre = str(song.get('genre', '')).lower().strip()
            normalized_genre = self._normalize_genre_name(song.get('genre', ''))

            # Check for direct or partial matches
            for target_genre in target_genres:
                target_lower = target_genre.lower()

                # Direct match
                if target_lower in song_genre or song_genre in target_lower:
                    song_dict = song.to_dict()
                    song_dict['personality_trait'] = trait
                    song_dict['trait_description'] = trait_config['description']
                    song_dict['match_reason'] = f"Genre match: {target_genre}"
                    song_dict['confidence'] = 1.0 if target_lower == song_genre else 0.8
                    matching_songs.append(song_dict)
                    break
                # Match with normalized genre
                elif target_lower in normalized_genre or normalized_genre in target_lower:
                    song_dict = song.to_dict()
                    song_dict['personality_trait'] = trait
                    song_dict['trait_description'] = trait_config['description']
                    song_dict['match_reason'] = f"Genre match: {target_genre}"
                    song_dict['confidence'] = 0.9
                    matching_songs.append(song_dict)
                    break

        # Sort by confidence and additional factors
        matching_songs.sort(key=lambda x: (
            x['confidence'],
            self._safe_float(x.get('Valence', 0.5)) if trait in ['agreeableness', 'conscientiousness'] else 0.0,
            self._safe_float(x.get('energy', 0.5)) if trait == 'extraversion' else 0.0,
            self._safe_float(x.get('acousticness', 0.5)) if trait == 'openness' else 0.0
        ), reverse=True)

        return matching_songs[:limit]

    def get_big_five_recommendations(self, big5_responses: List[int], total_recommendations: int = 20,
                                preferred_languages: List[str] = None, condition: str = "dementia") -> Dict[str, Any]:
        """
        Generate Big Five personality-based song recommendations

        Args:
            big5_responses: List of 10 responses on 1-7 scale (BFI-2-X format)
            total_recommendations: Total number of songs to recommend
            preferred_languages: List of preferred language codes (['en', 'bn'])
            condition: Condition type ('dementia', 'down_syndrome', 'generic')

        Returns:
            Dictionary containing recommendations and personality analysis
        """
        if self.songs_df is None:
            if not self.load_music_database():
                return {"error": "Could not load music database"}

        # Normalize condition input
        normalized_condition = ConditionNormalizer.normalize_condition(condition)
        condition_key = normalized_condition.lower()

        # Calculate Big Five scores from responses
        big5_scores = self._calculate_big_five_scores(big5_responses)

        # Sort traits by score to get dominant personality traits
        sorted_traits = sorted(big5_scores.items(), key=lambda x: x[1], reverse=True)
        dominant_traits = [trait[0] for trait in sorted_traits[:3]]  # Top 3 traits

        logger.info(f"Dominant personality traits: {dominant_traits} for condition: {condition_key}")

        # Select appropriate genre mapping based on condition
        if condition_key == "down_syndrome":
            # Use Down syndrome specific mapping
            original_mapping = self.personality_genre_mapping
            self.personality_genre_mapping = self.down_syndrome_genre_mapping
            logger.info("Using Down syndrome-specific genre mapping")
        elif condition_key == "dementia":
            # Use dementia specific mapping
            original_mapping = self.personality_genre_mapping
            self.personality_genre_mapping = self.dementia_genre_mapping
            logger.info("Using dementia-specific genre mapping")
        else:
            # Use generic mapping
            original_mapping = None
            logger.info("Using generic genre mapping")

        # Get recommendations for each dominant trait
        recommendations = []
        songs_per_trait = max(3, total_recommendations // len(dominant_traits))

        for i, trait in enumerate(dominant_traits):
            trait_songs = self.get_songs_by_personality_trait(trait, limit=songs_per_trait)

            for song in trait_songs:
                # Calculate trait score (0-1 scale)
                trait_score_1_7 = max(1, min(7, round(big5_scores[trait] * 6 + 1)))
                song['trait_score_1_7'] = trait_score_1_7
                song['trait_score_0_1'] = big5_scores[trait]
                song['trait_rank'] = i + 1  # Which dominant trait this represents
                song['trait_level'] = 'High' if big5_scores[trait] >= 0.67 else 'Medium' if big5_scores[trait] >= 0.33 else 'Low'

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
            before_filter = unique_recommendations
            filtered_recommendations = []

            # Map language preferences to catalog codes
            unique_lang_prefs = ConditionNormalizer.normalize_languages(preferred_languages)

            logger.info(f"Big Five service applying language filter: {unique_lang_prefs}")

            for song in unique_recommendations:
                song_language = song.get('language', '').lower()
                song_lang_code = (
                    ConditionNormalizer.normalize_languages([song_language])[0]
                    if song_language
                    else song_language
                )
                if song_lang_code in unique_lang_prefs:
                    filtered_recommendations.append(song)

            if filtered_recommendations:
                unique_recommendations = filtered_recommendations
                logger.info(f"Big Five language filtering: {len(filtered_recommendations)}/{len(before_filter)} songs kept")
            else:
                logger.warning(f"No Big Five songs match language preferences {unique_lang_prefs}, using all {len(before_filter)} songs as fallback")

        # Remove duplicates while preserving order
        seen_songs = set()
        final_recommendations = []
        for song in unique_recommendations:
            song_key = (song.get('song_name', ''), song.get('singer', ''))
            if song_key not in seen_songs:
                seen_songs.add(song_key)
                final_recommendations.append(song)

        # Limit to requested number
        final_recommendations = unique_recommendations[:total_recommendations]

        # Generate personality summary
        personality_summary = self._generate_personality_summary(big5_scores, dominant_traits)

        # Restore original mapping if it was changed
        if original_mapping:
            self.personality_genre_mapping = original_mapping

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
            "condition": condition_key,
            "algorithm_metadata": {
                "algorithm": f"Big_Five_Personality_Genre_Mapping_{condition_key}_v1.0",
                "total_songs_analyzed": len(self.songs_df),
                "traits_considered": len(dominant_traits),
                "genre_database": "d.xlsx",
                "condition_specific": condition_key,
                "psychological_basis": f"BFI-2-X with {condition_key}-specific genre-preference research"
            }
        }

    def _calculate_big_five_scores(self, responses: List[int]) -> Dict[str, float]:
        """
        Calculate Big Five scores from 10-question assessment

        Args:
            responses: List of 10 responses on 1-7 scale

        Returns:
            Dictionary of Big Five trait scores (0-1 scale)
        """
        if not responses or len(responses) < 10:
            # Return neutral scores if insufficient data
            return {
                'openness': 0.5,
                'conscientiousness': 0.5,
                'extraversion': 0.5,
                'agreeableness': 0.5,
                'neuroticism': 0.5
            }

        # BFI-2-X 10-item mapping
        trait_mapping = {
            'extraversion': [0, 1],
            'agreeableness': [2, 3],
            'conscientiousness': [4, 5],
            'neuroticism': [6, 7],
            'openness': [8, 9]
        }

        scores = {}
        for trait, indices in trait_mapping.items():
            trait_responses = [responses[i] for i in indices if i < len(responses)]
            if trait_responses:
                # Convert 1-7 scale to 0-1 scale
                avg_score = sum(trait_responses) / len(trait_responses)
                normalized_score = (avg_score - 1) / 6

                # Reverse score neuroticism (higher score = lower neuroticism = more stability)
                if trait == 'neuroticism':
                    normalized_score = 1 - normalized_score

                scores[trait] = max(0.0, min(1.0, normalized_score))
            else:
                scores[trait] = 0.5

        return scores

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

    def get_trait_genre_details(self, trait: str) -> Dict[str, Any]:
        """Get detailed information about genres for a specific trait"""
        if trait not in self.personality_genre_mapping:
            return {"error": f"Unknown trait: {trait}"}

        trait_info = self.personality_genre_mapping[trait]

        # Count songs available for this trait
        songs = self.get_songs_by_personality_trait(trait, limit=100)

        # Group songs by actual genres found
        genre_counts = {}
        for song in songs:
            genre = song.get('genre', 'Unknown')
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        return {
            "trait": trait,
            "description": trait_info["description"],
            "target_genres": trait_info["genres"],
            "available_songs": len(songs),
            "genres_in_database": list(genre_counts.keys()),
            "genre_distribution": genre_counts,
            "sample_songs": songs[:5]  # Show 5 sample songs
        }


def main():
    """Example usage of the Big Five recommendation service"""
    service = BigFiveRecommendationService("data/d.xlsx")

    # Example Big Five responses (1-7 scale for 10 questions)
    example_responses = [5, 6, 4, 5, 3, 4, 2, 3, 6, 5]

    print("🎵 Big Five Personality Music Recommendation Service")
    print("=" * 60)

    # Get recommendations
    recommendations = service.get_big_five_recommendations(example_responses, total_recommendations=15)

    print(f"\n🧠 Personality Profile:")
    for trait, scores in recommendations["personality_scores"].items():
        print(f"  {trait.title()}: {scores['score_1_7']}/7 ({scores['level']})")

    print(f"\n📝 Personality Summary:")
    print(f"  {recommendations['personality_summary']}")

    print(f"\n🎶 Dominant Traits: {', '.join(recommendations['dominant_traits']).title()}")

    print(f"\n📊 Recommendations ({len(recommendations['recommendations'])} songs):")
    for i, song in enumerate(recommendations['recommendations'], 1):
        print(f"\n{i}. {song.get('song_name', 'Unknown')} by {song.get('singer', 'Unknown')}")
        print(f"   Genre: {song.get('genre', 'Unknown')}")
        print(f"   Mood: {song.get('mood', 'Unknown')}")
        print(f"   Personality: {song['personality_trait'].title()} (Score: {song['trait_score_1_7']}/7)")
        print(f"   Match: {song['match_reason']}")
        print(f"   Spotify: {song.get('spotify link', 'N/A')}")

    print(f"\n🔧 Algorithm: {recommendations['algorithm_metadata']['algorithm']}")
    print(f"   Songs analyzed: {recommendations['algorithm_metadata']['total_songs_analyzed']}")
    print(f"   Traits considered: {recommendations['algorithm_metadata']['traits_considered']}")

    def get_down_syndrome_recommendations(
        self,
        big_five_scores: Dict[str, float],
        total_recommendations: int = 20,
        preferred_languages: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get Big Five recommendations specifically for Down syndrome patients.

        Args:
            big_five_scores: Dictionary of Big Five trait scores (0-1 scale)
            total_recommendations: Number of songs to recommend
            preferred_languages: List of preferred languages

        Returns:
            Dictionary containing Down syndrome-specific recommendations
        """
        if not self.down_syndrome_mapping:
            logger.error("Down syndrome mapping service not available")
            return {"recommendations": [], "error": "Down syndrome mapping not available"}

        if self.songs_df is None:
            if not self.load_music_database():
                return {"recommendations": [], "error": "Failed to load music database"}

        logger.info("Generating Down syndrome-specific Big Five recommendations")

        # Get Down syndrome genre preferences
        ds_genres = self.down_syndrome_mapping.get_down_syndrome_genres_for_personality(big_five_scores)

        # Get audio filters for Down syndrome
        audio_filters = self.down_syndrome_mapping.get_audio_filters_for_down_syndrome(big_five_scores)

        recommendations = []

        for genre_pref in ds_genres:
            trait = genre_pref["trait"]
            genre = genre_pref["genre"]
            weight = genre_pref["weight"]

            # Filter songs by genre from the database
            genre_songs = self.songs_df[
                self.songs_df['genre'].str.lower().str.contains(genre.lower(), na=False)
            ]

            # Apply Down syndrome specific audio filtering
            genre_songs = genre_songs[
                (genre_songs['tempo'].between(audio_filters['tempo_min'], audio_filters['tempo_max'])) &
                (genre_songs['valence'].between(audio_filters['valence_min'], audio_filters['valence_max'])) &
                (genre_songs['energy'].between(audio_filters['energy_min'], audio_filters['energy_max']))
            ]

            # Apply language filtering if specified
            if preferred_languages:
                language_mask = genre_songs['language'].isin(preferred_languages)
                genre_songs = genre_songs[language_mask]

            # Sort by trait score if available, otherwise by valence
            if trait in genre_songs.columns:
                genre_songs = genre_songs.sort_values(trait, ascending=False)
            else:
                genre_songs = genre_songs.sort_values('valence', ascending=False)

            # Take top songs for this genre
            songs_per_genre = max(1, total_recommendations // len(ds_genres))
            top_songs = genre_songs.head(songs_per_genre)

            for _, song in top_songs.iterrows():
                song_dict = {
                    "song_name": song.get('song_name', song.get('title', 'Unknown')),
                    "singer": song.get('singer', song.get('artist', 'Unknown')),
                    "genre": song.get('genre', 'Unknown'),
                    "language": song.get('language', 'English'),
                    "mood": song.get('mood', 'Unknown'),
                    "released_date": song.get('released_date', 2020),
                    "tempo": song.get('tempo', 120),
                    "valence": song.get('valence', 0.5),
                    "energy": song.get('energy', 0.5),
                    "danceability": song.get('danceability', 0.5),
                    "acousticness": song.get('acousticness', 0.5),
                    "match_score": weight,
                    "personality_trait": trait,
                    "trait_score_1_7": max(1, min(7, round(big_five_scores.get(trait, 0.5) * 6 + 1))),
                    "trait_score_0_1": big_five_scores.get(trait, 0.5),
                    "match_reason": self.down_syndrome_mapping.get_therapeutic_rationale(trait),
                    "confidence": weight,
                    "youtube link": song.get('youtube_link', ''),
                    "spotify link": song.get('spotify_link', ''),
                    "used instruments in the song": song.get('used_instruments', ''),
                    "condition_specific": "down_syndrome"
                }
                recommendations.append(song_dict)

        # Sort by match score and limit to requested number
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        recommendations = recommendations[:total_recommendations]

        # Add Down syndrome specific metadata
        dominant_traits = sorted(big_five_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "recommendations": recommendations,
            "personality_profile": {
                "big_five_scores": {
                    trait: {
                        "score_0_1": round(score, 3),
                        "score_1_7": max(1, min(7, round(score * 6 + 1))),
                        "level": "High" if score >= 0.67 else "Medium" if score >= 0.33 else "Low"
                    }
                    for trait, score in big_five_scores.items()
                },
                "dominant_traits": [trait for trait, score in dominant_traits],
                "personality_summary": "Down syndrome-specific personality profile for music therapy"
            },
            "recommendation_metadata": {
                "total_recommendations": len(recommendations),
                "algorithm": "Big_Five_Down_Syndrome_Specific_v1.0",
                "songs_analyzed": len(self.songs_df),
                "traits_considered": len(dominant_traits),
                "music_database": "d.xlsx",
                "personality_assessment": "Big_Five_0-1_scale",
                "condition_specific": "Down syndrome",
                "therapeutic_approach": "Research-based genre mapping for Down syndrome traits"
            }
        }


if __name__ == "__main__":
    main()
