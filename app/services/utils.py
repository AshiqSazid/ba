"""
Utility functions for condition normalization, genre mapping, and validation
"""

import re
from typing import Dict, List, Any, Optional, Tuple

class ConditionNormalizer:
    """Normalize condition names and language codes"""

    CONDITION_MAPPINGS = {
        # Dementia variations
        'dementia': 'DEMENTIA',
        'dementia_alzheimers': 'DEMENTIA',
        'alzheimer': 'DEMENTIA',
        'alzheimers': 'DEMENTIA',
        'memory_loss': 'DEMENTIA',
        'cognitive_decline': 'DEMENTIA',

        # Down syndrome variations
        'down_syndrome': 'DOWN_SYNDROME',
        'down syndrome': 'DOWN_SYNDROME',
        'downs_syndrome': 'DOWN_SYNDROME',
        'downs syndrome': 'DOWN_SYNDROME',
        'down syndrome': 'DOWN_SYNDROME',
        'trisomy_21': 'DOWN_SYNDROME',
        'trisomy21': 'DOWN_SYNDROME',

        # ADHD variations
        'adhd': 'ADHD',
        'attention_deficit': 'ADHD',
        'attention_deficit_hyperactivity_disorder': 'ADHD',
        'add': 'ADHD',
        'attention_deficit_disorder': 'ADHD',

        # Default fallback
        'unknown': 'DEMENTIA',
        'generic': 'DEMENTIA',
        'other': 'DEMENTIA'
    }

    LANGUAGE_MAPPINGS = {
        # English variations
        'english': 'en',
        'eng': 'en',
        'en': 'en',
        'en-us': 'en',
        'en_gb': 'en',
        'en_uk': 'en',

        # Bengali variations
        'bengali': 'bn',
        'bangla': 'bn',
        'bn': 'bn',
        'bn-bd': 'bn',
        'bn_in': 'bn',

        # Other common languages
        'hindi': 'hi',
        'hi': 'hi',
        'spanish': 'es',
        'es': 'es',
        'french': 'fr',
        'fr': 'fr'
    }

    @classmethod
    def normalize_condition(cls, condition: str) -> str:
        """Normalize condition name to standard format"""
        if not condition:
            return 'DEMENTIA'

        # Clean the input
        clean_condition = condition.lower().strip()
        clean_condition = re.sub(r'[^\w\s]', '_', clean_condition)
        clean_condition = re.sub(r'\s+', '_', clean_condition)

        # Try exact match first
        if clean_condition in cls.CONDITION_MAPPINGS:
            return cls.CONDITION_MAPPINGS[clean_condition]

        # Try partial matches
        for key, value in cls.CONDITION_MAPPINGS.items():
            if key in clean_condition or clean_condition in key:
                return value

        # Fallback to dementia
        return 'DEMENTIA'

    @classmethod
    def normalize_languages(cls, languages: List[str]) -> List[str]:
        """Normalize list of language names to standard codes"""
        if not languages:
            return ['en']  # Default to English

        normalized = []
        for lang in languages:
            if not lang:
                continue

            clean_lang = lang.lower().strip()

            # Try exact match
            if clean_lang in cls.LANGUAGE_MAPPINGS:
                normalized.append(cls.LANGUAGE_MAPPINGS[clean_lang])
            else:
                # Try partial match
                for key, code in cls.LANGUAGE_MAPPINGS.items():
                    if key in clean_lang or clean_lang in key:
                        normalized.append(code)
                        break
                else:
                    # Unknown language, keep as-is but lowercased
                    normalized.append(clean_lang)

        # Remove duplicates while preserving order
        seen = set()
        unique_normalized = []
        for lang in normalized:
            if lang not in seen:
                seen.add(lang)
                unique_normalized.append(lang)

        return unique_normalized


class GenreMapper:
    """Handle genre name mapping between different naming conventions"""

    # Common genre variations and normalizations
    GENRE_NORMALIZATIONS = {
        # Classical variations
        'classical': 'Classical',
        'orchestral': 'Classical',
        'symphony': 'Classical',
        'baroque': 'Classical',
        'romantic': 'Classical',
        'piano': 'Classical',
        'instrumental': 'Classical',

        # Rock variations
        'rock': 'Rock',
        'rock and roll': 'Rock',
        'rock_n_roll': 'Rock',
        'alternative rock': 'Rock',
        'soft rock': 'Rock',
        'hard rock': 'Rock',
        'pop rock': 'Rock',
        'indie rock': 'Rock',

        # Pop variations
        'pop': 'Pop',
        'pop music': 'Pop',
        'dance pop': 'Pop',
        'electropop': 'Pop',
        'synthpop': 'Pop',

        # Folk variations
        'folk': 'Folk',
        'folk music': 'Folk',
        'acoustic folk': 'Folk',
        'traditional folk': 'Folk',
        'contemporary folk': 'Folk',

        # Children's music variations
        "children's": "Children's Songs",
        'children': "Children's Songs",
        'kids': "Children's Songs",
        'nursery rhyme': "Nursery Rhymes",
        'nursery rhymes': "Nursery Rhymes",
        'lullaby': 'Lullabies',
        'lullabies': 'Lullabies',

        # Religious/Spiritual variations
        'religious': 'Religious/Spiritual',
        'spiritual': 'Religious/Spiritual',
        'devotional': 'Religious/Spiritual',
        'gospel': 'Religious/Spiritual',

        # Electronic variations
        'electronic': 'Electronic',
        'edm': 'Electronic',
        'ambient': 'Electronic',
        'techno': 'Electronic',
        'house': 'Electronic',

        # Jazz variations
        'jazz': 'Jazz',
        'smooth jazz': 'Jazz',
        'jazz fusion': 'Jazz',
        'bebop': 'Jazz',

        # Hip Hop/Rap variations
        'hip hop': 'Hip Hop',
        'hip-hop': 'Hip Hop',
        'rap': 'Hip Hop',
        'r&b': 'R&B',
        'rnb': 'R&B',
        'soul': 'R&B'
    }

    # Down syndrome specific genre mappings to database genres
    DOWN_SYNDROME_TO_DB_MAPPING = {
        "Children's Songs": ["Pop", "Folk", "Rock", "Traditional"],
        "Nursery Rhymes": ["Folk", "Traditional", "Classical"],
        "Repetitive Pop": ["Pop", "Dance", "Electronic"],
        "Simple Religious Music": ["Religious/Spiritual", "Gospel", "Classical"],
        "Soft World Music": ["Folk", "Classical", "Traditional"],
        "Movie Soundtracks": ["Classical", "Pop", "Rock"],
        "Upbeat Rock": ["Rock", "Pop", "Alternative"],
        "Group Songs": ["Pop", "Rock", "Folk"],
        "Karaoke-style Music": ["Pop", "Rock", "R&B"],
        "Calm Instrumental": ["Classical", "Electronic", "Jazz"],
        "Nature-sound Music": ["Electronic", "Ambient"],
        "Children's Bedtime Music": ["Classical", "Ambient", "Lullabies"]
    }

    @classmethod
    def normalize_genre_name(cls, genre: str) -> str:
        """Normalize genre name to standard format"""
        if not genre:
            return "Unknown"

        clean_genre = genre.lower().strip()

        # Try exact match
        if clean_genre in cls.GENRE_NORMALIZATIONS:
            return cls.GENRE_NORMALIZATIONS[clean_genre]

        # Try partial matches
        for key, value in cls.GENRE_NORMALIZATIONS.items():
            if key in clean_genre or clean_genre in key:
                return value

        # Return cleaned version if no mapping found
        return genre.title()

    @classmethod
    def map_down_syndrome_genre_to_database(cls, ds_genre: str) -> List[str]:
        """Map Down syndrome specific genre to available database genres"""
        return cls.DOWN_SYNDROME_TO_DB_MAPPING.get(ds_genre, [ds_genre])


class BigFiveValidator:
    """Validate and handle Big Five personality scores"""

    @staticmethod
    def normalize_score(score_1_7: float) -> float:
        """Convert 1-7 scale to 0-1 scale"""
        try:
            score = float(score_1_7)
            # Clamp to valid range
            score = max(1.0, min(7.0, score))
            # Normalize to 0-1
            return (score - 1) / 6.0
        except (ValueError, TypeError):
            return 0.5  # Default to neutral

    @staticmethod
    def calculate_big_five_scores(responses: List[int]) -> Dict[str, float]:
        """Calculate Big Five scores from BFI-2-X 10-item responses"""
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
                avg_score = sum(trait_responses) / len(trait_responses)
                normalized = BigFiveValidator.normalize_score(avg_score)

                # Reverse score neuroticism (higher score = lower neuroticism = more stability)
                if trait == 'neuroticism':
                    normalized = 1 - normalized

                scores[trait] = max(0.0, min(1.0, normalized))
            else:
                scores[trait] = 0.5

        return scores

    @staticmethod
    def validate_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate and clamp score to valid range"""
        try:
            return max(min_val, min(max_val, float(score)))
        except (ValueError, TypeError):
            return 0.5


class AudioValidator:
    """Validate audio features for therapeutic appropriateness"""

    # Default audio feature ranges
    DEFAULT_RANGES = {
        'tempo': (50, 180),
        'valence': (0.0, 1.0),
        'energy': (0.0, 1.0),
        'danceability': (0.0, 1.0),
        'acousticness': (0.0, 1.0),
        'loudness': (-60, 0)  # dB
    }

    # Down syndrome specific ranges
    DOWN_SYNDROME_RANGES = {
        'tempo': (50, 140),
        'valence': (0.4, 0.9),
        'energy': (0.2, 0.9),
        'loudness_max': -5  # Maximum loudness
    }

    # Dementia specific ranges
    DEMENTIA_RANGES = {
        'tempo': (60, 120),
        'valence': (0.3, 0.8),
        'energy': (0.3, 0.7),
        'complexity': (0.3, 0.9)  # Cognitive complexity
    }

    @staticmethod
    def validate_audio_features(
        song_features: Dict[str, Any],
        condition: str = 'DEMENTIA'
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate song audio features for condition-specific appropriateness

        Returns:
            Tuple of (is_valid: bool, reason: str, normalized_features: dict)
        """
        # Choose appropriate ranges
        if condition == 'DOWN_SYNDROME':
            ranges = {**AudioValidator.DEFAULT_RANGES, **AudioValidator.DOWN_SYNDROME_RANGES}
        elif condition == 'DEMENTIA':
            ranges = {**AudioValidator.DEFAULT_RANGES, **AudioValidator.DEMENTIA_RANGES}
        else:
            ranges = AudioValidator.DEFAULT_RANGES

        normalized_features = song_features.copy()
        validation_errors = []

        # Validate each feature
        for feature, value in song_features.items():
            if feature not in ranges:
                continue

            try:
                numeric_value = float(value)
                min_val, max_val = ranges[feature]

                # Clamp to valid range
                normalized_value = max(min_val, min(max_val, numeric_value))
                normalized_features[feature] = normalized_value

                # Check if original value was out of range
                if numeric_value < min_val or numeric_value > max_val:
                    validation_errors.append(f"{feature} {numeric_value:.2f} outside optimal range {min_val}-{max_val}")

            except (ValueError, TypeError):
                validation_errors.append(f"{feature} value '{value}' is not numeric")

        # Additional validation for specific conditions
        if condition == 'DOWN_SYNDROME':
            # Avoid overstimulating genres
            genre = song_features.get('genre', '').lower()
            avoid_genres = ['heavy metal', 'aggressive rap', 'hardcore punk', 'harsh noise']
            if any(avoid in genre for avoid in avoid_genres):
                validation_errors.append(f"Genre '{genre}' may be overstimulating")

        is_valid = len(validation_errors) == 0
        reason = "All features within therapeutic ranges" if is_valid else "; ".join(validation_errors)

        return is_valid, reason, normalized_features


class FallbackProvider:
    """Provide fallback recommendations when no matches are found"""

    # Safe, well-researched fallback songs for different conditions
    FALLBACK_SONGS = {
        'DOWN_SYNDROME': [
            {
                'song_name': 'Clair de Lune',
                'singer': 'Claude Debussy',
                'genre': 'Classical',
                'tempo': 66,
                'valence': 0.6,
                'energy': 0.3,
                'language': 'en',
                'trait': 'openness',
                'reason': 'Simple melodic structure supports emotional exploration'
            },
            {
                'song_name': 'Twinkle Twinkle Little Star',
                'singer': 'Traditional',
                'genre': 'Folk',
                'tempo': 120,
                'valence': 0.8,
                'energy': 0.4,
                'language': 'en',
                'trait': 'conscientiousness',
                'reason': 'Predictable structure for routine-based learning'
            },
            {
                'song_name': 'Happy',
                'singer': 'Pharrell Williams',
                'genre': 'Pop',
                'tempo': 160,
                'valence': 0.9,
                'energy': 0.8,
                'language': 'en',
                'trait': 'extraversion',
                'reason': 'Upbeat tempo reinforces positive mood'
            },
            {
                'song_name': 'Better Together',
                'singer': 'Jack Johnson',
                'genre': 'Rock',
                'tempo': 88,
                'valence': 0.7,
                'energy': 0.4,
                'language': 'en',
                'trait': 'agreeableness',
                'reason': 'Warm, gentle tone matches affiliative personality'
            },
            {
                'song_name': 'Weightless',
                'singer': 'Marconi Union',
                'genre': 'Electronic',
                'tempo': 60,
                'valence': 0.4,
                'energy': 0.2,
                'language': 'en',
                'trait': 'neuroticism',
                'reason': 'Scientifically designed for stress reduction'
            }
        ],
        'DEMENTIA': [
            {
                'song_name': 'Moon River',
                'singer': 'Audrey Hepburn',
                'genre': 'Classical',
                'tempo': 72,
                'valence': 0.6,
                'energy': 0.3,
                'language': 'en',
                'trait': 'openness',
                'reason': 'Familiar melody for cognitive stimulation'
            },
            {
                'song_name': 'What a Wonderful World',
                'singer': 'Louis Armstrong',
                'genre': 'Jazz',
                'tempo': 66,
                'valence': 0.7,
                'energy': 0.4,
                'language': 'en',
                'trait': 'agreeableness',
                'reason': 'Nostalgic melody for memory recall'
            }
        ],
        'ADHD': [
            {
                'song_name': 'Vivaldi Four Seasons - Spring',
                'singer': 'Antonio Vivaldi',
                'genre': 'Classical',
                'tempo': 120,
                'valence': 0.8,
                'energy': 0.6,
                'language': 'en',
                'trait': 'openness',
                'reason': 'Structured complexity for focus'
            }
        ]
    }

    @classmethod
    def get_fallback_songs(cls, condition: str, count: int = 5) -> List[Dict[str, Any]]:
        """Get fallback songs for a condition"""
        condition_key = ConditionNormalizer.normalize_condition(condition)
        fallbacks = cls.FALLBACK_SONGS.get(condition_key, cls.FALLBACK_SONGS['DEMENTIA'])
        return fallbacks[:count]