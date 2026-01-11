"""
Down Syndrome Music Preference Mapping

Based on research mapping Down syndrome traits → Big Five (OCEAN) → music genre preferences.
This module provides specialized music genre mappings for Down syndrome patients.
"""

from typing import Dict, List, Any, Tuple


class DownSyndromeMusicMapping:
    """
    Specialized music mapping service for Down syndrome patients.

    Research-based mapping of Down syndrome traits to Big Five personality traits
    and corresponding music genre preferences for therapeutic benefits.
    """

    # Down syndrome specific genre mapping based on research
    DOWN_SYNDROME_GENRE_MAPPING = {
        "openness": {
            "description": "Sensory openness rather than abstract curiosity; strong response to melody, harmony, and novelty when emotionally positive",
            "preferred_genres": ["Classical", "Instrumental", "Folk", "Soft World Music", "Movie Soundtracks"],
            "avoid_genres": ["Heavy Metal", "Aggressive Rap", "Highly Dissonant Electronic"],
            "audio_characteristics": {
                "tempo_range": (60, 120),
                "valence_range": (0.5, 0.8),
                "energy_range": (0.3, 0.7),
                "acousticness_min": 0.4,
                "complexity": "low_to_moderate"
            }
        },
        "conscientiousness": {
            "description": "Lower task persistence; better performance with structure and repetition; strong response to routine",
            "preferred_genres": ["Children's Songs", "Repetitive Pop", "Nursery Rhymes", "Simple Religious Music"],
            "avoid_genres": ["Complex Jazz", "Progressive Rock", "Experimental Music"],
            "audio_characteristics": {
                "tempo_range": (80, 110),
                "valence_range": (0.6, 0.9),
                "energy_range": (0.4, 0.6),
                "repetition": "high",
                "predictability": "high"
            }
        },
        "extraversion": {
            "description": "High sociability; expressive affect; enjoyment of group interaction and movement",
            "preferred_genres": ["Pop", "Dance", "Upbeat Rock", "Group Songs", "Karaoke-style Music"],
            "avoid_genres": ["Ambient", "Minimalist", "Highly Experimental"],
            "audio_characteristics": {
                "tempo_range": (100, 140),
                "valence_range": (0.7, 0.9),
                "energy_range": (0.6, 0.9),
                "danceability_min": 0.7,
                "vocals": "clear_pronounced"
            }
        },
        "agreeableness": {
            "description": "High empathy and warmth; cooperative and affiliative behavior; positive emotional tone",
            "preferred_genres": ["Soft Rock", "Folk", "Acoustic Pop", "Religious/Spiritual Music", "Love Songs"],
            "avoid_genres": ["Aggressive Metal", "Dark Ambient", "Harsh Noise"],
            "audio_characteristics": {
                "tempo_range": (70, 110),
                "valence_range": (0.6, 0.85),
                "energy_range": (0.4, 0.7),
                "warmth": "high",
                "emotional_tone": "positive"
            }
        },
        "neuroticism": {
            "description": "Generally lower emotional instability; heightened sensitivity to stress but quick emotional recovery",
            "preferred_genres": ["Calm Instrumental", "Lullabies", "Slow Melodic Pop", "Nature-sound Music"],
            "avoid_genres": ["Heavy Metal", "Aggressive Rap", "Highly Dissonant Electronic"],
            "audio_characteristics": {
                "tempo_range": (50, 90),
                "valence_range": (0.4, 0.7),
                "energy_range": (0.2, 0.5),
                "arousal": "low",
                "safety": "emotional_safety_high"
            }
        }
    }

    # Genre weights for prioritization (higher = more preferred)
    GENRE_WEIGHTS = {
        "Classical": 0.9,
        "Instrumental": 0.9,
        "Folk": 0.8,
        "Soft World Music": 0.7,
        "Movie Soundtracks": 0.7,
        "Children's Songs": 0.9,
        "Repetitive Pop": 0.8,
        "Nursery Rhymes": 0.8,
        "Simple Religious Music": 0.8,
        "Pop": 0.8,
        "Dance": 0.7,
        "Upbeat Rock": 0.7,
        "Group Songs": 0.8,
        "Karaoke-style Music": 0.7,
        "Soft Rock": 0.7,
        "Acoustic Pop": 0.8,
        "Religious/Spiritual Music": 0.8,
        "Love Songs": 0.7,
        "Calm Instrumental": 0.9,
        "Lullabies": 0.9,
        "Slow Melodic Pop": 0.8,
        "Nature-sound Music": 0.8
    }

    # Audio feature filters for Down syndrome preferences
    AUDIO_FEATURE_FILTERS = {
        "tempo_min": 50,
        "tempo_max": 140,
        "valence_min": 0.4,
        "valence_max": 0.9,
        "energy_min": 0.2,
        "energy_max": 0.9,
        "acousticness_min": 0.2,
        "loudness_max": -5,  # dB
        "avoid_dissonance": True,
        "prefer_clear_vocals": True
    }

    def get_down_syndrome_genres_for_personality(self, big5_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Get genre preferences for Down syndrome patients based on Big Five personality scores.

        Args:
            big5_scores: Dictionary of Big Five trait scores (0-1 scale)

        Returns:
            List of genre preferences with weights and characteristics
        """
        recommendations = []

        for trait, score in big5_scores.items():
            if trait not in self.DOWN_SYNDROME_GENRE_MAPPING:
                continue

            trait_config = self.DOWN_SYNDROME_GENRE_MAPPING[trait]
            genres = trait_config["preferred_genres"]

            for genre in genres:
                weight = self.GENRE_WEIGHTS.get(genre, 0.5)

                # Adjust weight based on trait score
                # Higher scores for traits that align well with Down syndrome characteristics
                trait_weight_modifier = 0.8 + (score * 0.4)  # 0.8 to 1.2 multiplier
                final_weight = weight * trait_weight_modifier

                recommendations.append({
                    "trait": trait,
                    "genre": genre,
                    "weight": min(final_weight, 1.0),  # Cap at 1.0
                    "score": score,
                    "description": trait_config["description"],
                    "audio_characteristics": trait_config["audio_characteristics"]
                })

        # Sort by weight (highest first)
        recommendations.sort(key=lambda x: x["weight"], reverse=True)

        return recommendations

    def get_audio_filters_for_down_syndrome(self, big5_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Get audio feature filters for Down syndrome patients.

        Args:
            big5_scores: Optional Big Five scores to customize filters

        Returns:
            Dictionary of audio feature filters
        """
        filters = self.AUDIO_FEATURE_FILTERS.copy()

        if big5_scores:
            # Customize filters based on dominant traits
            dominant_trait = max(big5_scores.items(), key=lambda x: x[1])[0]

            if dominant_trait == "extraversion":
                # More energetic for extraverted patients
                filters["tempo_min"] = 90
                filters["energy_min"] = 0.5
            elif dominant_trait == "neuroticism":
                # Calmer for patients with higher neuroticism
                filters["tempo_max"] = 100
                filters["energy_max"] = 0.6
                filters["valence_min"] = 0.3
            elif dominant_trait == "conscientiousness":
                # More structured and predictable
                filters["tempo_min"] = 70
                filters["tempo_max"] = 110

        return filters

    def get_therapeutic_rationale(self, trait: str) -> str:
        """
        Get the therapeutic rationale for genre preferences based on Down syndrome traits.

        Args:
            trait: Big Five trait name

        Returns:
            Therapeutic rationale string
        """
        rationales = {
            "openness": "Music with rich but non-complex structure supports emotional exploration without cognitive overload",
            "conscientiousness": "Predictable rhythm and lyrics align with structured behavioral patterns and routine-based learning",
            "extraversion": "Energetic and social music reinforces engagement and positive mood through social connection",
            "agreeableness": "Emotionally warm and prosocial music matches affiliative personality traits and cooperative nature",
            "neuroticism": "Low-arousal, emotionally safe music supports regulation and comfort for stress-sensitive patients"
        }

        return rationales.get(trait, "Music selected for therapeutic compatibility with Down syndrome traits")

    def get_down_syndrome_fallback_songs(self) -> List[Dict[str, Any]]:
        """
        Get curated fallback songs specifically chosen for Down syndrome patients.
        Based on research mapping of Down syndrome traits to Big Five and music preferences.
        These are safe, well-researched selections when no personalized matches are found.

        Returns:
            List of fallback song configurations
        """
        return [
            # Openness to Experience - Sensory openness rather than abstract curiosity
            # Classical (simple/melodic), Instrumental, Folk, Soft world music, Movie soundtracks
            {"title": "Clair de Lune", "artist": "Claude Debussy", "genre": "Classical", "tempo": 66, "valence": 0.6, "trait": "openness", "reasoning": "Simple melodic structure supports emotional exploration without cognitive overload"},
            {"title": "Für Elise", "artist": "Ludwig van Beethoven", "genre": "Classical", "tempo": 120, "valence": 0.7, "trait": "openness", "reasoning": "Predictable yet emotionally engaging melody"},
            {"title": "River Flows in You", "artist": "Yiruma", "genre": "Instrumental", "tempo": 98, "valence": 0.6, "trait": "openness", "reasoning": "Pure instrumental focus on harmonic richness"},
            {"title": "La Vie En Rose", "artist": "Édith Piaf", "genre": "Soft World Music", "tempo": 72, "valence": 0.8, "trait": "openness", "reasoning": "International melody with emotional warmth"},
            {"title": "Theme from 'The Sound of Music'", "artist": "Julie Andrews", "genre": "Movie Soundtracks", "tempo": 76, "valence": 0.8, "trait": "openness", "reasoning": "Familiar movie melody with positive associations"},

            # Conscientiousness - Better performance with structure and repetition
            # Children's songs, Repetitive pop, Nursery rhymes, Simple religious/devotional music
            {"title": "Twinkle Twinkle Little Star", "artist": "Traditional Nursery Rhyme", "genre": "Nursery Rhymes", "tempo": 120, "valence": 0.8, "trait": "conscientiousness", "reasoning": "Highly predictable rhythm and structure for routine-based learning"},
            {"title": "ABC Song", "artist": "Children's Educational", "genre": "Children's Songs", "tempo": 100, "valence": 0.9, "trait": "conscientiousness", "reasoning": "Repetitive pattern supports structured behavioral patterns"},
            {"title": "You Are My Sunshine", "artist": "Jimmie Davis", "genre": "Folk", "tempo": 110, "valence": 0.9, "trait": "conscientiousness", "reasoning": "Simple, repetitive lyrics align with routine preference"},
            {"title": "Row, Row, Row Your Boat", "artist": "Traditional", "genre": "Nursery Rhymes", "tempo": 90, "valence": 0.8, "trait": "conscientiousness", "reasoning": "Round structure provides predictable repetition"},
            {"title": "Amazing Grace", "artist": "Traditional", "genre": "Devotional", "tempo": 65, "valence": 0.7, "trait": "conscientiousness", "reasoning": "Simple religious music with comforting repetition"},

            # Extraversion - High sociability, enjoyment of group interaction and movement
            # Pop, Dance, Upbeat rock, Group songs, Karaoke-style music
            {"title": "Happy", "artist": "Pharrell Williams", "genre": "Pop", "tempo": 160, "valence": 0.9, "trait": "extraversion", "reasoning": "Upbeat tempo reinforces positive mood and movement"},
            {"title": "Can't Stop the Feeling", "artist": "Justin Timberlake", "genre": "Dance", "tempo": 113, "valence": 0.9, "trait": "extraversion", "reasoning": "Energetic dance music for social engagement"},
            {"title": "Walking on Sunshine", "artist": "Katrina & The Waves", "genre": "Pop", "tempo": 136, "valence": 0.9, "trait": "extraversion", "reasoning": "Group-friendly song encourages participation"},
            {"title": "Count on Me", "artist": "Bruno Mars", "genre": "Pop", "tempo": 77, "valence": 0.8, "trait": "extraversion", "reasoning": "Social theme reinforces connection and cooperation"},
            {"title": "I Want It That Way", "artist": "Backstreet Boys", "genre": "Pop", "tempo": 105, "valence": 0.8, "trait": "extraversion", "reasoning": "Karaoke-friendly group song for social interaction"},

            # Agreeableness - High empathy and warmth, cooperative behavior
            # Soft rock, Folk, Acoustic pop, Religious/spiritual music, Love songs
            {"title": "Better Together", "artist": "Jack Johnson", "genre": "Acoustic Pop", "tempo": 88, "valence": 0.8, "trait": "agreeableness", "reasoning": "Warm, gentle tone matches affiliative personality traits"},
            {"title": "What a Wonderful World", "artist": "Louis Armstrong", "genre": "Jazz", "tempo": 66, "valence": 0.7, "trait": "agreeableness", "reasoning": "Prosocial lyrics with positive emotional tone"},
            {"title": "You've Got a Friend", "artist": "Carole King", "genre": "Soft Rock", "tempo": 68, "valence": 0.8, "trait": "agreeableness", "reasoning": "Emotionally warm music for cooperative nature"},
            {"title": "Here Comes the Sun", "artist": "The Beatles", "genre": "Folk Rock", "tempo": 116, "valence": 0.9, "trait": "agreeableness", "reasoning": "Hopeful themes match empathetic personality"},
            {"title": "Kumbaya", "artist": "Traditional Spiritual", "genre": "Religious/Spiritual", "tempo": 72, "valence": 0.7, "trait": "agreeableness", "reasoning": "Group spiritual song for cooperative participation"},

            # Neuroticism - Lower emotional instability, stress-sensitive but quick recovery
            # Calm instrumental, Lullabies, Slow melodic pop, Nature-sound-infused music
            {"title": "Weightless", "artist": "Marconi Union", "genre": "Ambient", "tempo": 60, "valence": 0.4, "trait": "neuroticism", "reasoning": "Scientifically designed for stress reduction"},
            {"title": "Morning Mood", "artist": "Edvard Grieg", "genre": "Classical", "tempo": 60, "valence": 0.5, "trait": "neuroticism", "reasoning": "Gentle orchestral piece for emotional regulation"},
            {"title": "Hush Little Baby", "artist": "Traditional Lullaby", "genre": "Lullabies", "tempo": 60, "valence": 0.6, "trait": "neuroticism", "reasoning": "Traditional lullaby for comfort and security"},
            {"title": "Sunrise", "artist": "Norah Jones", "genre": "Slow Melodic Pop", "tempo": 72, "valence": 0.6, "trait": "neuroticism", "reasoning": "Gentle pop melody for emotional safety"},
            {"title": "Ocean Waves", "artist": "Nature Sounds", "genre": "Nature Music", "tempo": 50, "valence": 0.3, "trait": "neuroticism", "reasoning": "Nature sounds for quick emotional recovery"}
        ]

    def validate_song_for_down_syndrome(self, song_features: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if a song is appropriate for Down syndrome patients based on audio features.

        Args:
            song_features: Dictionary containing song audio features

        Returns:
            Tuple of (is_appropriate: bool, reason: str)
        """
        filters = self.AUDIO_FEATURE_FILTERS

        # Check tempo
        tempo = song_features.get('tempo', 120)
        if tempo < filters["tempo_min"] or tempo > filters["tempo_max"]:
            return False, f"Tempo {tempo} outside optimal range {filters['tempo_min']}-{filters['tempo_max']}"

        # Check valence (emotional positivity)
        valence = song_features.get('valence', 0.5)
        if valence < filters["valence_min"] or valence > filters["valence_max"]:
            return False, f"Valence {valence} outside optimal range {filters['valence_min']}-{filters['valence_max']}"

        # Check energy
        energy = song_features.get('energy', 0.5)
        if energy < filters["energy_min"] or energy > filters["energy_max"]:
            return False, f"Energy {energy} outside optimal range {filters['energy_min']}-{filters['energy_max']}"

        # Check for potentially problematic genres
        avoid_genres = ["heavy metal", "aggressive rap", "hardcore punk", "harsh noise"]
        genre = song_features.get('genre', '').lower()
        if any(avoid in genre for avoid in avoid_genres):
            return False, f"Genre '{genre}' may be overstimulating or inappropriate"

        return True, "Song is appropriate for Down syndrome patients"