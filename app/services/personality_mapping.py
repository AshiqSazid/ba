"""
Big Five Personality Mapping service for music recommendations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Set, Optional


class BigFivePersonalityMapping:
    """
    Maps Big Five personality traits to music preferences and audio characteristics.
    """

    BIG5_GENRE_MAPPING = {
        "openness": {
            (1, 2): ["Folk", "Contemporary Folk", "Soft Rock", "Traditional Pop"],
            (3, 4): ["Alternative Rock", "Folk Rock", "Fusion", "Progressive Rock"],
            (5, 7): ["Rabindra Sangeet", "Bn Classical", "Filmi Classical", "Experimental", "Psychedelic Soul"]
        },
        "conscientiousness": {
            (1, 2): ["Folk", "Bn Folk", "Traditional Pop"],
            (3, 4): ["Pop", "Soft Rock", "Filmi", "Adult Contemporary", "Country"],
            (5, 7): ["Classical", "Baroque", "Orchestral", "Classical Symphony"]
        },
        "extraversion": {
            (1, 2): ["Folk", "Acoustic", "Soft Rock"],
            (3, 4): ["Pop Rock", "Dance Pop", "Rock", "Funk", "Soul"],
            (5, 7): ["Dance Pop", "Hip Hop", "Rock and Roll", "EDM", "Club Music"]
        },
        "agreeableness": {
            (1, 2): ["Rock", "Hard Rock", "Alternative Rock"],
            (3, 4): ["Folk", "Acoustic", "Soul", "R&B", "Soft Rock"],
            (5, 7): ["Rabindra Sangeet", "Folk", "Devotional", "Romantic", "Pop Ballads"]
        },
        "neuroticism": {
            (1, 2): ["Heavy Metal", "Hard Rock", "Punk"],
            (3, 4): ["Grunge", "Alternative Rock", "Post Grunge", "Emo"],
            (5, 7): ["Ambient", "Classical", "Mediatative", "Pop Ballads", "Soul"]
        }
    }

    DOWN_SYNDROME_PERSONALITY_MAPPING = {
        "openness": {
            "tendency": "Sensory-emotional openness; responds to melody, harmony, and emotionally positive novelty",
            "preferred_genres": ["Classical", "Instrumental", "Folk", "Soft World Music", "Movie Soundtracks"],
            "avoid_genres": ["Heavy Metal", "Aggressive Rap", "Highly Dissonant Electronic"],
            "reasoning": "Rich but non-complex structure supports emotional exploration without cognitive overload."
        },
        "conscientiousness": {
            "tendency": "Lower task persistence; thrives with structure, repetition, and routine",
            "preferred_genres": ["Children's Songs", "Repetitive Pop", "Nursery Rhymes", "Simple Religious/Devotional"],
            "avoid_genres": ["Complex Jazz", "Progressive Rock", "Experimental"],
            "reasoning": "Predictable rhythm and lyrics align with structured behavioral patterns."
        },
        "extraversion": {
            "tendency": "High sociability and expressive affect; enjoys group interaction and movement",
            "preferred_genres": ["Pop", "Dance", "Upbeat Rock", "Group Songs", "Karaoke-style Music"],
            "avoid_genres": ["Ambient", "Minimalist", "Highly Experimental"],
            "reasoning": "Energetic social music reinforces engagement, movement, and positive mood."
        },
        "agreeableness": {
            "tendency": "High empathy, warmth, and positive emotional tone",
            "preferred_genres": ["Soft Rock", "Folk", "Acoustic Pop", "Religious/Spiritual", "Love Songs"],
            "avoid_genres": ["Aggressive Metal", "Dark Ambient", "Harsh Noise"],
            "reasoning": "Emotionally warm music mirrors affiliative tendencies and cooperative behavior."
        },
        "neuroticism": {
            "tendency": "Low-to-moderate emotional instability; sensitive to stress yet recovers quickly",
            "preferred_genres": ["Calm Instrumental", "Lullabies", "Slow Melodic Pop", "Nature-sound Music"],
            "avoid_genres": ["Heavy Metal", "Aggressive Rap", "Highly Dissonant Electronic"],
            "reasoning": "Low-arousal, emotionally safe music supports regulation and comfort."
        },
    }

    TRAIT_AUDIO_PROFILES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "extraversion": {
            "high": [
                {"key": "danceability", "min": 0.65, "weight": 3},
                {"key": "energy", "min": 0.65, "weight": 3},
                {"key": "tempo", "min": 100, "max": 180, "weight": 2},
                {"key": "speechiness", "max": 0.55, "weight": 1},
            ],
            "medium": [
                {"key": "danceability", "min": 0.5, "max": 0.75, "weight": 2},
                {"key": "energy", "min": 0.5, "max": 0.75, "weight": 2},
            ],
            "low": [
                {"key": "danceability", "max": 0.5, "weight": 2},
                {"key": "energy", "max": 0.5, "weight": 2},
                {"key": "acousticness", "min": 0.4, "weight": 1.5},
            ],
        },
        "openness": {
            "high": [
                {"key": "acousticness", "min": 0.3, "max": 0.85, "weight": 2},
                {"key": "energy", "min": 0.35, "max": 0.75, "weight": 1.5},
                {"key": "liveness", "min": 0.15, "weight": 1},
            ],
            "medium": [
                {"key": "acousticness", "min": 0.2, "max": 0.6, "weight": 1.5},
                {"key": "danceability", "min": 0.45, "max": 0.7, "weight": 1.5},
            ],
            "low": [
                {"key": "danceability", "min": 0.5, "weight": 1.5},
                {"key": "energy", "min": 0.55, "weight": 1.5},
            ],
        },
        "conscientiousness": {
            "high": [
                {"key": "tempo", "min": 60, "max": 140, "weight": 2},
                {"key": "danceability", "min": 0.45, "max": 0.65, "weight": 1.5},
                {"key": "liveness", "max": 0.4, "weight": 1},
            ],
            "medium": [
                {"key": "tempo", "min": 60, "max": 150, "weight": 1.5},
                {"key": "energy", "min": 0.4, "max": 0.7, "weight": 1.5},
            ],
            "low": [
                {"key": "tempo", "max": 120, "weight": 1},
                {"key": "speechiness", "max": 0.5, "weight": 1},
            ],
        },
        "agreeableness": {
            "high": [
                {"key": "acousticness", "min": 0.5, "weight": 2},
                {"key": "energy", "max": 0.6, "weight": 1.5},
                {"key": "loudness", "max": -6, "weight": 1},
            ],
            "medium": [
                {"key": "energy", "min": 0.4, "max": 0.65, "weight": 1.5},
                {"key": "danceability", "min": 0.5, "max": 0.75, "weight": 1},
            ],
            "low": [
                {"key": "energy", "min": 0.6, "weight": 1.5},
                {"key": "liveness", "min": 0.3, "weight": 1},
            ],
        },
        "neuroticism": {
            "high": [
                {"key": "energy", "max": 0.55, "weight": 2},
                {"key": "danceability", "max": 0.55, "weight": 1.5},
                {"key": "acousticness", "min": 0.45, "weight": 1.5},
                {"key": "tempo", "max": 120, "weight": 1},
            ],
            "medium": [
                {"key": "energy", "min": 0.4, "max": 0.7, "weight": 1.5},
                {"key": "danceability", "min": 0.45, "max": 0.7, "weight": 1.5},
            ],
            "low": [
                {"key": "energy", "min": 0.6, "weight": 2},
                {"key": "danceability", "min": 0.6, "weight": 1.5},
                {"key": "liveness", "min": 0.25, "weight": 1},
            ],
        },
    }

    def _to_1_7_scale(self, value: Optional[float]) -> float:
        """
        Convert normalized 0-1 scores to the 1-7 scale used by the research mappings.
        Values already in the target range pass through unchanged.
        """
        if value is None:
            return 4.0
        if 0.0 <= value <= 1.0:
            return (value * 6.0) + 1.0
        return max(1.0, min(7.0, value))

    def _score_band(self, value: float) -> str:
        """
        Determine the band (high/medium/low) for a given score.

        Args:
            value: Score value (1-7 scale)

        Returns:
            Band classification
        """
        value = self._to_1_7_scale(value)
        if value >= 5:
            return "high"
        if value <= 2:
            return "low"
        return "medium"

    def _get_audio_preferences(self, trait: str, score: float) -> List[Dict[str, Any]]:
        """
        Get audio preferences for a trait based on its score.

        Args:
            trait: Big Five trait name
            score: Trait score (1-7 scale)

        Returns:
            List of audio preference configurations
        """
        band = self._score_band(score)
        return self.TRAIT_AUDIO_PROFILES.get(trait, {}).get(band, [])

    def get_genres_for_personality(self, big5_scores: Dict) -> List[Dict[str, Any]]:
        """
        Get genre preferences based on Big Five personality scores.

        Args:
            big5_scores: Dictionary of Big Five trait scores

        Returns:
            List of genre preferences with audio characteristics
        """
        ordered_traits = sorted(big5_scores.items(), key=lambda item: item[1], reverse=True)
        genre_preferences: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        print(f"Processing Big 5 scores: {big5_scores}")

        for trait, score in ordered_traits:
            if trait not in self.BIG5_GENRE_MAPPING:
                continue

            score_1_7 = self._to_1_7_scale(score)
            print(f"  Processing {trait}: {score_1_7:.1f}")
            for (low, high), genres in self.BIG5_GENRE_MAPPING[trait].items():
                if low <= score_1_7 <= high:
                    print(f"    Range ({low}-{high}): {genres}")
                    for genre in genres:
                        pair = (trait, genre)
                        if pair in seen_pairs:
                            continue
                        genre_preferences.append(
                            {
                                "trait": trait,
                                "genre": genre,
                                "score": score_1_7,
                                "range": (low, high),
                                "audio_preferences": self._get_audio_preferences(trait, score),
                            }
                        )
                        seen_pairs.add(pair)
                    break

        print(f"🎵 Personality genres found ({len(genre_preferences)}). Top picks: {[entry['genre'] for entry in genre_preferences[:5]]}")
        return genre_preferences

    def _get_personality_interpretation(self, trait: str, score: float) -> str:
        """
        Get human-readable interpretation of a trait score.

        Args:
            trait: Big Five trait name
            score: Trait score (1-7 scale)

        Returns:
            Human-readable interpretation
        """
        interpretations = {
            "openness": {
                (1, 2): "Prefers familiar and conventional music",
                (3, 4): "Open to new musical experiences and diverse genres",
                (5, 7): "Highly creative, seeks innovative and experimental music"
            },
            "conscientiousness": {
                (1, 2): "Prefers spontaneous and relaxed musical styles",
                (3, 4): "Appreciates structured and organized musical compositions",
                (5, 7): "Prefers disciplined and complex musical arrangements"
            },
            "extraversion": {
                (1, 2): "Enjoys calm and introspective music",
                (3, 4): "Likes energetic and socially engaging music",
                (5, 7): "Drawn to highly stimulating and upbeat music"
            },
            "agreeableness": {
                (1, 2): "Enjoys intense and emotionally diverse music",
                (3, 4): "Prefers harmonious and warm musical content",
                (5, 7): "Seeks peaceful and cooperative musical themes"
            },
            "neuroticism": {
                (1, 2): "Emotionally stable, enjoys diverse musical moods",
                (3, 4): "Music helps manage stress and emotional expression",
                (5, 7): "Uses music for emotional regulation and comfort"
            }
        }

        value = self._to_1_7_scale(score)
        trait_interpretations = interpretations.get(trait, {})
        for range_key, interpretation in trait_interpretations.items():
            if range_key[0] <= value <= range_key[1]:
                return interpretation
        return "Moderate preference across musical styles"

    def calculate_big_five_from_responses(self, responses: List[int]) -> Dict[str, float]:
        """
        Calculate Big Five scores from 10-question assessment.
        Maps each response to 0-1 scale for personality trait scoring.

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

        # BFI-2-X 10-item mapping:
        # Items 1-2: Extraversion
        # Items 3-4: Agreeableness
        # Items 5-6: Conscientiousness
        # Items 7-8: Neuroticism (reverse scored)
        # Items 9-10: Openness

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

                # Reverse score neuroticism (higher score = lower neuroticism)
                if trait == 'neuroticism':
                    normalized_score = 1 - normalized_score

                scores[trait] = max(0.0, min(1.0, normalized_score))
            else:
                scores[trait] = 0.5

        return scores

    def get_dominant_traits(self, big5_scores: Dict[str, float], top_n: int = 2) -> List[str]:
        """
        Get the dominant personality traits based on scores.

        Args:
            big5_scores: Dictionary of Big Five trait scores
            top_n: Number of top traits to return

        Returns:
            List of dominant traits in descending order
        """
        sorted_traits = sorted(big5_scores.items(), key=lambda x: x[1], reverse=True)
        return [trait[0] for trait in sorted_traits[:top_n]]

    def get_personality_summary(self, big5_scores: Dict[str, float]) -> str:
        """
        Generate a human-readable personality summary.

        Args:
            big5_scores: Dictionary of Big Five trait scores

        Returns:
            Personality summary string
        """
        dominant_traits = self.get_dominant_traits(big5_scores, top_n=2)

        trait_descriptions = {
            'openness': 'Creative and curious',
            'conscientiousness': 'Organized and responsible',
            'extraversion': 'Energetic and social',
            'agreeableness': 'Warm and cooperative',
            'neuroticism': 'Emotionally stable'  # Reverse scored
        }

        if len(dominant_traits) >= 2:
            primary = trait_descriptions.get(dominant_traits[0], 'Balanced')
            secondary = trait_descriptions.get(dominant_traits[1], 'balanced')
            return f"{primary} with {secondary.lower()} tendencies"
        elif len(dominant_traits) == 1:
            return trait_descriptions.get(dominant_traits[0], 'Balanced personality')
        else:
            return "Balanced personality across all traits"

    def get_personality_insights(self, big5_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Get comprehensive insights about personality-based music preferences.

        Args:
            big5_scores: Dictionary of Big Five trait scores

        Returns:
            Dictionary containing personality insights and recommendations
        """
        insights = {}

        for trait, score in big5_scores.items():
            score_1_7 = self._to_1_7_scale(score)
            insights[trait] = {
                "score": score,
                "score_1_7": round(score_1_7, 2),
                "interpretation": self._get_personality_interpretation(trait, score),
                "audio_preferences": self._get_audio_preferences(trait, score),
                "genre_preferences": []
            }

        # Get genre preferences
        genre_prefs = self.get_genres_for_personality(big5_scores)
        for pref in genre_prefs:
            trait = pref["trait"]
            if trait in insights:
                insights[trait]["genre_preferences"].append(pref["genre"])

        # Add overall personality summary
        insights["personality_summary"] = self.get_personality_summary(big5_scores)
        insights["dominant_traits"] = self.get_dominant_traits(big5_scores)

        return insights

    def get_down_syndrome_personality_mapping(self, big5_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Provide conceptual mapping for Down syndrome traits → Big Five → music genres.
        Optionally annotates mapping with patient-specific Big Five scores.
        """
        trait_rows = []
        dominant_traits: List[str] = []

        if big5_scores:
            dominant_traits = self.get_dominant_traits(big5_scores, top_n=3)

        for trait, config in self.DOWN_SYNDROME_PERSONALITY_MAPPING.items():
            row: Dict[str, Any] = {
                "trait": trait,
                "down_syndrome_tendency": config["tendency"],
                "preferred_music_genres": config["preferred_genres"],
                "reasoning": config["reasoning"],
                "avoid_genres": config["avoid_genres"],
            }

            if big5_scores and trait in big5_scores:
                raw_score = big5_scores[trait]
                row.update(
                    {
                        "score_0_1": round(raw_score, 3),
                        "score_1_7": round(self._to_1_7_scale(raw_score), 2),
                        "level": self._score_band(raw_score).capitalize(),
                    }
                )

            trait_rows.append(row)

        summary_table = [
            {
                "trait": "Openness",
                "down_syndrome_tendency": "Sensory-emotional openness",
                "preferred_genres": ["Classical", "Instrumental", "Folk"],
            },
            {
                "trait": "Conscientiousness",
                "down_syndrome_tendency": "Prefers structure & repetition",
                "preferred_genres": ["Children's Songs", "Repetitive Pop"],
            },
            {
                "trait": "Extraversion",
                "down_syndrome_tendency": "High sociability",
                "preferred_genres": ["Pop", "Dance", "Upbeat Rock"],
            },
            {
                "trait": "Agreeableness",
                "down_syndrome_tendency": "Warm, empathetic",
                "preferred_genres": ["Folk", "Soft Rock", "Religious"],
            },
            {
                "trait": "Neuroticism",
                "down_syndrome_tendency": "Low–moderate emotional instability",
                "preferred_genres": ["Calm Instrumental", "Lullabies"],
            },
        ]

        return {
            "traits": trait_rows,
            "summary_table": summary_table,
            "dominant_traits": dominant_traits,
            "key_insight": (
                "Music preference in Down syndrome emphasizes emotional safety, social connection, "
                "and predictability more than complexity; aligning genres with Big Five tendencies "
                "captures those therapeutic needs."
            ),
        }
