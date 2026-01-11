import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


def _parse_key_field(key_value):
    """Parse key field that may be text like 'D major' or numeric."""
    if pd.isna(key_value):
        return 0
    if isinstance(key_value, (int, float)):
        return int(key_value)
    if isinstance(key_value, str):
        # Convert musical keys to numbers (simple mapping)
        key_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
            'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        # Handle "D major" format
        key_clean = key_value.replace(' major', '').replace(' minor', '').strip()
        return key_map.get(key_clean, 0)
    return 0


def _parse_valence_field(valence_value):
    """Parse valence field that may be text like 'D major' or numeric."""
    if pd.isna(valence_value):
        return 0.0
    if isinstance(valence_value, (int, float)):
        return float(valence_value)
    if isinstance(valence_value, str):
        # If it's text like "D major", return a neutral valence
        if 'major' in valence_value or 'minor' in valence_value:
            return 0.5  # Neutral valence for musical keys
        try:
            return float(valence_value)
        except ValueError:
            return 0.5  # Default neutral valence
    return 0.5


def _parse_numeric_field(value):
    """Parse numeric field that may contain time values or other non-numeric data."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    # Handle datetime.time objects
    if hasattr(value, 'hour'):  # datetime.time object
        # Convert time to a reasonable numeric value (e.g., seconds since midnight / 86400)
        total_seconds = value.hour * 3600 + value.minute * 60 + value.second
        return total_seconds / 86400.0  # Normalize to 0-1 range
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            # If it's a time string like "04:45:00", convert it
            if ':' in str(value):
                try:
                    parts = str(value).split(':')
                    if len(parts) >= 2:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = int(parts[2]) if len(parts) > 2 else 0
                        total_seconds = hours * 3600 + minutes * 60 + seconds
                        return total_seconds / 86400.0  # Normalize to 0-1 range
                except (ValueError, IndexError):
                    pass
            return 0.0  # Default fallback for int fields
    return 0.0


class MusicCatalogService:
    """
    Service for loading and querying the music catalog from Excel file.
    """

    def __init__(self, catalog_path: str = None):
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self.songs = []
        self.catalog_loaded = False

    def _get_language_filter(self, birthplace_country: str) -> str:
        """
        Apply global language selection logic based on birthplace country.

        Args:
            birthplace_country: Patient's birthplace country string

        Returns:
            Language filter code ('bn' for Bengali, 'en' for English, etc.)
        """
        if not birthplace_country:
            return "en"  # Default fallback

        country_lower = birthplace_country.lower().strip()

        # Bangladesh → Bengali
        if country_lower == "bangladesh":
            return "bn"

        # English-speaking countries → English
        english_countries = [
            "usa", "united states", "uk", "united kingdom", "australia",
            "canada", "new zealand", "ireland", "south africa"
        ]
        if country_lower in english_countries:
            return "en"

        # Default fallback to English
        return "en"

    def load_catalog(self) -> bool:
        """
        Load music catalog from Excel file.
        Returns True if successful, False otherwise.
        """
        try:
            if not self.catalog_path or not self.catalog_path.exists():
                logger.warning(f"Music catalog file not found: {self.catalog_path}")
                return False

            logger.info(f"Loading music catalog from {self.catalog_path}")

            # Read file (Excel or CSV)
            if self.catalog_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(self.catalog_path)
            elif self.catalog_path.suffix.lower() == '.csv':
                df = pd.read_csv(self.catalog_path)
            else:
                raise ValueError(f"Unsupported file format: {self.catalog_path.suffix}")

            # Convert to list of dictionaries
            self.songs = []
            for _, row in df.iterrows():
                song = {
                    'id': len(self.songs) + 1,
                    'title': str(row.get('song_name', '')),
                    'artist': str(row.get('singer', '')),
                    'language': str(row.get('language', '')),
                    'genre': str(row.get('genre', '')),
                    'mood': str(row.get('mood', '')),
                    'technique': str(row.get('technique', '')),
                    'class_and_technique': str(row.get('class and technique table tag', '')),
                    'duration': str(row.get('duration', '0:00')),
                    'danceability': _parse_numeric_field(row.get('danceability', 0.0)),
                    'acousticness': _parse_numeric_field(row.get('acousticness', 0.0)),
                    'energy': _parse_numeric_field(row.get('energy', 0.0)),
                    'liveness': _parse_numeric_field(row.get('liveness', 0.0)),
                    'loudness': _parse_numeric_field(row.get('loudness', 0.0)),
                    'speechiness': _parse_numeric_field(row.get('speechiness', 0.0)),
                    'tempo': _parse_numeric_field(row.get('tempo', 0.0)),
                    'mode': int(_parse_numeric_field(row.get('mode', 0))),
                    'key': _parse_key_field(row.get('key', 'C')),
                    'valence': _parse_valence_field(row.get('Valence', 0.0)),  # Note: capital V from Excel
                    'time_signature': int(_parse_numeric_field(row.get('time_signature', 4))),
                    'lyrics_reappraisal': _parse_numeric_field(row.get("Lyric's Reappraisal", 0.0)),
                    'lyrics_distracting': _parse_numeric_field(row.get("Lyric's Distracting", 0.0)),
                    'lyrics_uplifting': _parse_numeric_field(row.get("Lyric's Uplifting", 0.0)),
                    'lyrics_relaxing': _parse_numeric_field(row.get("Lyric's Relaxing", 0.0)),
                    'lyrics_suppressing': _parse_numeric_field(row.get("Lyric's Suppressing", 0.0)),
                    'lyrics_motivational': _parse_numeric_field(row.get("Lyric's Motivational", 0.0)),
                    'audio_reappraisal': _parse_numeric_field(row.get("audio's Reappraisal", 0.0)),
                    'audio_distracting': _parse_numeric_field(row.get("audio's Distracting", 0.0)),
                    'audio_uplifting': _parse_numeric_field(row.get("audio's Uplifting", 0.0)),
                    'audio_relaxing': _parse_numeric_field(row.get("audio's Relaxing", 0.0)),
                    'audio_suppressing': _parse_numeric_field(row.get("audio's Suppressing", 0.0)),
                    'audio_motivational': _parse_numeric_field(row.get("audio's Motivational", 0.0)),
                    'instruments': str(row.get('used instruments in the song', '')),
                    'openness': _parse_numeric_field(row.get('Openness', 0.0)),
                    'conscientiousness': _parse_numeric_field(row.get('Conscientiousness', 0.0)),
                    'extraversion': _parse_numeric_field(row.get('Extraversion', 0.0)),
                    'agreeableness': _parse_numeric_field(row.get('Agreeableness', 0.0)),
                    'neuroticism': _parse_numeric_field(row.get('Neuroticism', 0.0)),
                    'released_date': row.get('released_date', None),
                    'youtube_link': str(row.get('youtube link', '')),
                    'spotify_link': str(row.get('spotify link', ''))
                }

                # Extract year from released_date
                if song['released_date']:
                    try:
                        # Handle numeric year values (e.g., 1962.0, 1975.0)
                        if isinstance(song['released_date'], (int, float)):
                            song['year'] = int(float(song['released_date']))
                        # Handle date strings (YYYY-MM-DD format)
                        elif isinstance(song['released_date'], str) and '-' in song['released_date']:
                            year_str = song['released_date'].split('-')[0]
                            song['year'] = int(year_str)
                        # Handle year strings
                        elif isinstance(song['released_date'], str) and song['released_date'].isdigit():
                            song['year'] = int(song['released_date'])
                        else:
                            song['year'] = None
                    except (ValueError, IndexError, TypeError):
                        song['year'] = None
                else:
                    song['year'] = None

                self.songs.append(song)

            self.catalog_loaded = True
            logger.info(f"Successfully loaded {len(self.songs)} songs from catalog")
            return True

        except Exception as e:
            logger.error(f"Failed to load music catalog: {e}")
            return False

    def get_songs_by_preferences(self,
                               condition: str,
                               preferences: Dict[str, Any] = None,
                               birthplace_country: str = None) -> List[Dict[str, Any]]:
        """
        Get songs filtered by condition and patient preferences.
        Applies global language filtering based on birthplace country.
        """
        if not self.catalog_loaded:
            if not self.load_catalog():
                return []

        if preferences is None:
            preferences = {}

        filtered_songs = self.songs.copy()

        # Filter by language preference FIRST (case-insensitive + synonyms)
        # User's explicit language preferences should take precedence over birthplace defaults
        preferred_languages = [
            lang.strip().lower()
            for lang in (preferences.get('preferred_languages') or [])
            if isinstance(lang, str) and lang.strip()
        ]

        # Map language preferences to catalog language codes
        language_mapping = {
            'english': 'en',
            'en': 'en',
            'bangla': 'bn',
            'bengali': 'bn',
            'bn': 'bn',
            'bangali': 'bn'
        }
        normalized_languages = [
            language_mapping.get(lang, lang)
            for lang in preferred_languages
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_languages = []
        for lang in normalized_languages:
            if lang not in seen:
                seen.add(lang)
                unique_languages.append(lang)

        # Apply user's explicit language preferences first
        if unique_languages:
            before_filter = filtered_songs
            filtered_songs = [
                song for song in filtered_songs
                if song.get('language', '').lower() in unique_languages
            ]
            if not filtered_songs:
                filtered_songs = before_filter
                logger.warning(f"No songs match user's preferred languages {unique_languages}, using all songs as fallback")
        else:
            # Only apply birthplace country filtering if user hasn't specified language preferences
            if birthplace_country:
                language_filter = self._get_language_filter(birthplace_country)
                if language_filter:
                    before_filter = filtered_songs
                    filtered_songs = [
                        song for song in filtered_songs
                        if song.get('language', '').lower() == language_filter
                    ]
                    if not filtered_songs:
                        # If no songs match the birthplace language filter, use all songs as fallback
                        filtered_songs = before_filter
                        logger.warning(f"No songs found for birthplace language filter '{language_filter}', using all songs as fallback")

        # Filter by genre preference
        preferred_genres = [
            genre.strip().lower()
            for genre in (preferences.get('favorite_genres') or [])
            if isinstance(genre, str) and genre.strip()
        ]
        if preferred_genres:
            before_filter = filtered_songs
            filtered_songs = [
                song for song in filtered_songs
                if any(
                    genre in song.get('genre', '').lower()
                    or song.get('genre', '').lower() in genre
                    for genre in preferred_genres
                )
            ]
            if not filtered_songs:
                filtered_songs = before_filter

        # Filter by instruments preference
        preferred_instruments = [
            instrument.strip().lower()
            for instrument in (preferences.get('instruments') or [])
            if isinstance(instrument, str) and instrument.strip()
        ]
        if preferred_instruments:
            before_filter = filtered_songs
            filtered_songs = [
                song for song in filtered_songs
                if any(
                    instrument in song.get('instruments', '').lower()
                    or song.get('instruments', '').lower() in instrument
                    for instrument in preferred_instruments
                )
            ]
            if not filtered_songs:
                filtered_songs = before_filter

        # Apply condition-specific filtering
        if condition.lower() == 'dementia':
            # For dementia: prefer calming, familiar music
            # Filter by relaxing lyrics/audio and lower energy
            before_filter = filtered_songs
            filtered_songs = [
                song for song in filtered_songs
                if (song['lyrics_relaxing'] > 0.5 or song['audio_relaxing'] > 0.5)
                and song['energy'] < 0.7
            ]
            if not filtered_songs:
                filtered_songs = before_filter
        elif condition.lower() == 'adhd':
            # For ADHD: prefer structured, moderate energy music
            before_filter = filtered_songs
            filtered_songs = [
                song for song in filtered_songs
                if 0.4 < song['energy'] < 0.8 and song['danceability'] > 0.3
            ]
            if not filtered_songs:
                filtered_songs = before_filter
        elif condition.lower() == 'down_syndrome':
            # For Down syndrome: prefer uplifting, positive music
            before_filter = filtered_songs
            filtered_songs = [
                song for song in filtered_songs
                if (song['lyrics_uplifting'] > 0.5 or song['audio_uplifting'] > 0.5)
                or song['valence'] > 0.6
            ]
            if not filtered_songs:
                filtered_songs = before_filter

        # If no songs match preferences, return some anyway
        if len(filtered_songs) == 0:
            logger.warning(f"No songs match preferences, returning all songs as fallback")
            filtered_songs = self.songs[:]  # Return all songs as fallback
        elif len(filtered_songs) < 3:
            logger.warning(f"Only {len(filtered_songs)} songs match preferences, returning more")
            filtered_songs = self.songs[:min(20, len(self.songs))]  # Return up to 20 songs as fallback

        return filtered_songs

    def get_songs_by_birthplace(self, birthplace_city: str, birthplace_country: str = None) -> List[Dict[str, Any]]:
        """
        Get songs based on birthplace (cultural relevance).
        """
        if not self.catalog_loaded:
            if not self.load_catalog():
                return []

        # For Bangladeshi patients, prioritize Bengali songs
        if birthplace_country and 'bangladesh' in birthplace_country.lower():
            bengali_songs = [
                song for song in self.songs
                if song['language'].lower() == 'bengali'
            ]
            if bengali_songs:
                return bengali_songs

        return self.songs

    def get_song_features(self, song_dict: Dict[str, Any]) -> List[float]:
        """
        Extract features for ML model from song dictionary.
        """
        return [
            song_dict.get('danceability', 0.0),
            song_dict.get('acousticness', 0.0),
            song_dict.get('energy', 0.0),
            song_dict.get('liveness', 0.0),
            song_dict.get('loudness', 0.0) / 20.0,  # Normalize loudness
            song_dict.get('speechiness', 0.0),
            song_dict.get('tempo', 0.0) / 200.0,  # Normalize tempo
            song_dict.get('valence', 0.0),
            song_dict.get('lyrics_reappraisal', 0.0),
            song_dict.get('lyrics_uplifting', 0.0),
            song_dict.get('audio_uplifting', 0.0)
        ]

    def search_songs(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search songs by title, artist, or genre.
        """
        if not self.catalog_loaded:
            if not self.load_catalog():
                return []

        query = query.lower()
        matching_songs = []

        for song in self.songs:
            if (query in song['title'].lower() or
                query in song['artist'].lower() or
                query in song['genre'].lower() or
                query in song['instruments'].lower()):
                matching_songs.append(song)
                if len(matching_songs) >= limit:
                    break

        return matching_songs

    def get_catalog_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the music catalog.
        """
        if not self.catalog_loaded:
            if not self.load_catalog():
                return {}

        languages = {}
        genres = {}
        total_songs = len(self.songs)

        for song in self.songs:
            lang = song['language']
            if lang in languages:
                languages[lang] += 1
            else:
                languages[lang] = 1

            genre = song['genre']
            if genre in genres:
                genres[genre] += 1
            else:
                genres[genre] = 1

        return {
            'total_songs': total_songs,
            'languages': languages,
            'genres': genres,
            'avg_tempo': sum(s['tempo'] for s in self.songs) / total_songs if total_songs > 0 else 0,
            'avg_valence': sum(s['valence'] for s in self.songs) / total_songs if total_songs > 0 else 0,
            'avg_energy': sum(s['energy'] for s in self.songs) / total_songs if total_songs > 0 else 0
        }

    def get_songs_by_big_five_traits(self, target_traits: List[str], songs_per_trait: int = 6) -> List[Dict[str, Any]]:
        """
        Get songs from d.csv based on Big Five personality traits.
        Returns songs that have high scores for the specified traits.

        Args:
            target_traits: List of traits to focus on (e.g., ['openness', 'extraversion'])
            songs_per_trait: Number of songs to return per trait

        Returns:
            List of songs from catalog with high scores for specified traits
        """
        if not self.catalog_loaded:
            if not self.load_catalog():
                return []

        personality_songs = []
        trait_mapping = {
            'openness': 'openness',
            'conscientiousness': 'conscientiousness',
            'extraversion': 'extraversion',
            'agreeableness': 'agreeableness',
            'neuroticism': 'neuroticism'
        }

        for trait in target_traits:
            if trait not in trait_mapping:
                continue

            # Sort songs by the trait score in descending order and get top songs
            trait_songs = sorted(
                self.songs,
                key=lambda x: x.get(trait_mapping[trait], 0.0),
                reverse=True
            )[:songs_per_trait]

            # Add category information
            for song in trait_songs:
                song_copy = song.copy()
                song_copy['category'] = 'big_five_personality'
                song_copy['trait'] = trait.title()  # Store which trait this song represents
                personality_songs.append(song_copy)

        # Remove duplicates while preserving order
        seen_songs = set()
        unique_songs = []
        for song in personality_songs:
            song_key = (song['title'], song['artist'])
            if song_key not in seen_songs:
                seen_songs.add(song_key)
                unique_songs.append(song)

        logger.info(f"Found {len(unique_songs)} unique songs from catalog for Big Five traits: {target_traits}")
        return unique_songs
