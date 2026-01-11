from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import structlog

from sqlalchemy.orm import Session
from app.models.database import (
    Patient, Big5Score, TherapySession, TherapyRecommendation,
    TherapyFeedback, Song, BanditStats, UserActivity
)
from app.schemas.therapy import (
    TherapyCondition, FeedbackType, RecommendationRequest,
    RecommendationResponse, TherapyRecommendationCreate, TherapyFeedbackCreate
)
from app.services.ml_service import MLService
from app.services.music_catalog_service import MusicCatalogService
from app.services.big_five_recommendation_service import BigFiveRecommendationService
from app.services.down_syndrome_music_mapping import DownSyndromeMusicMapping
from app.services.utils import ConditionNormalizer, BigFiveValidator

logger = structlog.get_logger(__name__)


class TherapyService:
    """
    Core therapy service managing sessions and recommendations.
    """

    # Curated fallback songs grouped by trait so Personality Match always has content
    PERSONALITY_TRAIT_FALLBACKS = {
        "openness": [
            {"title": "Electric Feel", "artist": "MGMT", "genre": "Indie Fusion", "tempo": 102, "valence": 0.68, "arousal": 0.64, "language": "English", "year": 2007, "trait_score": 0.92},
            {"title": "Holocene", "artist": "Bon Iver", "genre": "Indie Folk", "tempo": 78, "valence": 0.45, "arousal": 0.38, "language": "English", "year": 2011, "trait_score": 0.81},
            {"title": "Everything in Its Right Place", "artist": "Radiohead", "genre": "Art Rock", "tempo": 74, "valence": 0.34, "arousal": 0.52, "language": "English", "year": 2000, "trait_score": 0.87},
            {"title": "Hyperballad", "artist": "Björk", "genre": "Electronic", "tempo": 111, "valence": 0.55, "arousal": 0.6, "language": "English", "year": 1995, "trait_score": 0.9},
            {"title": "Midnight City", "artist": "M83", "genre": "Dream Pop", "tempo": 104, "valence": 0.73, "arousal": 0.67, "language": "English", "year": 2011, "trait_score": 0.85},
            {"title": "Sunset Lover", "artist": "Petit Biscuit", "genre": "Chillwave", "tempo": 100, "valence": 0.57, "arousal": 0.42, "language": "Instrumental", "year": 2015, "trait_score": 0.8},
        ],
        "conscientiousness": [
            {"title": "Claire de Lune", "artist": "Claude Debussy", "genre": "Classical", "tempo": 66, "valence": 0.41, "arousal": 0.35, "language": "Instrumental", "year": 1905, "trait_score": 0.88},
            {"title": "Gymnopédie No.1", "artist": "Erik Satie", "genre": "Classical", "tempo": 72, "valence": 0.39, "arousal": 0.32, "language": "Instrumental", "year": 1888, "trait_score": 0.84},
            {"title": "River Flows in You", "artist": "Yiruma", "genre": "Piano", "tempo": 98, "valence": 0.56, "arousal": 0.4, "language": "Instrumental", "year": 2001, "trait_score": 0.8},
            {"title": "Canon in D", "artist": "Johann Pachelbel", "genre": "Classical", "tempo": 94, "valence": 0.6, "arousal": 0.45, "language": "Instrumental", "year": 1680, "trait_score": 0.82},
            {"title": "Kind of Blue", "artist": "Miles Davis", "genre": "Jazz", "tempo": 120, "valence": 0.48, "arousal": 0.53, "language": "Instrumental", "year": 1959, "trait_score": 0.79},
            {"title": "Time", "artist": "Hans Zimmer", "genre": "Soundtrack", "tempo": 90, "valence": 0.44, "arousal": 0.5, "language": "Instrumental", "year": 2010, "trait_score": 0.86},
        ],
        "extraversion": [
            {"title": "Can’t Stop the Feeling!", "artist": "Justin Timberlake", "genre": "Pop", "tempo": 113, "valence": 0.79, "arousal": 0.78, "language": "English", "year": 2016, "trait_score": 0.91},
            {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "genre": "Funk Pop", "tempo": 115, "valence": 0.84, "arousal": 0.83, "language": "English", "year": 2014, "trait_score": 0.92},
            {"title": "Levitating", "artist": "Dua Lipa", "genre": "Disco Pop", "tempo": 103, "valence": 0.76, "arousal": 0.74, "language": "English", "year": 2020, "trait_score": 0.88},
            {"title": "Feel It Still", "artist": "Portugal. The Man", "genre": "Indie Pop", "tempo": 79, "valence": 0.7, "arousal": 0.72, "language": "English", "year": 2017, "trait_score": 0.83},
            {"title": "On Top of the World", "artist": "Imagine Dragons", "genre": "Pop Rock", "tempo": 100, "valence": 0.85, "arousal": 0.77, "language": "English", "year": 2012, "trait_score": 0.81},
            {"title": "Good as Hell", "artist": "Lizzo", "genre": "Pop", "tempo": 79, "valence": 0.8, "arousal": 0.76, "language": "English", "year": 2016, "trait_score": 0.85},
        ],
        "agreeableness": [
            {"title": "Better Together", "artist": "Jack Johnson", "genre": "Acoustic", "tempo": 88, "valence": 0.73, "arousal": 0.42, "language": "English", "year": 2005, "trait_score": 0.83},
            {"title": "Banana Pancakes", "artist": "Jack Johnson", "genre": "Acoustic", "tempo": 88, "valence": 0.69, "arousal": 0.4, "language": "English", "year": 2005, "trait_score": 0.8},
            {"title": "Bloom", "artist": "The Paper Kites", "genre": "Indie Folk", "tempo": 118, "valence": 0.59, "arousal": 0.38, "language": "English", "year": 2010, "trait_score": 0.82},
            {"title": "First Day of My Life", "artist": "Bright Eyes", "genre": "Indie Folk", "tempo": 120, "valence": 0.52, "arousal": 0.36, "language": "English", "year": 2005, "trait_score": 0.79},
            {"title": "Good Old Fashioned Lover Boy", "artist": "Queen", "genre": "Pop", "tempo": 125, "valence": 0.8, "arousal": 0.55, "language": "English", "year": 1976, "trait_score": 0.77},
            {"title": "Home", "artist": "Edward Sharpe & The Magnetic Zeros", "genre": "Folk", "tempo": 88, "valence": 0.74, "arousal": 0.48, "language": "English", "year": 2009, "trait_score": 0.81},
        ],
        "neuroticism": [
            {"title": "Weightless", "artist": "Marconi Union", "genre": "Ambient", "tempo": 60, "valence": 0.34, "arousal": 0.2, "language": "Instrumental", "year": 2011, "trait_score": 0.9},
            {"title": "Breathe Me", "artist": "Sia", "genre": "Indie", "tempo": 112, "valence": 0.2, "arousal": 0.5, "language": "English", "year": 2004, "trait_score": 0.78},
            {"title": "Holocene", "artist": "Bon Iver", "genre": "Indie Folk", "tempo": 78, "valence": 0.45, "arousal": 0.38, "language": "English", "year": 2011, "trait_score": 0.82},
            {"title": "Night Owl", "artist": "Galimatias", "genre": "Chill", "tempo": 90, "valence": 0.37, "arousal": 0.33, "language": "Instrumental", "year": 2015, "trait_score": 0.76},
            {"title": "Sunrise", "artist": "Norah Jones", "genre": "Jazz Pop", "tempo": 110, "valence": 0.6, "arousal": 0.42, "language": "English", "year": 2004, "trait_score": 0.74},
            {"title": "Holocene (Acoustic)", "artist": "Amber Run", "genre": "Acoustic", "tempo": 86, "valence": 0.4, "arousal": 0.35, "language": "English", "year": 2017, "trait_score": 0.8},
        ],
    }

    def __init__(self, ml_service: MLService, catalog_path: str = None):
        self.ml_service = ml_service
        self.music_catalog = MusicCatalogService(catalog_path)
        self.big_five_recommendation_service = BigFiveRecommendationService(catalog_path)
        self.down_syndrome_mapping = DownSyndromeMusicMapping()
        # Try to load catalogs on initialization
        self.music_catalog.load_catalog()
        self.big_five_recommendation_service.load_music_database()

    def create_session(self, db: Session, patient_id: int, condition: TherapyCondition,
                      therapy_type: Optional[str] = None) -> TherapySession:
        """
        Create a new therapy session.
        """
        session_id = str(uuid.uuid4())
        db_session = TherapySession(
            patient_id=patient_id,
            session_id=session_id,
            condition=condition.value,
            therapy_type=therapy_type,
            status="active"
        )

        db.add(db_session)
        db.commit()
        db.refresh(db_session)

        logger.info(f"Created therapy session {session_id} for patient {patient_id}")
        return db_session

    def get_patient_with_scores(self, db: Session, patient_id: int) -> Optional[Patient]:
        """
        Get patient with their Big Five scores.
        """
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if patient:
            patient.big5_scores = db.query(Big5Score).filter(
                Big5Score.patient_id == patient_id
            ).all()
        return patient

    def get_available_songs(self, db: Session, condition: TherapyCondition,
                           preferences: Optional[Dict[str, Any]] = None) -> List[Song]:
        """
        Get available songs for recommendation based on condition and preferences.
        """
        query = db.query(Song).filter(Song.is_active == True)

        # Apply condition-based filtering
        if condition == TherapyCondition.DEMENTIA:
            # Prefer calm, familiar, culturally relevant music
            query = query.filter(
                (Song.genre.like('%classical%')) |
                (Song.genre.like('%folk%')) |
                (Song.genre.like('%traditional%')) |
                (Song.region.like('%Bangladesh%')) |
                (Song.region.like('%Bengali%'))
            )
        elif condition == TherapyCondition.ADHD:
            # Prefer structured, rhythmic music
            query = query.filter(
                (Song.genre.like('%classical%')) |
                (Song.genre.like('%instrumental%')) |
                (Song.tempo.between(60, 120))
            )
        elif condition == TherapyCondition.DOWN_SYNDROME:
            # Use Down syndrome specific music preferences based on research
            ds_filters = []

            # Classical and Instrumental (for Openness)
            ds_filters.extend([
                (Song.genre.like('%classical%')),
                (Song.genre.like('%instrumental%')),
                (Song.genre.like('%folk%')),
                (Song.genre.like('%movie soundtrack%')),
            ])

            # Children's and Repetitive (for Conscientiousness)
            ds_filters.extend([
                (Song.genre.like('%children%')),
                (Song.genre.like('%nursery%')),
                (Song.genre.like('%religious%')),
                (Song.genre.like('%repetitive%')),
            ])

            # Pop and Upbeat (for Extraversion)
            ds_filters.extend([
                (Song.genre.like('%pop%')),
                (Song.genre.like('%dance%')),
                (Song.genre.like('%upbeat%')),
                (Song.genre.like('%rock%') & ~Song.genre.like('%hard%')),
            ])

            # Soft and Warm (for Agreeableness)
            ds_filters.extend([
                (Song.genre.like('%soft%')),
                (Song.genre.like('%acoustic%')),
                (Song.genre.like('%religious%')),
                (Song.genre.like('%love%')),
            ])

            # Calm and Soothing (for Neuroticism)
            ds_filters.extend([
                (Song.genre.like('%calm%')),
                (Song.genre.like('%lullaby%')),
                (Song.genre.like('%ambient%')),
                (Song.genre.like('%nature%')),
            ])

            # Apply tempo and valence filters appropriate for Down syndrome
            tempo_filter = (Song.tempo.between(50, 140))
            valence_filter = (Song.valence >= 0.4)  # Generally positive
            energy_filter = (Song.energy.between(0.2, 0.9))

            # Build query with OR conditions for genres AND required audio characteristics
            if ds_filters:
                genre_query = query.filter(
                    tempo_filter,
                    valence_filter,
                    energy_filter
                ).filter(
                    # Any of the preferred genres
                    ds_filters[0]
                )

                # Add remaining genre filters with OR logic
                for genre_filter in ds_filters[1:]:
                    genre_query = genre_query.union(
                        query.filter(
                            tempo_filter,
                            valence_filter,
                            energy_filter
                        ).filter(genre_filter)
                    )

                query = genre_query
            else:
                # Fallback to basic filters if no genre filters
                query = query.filter(tempo_filter, valence_filter, energy_filter)

        # Apply user preferences
        if preferences:
            if 'genres' in preferences:
                query = query.filter(Song.genre.in_(preferences['genres']))
            if 'artists' in preferences:
                query = query.filter(Song.artist.in_(preferences['artists']))
            if 'language' in preferences:
                query = query.filter(Song.language == preferences['language'])
            if 'tempo_range' in preferences:
                min_tempo, max_tempo = preferences['tempo_range']
                query = query.filter(Song.tempo.between(min_tempo, max_tempo))

        # Limit results and return
        songs = query.limit(1000).all()
        logger.info(f"Found {len(songs)} available songs for {condition}")
        return songs

    def generate_recommendations(self, db: Optional[Session], request) -> RecommendationResponse:
        """
        Generate music recommendations for a patient.
        Accepts both RecommendationRequest and RecommendationRequestLegacy for backward compatibility.
        """
        try:
            # Generate a session ID without database
            session_id = str(uuid.uuid4())

            # Handle both new and legacy request formats
            if hasattr(request, 'intake'):
                # New format: RecommendationRequest
                intake_data = request.intake
                condition = intake_data.get('condition', 'dementia')
                patient_info = intake_data
                big_five_responses = intake_data.get('bigFiveResponses')
            else:
                # Legacy format: RecommendationRequestLegacy
                from app.schemas.therapy import RecommendationRequestLegacy
                if isinstance(request, RecommendationRequestLegacy):
                    intake_data = {
                        'name': request.patient_info.get('name', 'Unknown'),
                        'dateOfBirth': request.patient_info.get('dateOfBirth', '1950-01-01'),
                        'sex': request.patient_info.get('sex', 'other'),
                        'condition': request.condition.value if hasattr(request.condition, 'value') else str(request.condition),
                        'birthplaceCity': request.patient_info.get('birthplaceCity'),
                        'birthplaceCountry': request.patient_info.get('birthplaceCountry'),
                        'preferredLanguages': request.patient_info.get('preferredLanguages', []),
                        'favoriteGenres': request.patient_info.get('favoriteGenres', []),
                        'instruments': request.patient_info.get('instruments', []),
                        'favoriteMusician': request.patient_info.get('favoriteMusician'),
                        'favoriteSeason': request.patient_info.get('favoriteSeason'),
                        'naturalElements': request.patient_info.get('naturalElements', []),
                        'difficultySleeping': request.patient_info.get('difficultySleeping', False),
                        'troubleRemembering': request.patient_info.get('troubleRemembering', False),
                        'forgetsEverydayThings': request.patient_info.get('forgetsEverydayThings', False),
                        'difficultyRecallingOldMemories': request.patient_info.get('difficultyRecallingOldMemories', False),
                        'memoryWorseThanYearAgo': request.patient_info.get('memoryWorseThanYearAgo', False),
                        'visitedMentalHealthProfessional': request.patient_info.get('visitedMentalHealthProfessional', False),
                        'bigFiveResponses': request.big_five_responses,
                        'big_five': request.big_five_responses
                    }
                    condition = request.condition.value if hasattr(request.condition, 'value') else str(request.condition)
                    patient_info = request.patient_info
                    big_five_responses = request.big_five_responses
                else:
                    # Fallback for unknown request types
                    logger.warning(f"Unknown request type: {type(request)}")
                    intake_data = {
                        'name': 'Unknown',
                        'dateOfBirth': '1950-01-01',
                        'sex': 'other',
                        'condition': 'dementia'
                    }
                    condition = TherapyCondition.DEMENTIA
                    patient_info = {}
                    big_five_responses = None

            # Parse Big Five responses if provided
            big_five_scores = None
            if big_five_responses:
                try:
                    if isinstance(big_five_responses, dict):
                        # Already categorized format - calculate scores from categorized responses
                        # Flatten the categorized responses into a single list
                        all_responses = []
                        if isinstance(big_five_responses, dict):
                            for trait_responses in big_five_responses.values():
                                if isinstance(trait_responses, list):
                                    all_responses.extend(trait_responses)

                        if all_responses:
                            big_five_scores = self.ml_service.calculate_big_five_scores(all_responses)
                        else:
                            big_five_scores = None
                    elif isinstance(big_five_responses, list):
                        # List format - need to determine if it's BFI-2 or simplified
                        if len(big_five_responses) == 50:
                            big_five_scores = self.ml_service.calculate_big_five_scores(big_five_responses)
                        elif len(big_five_responses) == 10:
                            big_five_scores = BigFiveValidator.calculate_big_five_scores(big_five_responses)
                        else:
                            logger.warning(f"Unexpected Big Five response length: {len(big_five_responses)}")
                    logger.info(f"Calculated Big Five scores: {big_five_scores}")
                except Exception as e:
                    logger.error(f"Error calculating Big Five scores: {e}")
                    big_five_scores = None

            # Normalize condition string to handle variants like "Down Syndrome"
            condition_value = condition
            if not isinstance(condition, str):
                condition_value = condition.value if hasattr(condition, 'value') else str(condition)
            normalized_condition = ConditionNormalizer.normalize_condition(condition_value)

            # Convert to TherapyCondition enum
            try:
                if normalized_condition == 'DEMENTIA':
                    therapy_condition = TherapyCondition.DEMENTIA
                elif normalized_condition == 'ADHD':
                    therapy_condition = TherapyCondition.ADHD
                elif normalized_condition == 'DOWN_SYNDROME':
                    therapy_condition = TherapyCondition.DOWN_SYNDROME
                else:
                    therapy_condition = TherapyCondition.DEMENTIA
            except Exception as e:
                logger.error(f"Error parsing condition: {e}")
                therapy_condition = TherapyCondition.DEMENTIA
            condition_str = therapy_condition.value

            # Store birthplace country and patient info for global language filtering
            self.current_birthplace_country = patient_info.get('birthplaceCountry')
            self.current_patient_info = patient_info

            # Get personalized songs based on ALL Excel data
            personalized_songs = self._get_personalized_songs(therapy_condition, patient_info)

            # Ensure Personality Match always receives songs, even if catalog isn't loaded
            big_five_songs = self._get_personality_match_songs(big_five_scores, therapy_condition)

            # Combine all songs
            all_recommended_songs = personalized_songs + big_five_songs

            # Format songs for response
            formatted_recommendations = []
            for i, song in enumerate(all_recommended_songs):
                normalized_year = self._normalize_song_year(song.get('year'))
                if normalized_year is None:
                    normalized_year = self._normalize_song_year(song.get('released_date'))
                if normalized_year is None:
                    normalized_year = 2020
                youtube_link = song.get('youtube_link')
                if not youtube_link:
                    youtube_link = song.get('youtube link', song.get('youtube_url', ''))
                spotify_link = song.get('spotify_link')
                if not spotify_link:
                    spotify_link = song.get('spotify link', song.get('spotify_url', ''))
                # Create recommendation with proper URLs from Excel file
                recommendation = {
                    'id': i + 1,
                    'song_id': song.get('id', i + 1),
                    'song_title': song.get('title', ''),
                    'artist': song.get('artist', ''),
                    'genre': song.get('genre', ''),
                    'language': song.get('language', ''),
                    'year': normalized_year,
                    'tempo': song.get('tempo', 120.0),
                    'valence': song.get('valence', 0.5),
                    'arousal': (song.get('energy', 0.5) + song.get('valence', 0.5)) / 2,
                    'recommendation_score': max(0.0, min(1.0, 0.85 - (i * 0.01))),  # Descending scores within 0-1 range
                    'rank': i + 1,
                    'algorithm_used': 'big_five_personality_based' if song in big_five_songs else 'contextual_bandit',
                    'category': song.get('category', 'personalized'),
                    'youtube_url': self._normalize_song_link(youtube_link),
                    'spotify_url': self._normalize_song_link(spotify_link)
                }
                formatted_recommendations.append(recommendation)

            # Create patient summary
            age = 0
            if 'dateOfBirth' in intake_data:
                try:
                    birth_year = int(str(intake_data['dateOfBirth']).split('-')[0])
                    age = 2024 - birth_year
                except (ValueError, IndexError):
                    age = 65

            patient_summary = {
                'name': intake_data.get('name', 'Unknown'),
                'age': age,
                'condition': condition_str,
                'session_id': session_id,
                'recommendation_count': len(formatted_recommendations),
                'birth_date': intake_data.get('dateOfBirth', '1950-01-01'),
                'birthplace_city': intake_data.get('birthplaceCity', ''),
                'birthplace_country': intake_data.get('birthplaceCountry', ''),
                'sex': intake_data.get('sex', 'other')
            }

            algorithm_metadata = {
                'algorithm': f'big_five_condition_specific_genre_mapping_{condition_str}',
                'songs_considered': len(all_recommended_songs),
                'features_used': f'big_five_personality_condition_specific_{condition_str}_cultural_context_mood_energy',
                'condition': condition_str,
                'condition_specific_genre_mapping': True,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Generated {len(formatted_recommendations)} recommendations using Excel file songs only")

            # Create and return response
            response = RecommendationResponse(
                session_id=session_id,
                recommendations=formatted_recommendations,
                patient_summary=patient_summary,
                big_five_scores=big_five_scores,
                algorithm_metadata=algorithm_metadata
            )

            return response

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise

    def _extract_birth_year(self, date_of_birth: str) -> int:
        """
        Extract birth year from date string.
        Handles various date formats and returns a reasonable default for invalid dates.

        Args:
            date_of_birth: Date string in various formats (YYYY-MM-DD, YYYY, etc.)

        Returns:
            Birth year as integer
        """
        if not date_of_birth:
            return 1950  # Default fallback

        try:
            # Handle YYYY-MM-DD format
            if '-' in str(date_of_birth):
                year_str = str(date_of_birth).split('-')[0]
                birth_year = int(year_str)
            # Handle just year format
            elif str(date_of_birth).isdigit():
                birth_year = int(date_of_birth)
            else:
                # Try to parse as integer
                birth_year = int(str(date_of_birth))

            # Reasonable bounds check
            if birth_year < 1900:
                birth_year = 1950
            elif birth_year > 2024:
                birth_year = 2000

            return birth_year
        except (ValueError, IndexError):
            logger.warning(f"Could not parse birth date '{date_of_birth}', using default 1950")
            return 1950

    def _normalize_song_year(self, year_value: Any) -> Optional[int]:
        """Normalize year values (int/float/str/datetime) to an int or None."""
        if year_value is None:
            return None
        if isinstance(year_value, bool):
            return None
        if isinstance(year_value, (int, float)):
            try:
                return int(year_value)
            except (TypeError, ValueError):
                return None
        if hasattr(year_value, "year"):
            try:
                return int(year_value.year)
            except (TypeError, ValueError, AttributeError):
                return None
        if isinstance(year_value, str):
            year_str = year_value.strip()
            if not year_str:
                return None
            if "-" in year_str:
                year_str = year_str.split("-")[0]
            if year_str.isdigit():
                try:
                    return int(year_str)
                except (TypeError, ValueError):
                    return None
        return None

    def _normalize_song_link(self, link_value: Any) -> str:
        """Normalize link values from the catalog to clean strings."""
        if link_value is None:
            return ""
        if isinstance(link_value, float) and link_value != link_value:
            return ""
        link_str = str(link_value).strip()
        if not link_str or link_str.lower() == "nan":
            return ""
        return link_str

    def _apply_nostalgia_window_filter(self, songs: List[Dict[str, Any]], start_year: int, end_year: int) -> List[Dict[str, Any]]:
        """
        Filter songs based on release year within the nostalgia window.

        Args:
            songs: List of songs with year information
            start_year: Start of nostalgia window (birth_year + 5)
            end_year: End of nostalgia window (birth_year + 30)

        Returns:
            List of songs whose release years fall within the nostalgia window
        """
        filtered_songs = []
        for song in songs:
            song_year = song.get('year', None)

            if song_year and isinstance(song_year, (int, float)):
                if start_year <= song_year <= end_year:
                    song_copy = song.copy()
                    song_copy['nostalgia_window_match'] = True
                    song_copy['years_from_birth'] = song_year - (start_year - 5)  # Years since birth
                    filtered_songs.append(song_copy)

        # If no songs match the nostalgia window, return the first 20 songs as fallback
        if not filtered_songs:
            logger.warning(f"No songs found in nostalgia window {start_year}-{end_year}, returning first 20 songs as fallback")
            filtered_songs = songs[:20]
            for song in filtered_songs:
                song['nostalgia_window_match'] = False
                song['fallback_reason'] = 'no_nostalgia_matches'

        return filtered_songs

    def _get_country_songs_with_language_filtering(self, patient_info: Dict[str, Any], birthplace_country: str, preferred_languages: List[str]) -> List[Dict[str, Any]]:
        """
        Apply enhanced language filtering logic for Country Songs Recommendations.

        Logic:
        1. If user selects a Preferred Language, show only songs that match that choice
           - English → show only English songs (en)
           - Bangla → show only Bangla songs (bn)
        2. If no language is selected, apply fallback: recommend songs based on birthplace/country of origin

        Args:
            patient_info: Patient information including preferences
            birthplace_country: Patient's birthplace country
            preferred_languages: List of preferred languages from user selection

        Returns:
            List of songs filtered according to the language preference logic
        """
        logger.info(f"Applying enhanced language filtering for Country Songs - preferred: {preferred_languages}, birthplace: {birthplace_country}")

        # Load catalog if not already loaded
        if not self.music_catalog.catalog_loaded:
            self.music_catalog.load_catalog()

        if not self.music_catalog.catalog_loaded:
            logger.warning("Music catalog not loaded for country songs filtering, returning empty list")
            return []

        all_songs = self.music_catalog.songs.copy()
        filtered_songs = []

        # Step 1: Check if user has explicit language preferences
        if preferred_languages and len(preferred_languages) > 0:
            # User has selected preferred languages - apply strict filtering
            logger.info(f"User selected preferred languages: {preferred_languages}")

            # Map preferred languages to catalog language codes
            language_filters = []
            for lang in preferred_languages:
                lang_lower = lang.lower().strip()
                if lang_lower in ['english', 'en']:
                    language_filters.append('en')
                elif lang_lower in ['bangla', 'bengali', 'bn']:
                    language_filters.append('bn')
                elif lang_lower in ['bangali']:  # Handle alternative spelling
                    language_filters.append('bn')
                else:
                    # Use the language code as-is if it's a standard code
                    language_filters.append(lang_lower)

            # Remove duplicates while preserving order
            unique_filters = []
            for filt in language_filters:
                if filt not in unique_filters:
                    unique_filters.append(filt)

            logger.info(f"Mapped language preferences to filters: {unique_filters}")

            # Apply strict language filtering - only songs matching preferred languages
            for song in all_songs:
                song_language = str(song.get('language', '')).lower().strip()

                # Check if song matches any of the preferred languages
                matches_preference = False
                for lang_filter in unique_filters:
                    if song_language == lang_filter:
                        matches_preference = True
                        break
                    # Also handle common variations
                    elif lang_filter == 'en' and song_language in ['english', 'en']:
                        matches_preference = True
                        break
                    elif lang_filter == 'bn' and song_language in ['bangla', 'bengali', 'bn', 'bangali']:
                        matches_preference = True
                        break

                if matches_preference:
                    song_copy = song.copy()
                    song_copy['language_filter_type'] = 'user_preference'
                    song_copy['matched_languages'] = unique_filters
                    filtered_songs.append(song_copy)

            logger.info(f"Language preference filtering: {len(filtered_songs)}/{len(all_songs)} songs match {unique_filters}")

            # If no songs match the preferred languages, DO NOT fall back to birthplace-based filtering
            # This ensures user's language preferences are strictly respected
            if not filtered_songs:
                logger.warning(f"No songs match preferred languages {unique_filters}, returning empty list to respect user preferences")
                return []

            return filtered_songs

        else:
            # Step 2: No language preference selected - apply fallback based on birthplace/country
            logger.info("No language preference selected, applying birthplace/country fallback")
            return self._apply_birthplace_language_fallback(all_songs, birthplace_country)

    def _apply_birthplace_language_fallback(self, all_songs: List[Dict[str, Any]], birthplace_country: str) -> List[Dict[str, Any]]:
        """
        Apply fallback language filtering based on birthplace/country of origin.

        Args:
            all_songs: All available songs from catalog
            birthplace_country: Patient's birthplace country

        Returns:
            List of songs filtered based on birthplace country with fallback logic
        """
        if not birthplace_country:
            logger.info("No birthplace country provided, using all songs as fallback")
            # Add fallback metadata to all songs
            fallback_songs = []
            for song in all_songs:
                song_copy = song.copy()
                song_copy['language_filter_type'] = 'no_preference_no_birthplace'
                song_copy['fallback_reason'] = 'no_language_or_birthplace_specified'
                fallback_songs.append(song_copy)
            return fallback_songs

        # Get language filter for birthplace country
        language_filter = self.music_catalog._get_language_filter(birthplace_country)
        logger.info(f"Birthplace country '{birthplace_country}' → language filter '{language_filter}'")

        # Apply birthplace-based language filtering
        birthplace_filtered_songs = []
        for song in all_songs:
            song_language = str(song.get('language', '')).lower().strip()

            # Check if song matches birthplace country language
            if song_language == language_filter:
                song_copy = song.copy()
                song_copy['language_filter_type'] = 'birthplace_fallback'
                song_copy['birthplace_country'] = birthplace_country
                song_copy['language_filter_applied'] = language_filter
                birthplace_filtered_songs.append(song_copy)
            # Handle common language variations
            elif language_filter == 'en' and song_language in ['english', 'en']:
                song_copy = song.copy()
                song_copy['language_filter_type'] = 'birthplace_fallback'
                song_copy['birthplace_country'] = birthplace_country
                song_copy['language_filter_applied'] = language_filter
                birthplace_filtered_songs.append(song_copy)
            elif language_filter == 'bn' and song_language in ['bangla', 'bengali', 'bn', 'bangali']:
                song_copy = song.copy()
                song_copy['language_filter_type'] = 'birthplace_fallback'
                song_copy['birthplace_country'] = birthplace_country
                song_copy['language_filter_applied'] = language_filter
                birthplace_filtered_songs.append(song_copy)

        logger.info(f"Birthplace language filtering: {len(birthplace_filtered_songs)}/{len(all_songs)} songs match '{language_filter}' for '{birthplace_country}'")

        # If no songs match the birthplace language, use all songs as ultimate fallback
        if not birthplace_filtered_songs:
            logger.warning(f"No songs match birthplace language '{language_filter}' for '{birthplace_country}', using all songs as ultimate fallback")
            fallback_songs = []
            for song in all_songs:
                song_copy = song.copy()
                song_copy['language_filter_type'] = 'birthplace_fallback_failed'
                song_copy['birthplace_country'] = birthplace_country
                song_copy['language_filter_attempted'] = language_filter
                song_copy['fallback_reason'] = f'no_songs_for_{language_filter}_in_{birthplace_country}'
                fallback_songs.append(song_copy)
            return fallback_songs

        return birthplace_filtered_songs

    def _get_personalized_songs(self, condition: TherapyCondition, patient_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get ALL songs from Excel file d.xlsx and categorize them for different recommendation categories.
        Applies enhanced language filtering for Country Songs Recommendations based on user preferences.
        Total: All songs from Excel file (50 songs) with proper categories and URLs.
        """
        logger.info("Getting ALL songs from Excel file d.xlsx - categorizing for personalized recommendations")

        # Get birthplace country for fallback logic
        birthplace_country = patient_info.get('birthplaceCountry')

        # Get preferred languages from patient info for enhanced filtering
        preferred_languages = patient_info.get('preferredLanguages', [])

        # Apply enhanced language filtering logic for Country Songs Recommendations
        country_songs = self._get_country_songs_with_language_filtering(
            patient_info, birthplace_country, preferred_languages
        )

        # Prepare preferences dict with language preferences for other categories
        preferences = {}
        if preferred_languages:
            # Map language codes for better matching
            mapped_languages = []
            for lang in preferred_languages:
                if lang.lower() in ['english', 'en']:
                    mapped_languages.append('en')
                elif lang.lower() in ['bangla', 'bengali', 'bn']:
                    mapped_languages.append('bn')
                else:
                    mapped_languages.append(lang.lower())
            preferences['preferred_languages'] = mapped_languages

        # Convert condition to string value for music catalog
        condition_str = condition.value if hasattr(condition, 'value') else str(condition)
        catalog_condition_str = "dementia" if condition == TherapyCondition.DOWN_SYNDROME else condition_str

        # For Down syndrome, keep non-personality categories aligned to dementia behavior
        if condition == TherapyCondition.DOWN_SYNDROME:
            logger.info("Down syndrome selected: using dementia catalog filters for non-personality categories")

        all_excel_songs = self.music_catalog.get_songs_by_preferences(
            condition=catalog_condition_str,
            preferences=preferences,
            birthplace_country=birthplace_country
        )

        # Log language filtering result
        if birthplace_country:
            language_filter = self.music_catalog._get_language_filter(birthplace_country)
            logger.info(f"Applied global language filtering for birthplace '{birthplace_country}' → language filter '{language_filter}', found {len(all_excel_songs)} songs")

        # Calculate nostalgia window for birthplace location recommendations
        birth_year = self._extract_birth_year(patient_info.get('dateOfBirth', '1950-01-01'))
        nostalgia_start_year = birth_year + 5
        nostalgia_end_year = birth_year + 30

        # Apply nostalgia window filtering for Country Songs Recommendations
        nostalgia_filtered_songs = self._apply_nostalgia_window_filter(
            country_songs, nostalgia_start_year, nostalgia_end_year
        )

        logger.info(f"Applied nostalgia window filtering for birth year {birth_year}: {nostalgia_start_year}-{nostalgia_end_year}, found {len(nostalgia_filtered_songs)} songs")

        # Create category-specific filters while respecting language preferences
        category_songs = self._get_category_specific_songs(
            all_excel_songs, patient_info, catalog_condition_str
        )

        # Create categories from the Excel songs
        all_songs = []
        song_id = 1

        # Categorize songs for different recommendation types with category-specific filtering
        categories = {
            'birthplace_location': nostalgia_filtered_songs[:20],  # Language-filtered + nostalgia-filtered songs for location-based
            'favorite_genres': category_songs['favorite_genres'][:20],  # Genre-filtered songs
            'instruments': category_songs['instruments'][:20],  # Instrument-filtered songs
            'favorite_musician': category_songs['favorite_musician'][:20],  # Artist-filtered songs
            'favorite_season': category_songs['favorite_season'][:20],  # Seasonal mood-filtered songs
            'natural_elements': category_songs['natural_elements'][:20],  # Nature mood-filtered songs
            'cognitive_indicators': category_songs['cognitive_indicators'][:20],  # Cognitive support-filtered songs
        }

        for category, songs in categories.items():
            for song in songs:
                # Build song dict, preserving nostalgia metadata for birthplace_location category
                song_dict = {
                    'id': song_id,
                    'title': song['title'],
                    'artist': song['artist'],
                    'genre': song['genre'],
                    'language': song['language'],
                    'category': category,
                    'year': song.get('year', 2020),
                    'tempo': song.get('tempo', 120.0),
                    'valence': song.get('valence', 0.5),
                    'arousal': (song.get('energy', 0.5) + song.get('valence', 0.5)) / 2,
                    'youtube_link': song.get('youtube_link', ''),
                    'spotify_link': song.get('spotify_link', '')
                }

                # Preserve nostalgia window metadata for birthplace_location category
                if category == 'birthplace_location':
                    song_dict['nostalgia_window_match'] = song.get('nostalgia_window_match', False)
                    song_dict['years_from_birth'] = song.get('years_from_birth', None)
                    song_dict['fallback_reason'] = song.get('fallback_reason', None)
                    if birth_year and song.get('year'):
                        song_dict['years_from_birth'] = song.get('year') - birth_year

                all_songs.append(song_dict)
                song_id += 1

        logger.info(f"Categorized {len(all_songs)} songs from Excel file for personalized recommendations")
        return all_songs

    def _get_category_specific_songs(self, all_songs: List[Dict[str, Any]], patient_info: Dict[str, Any], condition: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate category-specific song lists that respect language preferences.

        This method ensures each recommendation category gets different songs based on
        category-specific criteria while maintaining the language filtering that users expect.

        Args:
            all_songs: Songs that have already been filtered by language preferences
            patient_info: Patient information including preferences
            condition: Therapy condition string

        Returns:
            Dictionary with category keys and corresponding filtered song lists
        """
        logger.info(f"Creating category-specific song lists from {len(all_songs)} language-filtered songs")

        # Extract user preferences for category-specific filtering
        favorite_genres = [g.strip().lower() for g in patient_info.get('favoriteGenres', []) if g.strip()]
        favorite_instruments = [i.strip().lower() for i in patient_info.get('instruments', []) if i.strip()]
        favorite_musician = patient_info.get('favoriteMusician', '').strip().lower()
        favorite_season = patient_info.get('favoriteSeason', '').strip().lower()
        natural_elements = [e.strip().lower() for e in patient_info.get('naturalElements', []) if e.strip()]

        # Initialize category dictionaries
        category_songs = {
            'favorite_genres': [],
            'instruments': [],
            'favorite_musician': [],
            'favorite_season': [],
            'natural_elements': [],
            'cognitive_indicators': []
        }

        # Process each song for category matching
        for song in all_songs:
            song_genres = str(song.get('genre', '')).lower()
            song_instruments = str(song.get('instruments', '')).lower()
            song_artist = str(song.get('artist', '')).lower()
            song_mood = str(song.get('mood', '')).lower()
            song_valence = song.get('valence', 0.5)
            song_energy = song.get('energy', 0.5)
            song_lyrics_uplifting = song.get('lyrics_uplifting', 0.5)
            song_audio_relaxing = song.get('audio_relaxing', 0.5)

            # Favorite Genres Category
            if favorite_genres:
                for genre in favorite_genres:
                    if genre in song_genres or song_genres in genre:
                        song_copy = song.copy()
                        song_copy['matched_genre'] = genre
                        category_songs['favorite_genres'].append(song_copy)
                        break
            else:
                # If no genre preference, include song with high musical relevance
                if song_valence > 0.4 or song_energy > 0.4:
                    song_copy = song.copy()
                    song_copy['match_reason'] = 'general_musical_relevance'
                    category_songs['favorite_genres'].append(song_copy)

            # Instruments Category
            if favorite_instruments:
                for instrument in favorite_instruments:
                    if instrument in song_instruments or song_instruments in instrument:
                        song_copy = song.copy()
                        song_copy['matched_instrument'] = instrument
                        category_songs['instruments'].append(song_copy)
                        break
            else:
                # If no instrument preference, include songs with distinctive instrumentation
                if song_instruments and len(song_instruments) > 10:  # Songs with rich instrumentation
                    song_copy = song.copy()
                    song_copy['match_reason'] = 'rich_instrumentation'
                    category_songs['instruments'].append(song_copy)

            # Favorite Musician Category
            if favorite_musician and favorite_musician in song_artist:
                song_copy = song.copy()
                song_copy['matched_artist'] = favorite_musician
                category_songs['favorite_musician'].append(song_copy)
            else:
                # If no artist preference or no match, include popular/classic songs
                if song_valence >= 0.6:  # Uplifting songs
                    song_copy = song.copy()
                    song_copy['match_reason'] = 'uplifting_popular'
                    category_songs['favorite_musician'].append(song_copy)

            # Favorite Season Category
            season_keywords = {
                'spring': ['fresh', 'renewal', 'bloom', 'uplifting', 'positive'],
                'summer': ['energetic', 'vibrant', 'lively', 'warm', 'bright'],
                'autumn': ['melancholy', 'nostalgic', 'reflective', 'warm', 'cozy'],
                'winter': ['calm', 'peaceful', 'serene', 'quiet', 'reflective'],
                'rainy': ['relaxing', 'calm', 'soothing', 'peaceful'],
                'sunny': ['bright', 'happy', 'energetic', 'uplifting', 'cheerful']
            }

            if favorite_season in season_keywords:
                season_moods = season_keywords[favorite_season]
                if any(mood in song_mood for mood in season_moods):
                    song_copy = song.copy()
                    song_copy['matched_season_mood'] = favorite_season
                    category_songs['favorite_season'].append(song_copy)
                else:
                    # Fallback: songs with seasonal-appropriate energy/valence
                    if favorite_season in ['winter', 'rainy'] and (song_audio_relaxing > 0.6 or song_energy < 0.5):
                        song_copy = song.copy()
                        song_copy['match_reason'] = f'{favorite_season}_energy_match'
                        category_songs['favorite_season'].append(song_copy)
                    elif favorite_season in ['summer', 'sunny'] and (song_energy > 0.6 or song_valence > 0.6):
                        song_copy = song.copy()
                        song_copy['match_reason'] = f'{favorite_season}_energy_match'
                        category_songs['favorite_season'].append(song_copy)
            else:
                # If no season preference, include seasonally appropriate songs
                song_copy = song.copy()
                song_copy['match_reason'] = 'seasonally_appropriate'
                category_songs['favorite_season'].append(song_copy)

            # Natural Elements Category
            if natural_elements:
                for element in natural_elements:
                    nature_keywords = {
                        'ocean': ['ocean', 'sea', 'wave', 'water', 'beach', 'calm'],
                        'forest': ['forest', 'trees', 'nature', 'natural', 'peaceful'],
                        'mountains': ['mountain', 'majestic', 'grand', 'inspiring'],
                        'rain': ['rain', 'gentle', 'soothing', 'peaceful', 'calm'],
                        'sunshine': ['bright', 'warm', 'happy', 'sunny', 'cheerful'],
                        'flowers': ['gentle', 'beautiful', 'delicate', 'bloom']
                    }

                    if element in nature_keywords:
                        element_moods = nature_keywords[element]
                        if any(mood in song_mood for mood in element_moods):
                            song_copy = song.copy()
                            song_copy['matched_nature_element'] = element
                            category_songs['natural_elements'].append(song_copy)
                            break
                # If no natural element matches, don't add to this category
            else:
                # If no nature preference, include calming/nature-appropriate songs
                if song_audio_relaxing > 0.5 or 'natural' in song_mood:
                    song_copy = song.copy()
                    song_copy['match_reason'] = 'natural_vibe'
                    category_songs['natural_elements'].append(song_copy)

            # Cognitive Indicators Category
            # For cognitive support: prefer calming, structured, and moderately engaging songs
            cognitive_score = 0
            if condition.lower() == 'dementia':
                # For dementia: prefer familiar, calming music
                if song_audio_relaxing > 0.6 and song_energy < 0.7:
                    cognitive_score += 2
                if 'classical' in song_genres or 'folk' in song_genres:
                    cognitive_score += 1
            elif condition.lower() == 'adhd':
                # For ADHD: prefer structured, moderate energy music
                if 0.4 < song_energy < 0.8:
                    cognitive_score += 2
                if song.get('danceability', 0) > 0.3:
                    cognitive_score += 1
            elif condition.lower() == 'down_syndrome':
                # For Down syndrome: prefer uplifting, positive music
                if song_lyrics_uplifting > 0.6 or song_valence > 0.6:
                    cognitive_score += 2

            # General cognitive support criteria
            if song_energy > 0.3 and song_energy < 0.8:  # Moderate energy
                cognitive_score += 1
            if song_audio_relaxing > 0.4:  # Calming audio
                cognitive_score += 1

            if cognitive_score >= 2:  # Only include songs that meet cognitive criteria
                song_copy = song.copy()
                song_copy['cognitive_score'] = cognitive_score
                song_copy['cognitive_match_reason'] = f'score_{cognitive_score}_for_{condition}'
                category_songs['cognitive_indicators'].append(song_copy)

        # Log results and ensure each category has at least some songs
        for category, songs in category_songs.items():
            if len(songs) < 5:
                logger.warning(f"Category '{category}' only has {len(songs)} songs, adding more from language-filtered pool")
                # Add more songs from the filtered pool to meet minimum requirements
                additional_songs = all_songs[:20]  # Take up to 20 more songs
                for song in additional_songs:
                    if len(category_songs[category]) >= 20:  # Stop when we have enough
                        break
                    song_copy = song.copy()
                    song_copy['match_reason'] = f'fallback_for_{category}'
                    category_songs[category].append(song_copy)

            logger.info(f"Category '{category}': {len(category_songs[category])} songs generated")

        return category_songs

    MIN_PERSONALITY_SONGS = 20

    def _get_personality_match_songs(
        self,
        big_five_scores: Optional[Dict[str, float]],
        condition: TherapyCondition = TherapyCondition.DEMENTIA,
        songs_per_trait: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Build the Personality Match category using either catalog songs or curated fallbacks.
        Uses score-driven rotation so changing scores immediately yields different tracks.
        Falls back to neutral trait scores when the user hasn't provided Big Five responses.
        For Down syndrome patients, uses specialized genre mappings.
        """
        normalized_scores = self._normalize_big_five_scores(big_five_scores)

        # Use Down syndrome specific mapping when condition is Down syndrome
        if condition == TherapyCondition.DOWN_SYNDROME:
            logger.info("Using Down syndrome-specific music mapping for personality recommendations")
            return self._get_down_syndrome_personality_songs(normalized_scores, songs_per_trait)

        # For all conditions (including dementia), use the updated Big Five service with condition parameter
        logger.info(f"Compiling personality match songs for condition: {condition}, scores: {normalized_scores}")

        # Convert normalized scores back to 1-7 scale for Big Five service
        big5_responses_1_7 = []
        for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            score = normalized_scores.get(trait, 0.55)  # Default to 0.55 if not found
            # Convert 0-1 scale to 1-7 scale
            score_1_7 = max(1, min(7, round(score * 6 + 1)))
            big5_responses_1_7.append(score_1_7)
            big5_responses_1_7.append(score_1_7)  # Add twice for BFI-2-X format (2 questions per trait)

        # Get preferred languages from current patient info
        current_patient_info = getattr(self, 'current_patient_info', {})
        preferred_languages = current_patient_info.get('preferredLanguages', [])
        normalized_languages = (
            ConditionNormalizer.normalize_languages(preferred_languages)
            if preferred_languages
            else []
        )

        catalog_songs: List[Dict[str, Any]] = []
        if self.music_catalog.catalog_loaded:
            catalog_songs = self.music_catalog.songs.copy()

        safe_catalog_songs: List[Dict[str, Any]] = []
        if catalog_songs:
            for song in catalog_songs:
                if normalized_languages:
                    song_language = str(song.get('language', '')).strip()
                    song_lang_code = (
                        ConditionNormalizer.normalize_languages([song_language])[0]
                        if song_language
                        else None
                    )
                    if not song_lang_code or song_lang_code not in normalized_languages:
                        continue
                song_features = {
                    'tempo': song.get('tempo', 120),
                    'valence': song.get('valence', 0.5),
                    'energy': song.get('energy', 0.5),
                    'genre': song.get('genre', '')
                }
                is_appropriate, _ = self.down_syndrome_mapping.validate_song_for_down_syndrome(song_features)
                if is_appropriate:
                    safe_catalog_songs.append(song)
        normalized_languages = (
            ConditionNormalizer.normalize_languages(preferred_languages)
            if preferred_languages
            else []
        )

        catalog_songs: List[Dict[str, Any]] = []
        if self.music_catalog.catalog_loaded:
            catalog_songs = self.music_catalog.songs.copy()

        safe_catalog_songs: List[Dict[str, Any]] = []
        if catalog_songs:
            for song in catalog_songs:
                if normalized_languages:
                    song_language = str(song.get('language', '')).strip()
                    song_lang_code = (
                        ConditionNormalizer.normalize_languages([song_language])[0]
                        if song_language
                        else None
                    )
                    if not song_lang_code or song_lang_code not in normalized_languages:
                        continue
                song_features = {
                    'tempo': song.get('tempo', 120),
                    'valence': song.get('valence', 0.5),
                    'energy': song.get('energy', 0.5),
                    'genre': song.get('genre', '')
                }
                is_appropriate, _ = self.down_syndrome_mapping.validate_song_for_down_syndrome(song_features)
                if is_appropriate:
                    safe_catalog_songs.append(song)
        normalized_languages = (
            ConditionNormalizer.normalize_languages(preferred_languages)
            if preferred_languages
            else []
        )

        # Map condition string for Big Five service
        condition_str = condition.value.lower() if hasattr(condition, 'value') else str(condition).lower()

        logger.info(f"DEBUG: Using Big Five recommendation service with condition '{condition_str}'")
        logger.info(f"DEBUG: Big Five scores being sent: {normalized_scores}")
        logger.info(f"DEBUG: Converted to 1-7 scale: {big5_responses_1_7}")
        logger.info(f"DEBUG: Preferred languages: {preferred_languages}")

        try:
            # Use the Big Five recommendation service with condition-specific genre mapping
            big_five_result = self.big_five_recommendation_service.get_big_five_recommendations(
                big5_responses=big5_responses_1_7,
                total_recommendations=self.MIN_PERSONALITY_SONGS,
                preferred_languages=preferred_languages,
                condition=condition_str
            )

            logger.info(f"DEBUG: Big Five service returned {len(big_five_result.get('recommendations', []))} recommendations")
            logger.info(f"DEBUG: Algorithm used: {big_five_result.get('algorithm_metadata', {}).get('algorithm', 'Unknown')}")

            personality_songs = []
            seen: set = set()

            for song in big_five_result.get('recommendations', []):
                youtube_link = song.get('youtube_link')
                if not youtube_link:
                    youtube_link = song.get('youtube link', song.get('youtube_url', ''))
                spotify_link = song.get('spotify_link')
                if not spotify_link:
                    spotify_link = song.get('spotify link', song.get('spotify_url', ''))
                # Convert to expected format
                song_copy = {
                    'title': song.get('song_name', song.get('title', 'Unknown')),
                    'artist': song.get('singer', song.get('artist', 'Unknown')),
                    'genre': song.get('genre', 'Unknown'),
                    'language': song.get('language', 'English'),
                    'year': song.get('released_date', song.get('year', 2020)),
                    'tempo': song.get('tempo', 120),
                    'valence': song.get('valence', 0.5),
                    'energy': song.get('energy', 0.5),
                    'danceability': song.get('danceability', 0.5),
                    'acousticness': song.get('acousticness', 0.5),
                    'category': 'big_five_personality',
                    'matched_trait': song.get('personality_trait', 'unknown'),
                    'patient_trait_score': song.get('trait_score_0_1', 0.5),
                    'song_trait_score': song.get('trait_score_0_1', 0.5),
                    'trait_level': song.get('trait_level', 'Medium'),
                    'match_reason': song.get('match_reason', 'Personality match'),
                    'confidence': song.get('confidence', 0.8),
                    'youtube_link': self._normalize_song_link(youtube_link),
                    'spotify_link': self._normalize_song_link(spotify_link),
                    'used instruments in the song': song.get('used instruments in the song', ''),
                }

                # Avoid duplicates
                key = (song_copy.get('title'), song_copy.get('artist'))
                if key not in seen:
                    seen.add(key)
                    personality_songs.append(song_copy)
                    logger.info(f"DEBUG: Added song '{song_copy.get('title')}' by {song_copy.get('artist')} with genre '{song_copy.get('genre')}' for trait '{song_copy.get('matched_trait')}'")

            # If we didn't get enough songs, supplement with fallbacks
            if len(personality_songs) < self.MIN_PERSONALITY_SONGS:
                filler_needed = self.MIN_PERSONALITY_SONGS - len(personality_songs)
                logger.info(f"Big Five service returned {len(personality_songs)} songs, adding {filler_needed} fallbacks")

                filler_tracks = self._get_global_personality_fallbacks(
                    normalized_scores,
                    filler_needed,
                    seen
                )
                personality_songs.extend(filler_tracks)

            logger.info(f"Personality match generated {len(personality_songs)} songs for condition {condition_str}")
            return personality_songs[:self.MIN_PERSONALITY_SONGS]

        except Exception as e:
            logger.error(f"Error using Big Five recommendation service: {e}")
            # Fallback to original method if Big Five service fails
            logger.info("Falling back to original personality matching method")
            sorted_traits = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
            personality_songs: List[Dict[str, Any]] = []
            seen: set = set()

            for trait_name, raw_score in sorted_traits:
                patient_score = float(raw_score)
                trait_key = trait_name.lower()

                trait_matches = self._get_songs_by_trait_score(trait_key, patient_score, songs_per_trait)
                if len(trait_matches) < songs_per_trait:
                    fallback_needed = songs_per_trait - len(trait_matches)
                    trait_matches.extend(
                        self._get_fallback_personality_songs(trait_key, patient_score, fallback_needed)
                    )

                for song in trait_matches:
                    song_copy = song.copy()
                    key = (song_copy.get('title'), song_copy.get('artist'))
                    if key in seen:
                        continue

                    song_copy['category'] = 'big_five_personality'
                    song_copy['matched_trait'] = trait_key
                    song_copy['patient_trait_score'] = patient_score
                    song_copy['song_trait_score'] = song_copy.get('song_trait_score') or \
                        song_copy.get('trait_score') or \
                        song_copy.get(trait_key, 0.0)
                    song_copy.setdefault('language', song_copy.get('language', 'English'))
                    song_copy.setdefault('year', song_copy.get('year', 2020))
                    personality_songs.append(song_copy)
                    seen.add(key)

            if len(personality_songs) < self.MIN_PERSONALITY_SONGS:
                filler_needed = self.MIN_PERSONALITY_SONGS - len(personality_songs)
                filler_tracks = self._get_global_personality_fallbacks(
                    normalized_scores,
                    filler_needed,
                    seen
                )
                personality_songs.extend(filler_tracks)

            logger.info(f"Fallback personality match generated {len(personality_songs)} songs")
            return personality_songs

    def _normalize_big_five_scores(self, raw_scores: Optional[Any]) -> Dict[str, float]:
        """
        Normalize Big Five payload (dict, model, None) to lower-case trait scores between 0 and 1.
        Returns neutral mid-scores when user data is unavailable so we can still populate the category.
        """
        trait_names = list(self.PERSONALITY_TRAIT_FALLBACKS.keys())
        default_score = 0.55
        normalized = {trait: default_score for trait in trait_names}

        if not raw_scores:
            return normalized

        for trait in trait_names:
            score = None
            if isinstance(raw_scores, dict):
                score = raw_scores.get(trait) or raw_scores.get(trait.capitalize())
            else:
                score = getattr(raw_scores, trait, None) or getattr(raw_scores, trait.capitalize(), None)

            if score is None:
                continue

            try:
                normalized[trait] = max(0.0, min(float(score), 1.0))
            except (TypeError, ValueError):
                continue

        return normalized

    def _get_global_personality_fallbacks(
        self,
        normalized_scores: Dict[str, float],
        songs_needed: int,
        seen_keys: set
    ) -> List[Dict[str, Any]]:
        """Collect fallback songs across traits to satisfy the minimum requirement."""
        if songs_needed <= 0:
            return []

        ordered_traits = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
        filler: List[Dict[str, Any]] = []

        for trait_name, score in ordered_traits:
            if len(filler) >= songs_needed:
                break

            candidates = self._get_fallback_personality_songs(
                trait_name,
                score,
                songs_needed - len(filler)
            )

            for track in candidates:
                key = (track.get('title'), track.get('artist'))
                if key in seen_keys:
                    continue
                track['category'] = 'big_five_personality'
                track['matched_trait'] = trait_name
                track['patient_trait_score'] = score
                track['song_trait_score'] = track.get('trait_score', 0.75)
                filler.append(track)
                seen_keys.add(key)
                if len(filler) >= songs_needed:
                    break

        return filler

    def _get_fallback_personality_songs(
        self,
        trait_name: str,
        patient_score: float,
        songs_needed: int
    ) -> List[Dict[str, Any]]:
        """Get fallback songs from Excel catalog instead of hardcoded pool."""
        if songs_needed <= 0:
            return []

        # Load catalog if not loaded
        if not self.music_catalog.catalog_loaded:
            self.music_catalog.load_catalog()

        if not self.music_catalog.catalog_loaded:
            logger.warning("Music catalog not loaded for fallback, returning empty list")
            return []

        # Get songs from catalog sorted by the specific trait
        all_songs = self.music_catalog.songs.copy()
        trait_songs = sorted(
            all_songs,
            key=lambda x: x.get(trait_name, 0.0),
            reverse=True
        )

        if not trait_songs:
            logger.warning(f"No songs found for trait '{trait_name}' in catalog")
            return []

        # Apply language filtering based on birthplace country AND preferred languages
        birthplace_country = getattr(self, 'current_birthplace_country', None)
        language_filter = None

        # Get language filter from birthplace country
        if birthplace_country:
            language_filter = self.music_catalog._get_language_filter(birthplace_country)

        # Get preferred languages from current patient info
        current_patient_info = getattr(self, 'current_patient_info', {})
        preferred_languages = current_patient_info.get('preferredLanguages', [])

        # Apply language filtering with strict preference for user's explicit language choices
        if preferred_languages:
            # User has explicit language preferences - apply strict filtering
            before_filter = trait_songs
            trait_songs = []

            for song in before_filter:
                song_language = song.get('language', '').lower()

                # Check if song matches any preferred language
                matches_preferred = False
                for pref_lang in preferred_languages:
                    pref_lang_lower = pref_lang.lower()
                    if (pref_lang_lower == 'english' and song_language == 'en') or \
                       (pref_lang_lower in ['bangla', 'bengali', 'bn'] and song_language == 'bn') or \
                       (pref_lang_lower == song_language):
                        matches_preferred = True
                        break

                # Only include songs that match user's preferred languages
                if matches_preferred:
                    trait_songs.append(song)

            if not trait_songs:
                # If no songs match user's language preferences, return empty to respect preferences
                logger.warning(f"No personality songs found for user's preferred languages {preferred_languages}, returning empty to respect preferences")
                return []
            else:
                logger.info(f"Applied strict language filtering for personality songs: kept {len(trait_songs)}/{len(before_filter)} songs for {preferred_languages}")
        elif language_filter:
            # No user preferences, apply birthplace country filtering
            before_filter = trait_songs
            trait_songs = []

            for song in before_filter:
                song_language = song.get('language', '').lower()

                # Check if song matches birthplace country language filter
                if song_language == language_filter:
                    trait_songs.append(song)

            if not trait_songs:
                # If no songs match birthplace language, use all trait songs as fallback
                trait_songs = before_filter
                logger.warning(f"No personality songs found for birthplace language '{language_filter}', using all trait songs as fallback")
            else:
                logger.info(f"Applied birthplace language filtering for personality songs: kept {len(trait_songs)}/{len(before_filter)} songs for '{language_filter}'")

        # Rotate through songs based on patient score for variety
        normalized = max(0.0, min(patient_score, 1.0))
        rotation = int(round(normalized * max(1, len(trait_songs) - songs_needed)))

        fallback_songs: List[Dict[str, Any]] = []
        for i in range(songs_needed):
            song_index = (rotation + i) % len(trait_songs)
            song = trait_songs[song_index]

            fallback_songs.append({
                'id': -(1000000 + (hash(trait_name) % 100000) * 1000 + rotation + i),  # Generate unique negative integers
                'title': song.get('title', ''),
                'artist': song.get('artist', ''),
                'genre': song.get('genre', ''),
                'language': song.get('language', 'English'),
                'year': song.get('year', 2020),
                'tempo': song.get('tempo', 120.0),
                'valence': song.get('valence', 0.5),
                'arousal': (song.get('energy', 0.5) + song.get('valence', 0.5)) / 2,
                'trait_score': song.get(trait_name, 0.75),
                'youtube_link': song.get('youtube_link', ''),
                'spotify_link': song.get('spotify_link', '')
            })

        logger.info(f"Generated {len(fallback_songs)} fallback songs for trait '{trait_name}' from Excel catalog")
        return fallback_songs

    def _get_songs_by_trait_score(self, trait_name: str, patient_score: float, songs_per_trait: int = 3) -> List[Dict[str, Any]]:
        """
        Get songs that have high scores for a specific personality trait.
        Matches patient trait score with song trait scores for optimal compatibility.
        Applies global language filtering based on patient's birthplace country.
        """
        if not self.music_catalog.catalog_loaded:
            self.music_catalog.load_catalog()
        if not self.music_catalog.catalog_loaded:
            return []

        # Use catalog service's built-in method to get songs by Big Five traits from Excel file
        trait_songs = self.music_catalog.get_songs_by_big_five_traits([trait_name], songs_per_trait * 3)  # Get more to filter

        if not trait_songs:
            logger.warning(f"No songs found for trait '{trait_name}' in catalog, falling back to catalog search")
            # Fallback: search all songs and sort by trait score
            if not self.music_catalog.catalog_loaded:
                self.music_catalog.load_catalog()
            all_songs = self.music_catalog.songs.copy()
            trait_songs = sorted(all_songs, key=lambda x: x.get(trait_name, 0.0), reverse=True)[:songs_per_trait * 3]

        # Apply language filtering based on birthplace country AND preferred languages
        birthplace_country = getattr(self, 'current_birthplace_country', None)
        language_filter = None

        # Get language filter from birthplace country
        if birthplace_country:
            language_filter = self.music_catalog._get_language_filter(birthplace_country)

        # Get preferred languages from current patient info
        current_patient_info = getattr(self, 'current_patient_info', {})
        preferred_languages = current_patient_info.get('preferredLanguages', [])

        # Apply language filtering with strict preference for user's explicit language choices
        if preferred_languages:
            # User has explicit language preferences - apply strict filtering
            before_filter = trait_songs
            trait_songs = []

            for song in before_filter:
                song_language = song.get('language', '').lower()

                # Check if song matches any preferred language
                matches_preferred = False
                for pref_lang in preferred_languages:
                    pref_lang_lower = pref_lang.lower()
                    if (pref_lang_lower == 'english' and song_language == 'en') or \
                       (pref_lang_lower in ['bangla', 'bengali', 'bn'] and song_language == 'bn') or \
                       (pref_lang_lower == song_language):
                        matches_preferred = True
                        break

                # Only include songs that match user's preferred languages
                if matches_preferred:
                    trait_songs.append(song)

            if not trait_songs:
                # If no songs match user's language preferences, return empty to respect preferences
                logger.warning(f"No personality songs found for user's preferred languages {preferred_languages}, returning empty to respect preferences")
                return []
            else:
                logger.info(f"Applied strict language filtering for personality match: kept {len(trait_songs)}/{len(before_filter)} songs for {preferred_languages}")
        elif language_filter:
            # No user preferences, apply birthplace country filtering
            before_filter = trait_songs
            trait_songs = []

            for song in before_filter:
                song_language = song.get('language', '').lower()

                # Check if song matches birthplace country language filter
                if song_language == language_filter:
                    trait_songs.append(song)

            if not trait_songs:
                # If no songs match birthplace language, use all trait songs as fallback
                trait_songs = before_filter
                logger.warning(f"No personality songs found for birthplace language '{language_filter}', using all trait songs as fallback")
            else:
                logger.info(f"Applied birthplace language filtering for personality match: kept {len(trait_songs)}/{len(before_filter)} songs for '{language_filter}'")

        # Calculate compatibility scores based on trait matching
        scored_songs = []
        for song in trait_songs:
            song_trait_score = song.get(trait_name, 0.0)

            # Calculate compatibility score based on how well the song's trait score matches the patient's
            if song_trait_score > 0.0:  # Consider all songs with trait scores
                # Compatibility = 1 - |patient_score - song_trait_score|
                # This rewards songs with similar trait scores to the patient
                compatibility = 1.0 - abs(patient_score - song_trait_score)

                song_copy = song.copy()
                song_copy['trait_compatibility'] = compatibility
                song_copy['matched_trait_name'] = trait_name
                song_copy['matched_patient_score'] = patient_score
                song_copy['matched_song_score'] = song_trait_score
                scored_songs.append(song_copy)

        # Sort by compatibility score (highest first) and return top songs
        scored_songs.sort(key=lambda x: x['trait_compatibility'], reverse=True)

        result = scored_songs[:songs_per_trait]
        logger.info(f"Found {len(result)} songs for trait '{trait_name}' with compatibility scores from Excel catalog")
        return result

    def _get_mock_songs_fallback(self, condition: TherapyCondition, patient_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get fallback songs from Excel catalog only.
        Use ALL songs from d.xlsx file.
        """
        logger.info("Using Excel catalog songs from d.xlsx file - returning all available songs")

        # Get ALL songs from the Excel catalog
        catalog_songs = self.music_catalog.get_songs_by_preferences(condition.value, {})

        # Return all songs from Excel file without filtering
        formatted_songs = []
        for i, song in enumerate(catalog_songs[:50]):  # Use up to 50 songs from Excel
            formatted_songs.append({
                'id': song.get('id', i + 1),
                'title': song.get('title', ''),
                'artist': song.get('artist', ''),
                'genre': song.get('genre', ''),
                'language': song.get('language', ''),
                'year': song.get('year', 2020),
                'tempo': song.get('tempo', 120.0),
                'valence': song.get('valence', 0.5),
                'arousal': (song.get('energy', 0.5) + song.get('valence', 0.5)) / 2,
                'youtube_link': song.get('youtube_link', ''),
                'spotify_link': song.get('spotify_link', '')
            })

        logger.info(f"Returning {len(formatted_songs)} songs from Excel catalog")
        return formatted_songs

    def record_feedback(self, db: Session, session_id: str, recommendation_id: int,
                       feedback_type: FeedbackType, user_comments: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record patient feedback for a recommendation.
        """
        try:
            # Check if session exists
            session = db.query(TherapySession).filter(
                TherapySession.session_id == session_id
            ).first()

            if not session:
                logger.warning(f"Session {session_id} not found, creating feedback record anyway")
                # Create a session record if it doesn't exist
                session = TherapySession(
                    patient_id=1,  # Default patient
                    session_id=session_id,
                    condition="dementia",
                    therapy_type="music",
                    status="completed"
                )
                db.add(session)
                db.commit()

            # Calculate feedback score
            feedback_score = self._calculate_feedback_score(feedback_type)

            # Create feedback record
            feedback = TherapyFeedback(
                session_id=session_id,
                recommendation_id=recommendation_id,
                feedback_type=feedback_type.value,
                feedback_score=feedback_score,
                user_comments=user_comments,
                context=context
            )

            db.add(feedback)
            db.commit()

            logger.info(f"Recorded feedback {feedback_type.value} for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            db.rollback()
            return False

    def _calculate_feedback_score(self, feedback_type: FeedbackType) -> float:
        """
        Calculate numerical feedback score from feedback type.
        """
        scores = {
            FeedbackType.LIKE: 1.0,
            FeedbackType.NEUTRAL: 0.5,
            FeedbackType.SKIP: 0.3,
            FeedbackType.DISLIKE: 0.0,
            FeedbackType.INAPPROPRIATE: -0.5
        }
        return scores.get(feedback_type, 0.5)

    def get_session_recommendations(self, db: Session, session_id: str) -> List[TherapyRecommendation]:
        """
        Get all recommendations for a therapy session.
        """
        try:
            recommendations = db.query(TherapyRecommendation).filter(
                TherapyRecommendation.session_id == session_id
            ).order_by(TherapyRecommendation.rank.asc()).all()

            logger.info(f"Found {len(recommendations)} recommendations for session {session_id}")
            return recommendations

        except Exception as e:
            logger.error(f"Error retrieving session recommendations: {e}")
            return []

    def get_patient_feedback_history(self, db: Session, patient_id: int,
                                   limit: int = 50) -> List[TherapyFeedback]:
        """
        Get feedback history for a patient across all sessions.
        """
        try:
            # Get all sessions for the patient
            sessions = db.query(TherapySession).filter(
                TherapySession.patient_id == patient_id
            ).all()

            session_ids = [s.session_id for s in sessions]

            if not session_ids:
                logger.info(f"No sessions found for patient {patient_id}")
                return []

            # Get feedback for all sessions
            feedback_list = db.query(TherapyFeedback).filter(
                TherapyFeedback.session_id.in_(session_ids)
            ).order_by(TherapyFeedback.timestamp.desc()).limit(limit).all()

            logger.info(f"Found {len(feedback_list)} feedback entries for patient {patient_id}")
            return feedback_list

        except Exception as e:
            logger.error(f"Error retrieving patient feedback history: {e}")
            return []

    def update_bandit_stats(self, db: Session, song_id: int, reward: float,
                          context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update contextual bandit statistics for a song.
        """
        try:
            # Check if stats exist for this song
            stats = db.query(BanditStats).filter(BanditStats.song_id == song_id).first()

            if stats:
                # Update existing stats
                stats.total_plays += 1
                stats.total_reward += reward
                stats.avg_reward = stats.total_reward / stats.total_plays
                stats.last_played = datetime.now()
                stats.play_count = stats.play_count + 1
            else:
                # Create new stats record
                stats = BanditStats(
                    song_id=song_id,
                    total_plays=1,
                    total_reward=reward,
                    avg_reward=reward,
                    play_count=1,
                    last_played=datetime.now(),
                    context_features=str(context) if context else "{}"
                )
                db.add(stats)

            db.commit()
            logger.debug(f"Updated bandit stats for song {song_id}: reward={reward}")
            return True

        except Exception as e:
            logger.error(f"Error updating bandit stats: {e}")
            db.rollback()
            return False

    def get_recommendations_with_feedback(self, db: Session, session_id: str) -> Dict[str, Any]:
        """
        Get recommendations along with their feedback for a session.
        """
        try:
            recommendations = self.get_session_recommendations(db, session_id)
            feedback_list = db.query(TherapyFeedback).filter(
                TherapyFeedback.session_id == session_id
            ).all()

            # Create a mapping of recommendation_id -> feedback
            feedback_map = {}
            for feedback in feedback_list:
                feedback_map[feedback.recommendation_id] = feedback

            # Combine recommendations with feedback
            recommendations_with_feedback = []
            for rec in recommendations:
                feedback = feedback_map.get(rec.id)
                recommendations_with_feedback.append({
                    'recommendation': rec,
                    'feedback': feedback,
                    'feedback_type': feedback.feedback_type if feedback else None,
                    'feedback_score': feedback.feedback_score if feedback else None
                })

            return {
                'session_id': session_id,
                'recommendations': recommendations_with_feedback,
                'total_recommendations': len(recommendations),
                'total_feedback': len(feedback_list)
            }

        except Exception as e:
            logger.error(f"Error retrieving recommendations with feedback: {e}")
            return {
                'session_id': session_id,
                'recommendations': [],
                'total_recommendations': 0,
                'total_feedback': 0,
                'error': str(e)
            }

    def get_popular_songs(self, db: Session, condition: Optional[TherapyCondition] = None,
                         limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get popular songs based on bandit stats and feedback.
        """
        try:
            query = db.query(BanditStats, Song).join(Song).filter(
                Song.is_active == True
            )

            if condition:
                # Filter by songs that were recommended for this condition
                condition_songs = db.query(TherapyRecommendation).filter(
                    TherapyRecommendation.context_features.like(f'%{condition.value}%')
                ).distinct(TherapyRecommendation.song_id).all()

                song_ids = [r.song_id for r in condition_songs]
                if song_ids:
                    query = query.filter(BanditStats.song_id.in_(song_ids))

            # Order by average reward and total plays
            results = query.order_by(
                BanditStats.avg_reward.desc(),
                BanditStats.total_plays.desc()
            ).limit(limit).all()

            popular_songs = []
            for stats, song in results:
                popular_songs.append({
                    'song': song,
                    'avg_reward': stats.avg_reward,
                    'total_plays': stats.total_plays,
                    'play_count': stats.play_count,
                    'last_played': stats.last_played
                })

            logger.info(f"Found {len(popular_songs)} popular songs for {condition or 'all conditions'}")
            return popular_songs

        except Exception as e:
            logger.error(f"Error retrieving popular songs: {e}")
            return []

    def get_session_analytics(self, db: Session, session_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific therapy session.
        """
        try:
            # Get session details
            session = db.query(TherapySession).filter(
                TherapySession.session_id == session_id
            ).first()

            if not session:
                logger.warning(f"Session {session_id} not found")
                return {}

            # Get recommendations and feedback
            recommendations_with_feedback = self.get_recommendations_with_feedback(db, session_id)

            # Calculate metrics
            total_recommendations = recommendations_with_feedback['total_recommendations']
            total_feedback = recommendations_with_feedback['total_feedback']

            feedback_counts = {'like': 0, 'dislike': 0, 'skip': 0, 'neutral': 0, 'inappropriate': 0}
            engagement_score = 0.0

            for item in recommendations_with_feedback['recommendations']:
                feedback_type = item['feedback_type']
                if feedback_type:
                    feedback_counts[feedback_type] += 1
                    engagement_score += 1

            # Calculate engagement rate
            engagement_rate = (engagement_score / total_recommendations) if total_recommendations > 0 else 0.0

            analytics = {
                'session_id': session_id,
                'patient_id': session.patient_id,
                'condition': session.condition,
                'therapy_type': session.therapy_type,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'total_recommendations': total_recommendations,
                'total_feedback': total_feedback,
                'feedback_counts': feedback_counts,
                'engagement_rate': round(engagement_rate, 2),
                'session_duration': (
                    (session.end_time - session.start_time).total_seconds() / 60
                    if session.end_time else None
                )
            }

            return analytics

        except Exception as e:
            logger.error(f"Error retrieving session analytics: {e}")
            return {}

    def track_user_activity(self, db: Session, session_id: str, activity_type: str,
                           details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Track user activity for analytics and improvement.
        """
        try:
            activity = UserActivity(
                session_id=session_id,
                activity_type=activity_type,
                details=details or {}
            )

            db.add(activity)
            db.commit()

            logger.debug(f"Tracked activity {activity_type} for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error tracking user activity: {e}")
            db.rollback()
            return False

    def get_therapy_outcomes(self, db: Session, patient_id: int,
                           days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze therapy outcomes for a patient over time.
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Get recent sessions
            sessions = db.query(TherapySession).filter(
                TherapySession.patient_id == patient_id,
                TherapySession.start_time >= cutoff_date
            ).order_by(TherapySession.start_time.desc()).all()

            if not sessions:
                logger.info(f"No recent sessions found for patient {patient_id}")
                return {}

            # Get feedback for these sessions
            session_ids = [s.session_id for s in sessions]
            feedback_list = db.query(TherapyFeedback).filter(
                TherapyFeedback.session_id.in_(session_ids)
            ).all()

            # Analyze outcomes
            total_sessions = len(sessions)
            total_recommendations = sum(
                len(self.get_session_recommendations(db, s.session_id))
                for s in sessions
            )
            total_feedback = len(feedback_list)

            # Calculate satisfaction rate
            positive_feedback = sum(
                1 for f in feedback_list if f.feedback_score >= 0.7
            )
            satisfaction_rate = (positive_feedback / total_feedback) if total_feedback > 0 else 0.0

            outcomes = {
                'patient_id': patient_id,
                'analysis_period_days': days_back,
                'total_sessions': total_sessions,
                'total_recommendations': total_recommendations,
                'total_feedback': total_feedback,
                'positive_feedback': positive_feedback,
                'satisfaction_rate': round(satisfaction_rate, 2),
                'avg_recommendations_per_session': round(
                    total_recommendations / total_sessions, 1
                ) if total_sessions > 0 else 0.0,
                'engagement_rate': round(
                    (total_feedback / total_recommendations) if total_recommendations > 0 else 0.0, 2
                )
            }

            return outcomes

        except Exception as e:
            logger.error(f"Error analyzing therapy outcomes: {e}")
            return {}

    def _get_down_syndrome_personality_songs(
        self,
        normalized_scores: Dict[str, float],
        songs_per_trait: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Generate personality-based song recommendations specifically for Down syndrome patients.
        Uses research-based genre mapping and therapeutic characteristics.
        """
        logger.info(f"Generating Down syndrome personality recommendations for scores: {normalized_scores}")

        personality_songs: List[Dict[str, Any]] = []
        seen: set = set()

        # Get Down syndrome specific genre preferences
        ds_genres = self.down_syndrome_mapping.get_down_syndrome_genres_for_personality(normalized_scores)

        # Load catalog for song matching
        if not self.music_catalog.catalog_loaded:
            self.music_catalog.load_catalog()

        # Get preferred languages from current patient info first
        current_patient_info = getattr(self, 'current_patient_info', {})
        preferred_languages = current_patient_info.get('preferredLanguages', [])
        normalized_languages = ConditionNormalizer.normalize_languages(preferred_languages)

        # Get songs from catalog for matching
        catalog_songs = self.music_catalog.get_songs_by_preferences("down_syndrome", {})

        # Create safe catalog songs filtered for Down syndrome appropriateness
        safe_catalog_songs: List[Dict[str, Any]] = []
        if catalog_songs:
            for song in catalog_songs:
                # Filter by language if preferences exist
                if normalized_languages:
                    song_language = str(song.get('language', '')).strip()
                    song_lang_code = (
                        ConditionNormalizer.normalize_languages([song_language])[0]
                        if song_language
                        else None
                    )
                    if not song_lang_code or song_lang_code not in normalized_languages:
                        continue

                # Validate song for Down syndrome appropriateness
                song_features = {
                    'tempo': song.get('tempo', 120),
                    'valence': song.get('valence', 0.5),
                    'energy': song.get('energy', 0.5),
                    'genre': song.get('genre', '')
                }
                is_appropriate, _ = self.down_syndrome_mapping.validate_song_for_down_syndrome(song_features)
                if is_appropriate:
                    safe_catalog_songs.append(song)

        # Process each genre preference
        for genre_pref in ds_genres:
            trait = genre_pref["trait"]
            genre = genre_pref["genre"]
            weight = genre_pref["weight"]

            logger.info(f"Processing genre '{genre}' for trait '{trait}' with weight {weight}")

            # Get songs from catalog matching the genre
            genre_songs = []
            if catalog_songs:
                # Filter by genre (case-insensitive partial match)
                for song in catalog_songs:
                    song_genre = str(song.get('genre', '')).lower()
                    if genre.lower() in song_genre or song_genre in genre.lower():
                        genre_songs.append(song)

            # Apply language filtering if preferences exist
            if normalized_languages and genre_songs:
                filtered_songs = []
                for song in genre_songs:
                    song_language = str(song.get('language', '')).strip()
                    song_lang_code = (
                        ConditionNormalizer.normalize_languages([song_language])[0]
                        if song_language
                        else None
                    )
                    if song_lang_code and song_lang_code in normalized_languages:
                        filtered_songs.append(song)

                genre_songs = filtered_songs

            # Validate songs for Down syndrome appropriateness
            validated_songs = []
            for song in genre_songs:
                song_features = {
                    'tempo': song.get('tempo', 120),
                    'valence': song.get('valence', 0.5),
                    'energy': song.get('energy', 0.5),
                    'genre': song.get('genre', '')
                }

                is_appropriate, reason = self.down_syndrome_mapping.validate_song_for_down_syndrome(song_features)
                if is_appropriate:
                    validated_songs.append(song)

            # If no catalog songs found, fall back to other safe catalog songs
            if not validated_songs:
                trait_genres = self.down_syndrome_mapping.DOWN_SYNDROME_GENRE_MAPPING.get(trait, {}).get(
                    "preferred_genres",
                    [],
                )
                if safe_catalog_songs:
                    alt_genre_matches = []
                    for song in safe_catalog_songs:
                        song_genre = str(song.get('genre', '')).lower()
                        if any(
                            pref.lower() in song_genre or song_genre in pref.lower()
                            for pref in trait_genres
                        ):
                            alt_genre_matches.append(song)
                    validated_songs = alt_genre_matches or safe_catalog_songs

            # Select top songs for this genre/trait
            songs_to_add = min(songs_per_trait // 2, len(validated_songs))  # Distribute songs across traits

            for i, song in enumerate(validated_songs[:songs_to_add]):
                # Create song copy with Down syndrome specific metadata
                song_copy = song.copy()

                # Standardize song fields
                song_copy['id'] = -(2000000 + hash(f"{trait}_{genre}") * 1000 + i)  # Unique negative IDs
                song_copy.setdefault('title', song_copy.get('song_title', song_copy.get('title', '')))
                song_copy.setdefault('artist', song_copy.get('artist', song_copy.get('singer', '')))
                song_copy.setdefault('language', song_copy.get('language', 'English'))
                song_copy.setdefault('year', song_copy.get('year', song_copy.get('released_date', 2020)))
                song_copy.setdefault('tempo', song_copy.get('tempo', 120.0))
                song_copy.setdefault('valence', song_copy.get('valence', 0.5))
                song_copy.setdefault('arousal', (song_copy.get('energy', 0.5) + song_copy.get('valence', 0.5)) / 2)
                youtube_link = (
                    song_copy.get('youtube_link')
                    or song_copy.get('youtube link')
                    or song_copy.get('youtube_url', '')
                )
                spotify_link = (
                    song_copy.get('spotify_link')
                    or song_copy.get('spotify link')
                    or song_copy.get('spotify_url', '')
                )
                song_copy['youtube_link'] = self._normalize_song_link(youtube_link)
                song_copy['spotify_link'] = self._normalize_song_link(spotify_link)

                # Add Down syndrome specific metadata
                song_copy['category'] = 'big_five_personality'
                song_copy['matched_trait'] = trait
                song_copy['patient_trait_score'] = normalized_scores.get(trait, 0.5)
                song_copy['down_syndrome_genre'] = genre
                song_copy['therapeutic_rationale'] = self.down_syndrome_mapping.get_therapeutic_rationale(trait)
                song_copy['recommendation_weight'] = weight
                song_copy['condition_specific'] = 'down_syndrome'

                # Avoid duplicates
                key = (song_copy.get('title'), song_copy.get('artist'))
                if key not in seen:
                    personality_songs.append(song_copy)
                    seen.add(key)

        # Ensure minimum number of songs using catalog sources
        if len(personality_songs) < self.MIN_PERSONALITY_SONGS:
            filler_needed = self.MIN_PERSONALITY_SONGS - len(personality_songs)
            fallback_candidates = safe_catalog_songs or catalog_songs

            for song in fallback_candidates:
                if filler_needed <= 0:
                    break
                key = (song.get('title'), song.get('artist'))
                if key in seen:
                    continue
                youtube_link = song.get('youtube_link') or song.get('youtube link') or song.get('youtube_url', '')
                spotify_link = song.get('spotify_link') or song.get('spotify link') or song.get('spotify_url', '')
                fallback_copy = song.copy()
                fallback_copy.update({
                    'id': -(3000000 + len(personality_songs)),
                    'category': 'big_five_personality',
                    'condition_specific': 'down_syndrome',
                    'youtube_link': self._normalize_song_link(youtube_link),
                    'spotify_link': self._normalize_song_link(spotify_link),
                })
                personality_songs.append(fallback_copy)
                seen.add(key)
                filler_needed -= 1

        # Sort by recommendation weight and trait score
        personality_songs.sort(key=lambda x: (
            x.get('recommendation_weight', 0.5),
            x.get('patient_trait_score', 0.5)
        ), reverse=True)

        logger.info(f"Generated {len(personality_songs)} Down syndrome-specific personality recommendations")
        return personality_songs

    def _get_down_syndrome_filtered_songs_from_catalog(
        self,
        preferences: Dict[str, Any],
        birthplace_country: str
    ) -> List[Dict[str, Any]]:
        """
        Get songs from Excel catalog filtered specifically for Down syndrome patients.
        Uses research-based genre preferences and therapeutic audio characteristics.
        """
        if not self.music_catalog.catalog_loaded:
            if not self.music_catalog.load_catalog():
                logger.warning("Failed to load music catalog for Down syndrome filtering")
                return []

        logger.info("Filtering Excel catalog for Down syndrome-specific genres and characteristics")

        # Get Down syndrome preferred genres from the mapping service
        ds_genres = set()
        for trait_config in self.down_syndrome_mapping.DOWN_SYNDROME_GENRE_MAPPING.values():
            ds_genres.update(trait_config["preferred_genres"])

        # Get audio filters for Down syndrome
        audio_filters = self.down_syndrome_mapping.get_audio_filters_for_down_syndrome()

        # Get all songs from catalog
        all_songs = self.music_catalog.songs.copy()
        filtered_songs = []

        for song in all_songs:
            # Get song features for validation
            song_features = {
                'tempo': song.get('tempo', 120),
                'valence': song.get('valence', 0.5),
                'energy': song.get('energy', 0.5),
                'genre': song.get('genre', '')
            }

            # Validate song is appropriate for Down syndrome
            is_appropriate, reason = self.down_syndrome_mapping.validate_song_for_down_syndrome(song_features)
            if not is_appropriate:
                continue

            # Check if song matches preferred genres (case-insensitive)
            song_genre = str(song.get('genre', '')).lower()
            genre_match = any(
                ds_genre.lower() in song_genre or song_genre in ds_genre.lower()
                for ds_genre in ds_genres
            )

            # Include song if it matches preferred genres OR if no specific preference (fallback logic)
            if genre_match or len(ds_genres) == 0:
                # Apply audio feature filtering
                tempo_ok = audio_filters['tempo_min'] <= song.get('tempo', 120) <= audio_filters['tempo_max']
                valence_ok = audio_filters['valence_min'] <= song.get('valence', 0.5) <= audio_filters['valence_max']
                energy_ok = audio_filters['energy_min'] <= song.get('energy', 0.5) <= audio_filters['energy_max']

                if tempo_ok and valence_ok and energy_ok:
                    # Add Down syndrome specific metadata
                    song_copy = song.copy()
                    song_copy['condition_specific'] = 'down_syndrome'
                    song_copy['therapeutically_appropriate'] = True
                    song_copy['validation_reason'] = reason

                    filtered_songs.append(song_copy)

        # Apply language filtering if preferences exist
        if preferences and 'preferred_languages' in preferences:
            preferred_languages = preferences['preferred_languages']
            if preferred_languages:
                before_filter = filtered_songs
                filtered_songs = []

                for song in before_filter:
                    song_language = str(song.get('language', '')).lower()

                    # Check if song matches any preferred language
                    for pref_lang in preferred_languages:
                        pref_lang_lower = pref_lang.lower()
                        if (pref_lang_lower == 'english' and song_language == 'en') or \
                           (pref_lang_lower in ['bangla', 'bengali', 'bn'] and song_language == 'bn') or \
                           (pref_lang_lower == song_language):
                            filtered_songs.append(song)
                            break

                logger.info(f"Applied language filtering for Down syndrome: {len(filtered_songs)}/{len(before_filter)} songs remaining")

        # If no songs found after filtering, use fallback approach with less strict filtering
        if len(filtered_songs) < 10:
            logger.warning(f"Only {len(filtered_songs)} songs found with strict Down syndrome filtering, applying fallback logic")

            # Fallback: use songs that match basic tempo/valence criteria
            filtered_songs = []
            for song in all_songs:
                tempo = song.get('tempo', 120)
                valence = song.get('valence', 0.5)
                energy = song.get('energy', 0.5)

                # Basic therapeutic filters for Down syndrome
                if 50 <= tempo <= 140 and 0.4 <= valence <= 0.9 and 0.2 <= energy <= 0.9:
                    # Avoid potentially problematic genres
                    song_genre = str(song.get('genre', '')).lower()
                    problematic_genres = ['heavy metal', 'hard rock', 'aggressive', 'scream', 'harsh']

                    if not any(pg in song_genre for pg in problematic_genres):
                        song_copy = song.copy()
                        song_copy['condition_specific'] = 'down_syndrome_fallback'
                        song_copy['therapeutically_appropriate'] = True
                        filtered_songs.append(song_copy)

                        if len(filtered_songs) >= 20:  # Limit fallback results
                            break

        logger.info(f"Down syndrome catalog filtering: {len(filtered_songs)} songs selected from {len(all_songs)} total songs")
        return filtered_songs
