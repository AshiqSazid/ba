from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, date
import structlog
from math import isnan

from app.core.database import get_db
from app.schemas.therapy import (
    RecommendationRequest, RecommendationResponse, CategorizedRecommendationResponse,
    CategorizedRecommendations, TherapyCondition,
    IntakeData, RecommendationSong, PatientSummary, AlgorithmMetadata,
    BigFiveScores, RecommendationsHealthResponse, ConditionsResponse
)
from app.services.ml_service import MLService
from app.services.therapy_service import TherapyService
from app.services.personality_mapping import BigFivePersonalityMapping
from app.services.music_feature_mapping_service import MusicFeatureMappingService
from app.services.a_py_adapter import get_a_py_adapter

logger = structlog.get_logger(__name__)


def _safe_float(value: Any, default: float = None) -> Any:
    """
    Convert a value to float when possible; return default otherwise.
    Avoids blowing up on strings like "C major" present in d.xlsx columns.
    """
    try:
        converted = float(value)
        return converted if not isnan(converted) else default
    except Exception:
        return default


def calculate_age(date_of_birth: str) -> int:
    """Calculate age from date of birth string (YYYY-MM-DD)."""
    try:
        dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except ValueError:
        return 75  # Default age if date parsing fails


def calculate_big_five_scores(big_five_responses: List[int]) -> BigFiveScores:
    """Calculate Big Five scores from questionnaire responses using enhanced personality mapping."""
    if not big_five_responses or len(big_five_responses) < 10:
        return None

    # Use the enhanced personality mapping service
    scores = personality_mapper.calculate_big_five_from_responses(big_five_responses)
    return BigFiveScores(**scores)


def categorize_recommendations(recommendations: List[RecommendationSong]) -> CategorizedRecommendations:
    """Categorize songs based on their category field from the therapy service"""
    categorized = CategorizedRecommendations()

    for song in recommendations:
        # Extract category from the song data - add a field to track this
        category = getattr(song, 'category', None)
        if hasattr(song, 'category'):
            category = song.category
        else:
            # Fallback: try to determine category from other attributes
            if hasattr(song, 'genre') and song.genre:
                genre = song.genre.lower()
                if genre in ['jazz', 'classical', 'traditional', 'folk']:
                    if hasattr(song, 'artist') and 'sinatra' in song.artist.lower():
                        category = 'favorite_musician'
                    else:
                        category = 'favorite_genres'
                elif genre in ['rock', 'pop']:
                    category = 'instruments' if 'piano' in song.song_title.lower() or 'guitar' in song.song_title.lower() else 'favorite_genres'
                else:
                    category = 'favorite_genres'

        # Add to appropriate category
        if category == 'birthplace_location':
            categorized.country_songs_recommendations.append(song)
        elif category == 'favorite_genres':
            categorized.favorite_genres_recommendations.append(song)
        elif category == 'instruments':
            categorized.instruments_recommendations.append(song)
        elif category == 'favorite_musician':
            categorized.favorite_musician_recommendations.append(song)
        elif category == 'favorite_season':
            categorized.favorite_season_recommendations.append(song)
        elif category == 'natural_elements':
            categorized.natural_elements_recommendations.append(song)
        elif category == 'cognitive_indicators':
            categorized.cognitive_indicators_recommendations.append(song)
        elif category == 'big_five_personality':
            categorized.big_five_personality_recommendations.append(song)
        else:
            # Default to country songs if no category
            categorized.country_songs_recommendations.append(song)

    return categorized


def convert_to_categorized_recommendation_response(therapy_response: Any, intake_data: Dict[str, Any]) -> CategorizedRecommendationResponse:
    """Convert therapy service response to categorized API format."""

    # Calculate Big Five scores from responses if available
    big_five_scores = None
    if 'bigFiveResponses' in intake_data and intake_data['bigFiveResponses']:
        big_five_scores = calculate_big_five_scores(intake_data['bigFiveResponses'])
    elif 'big_five' in intake_data and intake_data['big_five']:
        big_five_scores = BigFiveScores(**intake_data['big_five'])

    # Convert recommendations to new format
    recommendations = []
    if hasattr(therapy_response, 'recommendations') and therapy_response.recommendations:
        for rec in therapy_response.recommendations:
            song = RecommendationSong(
                id=getattr(rec, 'id', None),
                song_id=getattr(rec, 'song_id', None),
                song_title=getattr(rec, 'song_title', getattr(rec, 'title', 'Unknown')),
                artist=getattr(rec, 'artist', 'Unknown'),
                genre=getattr(rec, 'genre', None),
                language=getattr(rec, 'language', None),  # Include language field
                year=getattr(rec, 'year', None),
                tempo=getattr(rec, 'tempo', None),
                valence=getattr(rec, 'valence', None),
                arousal=getattr(rec, 'arousal', None),
                recommendation_score=getattr(rec, 'recommendation_score', 0.5),
                rank=getattr(rec, 'rank', 1),
                algorithm_used=getattr(rec, 'algorithm_used', 'contextual_bandit_algorithm'),
                created_at=getattr(rec, 'created_at', datetime.now()),
                category=getattr(rec, 'category', None),  # Include category field
                youtube_url=getattr(rec, 'youtube_url', None),
                spotify_url=getattr(rec, 'spotify_url', None)
            )
            recommendations.append(song)

    # Calculate age
    age = calculate_age(intake_data.get('dateOfBirth', '1950-01-01'))

    # Create patient summary
    patient_summary = PatientSummary(
        name=intake_data.get('name', 'Unknown'),
        age=age,
        condition=intake_data.get('condition', 'dementia'),
        session_id=getattr(therapy_response, 'session_id', 'unknown'),
        recommendation_count=len(recommendations),
        birth_date=intake_data.get('dateOfBirth'),
        birthplace_city=intake_data.get('birthplaceCity'),
        birthplace_country=intake_data.get('birthplaceCountry'),
        sex=intake_data.get('sex')
    )

    # Get algorithm metadata from therapy response
    if hasattr(therapy_response, 'algorithm_metadata') and therapy_response.algorithm_metadata:
        # If it's already an AlgorithmMetadata object, use it directly
        if isinstance(therapy_response.algorithm_metadata, AlgorithmMetadata):
            algorithm_metadata = therapy_response.algorithm_metadata
        else:
            # If it's a dict, unpack it
            algorithm_metadata = AlgorithmMetadata(**therapy_response.algorithm_metadata)
    else:
        # Create comprehensive algorithm metadata
        features_used = []
        if intake_data.get('bigFiveResponses'):
            features_used.append("big_five_personality")
        if intake_data.get('favoriteGenres'):
            features_used.append("genre_preferences")
        if intake_data.get('instruments'):
            features_used.append("instrument_preferences")
        if intake_data.get('birthplaceCountry'):
            features_used.append("cultural_context")
        if intake_data.get('favoriteMusician'):
            features_used.append("artist_preferences")
        if intake_data.get('favoriteSeason'):
            features_used.append("seasonal_context")
        if intake_data.get('naturalElements'):
            features_used.append("nature_preferences")
        if any([intake_data.get('difficultySleeping'), intake_data.get('troubleRemembering')]):
            features_used.append("cognitive_indicators")

        algorithm_metadata = AlgorithmMetadata(
            algorithm="Contextual Bandit with Personalization",
            songs_considered=len(recommendations),
            features_used=" + ".join(features_used) if features_used else "demographic + contextual",
            condition=intake_data.get('condition', 'dementia'),
            timestamp=datetime.now().isoformat()
        )

    # Categorize recommendations
    categorized_recommendations = categorize_recommendations(recommendations)

    return CategorizedRecommendationResponse(
        session_id=getattr(therapy_response, 'session_id', 'unknown'),
        patient_summary=patient_summary,
        big_five_scores=big_five_scores,
        algorithm_metadata=algorithm_metadata,
        recommendations=categorized_recommendations
    )


router = APIRouter()

# Initialize services
ml_service = MLService()
from app.core.config import settings
therapy_service = TherapyService(ml_service, catalog_path=settings.MUSIC_CATALOG_PATH)
personality_mapper = BigFivePersonalityMapping()
music_feature_mapping_service = MusicFeatureMappingService(settings.MUSIC_CATALOG_PATH)


@router.post("/recommendations", response_model=CategorizedRecommendationResponse)
async def create_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate categorized music recommendations for a patient based on their intake form.

    This endpoint returns recommendations grouped by categories:
    - country_songs_recommendations: Songs based on user's location/country
    - favorite_genres_recommendations: Songs matching favorite genres
    - instruments_recommendations: Songs featuring preferred instruments
    - favorite_musician_recommendations: Songs by favorite artists
    - favorite_season_recommendations: Songs matching seasonal preferences
    - natural_elements_recommendations: Songs inspired by nature
    - cognitive_indicators_recommendations: Songs for cognitive support
    - big_five_personality_recommendations: Songs based on personality traits

    The system uses machine learning to personalize music recommendations based on:
    - Patient demographics and clinical information from intake form
    - Big Five personality traits (if questionnaire responses provided)
    - Therapeutic condition (dementia, ADHD, Down syndrome)
    - Music preferences and constraints
    """
    return await _create_categorized_recommendations_impl(request, db)


@router.post("/recommendations/personality-assessment")
async def assess_big_five_personality(responses: List[int]):
    """
    Assess Big Five personality from 10-question response format.

    Args:
        responses: List of 10 responses on 1-7 scale

    Returns:
        Dictionary with Big Five scores and personality insights
    """
    try:
        if len(responses) != 10:
            raise HTTPException(status_code=400, detail="Exactly 10 responses required for Big Five assessment")

        if any(not isinstance(r, int) or r < 1 or r > 7 for r in responses):
            raise HTTPException(status_code=400, detail="All responses must be integers between 1 and 7")

        # Calculate Big Five scores
        scores = personality_mapper.calculate_big_five_from_responses(responses)
        dominant_traits = personality_mapper.get_dominant_traits(scores, top_n=3)
        personality_summary = personality_mapper.get_personality_summary(scores)
        genre_recommendations = personality_mapper.get_genres_for_personality(scores)
        detailed_insights = personality_mapper.get_personality_insights(scores)

        return {
            "big_five_scores": {
                trait: {
                    "score_0_1": round(score, 3),
                    "score_1_7": max(1, min(7, round(score * 6 + 1))),
                    "level": "High" if score >= 0.67 else "Medium" if score >= 0.33 else "Low"
                }
                for trait, score in scores.items()
            },
            "dominant_traits": dominant_traits,
            "personality_summary": personality_summary,
            "recommended_genres": [
                {
                    "trait": rec["trait"],
                    "genre": rec["genre"],
                    "score": round(rec["score"], 2),
                    "interpretation": personality_mapper._get_personality_interpretation(rec["trait"], rec["score"])
                }
                for rec in genre_recommendations[:8]  # Top 8 recommendations
            ],
            "detailed_insights": detailed_insights,
            "assessment_metadata": {
                "total_questions": 10,
                "scale_used": "1-7 Likert scale",
                "assessment_type": "BFI-2-X (Big Five Inventory-2-Short Form)",
                "personality_traits": ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism (Emotional Stability)"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in personality assessment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during personality assessment")


@router.post("/recommendations/big-five-songs")
async def get_big_five_song_recommendations(responses: List[int], total_recommendations: int = 20):
    """
    Get song recommendations based on Big Five personality from d.xlsx music database.
    Maps personality traits to genres using psychological research.

    Args:
        responses: List of 10 responses on 1-7 scale (BFI-2-X format)
        total_recommendations: Number of songs to recommend (default: 20)

    Returns:
        Dictionary containing personality-based song recommendations from d.xlsx
    """
    try:
        if len(responses) != 10:
            raise HTTPException(status_code=400, detail="Exactly 10 responses required for Big Five song recommendations")

        if any(not isinstance(r, int) or r < 1 or r > 7 for r in responses):
            raise HTTPException(status_code=400, detail="All responses must be integers between 1 and 7")

        if total_recommendations < 1 or total_recommendations > 100:
            raise HTTPException(status_code=400, detail="Total recommendations must be between 1 and 100")

        # Import and use the Big Five recommendation service
        from app.services.big_five_recommendation_service import BigFiveRecommendationService

        # Initialize the service with the music database path
        from app.core.config import settings
        recommendation_service = BigFiveRecommendationService(settings.MUSIC_CATALOG_PATH)

        # Extract preferred languages from the responses (if available) or use English as default
        preferred_languages = ['english']  # Default fallback

        # Try to get preferred languages from context or use English as default
        # Note: This endpoint doesn't have access to intake data, so we use English as default
        # The main /recommendations endpoint properly handles preferred languages

        # Get personality-based song recommendations
        recommendations = recommendation_service.get_big_five_recommendations(
            responses,
            total_recommendations,
            preferred_languages
        )

        # Format the response
        formatted_songs = []
        for i, song in enumerate(recommendations.get('recommendations', []), 1):
            formatted_song = {
                "id": i,
                "song_id": song.get('id', i),
                "song_title": song.get('song_name', ''),
                "artist": song.get('singer', ''),
                "genre": song.get('genre', ''),
                "language": song.get('language', ''),
                "mood": song.get('mood', ''),
                "released_date": song.get('released_date', ''),
                "year": song.get('released_date', ''),
                "tempo": song.get('tempo', 0.0),
                "valence": song.get('Valence', 0.5),
                "energy": song.get('energy', 0.5),
                "danceability": song.get('danceability', 0.5),
                "acousticness": song.get('acousticness', 0.5),
                "recommendation_score": song.get('match_score', 0.8),
                "rank": i,
                "algorithm_used": recommendations.get('algorithm_metadata', {}).get('algorithm', 'Big_Five_Personality_Genre_Mapping'),
                "category": "big_five_personality",
                "personality_trait": song.get('personality_trait', ''),
                "trait_score_1_7": song.get('trait_score_1_7', 4),
                "trait_score_0_1": song.get('trait_score_0_1', 0.5),
                "trait_level": song.get('trait_level', 'Medium'),
                "match_reason": song.get('match_reason', ''),
                "confidence": song.get('confidence', 0.8),
                "youtube_url": song.get('youtube link', ''),
                "spotify_url": song.get('spotify link', ''),
                "used_instruments": song.get('used instruments in the song', '')
            }
            formatted_songs.append(formatted_song)

        # Calculate Big Five scores for response
        personality_scores = recommendation_service._calculate_big_five_scores(responses)

        return {
            "personality_profile": {
                "big_five_scores": {
                    trait: {
                        "score_0_1": round(score, 3),
                        "score_1_7": max(1, min(7, round(score * 6 + 1))),
                        "level": "High" if score >= 0.67 else "Medium" if score >= 0.33 else "Low"
                    }
                    for trait, score in personality_scores.items()
                },
                "dominant_traits": recommendations.get('dominant_traits', []),
                "personality_summary": recommendations.get('personality_summary', '')
            },
            "recommendations": formatted_songs,
            "recommendation_metadata": {
                "total_recommendations": len(formatted_songs),
                "algorithm": recommendations.get('algorithm_metadata', {}).get('algorithm', 'Big_Five_Personality_Genre_Mapping_v1.0'),
                "songs_analyzed": recommendations.get('algorithm_metadata', {}).get('total_songs_analyzed', 0),
                "traits_considered": len(recommendations.get('dominant_traits', [])),
                "music_database": "d.xlsx",
                "personality_assessment": "BFI-2-X (10-item)",
                "timestamp": datetime.now().isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Big Five song recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during Big Five song recommendations")


@router.post("/recommend", response_model=CategorizedRecommendationResponse)
async def create_recommendations_alias(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Alias for /recommendations endpoint for frontend compatibility.
    Returns categorized music recommendations.
    """
    return await _create_categorized_recommendations_impl(request, db)




async def _create_recommendations_impl(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    try:
        intake = request.intake
        logger.info(
            "Received recommendation request",
            condition=intake.get('condition'),
            has_big_five='bigFiveResponses' in intake
        )

        # Validate request data
        if not intake:
            raise HTTPException(status_code=400, detail="Intake information is required")

        # Convert intake format to internal format
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        from app.schemas.therapy import TherapyCondition, RecommendationRequestLegacy

        # Calculate age from dateOfBirth
        date_of_birth = datetime.strptime(intake['dateOfBirth'], '%Y-%m-%d').date()
        today = datetime.now().date()
        age = relativedelta(today, date_of_birth).years

        # Extract patient info
        patient_info = {
            'name': intake['name'],
            'age': age,
            'gender': intake['sex'],
            'condition': intake['condition'],
            'birth_date': intake['dateOfBirth'],
            'birthplace_city': intake.get('birthplaceCity'),
            'birthplaceCountry': intake.get('birthplaceCountry'),  # Keep camelCase for compatibility
            'instruments': intake.get('instruments', []),
            'preferred_languages': intake.get('preferredLanguages', []),
            'favorite_genres': intake.get('favoriteGenres', []),
            'favorite_musician': intake.get('favoriteMusician'),
            'favorite_season': intake.get('favoriteSeason'),
            'natural_elements': intake.get('naturalElements', []),
            'difficulty_sleeping': intake.get('difficultySleeping', False),
            'trouble_remembering': intake.get('troubleRemembering', False),
            'forgets_everyday_things': intake.get('forgetsEverydayThings', False),
            'difficulty_recalling_old_memories': intake.get('difficultyRecallingOldMemories', False),
            'memory_worse_than_year_ago': intake.get('memoryWorseThanYearAgo', False),
            'visited_mental_health_professional': intake.get('visitedMentalHealthProfessional', False)
        }
        # Duplicate key styles for compatibility across services
        patient_info['preferredLanguages'] = patient_info['preferred_languages']
        patient_info['favoriteGenres'] = patient_info['favorite_genres']
        patient_info['birthplaceCountry'] = patient_info['birthplaceCountry']  # Ensure camelCase exists

        # Extract preferences
        preferences = {
            'genres': intake.get('favoriteGenres', []),
            'artists': [intake.get('favoriteMusician')] if intake.get('favoriteMusician') else [],
            'instruments': intake.get('instruments', []),
            'languages': intake.get('preferredLanguages', [])
        }

        # Map condition string to enum
        condition_map = {
            'dementia': TherapyCondition.DEMENTIA,
            'adhd': TherapyCondition.ADHD,
            'down_syndrome': TherapyCondition.DOWN_SYNDROME
        }
        condition = condition_map.get(intake['condition'].lower(), TherapyCondition.DEMENTIA)

        # Extract Big Five responses if available
        big_five_responses = intake.get('bigFiveResponses')

        # Create legacy request format
        legacy_request = RecommendationRequestLegacy(
            patient_info=patient_info,
            condition=condition,
            preferences=preferences if any(preferences.values()) else None,
            big_five_responses=big_five_responses
        )

        # Generate recommendations using therapy service with real database connection
        response = therapy_service.generate_recommendations(db, legacy_request)

        logger.info(
            "Successfully generated recommendations",
            session_id=response.session_id,
            num_recommendations=len(response.recommendations)
        )

        # Convert to new API response format
        api_response = convert_to_recommendation_response(response, intake)
        return api_response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _create_categorized_recommendations_impl(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """Implementation for categorized recommendations endpoint."""
    try:
        intake = request.intake
        logger.info(
            "Received categorized recommendation request",
            condition=intake.get('condition'),
            has_big_five='bigFiveResponses' in intake
        )

        # Validate request data
        if not intake:
            raise HTTPException(status_code=400, detail="Intake information is required")

        # Convert intake to IntakeData model
        intake_data = IntakeData(**intake)

        # Process Big Five responses if provided
        big_five_scores = None
        if intake_data.bigFiveResponses:
            # Convert BigFiveResponses object to flat list of integers
            if isinstance(intake_data.bigFiveResponses, dict):
                # If it's already a dict, convert to flat list in the correct order
                responses_list = []
                trait_order = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
                for trait in trait_order:
                    if trait in intake_data.bigFiveResponses:
                        responses_list.extend(intake_data.bigFiveResponses[trait])
                big_five_scores = calculate_big_five_scores(responses_list)
            elif hasattr(intake_data.bigFiveResponses, 'extraversion'):
                # If it's a BigFiveResponses object, convert to flat list
                responses_list = []
                responses_list.extend(intake_data.bigFiveResponses.extraversion)
                responses_list.extend(intake_data.bigFiveResponses.agreeableness)
                responses_list.extend(intake_data.bigFiveResponses.conscientiousness)
                responses_list.extend(intake_data.bigFiveResponses.neuroticism)
                responses_list.extend(intake_data.bigFiveResponses.openness)
                big_five_scores = calculate_big_five_scores(responses_list)
            else:
                # Assume it's already a flat list
                big_five_scores = calculate_big_five_scores(intake_data.bigFiveResponses)

            intake_data.big_five = big_five_scores

        # Generate therapy recommendations using the service
        therapy_response = therapy_service.generate_recommendations(
            db=db,
            request=request
        )

        # Generate cognitive indicators recommendations (only when indicators are selected and catalog is available)
        cognitive_recommendations: List[RecommendationSong] = []

        # Calculate which cognitive indicators are selected
        target_indicators = [
            name for name, flag in [
                ("Difficulty Sleeping", intake.get('difficultySleeping')),
                ("Trouble Remembering Recent Events", intake.get('troubleRememberingRecentEvents') or intake.get('troubleRemembering')),
                ("Forgets Everyday Tasks", intake.get('forgetsEverydayTasks') or intake.get('forgetsEverydayThings')),
                ("Difficulty Recalling Older Memories", intake.get('difficultyRecallingOldMemories')),
                ("Memory Worse Than a Year Ago", intake.get('memoryWorseThanYearAgo')),
                ("Visited Mental Health Professional", intake.get('visitedMentalHealthProfessional')),
            ] if flag
        ]

        if music_feature_mapping_service.catalog_loaded and target_indicators:
            try:
                preferred_languages = intake.get('preferredLanguages', ['english'])

                filtered_songs: List[Dict[str, Any]] = []
                for song in music_feature_mapping_service.songs_database:
                    song_lang = str(song.get('language', '')).lower()
                    if preferred_languages:
                        if any(
                            song_lang == pref.lower()
                            or song_lang == pref[:2].lower()
                            or (pref.lower() == 'english' and song_lang == 'en')
                            or (pref.lower() in ['bengali', 'bangla'] and song_lang == 'bn')
                            for pref in preferred_languages
                        ):
                            filtered_songs.append(song)
                    else:
                        filtered_songs.append(song)

                candidates: List[Dict[str, Any]] = []
                for song in filtered_songs:
                    # Use only the 12 therapeutic dimensions; coerce non-numeric values safely.
                    song_features = {
                        feature: _safe_float(song.get(feature, 0.5), 0.5)
                        for feature in music_feature_mapping_service.FEATURES
                    }
                    indicator_scores = music_feature_mapping_service.map_music_to_indicators(song_features)

                    # Get additional audio features for scoring
                    tempo = _safe_float(song.get('tempo', 120), 120)
                    valence = _safe_float(song.get('Valence', song.get('valence', 0.5)), 0.5)
                    energy = _safe_float(song.get('energy', song.get('Arousal', 0.5)), 0.5)
                    acousticness = _safe_float(song.get('acousticness', 0.5), 0.5)

                    # Calculate specialized scores based on specific cognitive indicators
                    if target_indicators:
                        # Weight scores differently based on which indicators are selected
                        indicator_weights = {
                            "Difficulty Sleeping": {
                                "audio_relaxing": 0.4, "lyric_relaxing": 0.3, "tempo_slow": 0.2, "acousticness": 0.1
                            },
                            "Trouble Remembering Recent Events": {
                                "audio_motivational": 0.3, "lyric_reappraisal": 0.3, "valence_positive": 0.2, "tempo_medium": 0.2
                            },
                            "Forgets Everyday Tasks": {
                                "audio_motivational": 0.35, "lyric_motivational": 0.35, "tempo_steady": 0.2, "energy_moderate": 0.1
                            },
                            "Difficulty Recalling Older Memories": {
                                "audio_reappraisal": 0.4, "lyric_reappraisal": 0.3, "familiarity_boost": 0.2, "valence_positive": 0.1
                            },
                            "Memory Worse Than a Year Ago": {
                                "audio_reappraisal": 0.35, "audio_motivational": 0.25, "lyric_reappraisal": 0.2, "valence_positive": 0.2
                            },
                            "Visited Mental Health Professional": {
                                "audio_relaxing": 0.25, "audio_reappraisal": 0.25, "lyric_reappraisal": 0.25, "valence_balanced": 0.25
                            }
                        }

                        # Calculate weighted score based on selected indicators
                        total_score = 0.0
                        total_weight = 0.0

                        for indicator in target_indicators:
                            if indicator in indicator_weights:
                                weights = indicator_weights[indicator]

                                # Calculate individual component scores
                                audio_relaxing = song_features.get("audio's Relaxing", 0.5)
                                lyric_relaxing = song_features.get("Lyric's Relaxing", 0.5)
                                audio_motivational = song_features.get("audio's Motivational", 0.5)
                                lyric_motivational = song_features.get("Lyric's Motivational", 0.5)
                                audio_reappraisal = song_features.get("audio's Reappraisal", 0.5)
                                lyric_reappraisal = song_features.get("Lyric's Reappraisal", 0.5)

                                # Tempo scoring (0-1 normalized)
                                tempo_slow = max(0, (100 - tempo) / 100) if tempo < 100 else 0  # Slow is good for sleep
                                tempo_medium = 1.0 if 80 <= tempo <= 120 else max(0, 1 - abs(tempo - 100) / 40)  # Medium is good
                                tempo_steady = 1.0 if 90 <= tempo <= 130 else max(0, 1 - abs(tempo - 110) / 40)  # Steady is good

                                # Energy scoring
                                energy_moderate = 1.0 if 0.3 <= energy <= 0.7 else max(0, 1 - abs(energy - 0.5) * 2)

                                # Valence scoring
                                valence_positive = valence
                                valence_balanced = 1.0 if 0.4 <= valence <= 0.7 else max(0, 1 - abs(valence - 0.55) * 2)

                                # Familiarity boost (for older songs)
                                year = song.get('released_date', 2025)
                                familiarity_boost = 1.0 if year and year < 2000 else 0.7

                                # Calculate weighted sum for this indicator
                                indicator_score = (
                                    weights.get("audio_relaxing", 0) * audio_relaxing +
                                    weights.get("lyric_relaxing", 0) * lyric_relaxing +
                                    weights.get("audio_motivational", 0) * audio_motivational +
                                    weights.get("lyric_motivational", 0) * lyric_motivational +
                                    weights.get("audio_reappraisal", 0) * audio_reappraisal +
                                    weights.get("lyric_reappraisal", 0) * lyric_reappraisal +
                                    weights.get("tempo_slow", 0) * tempo_slow +
                                    weights.get("tempo_medium", 0) * tempo_medium +
                                    weights.get("tempo_steady", 0) * tempo_steady +
                                    weights.get("energy_moderate", 0) * energy_moderate +
                                    weights.get("valence_positive", 0) * valence_positive +
                                    weights.get("valence_balanced", 0) * valence_balanced +
                                    weights.get("familiarity_boost", 0) * familiarity_boost +
                                    weights.get("acousticness", 0) * acousticness
                                )

                                # Normalize by sum of weights for this indicator
                                weight_sum = sum(weights.values())
                                if weight_sum > 0:
                                    indicator_score = indicator_score / weight_sum
                                    total_score += indicator_score
                                    total_weight += 1

                        # Final support score (0-1 scale)
                        support_score = min(1.0, total_score / max(total_weight, 1)) if total_weight > 0 else 0.5
                    else:
                        # Fallback if no indicators selected
                        support_score = 0.5

                    # Detect if this song already carries a match score tag (to surface it first)
                    match_score = song.get('match_score') or song.get('Match Score')

                    candidates.append({
                        "song": song,
                        "support_score": support_score,
                        "has_match_score": match_score is not None,
                        "indicator_match": target_indicators
                    })

                # Sort so that songs with a match_score tag appear first, then by support score
                candidates.sort(key=lambda x: (not x["has_match_score"], -x["support_score"]))
                top_candidates = candidates[:10]

                for idx, item in enumerate(top_candidates, 1):
                    song = item["song"]
                    cognitive_song = RecommendationSong(
                        id=song.get('id', idx),
                        song_id=song.get('id'),
                        song_title=song.get('song_name', song.get('title', '')),
                        artist=song.get('singer', song.get('artist', '')),
                        genre=song.get('genre', ''),
                        language=song.get('language', ''),
                        year=song.get('released_date'),
                        tempo=song.get('tempo'),
                        # Only include numeric valence/arousal if coercible; otherwise drop to None
                        valence=_safe_float(song.get('Valence', song.get('valence')), None),
                        arousal=_safe_float(song.get('energy', song.get('Arousal')), None),
                        recommendation_score=round(item["support_score"], 3),
                        rank=idx,
                        algorithm_used='Cognitive_Indicators_Music_Mapping_v1.1',
                        created_at=datetime.now(),
                        category='cognitive_indicators',
                        youtube_url=song.get('youtube link', song.get('youtube_url', '')),
                        spotify_url=song.get('spotify link', song.get('spotify_url', ''))
                    )
                    cognitive_recommendations.append(cognitive_song)

                logger.info(
                    "Generated %d cognitive indicator recommendations using d.xlsx features",
                    len(cognitive_recommendations)
                )
            except Exception as e:
                logger.error("Failed to generate cognitive indicators recommendations: %s", e)

        # Convert to categorized response format
        response = convert_to_categorized_recommendation_response(therapy_response, intake)

        # If no cognitive indicators were selected, clear the cognitive recommendations from therapy service
        if not target_indicators:
            response.recommendations.cognitive_indicators_recommendations = []
        else:
            # Add our improved cognitive recommendations with proper scoring
            start_rank = 30
            ranked_recs: List[RecommendationSong] = []
            for idx, rec in enumerate(reversed(cognitive_recommendations), 1):
                try:
                    rec.rank = max(1, start_rank - (idx - 1))
                except Exception:
                    pass
                ranked_recs.append(rec)
            # Replace the therapy service recommendations with our improved ones
            response.recommendations.cognitive_indicators_recommendations = ranked_recs

        logger.info(
            "Generated categorized recommendations",
            session_id=response.session_id,
            total_recommendations=len(response.recommendations.country_songs_recommendations) +
                               len(response.recommendations.favorite_genres_recommendations) +
                               len(response.recommendations.instruments_recommendations) +
                               len(response.recommendations.favorite_musician_recommendations) +
                               len(response.recommendations.favorite_season_recommendations) +
                               len(response.recommendations.natural_elements_recommendations) +
                               len(response.recommendations.cognitive_indicators_recommendations) +
                               len(response.recommendations.big_five_personality_recommendations)
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate categorized recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recommendations/health", response_model=RecommendationsHealthResponse)
async def health_check():
    """
    Health check endpoint for the recommendations service.
    """
    return RecommendationsHealthResponse(
        status="healthy",
        service="recommendations",
        model_loaded=hasattr(ml_service, 'bandits') and len(ml_service.bandits) > 0
    )


@router.get("/recommendations/conditions", response_model=ConditionsResponse)
async def get_supported_conditions():
    """
    Get list of supported therapy conditions.
    """
    from app.schemas.therapy import ConditionInfo

    return ConditionsResponse(
        conditions=[
            ConditionInfo(
                id=TherapyCondition.DEMENTIA.value,
                name="Dementia",
                description="Music therapy for dementia patients focusing on cognitive stimulation and emotional regulation"
            ),
            ConditionInfo(
                id=TherapyCondition.ADHD.value,
                name="ADHD",
                description="Music therapy for ADHD patients focusing on attention and focus enhancement"
            ),
            ConditionInfo(
                id=TherapyCondition.DOWN_SYNDROME.value,
                name="Down Syndrome",
                description="Music therapy for Down syndrome patients focusing on development and social engagement"
            )
        ]
    )


@router.post("/recommendations/a-py", response_model=Dict[str, Any])
async def create_recommendations_with_a_py(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate music recommendations using the a.py standalone script.

    This endpoint uses the TheramuseRecommender class from a.py to generate
    personalized music therapy recommendations based on the patient's intake data.

    The a.py script provides:
    - Condition-specific music profiles (ADHD, Dementia, Down Syndrome)
    - Audio feature filtering (tempo, energy, valence, danceability, etc.)
    - Genre, instrument, language, and artist preferences
    - Big Five personality integration
    - LLM-based personalized recommendations

    Returns recommendations with detailed audio features and therapeutic rationale.
    """
    try:
        intake = request.intake
        logger.info(
            "Received a.py recommendation request",
            condition=intake.get('condition'),
            has_big_five='bigFiveResponses' in intake
        )

        # Get the a.py adapter
        a_py_adapter = get_a_py_adapter()
        if a_py_adapter is None:
            raise HTTPException(
                status_code=503,
                detail="a.py adapter is not available. Please ensure a.py exists and all dependencies are installed."
            )

        # Check if adapter is functional
        if not a_py_adapter.is_available():
            raise HTTPException(
                status_code=503,
                detail="a.py recommender is not available. Check that required libraries (pandas, requests) are installed."
            )

        # Generate recommendations using a.py
        n_recommendations = intake.get('recommendationCount', 60)
        result = a_py_adapter.generate_recommendations(intake, n_recommendations)

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate recommendations from a.py"
            )

        logger.info(
            "Successfully generated a.py recommendations",
            total_count=result['total_count'],
            source=result['source']
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate a.py recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/recommendations/a-py/profile/{condition}", response_model=Dict[str, Any])
async def get_music_profile(condition: str):
    """
    Get the music profile for a specific therapeutic condition from a.py.

    Returns the audio feature ranges and preferences used for recommendations:
    - tempo_range: Target BPM range
    - energy_range: Energy level (0-1)
    - valence_range: Musical positivity (0-1)
    - danceability_range: Danceability score (0-1)
    - acousticness_range: Acoustic content (0-1)
    - speechiness_max: Maximum speech content
    - loudness_range: Loudness in dB
    - preferred_genres: List of preferred genres
    - therapeutic_tags: Clinical goals for the condition
    """
    try:
        a_py_adapter = get_a_py_adapter()
        if a_py_adapter is None:
            raise HTTPException(
                status_code=503,
                detail="a.py adapter is not available"
            )

        profile = a_py_adapter.get_music_profile(condition)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Music profile not found for condition: {condition}"
            )

        return {
            'condition': condition,
            'profile': profile,
            'source': 'a_py_theramuse_recommender'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get music profile: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/recommendations/a-py/health", response_model=Dict[str, Any])
async def a_py_health_check():
    """
    Health check for the a.py adapter service.

    Returns the status of a.py integration and whether it's ready to use.
    """
    a_py_adapter = get_a_py_adapter()

    is_available = a_py_adapter is not None and a_py_adapter.is_available()

    return {
        'status': 'available' if is_available else 'unavailable',
        'adapter_loaded': a_py_adapter is not None,
        'recommender_ready': is_available,
        'source': 'a_py_theramuse_recommender'
    }

