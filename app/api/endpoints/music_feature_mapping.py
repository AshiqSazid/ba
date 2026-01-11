"""
Music Feature Mapping API Endpoints

Provides endpoints to map music features (lyrics and audio characteristics)
to Cognitive & Lifestyle Indicators and generate therapeutic song suggestions.

Main endpoint:
- POST /music-features/map: Map music features to cognitive indicators and get song suggestions
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime
import structlog

from app.core.database import get_db
from app.schemas.therapy import (
    MusicFeatureRequest, MusicFeatureMappingResponse
)
from app.services.music_feature_mapping_service import MusicFeatureMappingService

logger = structlog.get_logger(__name__)

router = APIRouter()

# Initialize the music feature mapping service
def get_music_feature_mapping_service():
    """Get an instance of the music feature mapping service."""
    from app.core.config import settings
    return MusicFeatureMappingService(settings.MUSIC_CATALOG_PATH)


@router.post("/music-features/map", response_model=MusicFeatureMappingResponse)
async def map_music_features_to_cognitive_indicators(
    request: MusicFeatureRequest,
    db: Session = Depends(get_db)
):
    """
    Map music features to Cognitive & Lifestyle Indicators and generate therapeutic song suggestions.

    This endpoint accepts 12 music features on a 0.0-1.0 scale:

    **Input Features:**
    - Lyric's Reappraisal, Lyric's Distracting, Lyric's Uplifting, Lyric's Relaxing
    - Lyric's Suppressing, Lyric's Motivational
    - audio's Reappraisal, audio's Distracting, audio's Uplifting, audio's Relaxing
    - audio's Suppressing, audio's Motivational

    **Output Cognitive Indicators:**
    - Difficulty Sleeping
    - Trouble Remembering Recent Events
    - Forgets Everyday Tasks
    - Difficulty Recalling Older Memories
    - Memory Worse Than a Year Ago
    - Visited Mental Health Professional

    **Returns:**
    - Cognitive indicator scores (0.0-1.0) with interpretation levels
    - Personalized song suggestions based on cognitive profile
    - Therapeutic approach recommendations
    - Complete processing metadata

    **Algorithm:**
    Uses weighted logistic regression with sensitivity adjustment to map features
    to cognitive indicators, then selects therapeutic songs based on the cognitive profile.

    **Song Selection Criteria:**
    - Moderate tempo (60-100 BPM) for cognitive engagement
    - Positive emotional content (high valence)
    - Appropriate energy levels to avoid overstimulation
    - Cognitive-friendly genres (classical, jazz, folk, ambient, instrumental)
    - Supportive moods (calm, peaceful, soothing, uplifting, gentle)
    - Language preferences when specified
    """
    try:
        logger.info(
            "Received music feature mapping request",
            total_features=len(request.features),
            sensitivity=request.sensitivity,
            total_suggestions=request.total_suggestions,
            preferred_languages=request.preferred_languages
        )

        # Validate that all required features are present
        required_features = [
            "Lyric's Reappraisal", "Lyric's Distracting", "Lyric's Uplifting", "Lyric's Relaxing",
            "Lyric's Suppressing", "Lyric's Motivational", "audio's Reappraisal", "audio's Distracting",
            "audio's Uplifting", "audio's Relaxing", "audio's Suppressing", "audio's Motivational"
        ]

        missing_features = [f for f in required_features if f not in request.features]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}. All 12 features must be provided."
            )

        # Get the music feature mapping service
        mapping_service = get_music_feature_mapping_service()

        # Process music features and get results
        result = mapping_service.process_music_features_and_get_songs(
            music_features=request.features,
            total_suggestions=request.total_suggestions,
            preferred_languages=request.preferred_languages,
            sensitivity=request.sensitivity
        )

        # Convert the result to match the response schema
        # Convert cognitive indicators to proper schema format
        cognitive_indicators = {}
        for indicator, data in result["cognitive_indicators"].items():
            cognitive_indicators[indicator] = {
                "score": data["score"],
                "level": data["level"],
                "interpretation": data["interpretation"]
            }

        # Convert song recommendations metadata
        song_recommendations = result["song_recommendations"]
        therapeutic_approach = song_recommendations.get("therapeutic_approach", {})
        cognitive_profile = song_recommendations.get("cognitive_profile", {})

        formatted_recommendations = {
            "recommendations": song_recommendations.get("recommendations", []),
            "cognitive_profile": cognitive_profile,
            "therapeutic_approach": {
                "memory_support_level": therapeutic_approach.get("memory_support_level", "Moderate"),
                "sleep_support_level": therapeutic_approach.get("sleep_support_level", "Moderate"),
                "overall_strategy": therapeutic_approach.get("overall_strategy", "General wellness")
            },
            "recommendation_metadata": song_recommendations.get("recommendation_metadata", {
                "total_recommendations": 0,
                "algorithm": "Cognitive_Indicators_Music_Mapping_v1.0",
                "songs_analyzed": 0,
                "music_database": "Not available",
                "cognitive_indicators_considered": list(cognitive_indicators.keys()),
                "therapeutic_approach": "Evidence-based music selection",
                "timestamp": result["processing_metadata"]["timestamp"]
            })
        }

        # Create the response
        response = MusicFeatureMappingResponse(
            input_features=result["input_features"],
            cognitive_indicators=cognitive_indicators,
            song_recommendations=formatted_recommendations,
            processing_metadata=result["processing_metadata"]
        )

        logger.info(
            "Successfully processed music feature mapping",
            cognitive_indicators_generated=len(cognitive_indicators),
            song_recommendations_generated=len(formatted_recommendations["recommendations"]),
            overall_risk_score=cognitive_profile.get("overall_cognitive_risk_score", 0.0)
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process music feature mapping: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during music feature mapping"
        )


@router.post("/music-features/indicators-only")
async def map_music_features_to_indicators_only(
    features: Dict[str, float],
    sensitivity: float = 4.0
):
    """
    Map music features to Cognitive & Lifestyle Indicators only (no song suggestions).

    This lightweight endpoint returns only the cognitive indicator scores
    without generating song recommendations.

    Args:
        features: Dictionary of 12 music feature scores (0.0-1.0)
        sensitivity: Sensitivity multiplier (default: 4.0)

    Returns:
        Dictionary with cognitive indicator scores and interpretations
    """
    try:
        # Validate required features
        required_features = [
            "Lyric's Reappraisal", "Lyric's Distracting", "Lyric's Uplifting", "Lyric's Relaxing",
            "Lyric's Suppressing", "Lyric's Motivational", "audio's Reappraisal", "audio's Distracting",
            "audio's Uplifting", "audio's Relaxing", "audio's Suppressing", "audio's Motivational"
        ]

        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )

        # Validate feature ranges
        invalid_features = [(f, val) for f, val in features.items() if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0]
        if invalid_features:
            raise HTTPException(
                status_code=400,
                detail=f"All features must be numbers between 0.0 and 1.0. Invalid: {invalid_features}"
            )

        # Get mapping service
        mapping_service = get_music_feature_mapping_service()

        # Map features to indicators
        indicators = mapping_service.map_music_to_indicators(features, sensitivity)

        # Add interpretations
        interpreted_indicators = {}
        for indicator, score in indicators.items():
            interpreted_indicators[indicator] = {
                "score": round(score, 3),
                "level": mapping_service.get_cognitive_level_interpretation(score),
                "interpretation": f"{mapping_service.get_cognitive_level_interpretation(score).lower()} of {indicator.lower()}"
            }

        return {
            "cognitive_indicators": interpreted_indicators,
            "input_features": {feature: round(features.get(feature, 0.5), 3) for feature in required_features},
            "processing_metadata": {
                "algorithm": "Music_Feature_to_Cognitive_Indicators_Mapping_v1.0",
                "sensitivity_used": sensitivity,
                "features_processed": len(features),
                "indicators_generated": len(indicators),
                "timestamp": datetime.now().isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to map music features to indicators: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during music feature mapping"
        )


@router.get("/music-features/info")
async def get_music_feature_mapping_info():
    """
    Get information about the music feature mapping system.

    Returns details about:
    - Required input features
    - Output cognitive indicators
    - Algorithm approach
    - Feature weighting information
    """
    return {
        "service_name": "Music Feature to Cognitive Indicators Mapping",
        "version": "1.0",
        "description": "Maps music features (lyrics and audio characteristics) to Cognitive & Lifestyle Indicators using weighted logistic regression",
        "input_features": {
            "total_features": 12,
            "feature_scale": "0.0 to 1.0",
            "required_features": [
                "Lyric's Reappraisal",
                "Lyric's Distracting",
                "Lyric's Uplifting",
                "Lyric's Relaxing",
                "Lyric's Suppressing",
                "Lyric's Motivational",
                "audio's Reappraisal",
                "audio's Distracting",
                "audio's Uplifting",
                "audio's Relaxing",
                "audio's Suppressing",
                "audio's Motivational"
            ]
        },
        "output_indicators": {
            "total_indicators": 6,
            "indicator_scale": "0.0 to 1.0",
            "indicators": [
                "Difficulty Sleeping",
                "Trouble Remembering Recent Events",
                "Forgets Everyday Tasks",
                "Difficulty Recalling Older Memories",
                "Memory Worse Than a Year Ago",
                "Visited Mental Health Professional"
            ],
            "interpretation_levels": {
                "0.70-1.00": "Strong indication",
                "0.40-0.70": "Moderate indication",
                "0.00-0.40": "Low indication"
            }
        },
        "algorithm": {
            "name": "Weighted Logistic Regression with Sensitivity Adjustment",
            "mapping_function": "sigmoid(intercept + sensitivity * Σ(weight * (feature - 0.5)))",
            "default_sensitivity": 4.0,
            "sensitivity_range": "0.1 to 10.0"
        },
        "song_recommendations": {
            "approach": "Therapeutic music selection based on cognitive profile",
            "selection_criteria": [
                "Moderate tempo (60-100 BPM) for cognitive engagement",
                "Positive emotional content (high valence)",
                "Appropriate energy levels to avoid overstimulation",
                "Cognitive-friendly genres (classical, jazz, folk, ambient, instrumental)",
                "Supportive moods (calm, peaceful, soothing, uplifting, gentle)",
                "Language preferences when specified"
            ],
            "cognitive_targets": [
                "Sleep support",
                "Memory engagement",
                "Cognitive relaxation",
                "Mood enhancement",
                "General cognitive support"
            ]
        },
        "endpoints": {
            "map_features_and_songs": "POST /music-features/map",
            "map_indicators_only": "POST /music-features/indicators-only",
            "service_info": "GET /music-features/info"
        }
    }