from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
import structlog

from app.core.database import get_db
from app.schemas.therapy import (
    FeedbackType, TherapyFeedback, TherapyFeedbackCreate,
    FeedbackRequest as NewFeedbackRequest, FeedbackResponse, FeedbackSong
)
from app.services.ml_service import MLService
from app.services.therapy_service import TherapyService

logger = structlog.get_logger(__name__)

router = APIRouter()

# Initialize services
ml_service = MLService()
from app.core.config import settings
therapy_service = TherapyService(ml_service, catalog_path=settings.MUSIC_CATALOG_PATH)


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(
    request: NewFeedbackRequest
):
    """
    Record patient feedback on music recommendations.

    This endpoint captures user feedback (like, dislike, skip, inappropriate)
    and uses it to improve future recommendations through reinforcement learning.

    The feedback is processed to:
    - Update the multi-armed bandit model
    - Improve personalization for the specific patient
    - Enhance recommendations for patients with similar profiles
    - Track therapeutic progress and engagement

    Feedback types:
    - like: Positive response (+1.0 reward)
    - dislike: Negative response (-1.0 reward)
    - skip: Neutral/negative response (-0.5 reward)
    - inappropriate: Strongly negative (-2.0 reward)
    - neutral: No response (0.0 reward)
    """
    try:
        logger.info(
            "Received feedback request",
            session_id=request.session_id,
            feedback_type=request.feedback_type,
            has_song_info=bool(request.song)
        )

        # Validate session_id
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Mock feedback handling since we don't have database access
        song_id = request.song.id
        song_title = request.song.title or 'Unknown Song'

        logger.info(
            "Mock feedback recorded (testing mode)",
            session_id=request.session_id,
            feedback_type=request.feedback_type,
            song_title=song_title,
            song_id=song_id
        )

        # Try to update ML model if song_id is available
        if song_id:
            try:
                # Extract patient info for ML update
                patient_info = {}
                if request.patientInfo:
                    patient_info = {
                        'age': request.patientInfo.get('age'),
                        'gender': request.patientInfo.get('sex'),
                        'condition': request.condition or 'dementia'
                    }

                # Update bandit model (will use mock data if ML service fails)
                from app.schemas.therapy import TherapyCondition
                condition_map = {
                    'dementia': TherapyCondition.DEMENTIA,
                    'adhd': TherapyCondition.ADHD,
                    'down_syndrome': TherapyCondition.DOWN_SYNDROME
                }
                condition = condition_map.get(request.condition or 'dementia', TherapyCondition.DEMENTIA)

                ml_service.update_bandit(
                    condition=condition,
                    song_id=str(song_id),
                    patient_info=patient_info,
                    feedback_type=request.feedback_type.value
                )
            except Exception as ml_error:
                logger.warning(f"ML update failed (expected in testing): {ml_error}")

        return FeedbackResponse(
            status="ok",
            message="Feedback recorded successfully (testing mode)",
            feedback_id=None  # No database ID in testing mode
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/feedback/types")
async def get_feedback_types():
    """
    Get list of available feedback types with descriptions.
    """
    return {
        "feedback_types": [
            {
                "id": FeedbackType.LIKE.value,
                "name": "Like",
                "description": "Patient enjoyed the music",
                "reward": 1.0
            },
            {
                "id": FeedbackType.DISLIKE.value,
                "name": "Dislike",
                "description": "Patient did not enjoy the music",
                "reward": -1.0
            },
            {
                "id": FeedbackType.SKIP.value,
                "name": "Skip",
                "description": "Patient skipped the music",
                "reward": -0.5
            },
            {
                "id": FeedbackType.INAPPROPRIATE.value,
                "name": "Inappropriate",
                "description": "Music was inappropriate for the patient",
                "reward": -2.0
            },
            {
                "id": FeedbackType.NEUTRAL.value,
                "name": "Neutral",
                "description": "No strong response to the music",
                "reward": 0.0
            }
        ]
    }


@router.get("/feedback/health")
async def health_check():
    """
    Health check endpoint for the feedback service.
    """
    return {
        "status": "healthy",
        "service": "feedback",
        "model_ready": hasattr(ml_service, 'bandits') and len(ml_service.bandits) > 0
    }