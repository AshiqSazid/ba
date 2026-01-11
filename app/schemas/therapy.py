from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class TherapyCondition(str, Enum):
    DEMENTIA = "dementia"
    ADHD = "adhd"
    DOWN_SYNDROME = "down_syndrome"


class FeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    SKIP = "skip"
    INAPPROPRIATE = "inappropriate"
    NEUTRAL = "neutral"


class TherapySessionBase(BaseModel):
    condition: TherapyCondition
    therapy_type: Optional[str] = Field(None, max_length=100)


class TherapySessionCreate(TherapySessionBase):
    patient_id: int
    session_id: Optional[str] = None


class TherapySession(TherapySessionBase):
    id: int
    patient_id: int
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    metadata: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class MusicFeatureRequest(BaseModel):
    """
    Request model for music feature to cognitive indicators mapping.

    Accepts 12 music features on 0.0-1.0 scale:
    - Lyric's Reappraisal, Distracting, Uplifting, Relaxing, Suppressing, Motivational
    - audio's Reappraisal, Distracting, Uplifting, Relaxing, Suppressing, Motivational
    """
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of 12 music feature scores (0.0-1.0 scale)"
    )
    sensitivity: Optional[float] = Field(
        4.0,
        ge=0.1,
        le=10.0,
        description="Sensitivity multiplier for the mapping algorithm (default: 4.0)"
    )
    total_suggestions: Optional[int] = Field(
        20,
        ge=1,
        le=100,
        description="Number of song suggestions to generate (default: 20)"
    )
    preferred_languages: Optional[List[str]] = Field(
        None,
        description="Optional list of preferred languages (e.g., ['English'], ['Bengali'])"
    )

    @validator('features')
    def validate_features(cls, v):
        """Validate that all required features are present and in valid range."""
        required_features = [
            "Lyric's Reappraisal", "Lyric's Distracting", "Lyric's Uplifting", "Lyric's Relaxing",
            "Lyric's Suppressing", "Lyric's Motivational", "audio's Reappraisal", "audio's Distracting",
            "audio's Uplifting", "audio's Relaxing", "audio's Suppressing", "audio's Motivational"
        ]

        # Check for missing features
        missing_features = [f for f in required_features if f not in v]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Check value ranges
        invalid_features = [(f, val) for f, val in v.items() if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0]
        if invalid_features:
            invalid_list = [f"{f}={val}" for f, val in invalid_features]
            raise ValueError(f"Features must be numbers between 0.0 and 1.0. Invalid: {invalid_list}")

        return v


class CognitiveIndicator(BaseModel):
    """Individual cognitive indicator with interpretation."""
    score: float = Field(..., ge=0.0, le=1.0, description="Cognitive indicator score")
    level: str = Field(..., description="Interpretation level (Strong/Moderate/Low indication)")
    interpretation: str = Field(..., description="Human-readable interpretation")


class CognitiveProfile(BaseModel):
    """Cognitive profile summary."""
    overall_cognitive_risk_score: float = Field(..., ge=0.0, le=1.0)
    overall_risk_level: str = Field(..., description="Overall risk level interpretation")
    high_risk_areas: List[str] = Field(default_factory=list, description="Indicators with strong indication")
    low_risk_areas: List[str] = Field(default_factory=list, description="Indicators with low indication")
    primary_concerns: List[Dict[str, Any]] = Field(..., description="Top 3 highest-scoring indicators")
    total_indicators_assessed: int = Field(..., description="Number of indicators evaluated")
    assessment_timestamp: str = Field(..., description="When the assessment was performed")


class TherapeuticApproach(BaseModel):
    """Therapeutic approach based on cognitive profile."""
    memory_support_level: str = Field(..., description="Level of memory support needed")
    sleep_support_level: str = Field(..., description="Level of sleep support needed")
    overall_strategy: str = Field(..., description="Overall therapeutic strategy")


class SongSuggestion(BaseModel):
    """Individual song suggestion with cognitive therapeutic properties."""
    id: int
    song_id: Optional[int]
    song_title: str
    artist: str
    genre: Optional[str]
    language: Optional[str]
    mood: Optional[str]
    released_date: Optional[str]
    tempo: Optional[float]
    valence: Optional[float]
    energy: Optional[float]
    therapeutic_score: float = Field(..., ge=0.0, le=1.0, description="Therapeutic suitability score")
    cognitive_target: str = Field(..., description="Primary cognitive target for this song")
    therapeutic_reasons: List[str] = Field(..., description="Why this song was recommended")
    recommendation_rank: int
    algorithm_used: str
    category: str = Field(default="cognitive_indicators")
    youtube_url: Optional[str] = None
    spotify_url: Optional[str] = None


class SongRecommendationMetadata(BaseModel):
    """Metadata for song recommendations."""
    total_recommendations: int
    algorithm: str
    songs_analyzed: int
    music_database: str
    cognitive_indicators_considered: List[str]
    therapeutic_approach: str
    timestamp: str


class SongRecommendations(BaseModel):
    """Complete song recommendation package."""
    recommendations: List[SongSuggestion]
    cognitive_profile: CognitiveProfile
    therapeutic_approach: TherapeuticApproach
    recommendation_metadata: SongRecommendationMetadata


class MusicFeatureMappingResponse(BaseModel):
    """Complete response for music feature mapping request."""
    input_features: Dict[str, float]
    cognitive_indicators: Dict[str, CognitiveIndicator]
    song_recommendations: SongRecommendations
    processing_metadata: Dict[str, Any]


class SongBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    artist: Optional[str] = Field(None, max_length=255)
    album: Optional[str] = Field(None, max_length=255)
    genre: Optional[str] = Field(None, max_length=100)
    year: Optional[int] = Field(None, ge=1500, le=2100)


class Song(SongBase):
    id: int
    duration: Optional[int]
    tempo: Optional[float]
    valence: Optional[float]
    arousal: Optional[float]
    energy: Optional[float]
    danceability: Optional[float]
    acousticness: Optional[float]
    instrumentalness: Optional[float]
    language: Optional[str]
    region: Optional[str]
    cultural_context: Optional[str]
    therapeutic_tags: Optional[str]
    file_path: Optional[str]
    spotify_id: Optional[str]
    youtube_id: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TherapyRecommendationBase(BaseModel):
    song_id: Optional[int] = None
    song_title: str = Field(..., min_length=1, max_length=255)
    artist: Optional[str] = Field(None, max_length=255)
    genre: Optional[str] = Field(None, max_length=100)
    year: Optional[int] = Field(None, ge=1500, le=2100)
    recommendation_score: Optional[float] = Field(None, ge=0, le=1)
    rank: Optional[int] = Field(None, ge=1)


class TherapyRecommendationCreate(TherapyRecommendationBase):
    session_id: str


class TherapyRecommendation(TherapyRecommendationBase):
    id: int
    session_id: str
    tempo: Optional[float]
    valence: Optional[float]
    arousal: Optional[float]
    context_features: Optional[str]
    algorithm_used: Optional[str]
    youtube_url: Optional[str] = Field(None, max_length=500)
    spotify_url: Optional[str] = Field(None, max_length=500)
    created_at: datetime

    class Config:
        from_attributes = True


class TherapyFeedbackBase(BaseModel):
    feedback_type: FeedbackType
    user_comments: Optional[str] = Field(None, max_length=1000)
    context: Optional[Dict[str, Any]] = None


class TherapyFeedbackCreate(TherapyFeedbackBase):
    session_id: str
    recommendation_id: int


class TherapyFeedback(TherapyFeedbackBase):
    id: int
    session_id: str
    recommendation_id: int
    feedback_score: Optional[float]
    timestamp: datetime

    class Config:
        from_attributes = True


class RecommendationRequest(BaseModel):
    intake: Dict[str, Any] = Field(..., description="Patient intake form data")

    @validator('intake')
    def validate_intake(cls, v):
        required_fields = ['name', 'dateOfBirth', 'sex', 'condition']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v


# Legacy schema for backward compatibility
class RecommendationRequestLegacy(BaseModel):
    patient_info: Dict[str, Any] = Field(..., description="Patient demographic and clinical information")
    condition: TherapyCondition = Field(..., description="Therapy condition")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Music preferences and constraints")
    big_five_responses: Optional[List[int]] = Field(None, description="Big Five personality questionnaire responses")

    @validator('big_five_responses')
    def validate_big_five_responses(cls, v):
        if v is not None:
            # Accept both 50-item BFI-2 format and simplified 10-item format
            if len(v) != 50 and len(v) != 10:
                raise ValueError("Big Five responses must contain either 50 items (BFI-2) or 10 items (simplified format)")
            # For 50-item format, expect 1-5 scale; for 10-item format, expect 1-7 scale
            max_scale = 5 if len(v) == 50 else 7
            for item in v:
                if not isinstance(item, int) or item < 1 or item > max_scale:
                    raise ValueError(f"Each Big Five response must be an integer between 1 and {max_scale}")
        return v


class RecommendationResponse(BaseModel):
    session_id: str
    recommendations: List[TherapyRecommendation]
    patient_summary: Dict[str, Any]
    big_five_scores: Optional[Dict[str, float]]
    algorithm_metadata: Dict[str, Any]


class ExportRequest(BaseModel):
    format: str = Field(..., pattern="^(pdf|docx|csv|json)$")
    patient_info: Dict[str, Any]
    recommendations: List[TherapyRecommendation]
    big_five: Optional[Dict[str, float]]
    patient_summary: Dict[str, Any]


class ExportResponse(BaseModel):
    filename: str
    content_type: str
    file_size: int
    download_url: Optional[str] = None
    base64_data: Optional[str] = None


# API Request/Response Models matching the specified format

class BigFiveScores(BaseModel):
    openness: float = Field(..., ge=0, le=1)
    conscientiousness: float = Field(..., ge=0, le=1)
    extraversion: float = Field(..., ge=0, le=1)
    agreeableness: float = Field(..., ge=0, le=1)
    neuroticism: float = Field(..., ge=0, le=1)


class BigFiveResponses(BaseModel):
    openness: List[int] = Field(default_factory=list)
    conscientiousness: List[int] = Field(default_factory=list)
    extraversion: List[int] = Field(default_factory=list)
    agreeableness: List[int] = Field(default_factory=list)
    neuroticism: List[int] = Field(default_factory=list)


class IntakeData(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dateOfBirth: str = Field(..., description="Date in YYYY-MM-DD format")
    sex: str = Field(..., pattern="^(male|female|other|Male|Female|Other)$")
    condition: TherapyCondition = Field(...)
    birthplaceCity: Optional[str] = Field(None, max_length=255)
    birthplaceCountry: Optional[str] = Field(None, max_length=255)
    preferredLanguages: List[str] = Field(default_factory=list)
    favoriteGenres: List[str] = Field(default_factory=list)
    instruments: List[str] = Field(default_factory=list)
    favoriteMusician: Optional[str] = Field(None, max_length=255)
    favoriteSeason: Optional[str] = Field(None, max_length=100)
    naturalElements: List[str] = Field(default_factory=list)
    difficultySleeping: bool = False
    troubleRemembering: bool = False
    forgetsEverydayThings: bool = False
    difficultyRecallingOldMemories: bool = False
    memoryWorseThanYearAgo: bool = False
    visitedMentalHealthProfessional: bool = False
    bigFiveResponses: Optional[Union[BigFiveResponses, List[int], Dict[str, List[int]]]] = None
    big_five: Optional[BigFiveScores] = None


class RecommendationSong(BaseModel):
    id: Optional[int] = None
    song_id: Optional[int] = None
    song_title: str
    artist: Optional[str] = None
    genre: Optional[str] = None
    language: Optional[str] = None  # Add language field for proper filtering
    year: Optional[int] = None
    tempo: Optional[float] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    recommendation_score: float = Field(..., ge=0, le=1)
    rank: int = Field(..., ge=1)
    algorithm_used: Optional[str] = None
    created_at: Optional[datetime] = None
    category: Optional[str] = None  # Add category field for proper categorization
    youtube_url: Optional[str] = Field(None, max_length=500)
    spotify_url: Optional[str] = Field(None, max_length=500)


class PatientSummary(BaseModel):
    name: str
    age: Optional[int] = None
    condition: str
    session_id: str
    recommendation_count: int
    birth_date: Optional[str] = None
    birthplace_city: Optional[str] = None
    birthplace_country: Optional[str] = None
    sex: Optional[str] = None


class AlgorithmMetadata(BaseModel):
    algorithm: str
    songs_considered: int
    features_used: str
    condition: str
    timestamp: Optional[str] = None


class RecommendationResponse(BaseModel):
    session_id: str
    recommendations: List[RecommendationSong]
    patient_summary: PatientSummary
    big_five_scores: Optional[BigFiveScores]
    algorithm_metadata: AlgorithmMetadata


class CategorizedRecommendations(BaseModel):
    """Schema for categorized recommendations by user preferences"""
    country_songs_recommendations: List[RecommendationSong] = Field(default_factory=list)
    big_five_personality_recommendations: List[RecommendationSong] = Field(default_factory=list)
    favorite_genres_recommendations: List[RecommendationSong] = Field(default_factory=list)
    instruments_recommendations: List[RecommendationSong] = Field(default_factory=list)
    favorite_musician_recommendations: List[RecommendationSong] = Field(default_factory=list)
    favorite_season_recommendations: List[RecommendationSong] = Field(default_factory=list)
    natural_elements_recommendations: List[RecommendationSong] = Field(default_factory=list)
    cognitive_indicators_recommendations: List[RecommendationSong] = Field(default_factory=list)


class CategorizedRecommendationResponse(BaseModel):
    """Response schema with categorized recommendations"""
    session_id: str
    patient_summary: PatientSummary
    big_five_scores: Optional[BigFiveScores]
    algorithm_metadata: AlgorithmMetadata
    recommendations: CategorizedRecommendations


class FeedbackSong(BaseModel):
    id: Optional[int] = None
    title: str
    artist: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from recommendations response")
    recommendation_id: Optional[int] = Field(
        default=None,
        description="ID of the recommendation (if tracking specific recommendation)"
    )
    feedback_type: FeedbackType = Field(..., description="Feedback type provided by the user")
    song: FeedbackSong = Field(..., description="Song information")
    patientInfo: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Patient demographic info (optional for minimal payloads)"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Therapy condition (defaults to dementia when omitted)"
    )
    comments: Optional[str] = Field(None, max_length=1000, description="Optional feedback comments")


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: Optional[str] = None


class ExportRecommendation(BaseModel):
    id: Optional[int] = None
    session_id: str
    song_id: Optional[int] = None
    song_title: str
    artist: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    tempo: Optional[float] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    recommendation_score: Optional[float] = None
    rank: Optional[int] = None
    algorithm_used: Optional[str] = None
    context_features: Optional[str] = None
    youtube_url: Optional[str] = Field(None, max_length=500)
    spotify_url: Optional[str] = Field(None, max_length=500)
    created_at: Optional[datetime] = None


class ExportRequest(BaseModel):
    format: str = Field(..., pattern="^(pdf|docx|csv|json)$", description="Export format")
    patient_info: Dict[str, Any] = Field(..., description="Patient demographic information")
    recommendations: List[ExportRecommendation] = Field(..., description="Recommendation list")
    big_five: Optional[BigFiveScores] = Field(None, description="Big Five personality scores")
    patient_summary: Dict[str, Any] = Field(..., description="Patient session summary")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: Optional[str] = None
    environment: Optional[str] = None


class RecommendationsHealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    model_config = ConfigDict(protected_namespaces=())


class ConditionInfo(BaseModel):
    id: str
    name: str
    description: str


class ConditionsResponse(BaseModel):
    conditions: List[ConditionInfo]


class RootResponse(BaseModel):
    message: str
    version: str
    docs_url: Optional[str] = None
    health_url: Optional[str] = None
