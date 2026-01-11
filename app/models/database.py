from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, Table, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


# Association tables for many-to-many relationships
song_instrument_association = Table(
    'song_instruments',
    Base.metadata,
    Column('song_id', Integer, ForeignKey('songs.id'), primary_key=True),
    Column('instrument_id', Integer, ForeignKey('instruments.id'), primary_key=True)
)

class Technique(Base):
    __tablename__ = "techniques"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    category = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    category = Column(String(100))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ClassTechniqueTag(Base):
    __tablename__ = "class_technique_tags"

    id = Column(Integer, primary_key=True, index=True)
    tag_name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    age = Column(Integer)
    gender = Column(String(50))
    condition = Column(String(100))
    birth_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    therapy_sessions = relationship("TherapySession", back_populates="patient")
    big5_scores = relationship("Big5Score", back_populates="patient")


class Big5Score(Base):
    __tablename__ = "big5_scores"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    openness = Column(Float)
    conscientiousness = Column(Float)
    extraversion = Column(Float)
    agreeableness = Column(Float)
    neuroticism = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="big5_scores")


class TherapySession(Base):
    __tablename__ = "therapy_sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    condition = Column(String(100), nullable=False)
    therapy_type = Column(String(100))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String(50), default="active")
    session_metadata = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="therapy_sessions")
    recommendations = relationship("TherapyRecommendation", back_populates="session")


class TherapyRecommendation(Base):
    __tablename__ = "therapy_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), ForeignKey("therapy_sessions.session_id"), nullable=False)
    song_id = Column(Integer, ForeignKey("songs.id"))
    song_title = Column(String(255))
    artist = Column(String(255))
    genre = Column(String(100))
    year = Column(Integer)
    tempo = Column(Float)
    valence = Column(Float)
    arousal = Column(Float)
    recommendation_score = Column(Float)
    context_features = Column(Text)  # JSON string
    algorithm_used = Column(String(100))
    rank = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("TherapySession", back_populates="recommendations")
    song = relationship("Song")
    feedback = relationship("TherapyFeedback", back_populates="recommendation")


class TherapyFeedback(Base):
    __tablename__ = "therapy_feedback"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), ForeignKey("therapy_sessions.session_id"), nullable=False)
    recommendation_id = Column(Integer, ForeignKey("therapy_recommendations.id"), nullable=False)
    feedback_type = Column(String(50), nullable=False)  # like, dislike, skip, inappropriate
    feedback_score = Column(Float)  # Reward value for ML
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_comments = Column(Text)
    context = Column(Text)  # Additional context data

    # Relationships
    recommendation = relationship("TherapyRecommendation", back_populates="feedback")


class Song(Base):
    __tablename__ = "songs"

    id = Column(Integer, primary_key=True, index=True)

    # Basic song information
    song_name = Column(String(255), nullable=False, index=True)
    singer = Column(String(255), index=True)
    language = Column(String(100))
    released_date = Column(DateTime)
    genre = Column(String(100), index=True)
    mood = Column(String(100))
    duration = Column(Integer)  # seconds

    # Audio features
    danceability = Column(Float)
    acousticness = Column(Float)
    energy = Column(Float)
    liveness = Column(Float)
    loudness = Column(Float)
    speechiness = Column(Float)
    tempo = Column(Float)
    mode = Column(Integer)  # 0=minor, 1=major
    key = Column(Integer)
    valence = Column(Float)
    time_signature = Column(Integer)

    # Lyric analysis scores (0-1 scale)
    lyrics_reappraisal = Column(Float, comment="Lyric's Reappraisal")
    lyrics_distracting = Column(Float, comment="Lyric's Distracting")
    lyrics_uplifting = Column(Float, comment="Lyric's Uplifting")
    lyrics_relaxing = Column(Float, comment="Lyric's Relaxing")
    lyrics_suppressing = Column(Float, comment="Lyric's Suppressing")
    lyrics_motivational = Column(Float, comment="Lyric's Motivational")

    # Audio analysis scores (0-1 scale)
    audio_reappraisal = Column(Float, comment="Audio's Reappraisal")
    audio_distracting = Column(Float, comment="Audio's Distracting")
    audio_uplifting = Column(Float, comment="Audio's Uplifting")
    audio_relaxing = Column(Float, comment="Audio's Relaxing")
    audio_suppressing = Column(Float, comment="Audio's Suppressing")
    audio_motivational = Column(Float, comment="Audio's Motivational")

    # Personality traits (Big Five scores)
    openness = Column(Float)
    conscientiousness = Column(Float)
    extraversion = Column(Float)
    agreeableness = Column(Float)
    neuroticism = Column(Float)

    # Additional metadata
    technique_description = Column(Text)
    class_technique_tags = Column(JSON)  # Array of technique tags
    used_instruments = Column(JSON)  # Array of instrument names
    file_path = Column(String(500))
    spotify_id = Column(String(100))
    youtube_id = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    instruments = relationship("Instrument", secondary=song_instrument_association, backref="songs")

    # For backward compatibility with existing code
    @property
    def title(self):
        return self.song_name

    @property
    def artist(self):
        return self.singer

    @title.setter
    def title(self, value):
        self.song_name = value

    @artist.setter
    def artist(self, value):
        self.singer = value


class BanditStats(Base):
    __tablename__ = "bandit_stats"

    id = Column(Integer, primary_key=True, index=True)
    condition = Column(String(100), nullable=False)
    song_id = Column(Integer, ForeignKey("songs.id"), nullable=False)
    arm_id = Column(String(255), nullable=False)  # Bandit arm identifier
    pulls = Column(Integer, default=0)
    rewards = Column(Float, default=0.0)
    avg_reward = Column(Float, default=0.0)
    confidence = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    model_parameters = Column(Text)  # JSON string

    # Relationships
    song = relationship("Song")


class UserActivity(Base):
    __tablename__ = "user_activity"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    action_type = Column(String(100), nullable=False)  # recommend, feedback, export, etc.
    endpoint = Column(String(255))
    request_data = Column(Text)
    response_data = Column(Text)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient")