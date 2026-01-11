from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime, date


class PatientBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, max_length=50)
    condition: str = Field(..., max_length=100)
    birth_date: Optional[date] = None

    @validator('age')
    def validate_age(cls, v, values):
        if 'birth_date' in values and values['birth_date']:
            # Calculate age from birth date if provided
            today = date.today()
            birth_date = values['birth_date']
            calculated_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            if v and v != calculated_age:
                raise ValueError("Age does not match birth date")
        return v


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, max_length=50)
    condition: Optional[str] = Field(None, max_length=100)
    birth_date: Optional[date] = None


class Patient(PatientBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Big5ScoreBase(BaseModel):
    openness: Optional[float] = Field(None, ge=0, le=1)
    conscientiousness: Optional[float] = Field(None, ge=0, le=1)
    extraversion: Optional[float] = Field(None, ge=0, le=1)
    agreeableness: Optional[float] = Field(None, ge=0, le=1)
    neuroticism: Optional[float] = Field(None, ge=0, le=1)

    @validator('openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism')
    def validate_scores(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Personality scores must be between 0 and 1")
        return v


class Big5ScoreCreate(Big5ScoreBase):
    patient_id: int


class Big5Score(Big5ScoreBase):
    id: int
    patient_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class PatientWithScores(Patient):
    big5_scores: Optional[List[Big5Score]] = []