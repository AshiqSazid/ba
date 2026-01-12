from typing import Optional, List
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
import os


class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "TheraMuse API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"

    # Database - PostgreSQL Only
    DATABASE_URL: Optional[str] = None
    # PostgreSQL Configuration
    DB_HOST: str = "moodsinger-mlm-postgressql.postgres.database.azure.com"
    DB_PORT: int = 5432
    DB_USER: str = "moodroot"
    DB_PASSWORD: str = "a6amvy76wM7mA-$"
    DB_NAME: str = "theramuse_backend"

  
    # ML Model
    MODEL_PATH: str = "theramuse_model.pkl"
    MUSIC_CATALOG_PATH: str = "./data/d.xlsx"

    # File Storage
    UPLOAD_DIR: str = "uploads"
    EXPORT_DIR: str = "exports"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "https://f-9gz1.vercel.app", 
        "https://ba-beryl.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://localhost:8000"
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        elif isinstance(v, str) and v.startswith("["):
            return v
        return v

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    @validator('DEBUG', pre=True)
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return v

    # External Services
    PYTHON_BINARY: str = "python"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    @validator("DATABASE_URL", pre=True)
    def assemble_database_url(cls, v, values):
        if v and isinstance(v, str) and v.strip():
            return v

        # PostgreSQL Configuration
        user = values.get("DB_USER", "moodroot")
        password = quote_plus(values.get("DB_PASSWORD", "a6amvy76wM7mA-$"))
        host = values.get("DB_HOST", "moodsinger-mlm-postgressql.postgres.database.azure.com")
        port = values.get("DB_PORT", 5432)
        db_name = values.get("DB_NAME", "theramuse_backend")

        # Use clean connection string with only essential parameters
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}?sslmode=require"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
