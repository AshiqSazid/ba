from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
import logging
import os

logger = logging.getLogger(__name__)

# Memory-efficient database configuration
def get_db_config():
    """Get database configuration optimized for memory efficiency."""

    # Adjust pool size based on environment and available workers
    workers = int(os.getenv("WORKERS", "1"))  # Default to 1 for safety

    # Conservative pool sizing to prevent memory bloat
    pool_size = max(2, min(5, workers))  # 2-5 connections
    max_overflow = max(5, min(10, workers * 2))  # 5-10 overflow

    # Environment-specific tuning
    if settings.DEBUG:
        pool_size = 2
        max_overflow = 5
    elif workers > 4:  # Production with many workers
        pool_size = 3
        max_overflow = 8

    return {
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_pre_ping": True,
        "pool_recycle": 300,  # 5 minutes
        "pool_timeout": 30,   # 30 seconds
        "echo": settings.DEBUG,
        # Memory-efficient settings
        "connect_args": {
            "application_name": f"theramuse_{settings.ENVIRONMENT}",
            "connect_timeout": 10,
            # Optimize for memory usage
            "options": "-c statement_timeout=30s -c idle_in_transaction_session_timeout=10s"
        }
    }

logger.info("Initializing PostgreSQL database connection with memory-efficient configuration...")
logger.info(f"PostgreSQL Host: {settings.DB_HOST}")
logger.info(f"PostgreSQL Database: {settings.DB_NAME}")
logger.info(f"PostgreSQL User: {settings.DB_USER}")

# Get optimized configuration
db_config = get_db_config()
logger.info(f"Database pool configuration: size={db_config['pool_size']}, overflow={db_config['max_overflow']}")

engine = create_engine(settings.DATABASE_URL, **db_config)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()


def get_db():
    """
    Dependency to get DB session.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    """
    try:
        logger.info("Initializing PostgreSQL database schema...")

        # Import ALL models in dependency order to ensure they're registered with SQLAlchemy
        from app.models.database import (
            Technique, Instrument, ClassTechniqueTag,  # No dependencies
            Patient,  # Base table for foreign keys
            Song, song_instrument_association,  # Song and association table
            Big5Score, TherapySession,  # Depend on Patient
            TherapyRecommendation,  # Depends on TherapySession and Song
            TherapyFeedback, BanditStats, UserActivity  # Final tables
        )

        # Drop existing tables that might have incompatible schemas
        # Only in development mode to avoid data loss in production
        if settings.DEBUG:
            logger.info("Development mode detected - cleaning up existing tables...")
            tables_to_drop = [
                UserActivity.__table__,
                song_instrument_association,
                TherapyFeedback.__table__,
                TherapyRecommendation.__table__,
                BanditStats.__table__,
                TherapySession.__table__,
                Big5Score.__table__,
                Song.__table__,
                Patient.__table__,
                ClassTechniqueTag.__table__,
                Instrument.__table__,
                Technique.__table__
            ]

            for table in tables_to_drop:
                try:
                    table.drop(bind=engine, checkfirst=True)
                    logger.info(f"Dropped table: {table.name}")
                except Exception as drop_error:
                    logger.debug(f"Could not drop table {table.name}: {drop_error}")

        # Create tables in explicit dependency order to avoid foreign key issues
        tables_to_create = [
            Technique.__table__,
            Instrument.__table__,
            ClassTechniqueTag.__table__,
            Patient.__table__,
            Song.__table__,
            song_instrument_association,
            Big5Score.__table__,
            TherapySession.__table__,
            TherapyRecommendation.__table__,
            TherapyFeedback.__table__,
            BanditStats.__table__,
            UserActivity.__table__
        ]

        # Create each table individually in the correct order
        for table in tables_to_create:
            try:
                table.create(bind=engine, checkfirst=True)
                logger.info(f"Created table: {table.name}")
            except Exception as table_error:
                logger.warning(f"Could not create table {table.name}: {table_error}")

        logger.info("PostgreSQL database tables created successfully")
    except Exception as e:
        # Suppress connection errors during startup to reduce noise
        if "invalid connection option" in str(e) or "dsn" in str(e).lower():
            # Don't print connection parameter errors to reduce startup noise
            logger.debug("PostgreSQL connection issue: %s", e)
        else:
            logger.error("PostgreSQL database initialization failed: %s", e)
        raise