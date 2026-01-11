"""
TheraMuse Backend API - Main FastAPI Application

This is the main entry point for the TheraMuse API service.
Provides personalized music therapy recommendations using machine learning
bandit algorithms and Big Five personality profiling.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder
from sqlalchemy import text

# Application imports
from app.core.config import settings
from app.core.database import init_db, SessionLocal
from app.core.logging import configure_logging
from app.core.memory_manager_fallback import initialize_memory_monitoring, cleanup_memory_manager
from app.api.api import api_router
from app.utils.middleware import (
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    ErrorHandlingMiddleware
)

# Configure structured logging
configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """
    Application lifespan manager with memory-efficient resource handling.
    Handles startup and shutdown events.
    """
    # Startup phase
    logger.info(f"Starting TheraMuse API v{settings.VERSION} - {settings.ENVIRONMENT} (debug={settings.DEBUG})")

    # Track DB availability for health checks
    app.state.db_available = False

    try:
        # Initialize memory monitoring first
        logger.info("Initializing memory monitoring...")
        initialize_memory_monitoring()

        # Initialize database connections with memory-efficient configuration
        if settings.ENVIRONMENT.lower() != "testing":
            try:
                logger.info("Starting PostgreSQL database initialization...")
                await asyncio.to_thread(init_db)
                app.state.db_available = True
                logger.info("PostgreSQL database initialized successfully")
            except Exception as db_error:
                app.state.db_available = False
                # Only log database errors at debug level to hide startup noise
                if settings.DEBUG:
                    logger.debug("PostgreSQL initialization failed: %s", db_error)
        else:
            logger.info("Running in testing mode - skipping PostgreSQL database initialization")

        # Initialize shared services with lazy loading
        logger.info("Initializing shared services...")

        # Import here to avoid circular imports and enable lazy loading
        from app.services.memory_efficient_ml_service import get_ml_service, cleanup_ml_service
        from app.services.shared_music_catalog_service import get_shared_music_catalog, cleanup_shared_music_catalog

        # Initialize shared music catalog (lazy loading will happen on first use)
        music_catalog = get_shared_music_catalog()
        logger.info("Shared music catalog service initialized")

        # Initialize ML service (lazy loading will happen on first request)
        ml_service = get_ml_service()
        logger.info("ML service initialized")

        # Store services in app state for cleanup
        app.state.cleanup_functions = [
            cleanup_ml_service,
            cleanup_shared_music_catalog,
            cleanup_memory_manager
        ]

        # Log initial memory status
        from app.core.memory_manager_fallback import get_memory_manager
        memory_manager = get_memory_manager()
        memory_status = memory_manager.get_health_status()
        logger.info(f"Initial memory status: {memory_status['memory']['process']['rss_mb']:.1f}MB")

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise

    yield

    # Shutdown phase
    logger.info("Shutting down TheraMuse API")

    try:
        # Cleanup all registered services
        if hasattr(app.state, 'cleanup_functions'):
            for cleanup_func in app.state.cleanup_functions:
                try:
                    cleanup_func()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")

        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI application instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="TheraMuse API - Personalized music therapy recommendations using machine learning",
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
    debug=settings.DEBUG,
    contact={
        "name": "TheraMuse Team",
        "email": "support@theramuse.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Add trusted host middleware for security
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.theramuse.com"]
    )

# Add CORS middleware with configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With"
    ],
    expose_headers=["X-Total-Count", "X-Rate-Limit-Remaining"]
)

# Add custom middleware in the correct order
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
# TODO: Fix RateLimitMiddleware parameter issue
# app.add_middleware(RateLimitMiddleware, per_minute=settings.RATE_LIMIT_PER_MINUTE)
app.add_middleware(SecurityHeadersMiddleware)

# Include API routers
app.include_router(api_router, prefix=settings.API_V1_STR)


# Health and Status Endpoints
@app.get("/", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Any]:
    """
    Root endpoint with basic API information.
    """
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs_url": f"{settings.API_V1_STR}/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Basic health check endpoint.
    Returns service status and basic system information.
    """
    try:
        return {
            "status": "healthy",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )


@app.get("/version", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
async def version_info() -> Dict[str, str]:
    """
    Returns detailed version information.
    """
    return {
        "api_version": settings.VERSION,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "fastapi_version": "0.104.1",
        "environment": settings.ENVIRONMENT,
        "build_date": "2025-11-14"
    }


@app.get("/memory", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def memory_status() -> Dict[str, Any]:
    """
    Get comprehensive memory usage and system status.
    """
    from datetime import datetime

    try:
        # Get memory manager status
        from app.core.memory_manager_fallback import get_memory_manager
        memory_manager = get_memory_manager()
        memory_status = memory_manager.get_health_status()

        # Get ML service status
        from app.services.memory_efficient_ml_service import get_ml_service
        ml_service = get_ml_service()
        ml_memory = ml_service.get_memory_status()

        # Get music catalog status
        from app.services.shared_music_catalog_service import get_shared_music_catalog
        catalog_service = get_shared_music_catalog()
        catalog_status = catalog_service.get_health_status()

        return {
            "timestamp": datetime.now().isoformat(),
            "memory_manager": memory_status,
            "ml_service": ml_memory,
            "music_catalog": catalog_status,
            "recommendations": {
                "status": "healthy" if memory_status["status"] == "healthy" else "warning",
                "actions": []
            }
        }

    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "error"
        }


@app.post("/memory/cleanup", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def trigger_memory_cleanup() -> Dict[str, Any]:
    """
    Trigger manual memory cleanup.
    """
    from datetime import datetime

    try:
        from app.core.memory_manager_fallback import get_memory_manager
        memory_manager = get_memory_manager()

        # Get memory before cleanup
        before_stats = memory_manager.get_memory_usage()
        before_mb = before_stats["process"]["rss_mb"]

        # Trigger cleanup
        memory_manager.force_cleanup("api_request")

        # Get memory after cleanup
        after_stats = memory_manager.get_memory_usage()
        after_mb = after_stats["process"]["rss_mb"]

        memory_saved = before_mb - after_mb

        return {
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "memory_saved_mb": max(0, memory_saved),
            "before_mb": before_mb,
            "after_mb": after_mb,
            "message": f"Cleanup completed. Memory saved: {max(0, memory_saved):.1f}MB"
        }

    except Exception as e:
        logger.error(f"Failed to trigger memory cleanup: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e),
            "message": "Cleanup failed"
        }


# Exception Handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler for unexpected errors.
    """
    logger.error(
        "Unexpected error occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        exc_info=True
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "error_type": type(exc).__name__,
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
        }
    )


@app.exception_handler(status.HTTP_404_NOT_FOUND)
async def http_not_found_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTP 404 Not Found errors.
    """
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Not Found",
            "message": f"Endpoint {request.method} {request.url.path} not found",
            "error_type": "HTTPError",
            "status_code": status.HTTP_404_NOT_FOUND,
            "available_endpoints": [
                "/",
                "/health",
                "/version",
                f"{settings.API_V1_STR}/recommendations",
                f"{settings.API_V1_STR}/feedback",
                f"{settings.API_V1_STR}/docs"
            ]
        }
    )


@app.exception_handler(status.HTTP_500_INTERNAL_SERVER_ERROR)
async def http_internal_server_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTP 500 Internal Server Error.
    """
    logger.error(
        "Internal server error occurred",
        error=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. The development team has been notified.",
            "error_type": "HTTPError",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
        }
    )


# Application startup configuration
if __name__ == "__main__":
    import uvicorn

    # Memory-efficient worker configuration
    # Reduced from 4 to 2 workers in production to prevent memory bloat
    # Each worker loads ML models and music catalog separately
    workers = 1 if settings.DEBUG else 2

    logger.info(f"Starting with {workers} workers (memory-efficient configuration)")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        use_colors=True,
        workers=workers,
        # Memory optimization settings
        limit_concurrency=20,  # Limit concurrent requests
        timeout_keep_alive=30,  # Shorter keep-alive timeout
        timeout_graceful_shutdown=10,  # Faster shutdown
    )
