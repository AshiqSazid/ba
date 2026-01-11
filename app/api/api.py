"""
Central API router configuration.

Base URL: http://localhost:8000/api

Available entry points:
- Generate recommendations → POST /recommendations
- Submit feedback → POST /feedback
- Export reports → POST /export
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fastapi import APIRouter

from app.api.endpoints import export, feedback, recommendations, music_feature_mapping


BASE_URL = "http://localhost:8000/api"


@dataclass(frozen=True)
class EndpointGroup:
    """Metadata describing an endpoint collection."""

    description: str
    method: str
    path: str
    router: APIRouter
    tag: str


ENDPOINT_GROUPS: Iterable[EndpointGroup] = (
    EndpointGroup(
        description="Generate recommendations",
        method="POST",
        path="/recommendations",
        router=recommendations.router,
        tag="recommendations",
    ),
    EndpointGroup(
        description="Submit feedback",
        method="POST",
        path="/feedback",
        router=feedback.router,
        tag="feedback",
    ),
    EndpointGroup(
        description="Export reports",
        method="POST",
        path="/export",
        router=export.router,
        tag="export",
    ),
    EndpointGroup(
        description="Map music features to cognitive indicators",
        method="POST",
        path="/music-features",
        router=music_feature_mapping.router,
        tag="music-feature-mapping",
    ),
)


def create_api_router() -> APIRouter:
    """
    Build the root API router that aggregates all endpoint groups.
    """
    router = APIRouter()
    for group in ENDPOINT_GROUPS:
        router.include_router(
            group.router,
            prefix="",
            tags=[group.tag],
        )
    return router


# Expose a ready-to-use router instance for FastAPI apps.
api_router = create_api_router()
