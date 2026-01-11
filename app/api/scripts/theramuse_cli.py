#!/usr/bin/env python3

"""
CLI adapter that mirrors the behaviour of the legacy TheraMuse Python script
but routes every action through the modern FastAPI service layer. The Next.js
frontend streams JSON payloads via stdin/stdout, so this module keeps I/O
minimal and deterministic.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
APP_API_DIR = SCRIPT_DIR.parent
APP_DIR = APP_API_DIR.parent
REPO_ROOT = APP_DIR.parent
for candidate in (SCRIPT_DIR, APP_API_DIR, APP_DIR, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

# Ensure structured logging is configured so CLI runs keep stdout JSON-only.
import app.core.logging  # noqa: F401

from app.core.config import settings
from app.schemas.therapy import (
    TherapyCondition,
    RecommendationRequestLegacy,
    TherapyRecommendation,
)
from app.services.ml_service import MLService
from app.services.therapy_service import TherapyService
from app.services.export_service import ExportService
from app.services.legacy_theramuse_adapter import LegacyTheraMuseAdapter

ml_service = MLService()
therapy_service = TherapyService(ml_service, catalog_path=settings.MUSIC_CATALOG_PATH)
export_service = ExportService()
legacy_adapter = LegacyTheraMuseAdapter()


@dataclass
class Payload:
    action: str
    data: Dict[str, Any]
    db_path: Optional[str] = None
    model_path: Optional[str] = None

    @staticmethod
    def from_json(raw: str) -> "Payload":
        parsed = json.loads(raw)
        return Payload(
            action=parsed.get("action", ""),
            data=parsed.get("data", {}) or {},
            db_path=parsed.get("db_path"),
            model_path=parsed.get("model_path"),
        )


def handle_recommendations(payload: Payload) -> Dict[str, Any]:
    patient_info = dict(payload.data.get("patient_info") or {})
    condition = _resolve_condition(payload.data.get("condition"))
    patient_id = payload.data.get("patient_id") or f"patient_{int(datetime.utcnow().timestamp())}"
    patient_info = _augment_patient_profile(patient_info, patient_id)

    legacy_result = legacy_adapter.generate_recommendations(patient_info, condition.value, patient_id)
    if legacy_result:
        return {"recommendations": legacy_result}

    preferences = _derive_preferences(patient_info)
    request = RecommendationRequestLegacy(
        patient_info=patient_info,
        condition=condition,
        preferences=preferences if any(preferences.values()) else None,
        big_five_responses=None,
    )

    response = therapy_service.generate_recommendations(None, request)
    recommendation_payload = _build_recommendation_payload(response, patient_info, patient_id, condition)
    return {"recommendations": recommendation_payload}


def handle_feedback(payload: Payload) -> Dict[str, Any]:
    session_id = payload.data.get("session_id")
    patient_id = payload.data.get("patient_id")
    condition = _resolve_condition(payload.data.get("condition"))
    feedback_type = payload.data.get("feedback_type")
    song = dict(payload.data.get("song") or {})
    patient_info = dict(payload.data.get("patient_info") or {})
    patient_info = _augment_patient_profile(patient_info, patient_id)

    if not all([session_id, patient_id, feedback_type]):
        raise ValueError("Feedback request missing required fields")

    song_id = song.get("id") or song.get("song_id") or song.get("video_id") or song.get("title") or "unknown"
    ml_context = {
        "age": patient_info.get("age"),
        "gender": patient_info.get("sex") or patient_info.get("gender"),
        "condition": condition.value,
    }

    legacy_adapter.record_feedback(
        patient_id=str(patient_id),
        session_id=str(session_id),
        condition=condition.value,
        song=song,
        feedback_type=str(feedback_type),
        patient_info=patient_info,
    )

    ml_service.update_bandit(
        condition=condition,
        song_id=str(song_id),
        patient_info=ml_context,
        feedback_type=str(feedback_type),
    )

    return {"status": "ok"}


def handle_export(payload: Payload) -> Dict[str, Any]:
    export_format = (payload.data.get("format") or "").lower()
    if export_format not in {"pdf", "docx", "csv", "json"}:
        raise ValueError("Unsupported export format")

    patient_info = dict(payload.data.get("patient_info") or {})
    recommendations_payload = dict(payload.data.get("recommendations") or {})
    big_five = payload.data.get("big5_scores") or payload.data.get("big_five")
    patient_summary = payload.data.get("patient_summary") or {}

    therapy_recommendations = _convert_to_therapy_recommendations(recommendations_payload)
    export_result = export_service.export_data(
        export_format=export_format,
        patient_info=patient_info,
        recommendations=therapy_recommendations,
        big_five=big_five,
        patient_summary=patient_summary,
    )

    return {
        "content": export_result["base64_data"],
        "filename": export_result["filename"],
        "mimeType": export_result["content_type"],
        "fileSize": export_result["file_size"],
    }


def handle_analytics(_: Payload) -> Dict[str, Any]:
    legacy_stats = legacy_adapter.get_analytics()
    if legacy_stats:
        return {"analytics": legacy_stats}

    bandit_stats = _collect_bandit_metrics()
    totals = {
        "totalSessions": 0,
        "totalFeedback": 0,
        "totalPatients": 0,
    }
    if bandit_stats:
        totals["totalSessions"] = sum(item["count"] for item in bandit_stats)

    return {
        "analytics": {
            "totals": totals,
            "rewardsByCondition": [
                {
                    "condition": stats["condition"],
                    "averageReward": stats["average_reward"],
                    "count": stats["count"],
                }
                for stats in bandit_stats
            ],
            "sessionsOverTime": [],
        }
    }


def _build_recommendation_payload(response, patient_info: Dict[str, Any], patient_id: str,
                                   condition: TherapyCondition) -> Dict[str, Any]:
    data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    songs = [_normalize_song_entry(item) for item in data.get("recommendations", [])]
    algorithm_metadata = data.get("algorithm_metadata") or {}

    payload: Dict[str, Any] = {
        "session_id": data.get("session_id"),
        "patient_id": patient_id,
        "condition": condition.value,
        "total_songs": len(songs),
        "categories": _build_cli_categories(songs, patient_info),
        "patient_context": patient_info,
        "method": algorithm_metadata.get("algorithm", "theramuse_mock"),
        "bandit_stats": {
            "n_interactions": algorithm_metadata.get("songs_considered", len(songs)),
            "avg_reward": algorithm_metadata.get("avg_reward", 0.0),
            "exploration_rate": algorithm_metadata.get("exploration_rate", 0.0),
        },
    }
    if "generated_at" in data:
        payload["generated_at"] = data["generated_at"]
    return payload


def _build_cli_categories(songs: List[Dict[str, Any]], patient_info: Dict[str, Any]) -> Dict[str, Any]:
    if not songs:
        return {}

    favorite_genre = patient_info.get("favorite_genre") or ", ".join(patient_info.get("favorite_genres", []))
    return {
        "recommended": {
            "label": "Recommended Songs",
            "query": favorite_genre,
            "songs": songs,
        }
    }


def _normalize_song_entry(raw_song: Any) -> Dict[str, Any]:
    song = _as_dict(raw_song)
    title = song.get("song_title") or song.get("title") or "Untitled"
    description_parts = []
    if song.get("genre"):
        description_parts.append(f"Genre: {song['genre']}")
    if song.get("tempo"):
        description_parts.append(f"Tempo: {song['tempo']} BPM")
    if song.get("valence") is not None:
        description_parts.append(f"Valence: {song['valence']:.2f}")
    if song.get("arousal") is not None:
        description_parts.append(f"Arousal: {song['arousal']:.2f}")

    return {
        "id": song.get("song_id") or song.get("id"),
        "title": title,
        "artist": song.get("artist"),
        "genre": song.get("genre"),
        "tempo": song.get("tempo"),
        "valence": song.get("valence"),
        "arousal": song.get("arousal"),
        "score": song.get("recommendation_score"),
        "rank": song.get("rank"),
        "description": " | ".join(description_parts) if description_parts else None,
    }


def _derive_preferences(patient_info: Dict[str, Any]) -> Dict[str, List[str]]:
    genres: List[str] = []
    favorite_genre = patient_info.get("favorite_genre")
    if isinstance(favorite_genre, str):
        genres.extend([value.strip() for value in favorite_genre.split(",") if value.strip()])
    favorite_genres = patient_info.get("favorite_genres")
    if isinstance(favorite_genres, list):
        genres.extend([value.strip() for value in favorite_genres if isinstance(value, str)])

    artists = []
    favorite_musician = patient_info.get("favorite_musician")
    if isinstance(favorite_musician, str) and favorite_musician.strip():
        artists.append(favorite_musician.strip())

    preferred_languages = patient_info.get("preferred_languages")
    if not isinstance(preferred_languages, list):
        preferred_languages = []

    instruments = patient_info.get("instruments")
    if not isinstance(instruments, list):
        instruments = []

    return {
        "genres": list(dict.fromkeys(genres)),
        "artists": artists,
        "languages": preferred_languages,
        "instruments": instruments,
    }


def _resolve_condition(raw: Optional[str]) -> TherapyCondition:
    if not raw:
        return TherapyCondition.DEMENTIA
    normalized = str(raw).lower()
    for condition in TherapyCondition:
        if condition.value == normalized:
            return condition
    return TherapyCondition.DEMENTIA


def _convert_to_therapy_recommendations(recommendations: Dict[str, Any]) -> List[TherapyRecommendation]:
    session_id = recommendations.get("session_id") or f"session_{int(datetime.utcnow().timestamp())}"
    created_at = _ensure_datetime(recommendations.get("generated_at"))
    flattened = _flatten_cli_recommendations(recommendations, session_id, created_at)

    converted: List[TherapyRecommendation] = []
    for idx, rec in enumerate(flattened, start=1):
        raw_song_id = rec.get("song_id") or rec.get("id")
        try:
            song_id = int(raw_song_id) if raw_song_id is not None else None
        except (TypeError, ValueError):
            song_id = None

        payload = {
            "id": rec.get("id") or idx,
            "session_id": rec.get("session_id", session_id),
            "song_id": song_id,
            "song_title": rec.get("song_title") or rec.get("title") or f"Song {idx}",
            "artist": rec.get("artist"),
            "genre": rec.get("genre"),
            "year": rec.get("year"),
            "tempo": rec.get("tempo"),
            "valence": rec.get("valence"),
            "arousal": rec.get("arousal"),
            "recommendation_score": rec.get("score") or rec.get("recommendation_score"),
            "rank": rec.get("rank") or idx,
            "context_features": rec.get("context_features"),
            "algorithm_used": rec.get("algorithm_used"),
            "created_at": _ensure_datetime(rec.get("created_at")) or created_at,
        }
        converted.append(TherapyRecommendation(**payload))
    return converted


def _flatten_cli_recommendations(recommendations: Dict[str, Any], session_id: str,
                                 fallback_created_at: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    categories = recommendations.get("categories")
    if not isinstance(categories, dict):
        return rows

    for key, category in categories.items():
        category_data = _as_dict(category)
        songs = category_data.get("songs")
        if not isinstance(songs, list):
            continue
        for rank, song in enumerate(songs, start=1):
            song_data = _as_dict(song)
            rows.append(
                {
                    "session_id": session_id,
                    "song_id": song_data.get("id"),
                    "song_title": song_data.get("title"),
                    "artist": song_data.get("artist"),
                    "genre": song_data.get("genre"),
                    "year": song_data.get("year"),
                    "tempo": song_data.get("tempo"),
                    "valence": song_data.get("valence"),
                    "arousal": song_data.get("arousal"),
                    "score": song_data.get("score"),
                    "rank": rank,
                    "algorithm_used": category_data.get("label") or key,
                    "created_at": song_data.get("created_at") or fallback_created_at.isoformat(),
                }
            )
    return rows


def _ensure_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()


def _collect_bandit_metrics() -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for condition, bandit in ml_service.bandits.items():
        pulls = sum(bandit.pulls.values()) if hasattr(bandit, "pulls") else 0
        total_reward = sum(bandit.rewards.values()) if hasattr(bandit, "rewards") else 0.0
        average_reward = round(total_reward / pulls, 3) if pulls else 0.0
        label = condition.value if isinstance(condition, TherapyCondition) else str(condition)
        stats.append(
            {
                "condition": label,
                "average_reward": average_reward,
                "count": pulls,
            }
        )
    return stats


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _augment_patient_profile(patient_info: Dict[str, Any], patient_id: Optional[str]) -> Dict[str, Any]:
    """
    Backfill duplicated field names and normalise formats expected by the legacy engine.
    """
    profile = dict(patient_info or {})
    if patient_id and not profile.get("patient_id"):
        profile["patient_id"] = patient_id

    favorite_genres = profile.get("favorite_genres")
    favorite_genre = profile.get("favorite_genre")
    if not favorite_genres and isinstance(favorite_genre, str):
        parsed = [value.strip() for value in favorite_genre.split(",") if value.strip()]
        if parsed:
            profile["favorite_genres"] = parsed

    instruments = profile.get("instruments")
    if isinstance(instruments, str):
        profile["instruments"] = [value.strip() for value in instruments.split(",") if value.strip()]

    for new_key, legacy_key in (
        ("preferredLanguages", "preferred_languages"),
        ("favoriteGenres", "favorite_genres"),
    ):
        if legacy_key in profile and new_key not in profile:
            profile[new_key] = profile[legacy_key]

    if "birthplace" not in profile:
        city = profile.get("birthplace_city")
        country = profile.get("birthplace_country")
        if city and country:
            profile["birthplace"] = f"{city}, {country}"
        elif city:
            profile["birthplace"] = city
        elif country:
            profile["birthplace"] = country

    return profile


def main() -> int:
    raw_input = sys.stdin.read()
    if not raw_input:
        print(json.dumps({"error": "Empty payload"}))
        return 1

    try:
        payload = Payload.from_json(raw_input)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON payload: {exc}"}))
        return 1

    try:
        if payload.action == "recommend":
            result = handle_recommendations(payload)
        elif payload.action == "feedback":
            result = handle_feedback(payload)
        elif payload.action == "export":
            result = handle_export(payload)
        elif payload.action == "analytics":
            result = handle_analytics(payload)
        else:
            raise ValueError(f"Unsupported action: {payload.action}")
    except Exception as exc:  # pragma: no cover - defensive
        print(json.dumps({"error": str(exc)}))
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
