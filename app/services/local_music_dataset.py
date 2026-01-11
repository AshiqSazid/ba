"""
Local Music Dataset service for loading and searching music from CSV files.
"""

from __future__ import annotations

import csv
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class LocalMusicDataset:
    """
    Offline music catalogue that loads rows from a curated CSV file and enables
    fuzzy matching against song titles, singers, languages, and genres.
    """

    AUDIO_FEATURE_FIELDS = {
        "danceability": "danceability",
        "acousticness": "acousticness",
        "energy": "energy",
        "liveness": "liveness",
        "loudness": "loudness",
        "speechiness": "speechiness",
        "tempo": "tempo",
    }

    PERSONALITY_FIELDS: Dict[str, Tuple[str, ...]] = {
        "openness": ("openness", "Openness"),
        "conscientiousness": ("conscientiousness", "Conscientiousness"),
        "extraversion": ("extraversion", "Extraversion"),
        "agreeableness": ("agreeableness", "Agreeableness"),
        "neuroticism": ("neuroticism", "Neuroticism"),
    }

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = Path(
            csv_path
            or os.getenv(
                "LOCAL_MUSIC_CSV_PATH",
                "./data/d.xlsx",
            )
        ).expanduser()
        self._catalog: List[Dict[str, Any]] = []
        self._load_catalog()

    def _parse_release_date(self, raw_value: str) -> Optional[str]:
        raw_value = (raw_value or "").strip()
        if not raw_value:
            return None

        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%Y",
        ]
        for fmt in formats:
            try:
                parsed = datetime.strptime(raw_value, fmt)
                if fmt == "%Y":
                    return f"{parsed.year}-01-01"
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        digits_only = "".join(ch for ch in raw_value if ch.isdigit())
        if len(digits_only) == 4:
            return f"{digits_only}-01-01"
        return None

    def _parse_duration_ms(self, raw_value: str) -> Optional[int]:
        raw_value = (raw_value or "").strip()
        if not raw_value:
            return None
        if raw_value.isdigit():
            seconds = int(raw_value)
            return seconds * 1000

        parts = raw_value.split(":")
        if all(part.isdigit() for part in parts):
            parts = [int(part) for part in parts]
            if len(parts) == 3:
                hours, minutes, seconds = parts
                total_seconds = hours * 3600 + minutes * 60 + seconds
                return total_seconds * 1000
            if len(parts) == 2:
                minutes, seconds = parts
                total_seconds = minutes * 60 + seconds
                return total_seconds * 1000
        return None

    def _parse_float_value(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        text = text.replace("%", "")
        try:
            return float(text)
        except ValueError:
            return None

    def _extract_audio_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        for feature_key, column in self.AUDIO_FEATURE_FIELDS.items():
            raw_value = row.get(column)
            numeric = self._parse_float_value(raw_value)
            if numeric is None:
                continue
            if feature_key in {"danceability", "acousticness", "energy", "liveness", "speechiness"}:
                numeric = max(0.0, min(1.0, numeric))
            features[feature_key] = numeric
        return features

    def _extract_personality_vector(self, row: Dict[str, Any]) -> Optional[Dict[str, float]]:
        vector: Dict[str, float] = {}
        for trait, columns in self.PERSONALITY_FIELDS.items():
            value: Optional[float] = None
            for column in columns:
                if column in row and row[column] not in (None, ""):
                    numeric = self._parse_float_value(row[column])
                    if numeric is not None:
                        value = numeric
                        break
            if value is not None:
                vector[trait] = value
        return vector or None

    def _build_track_record(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        normalized_row = {
            key: (value.strip() if isinstance(value, str) else value)
            for key, value in row.items()
        }

        song_name = (normalized_row.get("song_name") or "").strip()
        singer = (normalized_row.get("singer") or "").strip() or "Unknown Artist"
        if not song_name:
            return None

        release_value = self._parse_release_date(normalized_row.get("released_date") or "")
        duration_ms = self._parse_duration_ms(normalized_row.get("duration") or "")
        genre = (normalized_row.get("genre") or "").strip()
        language = (normalized_row.get("language") or "").strip()

        track_id = f"local::{hash((song_name.lower(), singer.lower(), release_value or ''))}"

        audio_features = self._extract_audio_features(normalized_row)
        personality_vector = self._extract_personality_vector(normalized_row)

        track = {
            "id": track_id,
            "title": song_name,
            "name": song_name,
            "url": f"https://theramuse.local/{track_id}",
            "artists": [singer],
            "artist": singer,
            "album": genre or "Local Catalogue",
            "duration_ms": duration_ms,
            "duration": duration_ms,
            "language": language,
            "genre": genre,
            "popularity": 50,
            "explicit": False,
            "metadata": {
                "release_value": release_value,
                "source": "local_csv_catalog",
                "csv_row": normalized_row,
                "audio_features": audio_features,
                "personality_vector": personality_vector,
            },
        }

        searchable_text = " ".join(
            filter(
                None,
                [
                    song_name.lower(),
                    singer.lower(),
                    genre.lower() if genre else "",
                    language.lower() if language else "",
                ],
            )
        )

        return {
            "track": track,
            "search_text": searchable_text,
        }

    def _load_catalog(self) -> None:
        if not self.csv_path.exists():
            print(f"ℹ️  Local music file not found at {self.csv_path}")
            self._catalog = []
            return

        catalog: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        try:
            # Handle both CSV and Excel files
            if self.csv_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(self.csv_path)
                # Convert DataFrame rows to dictionaries
                rows = df.to_dict('records')
            else:
                # Original CSV logic
                with self.csv_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)

            for row in rows:
                record = self._build_track_record(row)
                if not record:
                    continue
                track_id = record["track"]["id"]
                if track_id in seen_ids:
                    continue
                seen_ids.add(track_id)
                catalog.append(record)

            print(f"ℹ️  Loaded {len(catalog)} songs from {self.csv_path}")
        except Exception as exc:
            print(f"⚠️  Failed to load local music file: {exc}")
            catalog = []

        self._catalog = catalog

    def search_tracks(self, query: str, max_results: int = 20) -> List[Dict]:
        if not query:
            return []
        query_lower = query.lower()
        terms = [term for term in re.split(r"\s+", query_lower) if term]

        if not self._catalog:
            return []

        matches: List[Tuple[int, Dict[str, Any]]] = []
        for record in self._catalog:
            text = record["search_text"]
            score = 0
            if query_lower in text:
                score += 5
            for term in terms:
                if term in text:
                    score += 1
            if score > 0:
                matches.append((score, record["track"]))

        if not matches:
            # fallback: return first N songs for deterministic behavior
            return [record["track"] for record in self._catalog[:max_results]]

        matches.sort(key=lambda item: item[0], reverse=True)
        return [track for _, track in matches[:max_results]]

    def _normalise_preference_values(self, raw_value: Any) -> List[str]:
        values: List[str] = []
        if isinstance(raw_value, list):
            for item in raw_value:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        values.append(cleaned)
        elif isinstance(raw_value, str):
            parts = [raw_value]
            if "," in raw_value:
                parts = raw_value.split(",")
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    values.append(cleaned)
        return values

    def _split_csv_list(self, raw_value: Optional[str]) -> List[str]:
        if not raw_value:
            return []
        return [segment.strip() for segment in raw_value.split(",") if segment.strip()]

    def search_by_patient_preferences(self, patient_info: Dict[str, Any], max_results: int = 8) -> List[Dict[str, Any]]:
        if not patient_info or not self._catalog:
            return []

        favorite_genre_input = patient_info.get("favorite_genre") or patient_info.get("favoriteGenres")
        favorite_genres = self._normalise_preference_values(favorite_genre_input)
        instrument_preferences = self._normalise_preference_values(patient_info.get("instruments"))
        preferred_languages = self._normalise_preference_values(
            patient_info.get("preferred_languages") or patient_info.get("preferredLanguages")
        )
        favorite_musician = (patient_info.get("favorite_musician") or "").strip().lower()

        has_preferences = any([favorite_genres, instrument_preferences, preferred_languages, favorite_musician])
        if not has_preferences:
            return []

        genre_terms = [genre.lower() for genre in favorite_genres]
        instrument_terms = {instrument.lower() for instrument in instrument_preferences}
        language_terms = [language.lower() for language in preferred_languages]

        matches: List[Tuple[int, Dict[str, Any]]] = []
        for record in self._catalog:
            track = record["track"]
            metadata = track.get("metadata", {}) or {}
            csv_row = metadata.get("csv_row") or {}

            score = 0
            matched_instruments: List[str] = []
            matched_genre: Optional[str] = None
            matched_language: Optional[str] = None

            track_genre = (csv_row.get("genre") or track.get("genre") or "").lower()
            if genre_terms and track_genre:
                for genre in genre_terms:
                    if genre and genre in track_genre:
                        score += 4
                        matched_genre = genre
                        break

            csv_instruments = self._split_csv_list(csv_row.get("used instruments in the song"))
            if instrument_terms and csv_instruments:
                for instrument in csv_instruments:
                    if instrument.lower() in instrument_terms:
                        matched_instruments.append(instrument)
                if matched_instruments:
                    score += 3 * len(matched_instruments)

            track_language = (csv_row.get("language") or "").lower()
            if language_terms and track_language:
                for language in language_terms:
                    if language and language in track_language:
                        score += 2
                        matched_language = language
                        break

            artist_name = (track.get("artist") or "").lower()
            if favorite_musician and artist_name and favorite_musician in artist_name:
                score += 3

            if score <= 0:
                continue

            description_parts: List[str] = []
            if csv_row.get("genre"):
                description_parts.append(csv_row["genre"])
            if csv_row.get("language"):
                description_parts.append(csv_row["language"])
            if matched_instruments:
                description_parts.append(", ".join(matched_instruments))

            match_payload = {
                "id": track.get("id"),
                "title": track.get("title"),
                "url": track.get("url"),
                "artist": track.get("artist"),
                "channel": "Local CSV Catalogue",
                "description": " • ".join(description_parts) if description_parts else None,
                "metadata": {
                    "genre": csv_row.get("genre"),
                    "language": csv_row.get("language"),
                    "instruments": csv_instruments,
                    "release_value": metadata.get("release_value"),
                    "match_score": score,
                    "matched_genre": matched_genre,
                    "matched_instruments": matched_instruments,
                    "matched_language": matched_language,
                },
                "source": "local_csv_preferences",
            }
            matches.append((score, match_payload))

        if not matches:
            return []

        matches.sort(
            key=lambda item: (
                item[0],
                (item[1].get("metadata", {}).get("release_value") or ""),
            ),
            reverse=True,
        )

        return [payload for _, payload in matches[:max_results]]

    def get_catalog_size(self) -> int:
        """Return the number of songs in the catalog."""
        return len(self._catalog)

    def get_api_health_status(self) -> Dict[str, Any]:
        return {
            "service": "LocalMusicDataset",
            "csv_path": str(self.csv_path),
            "catalog_size": len(self._catalog),
        }