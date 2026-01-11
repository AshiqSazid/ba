"""
Shared Music Catalog Service
Eliminates duplicate catalog loading and implements memory-efficient music management.
"""

import gc
import os
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
import structlog
from threading import Lock
import pandas as pd
import csv
import re

# Try to import psutil, use fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = structlog.get_logger(__name__)


class SharedMusicCatalogService:
    """
    Singleton music catalog service that eliminates duplicate loading
    and implements memory-efficient management.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._catalog: List[Dict[str, Any]] = []
        self._catalog_index: Dict[str, int] = {}  # For O(1) lookups
        self._genre_index: Dict[str, Set[str]] = {}  # Genre to track IDs mapping
        self._artist_index: Dict[str, Set[str]] = {}  # Artist to track IDs mapping
        self._csv_path: Optional[Path] = None
        self._last_loaded = None
        self._lock = Lock()
        self._initialized = True
        self._memory_threshold_mb = 200  # Reload if catalog exceeds this

    @contextmanager
    def _monitor_memory(self, operation: str):
        """Monitor memory usage during operations."""
        try:
            start_memory = 0.0
            if PSUTIL_AVAILABLE:
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss / 1024 / 1024  # MB

            yield

            end_memory = 0.0
            if PSUTIL_AVAILABLE:
                process = psutil.Process(os.getpid())
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - start_memory

            if abs(memory_delta) > 20:  # Log if > 20MB change
                logger.info(f"Music catalog memory change during {operation}: {memory_delta:+.1f}MB")

        except Exception as e:
            logger.warning(f"Memory monitoring failed for {operation}: {e}")

    def _build_search_indexes(self):
        """Build efficient search indexes for the catalog."""
        self._genre_index.clear()
        self._artist_index.clear()
        self._catalog_index.clear()

        for i, record in enumerate(self._catalog):
            track_id = record["track"]["id"]
            self._catalog_index[track_id] = i

            # Build genre index
            genre = record["track"].get("genre", "").lower()
            if genre:
                if genre not in self._genre_index:
                    self._genre_index[genre] = set()
                self._genre_index[genre].add(track_id)

            # Build artist index
            artists = record["track"].get("artists", [])
            for artist in artists:
                artist_key = artist.lower()
                if artist_key not in self._artist_index:
                    self._artist_index[artist_key] = set()
                self._artist_index[artist_key].add(track_id)

        logger.info(f"Built indexes: {len(self._catalog_index)} tracks, "
                   f"{len(self._genre_index)} genres, {len(self._artist_index)} artists")

    def _parse_release_date(self, raw_value: str) -> Optional[str]:
        """Parse release date from various formats."""
        raw_value = (raw_value or "").strip()
        if not raw_value:
            return None

        formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y"]
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
        """Parse duration from various formats."""
        raw_value = (raw_value or "").strip()
        if not raw_value:
            return None
        if raw_value.isdigit():
            return int(raw_value) * 1000

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
        """Parse float value from string."""
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

    def _build_track_record(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Build a normalized track record from CSV row."""
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

        track_id = f"shared::{hash((song_name.lower(), singer.lower(), release_value or ''))}"

        track = {
            "id": track_id,
            "title": song_name,
            "name": song_name,
            "url": f"https://theramuse.shared/{track_id}",
            "artists": [singer],
            "artist": singer,
            "album": genre or "Shared Catalogue",
            "duration_ms": duration_ms,
            "duration": duration_ms,
            "language": language,
            "genre": genre,
            "popularity": 50,
            "explicit": False,
            "metadata": {
                "release_value": release_value,
                "source": "shared_catalog",
                "csv_row": normalized_row,
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

    def load_catalog(self, csv_path: Optional[str] = None, force_reload: bool = False) -> bool:
        """
        Load music catalog with memory management and caching.

        Args:
            csv_path: Path to CSV/Excel file
            force_reload: Force reload even if already loaded

        Returns:
            True if loaded successfully
        """
        with self._lock:
            # Check if already loaded
            if not force_reload and self._catalog and self._csv_path:
                logger.info(f"Catalog already loaded with {len(self._catalog)} tracks")
                return True

            with self._monitor_memory("load_catalog"):
                try:
                    # Determine path
                    path = Path(csv_path) if csv_path else self._csv_path
                    if not path:
                        path = Path(os.getenv(
                            "LOCAL_MUSIC_CSV_PATH",
                            "./data/d.xlsx",
                        )).expanduser()

                    if not path.exists():
                        logger.warning(f"Music catalog file not found at {path}")
                        self._catalog = []
                        return False

                    catalog: List[Dict[str, Any]] = []
                    seen_ids: Set[str] = set()

                    # Handle both CSV and Excel files
                    if path.suffix.lower() == '.xlsx':
                        # Load Excel file efficiently
                        df = pd.read_excel(
                            path,
                            dtype=str,  # Read all as strings to prevent type inference
                            na_values=['', 'NULL', 'null', 'None']
                        )
                        rows = df.to_dict('records')
                    else:
                        # Load CSV file efficiently
                        with path.open(newline="", encoding="utf-8") as handle:
                            reader = csv.DictReader(handle)
                            rows = list(reader)

                    # Process rows
                    for row in rows:
                        record = self._build_track_record(row)
                        if not record:
                            continue
                        track_id = record["track"]["id"]
                        if track_id in seen_ids:
                            continue
                        seen_ids.add(track_id)
                        catalog.append(record)

                    # Check memory usage if psutil is available
                    if PSUTIL_AVAILABLE:
                        process = psutil.Process(os.getpid())
                        memory_mb = process.memory_info().rss / 1024 / 1024

                        if memory_mb > self._memory_threshold_mb:
                            logger.warning(f"High memory usage after loading catalog: {memory_mb:.1f}MB")
                            # Force garbage collection
                            gc.collect()

                    self._catalog = catalog
                    self._csv_path = path
                    self._last_loaded = datetime.now()

                    # Build search indexes
                    self._build_search_indexes()

                    logger.info(f"Loaded {len(self._catalog)} tracks from {path}")
                    return True

                except Exception as e:
                    logger.error(f"Failed to load music catalog: {e}")
                    self._catalog = []
                    return False

    def search_tracks(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search tracks using efficient indexing.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of matching tracks
        """
        if not query or not self._catalog:
            return []

        query_lower = query.lower()
        terms = [term for term in re.split(r"\s+", query_lower) if term]

        # Use indexes for faster search
        matching_ids = set()

        # Check genre index
        for term in terms:
            if term in self._genre_index:
                matching_ids.update(self._genre_index[term])

        # Check artist index
        for term in terms:
            if term in self._artist_index:
                matching_ids.update(self._artist_index[term])

        # If we have matches from indexes, use them
        if matching_ids:
            matches = []
            for track_id in matching_ids:
                if track_id in self._catalog_index:
                    idx = self._catalog_index[track_id]
                    record = self._catalog[idx]

                    # Verify match with full text search
                    score = 0
                    text = record["search_text"]
                    if query_lower in text:
                        score += 5
                    for term in terms:
                        if term in text:
                            score += 1

                    if score > 0:
                        matches.append((score, record["track"]))

            matches.sort(key=lambda item: item[0], reverse=True)
            return [track for _, track in matches[:max_results]]

        # Fallback to full text search
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
            # Return random tracks if no matches
            import random
            sample_size = min(max_results, len(self._catalog))
            sampled_records = random.sample(self._catalog, sample_size)
            return [record["track"] for record in sampled_records]

        matches.sort(key=lambda item: item[0], reverse=True)
        return [track for _, track in matches[:max_results]]

    def get_track_by_id(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get track by ID using index for O(1) lookup."""
        if track_id in self._catalog_index:
            idx = self._catalog_index[track_id]
            return self._catalog[idx]["track"]
        return None

    def get_tracks_by_genre(self, genre: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get tracks by genre using index."""
        genre_key = genre.lower()
        if genre_key not in self._genre_index:
            return []

        track_ids = list(self._genre_index[genre_key])[:max_results]
        tracks = []
        for track_id in track_ids:
            track = self.get_track_by_id(track_id)
            if track:
                tracks.append(track)

        return tracks

    def get_tracks_by_artist(self, artist: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get tracks by artist using index."""
        artist_key = artist.lower()
        if artist_key not in self._artist_index:
            return []

        track_ids = list(self._artist_index[artist_key])[:max_results]
        tracks = []
        for track_id in track_ids:
            track = self.get_track_by_id(track_id)
            if track:
                tracks.append(track)

        return tracks

    def search_by_patient_preferences(self, patient_info: Dict[str, Any], max_results: int = 8) -> List[Dict[str, Any]]:
        """Search by patient preferences using efficient lookups."""
        if not patient_info or not self._catalog:
            return []

        favorite_genres = self._normalize_preference_values(patient_info.get("favorite_genre") or patient_info.get("favoriteGenres"))
        favorite_artists = self._normalize_preference_values(patient_info.get("favorite_musician") or patient_info.get("favoriteArtists"))

        if not favorite_genres and not favorite_artists:
            return []

        matching_ids = set()

        # Add genre matches
        for genre in favorite_genres:
            genre_key = genre.lower()
            if genre_key in self._genre_index:
                matching_ids.update(self._genre_index[genre_key])

        # Add artist matches
        for artist in favorite_artists:
            artist_key = artist.lower()
            if artist_key in self._artist_index:
                matching_ids.update(self._artist_index[artist_key])

        # Convert IDs to tracks
        tracks = []
        for track_id in matching_ids:
            track = self.get_track_by_id(track_id)
            if track:
                tracks.append(track)

        return tracks[:max_results]

    def _normalize_preference_values(self, raw_value: Any) -> List[str]:
        """Normalize preference values to list of strings."""
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

    def get_catalog_size(self) -> int:
        """Return the number of songs in the catalog."""
        return len(self._catalog)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            process_memory_mb = memory_info.rss / 1024 / 1024
        else:
            process_memory_mb = 0.0

        return {
            "catalog_size": len(self._catalog),
            "indexed_tracks": len(self._catalog_index),
            "indexed_genres": len(self._genre_index),
            "indexed_artists": len(self._artist_index),
            "process_memory_rss_mb": process_memory_mb,
            "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
            "catalog_path": str(self._csv_path) if self._csv_path else None
        }

    def cleanup(self):
        """Cleanup catalog data and free memory."""
        with self._lock:
            logger.info("Cleaning up shared music catalog")

            self._catalog.clear()
            self._catalog_index.clear()
            self._genre_index.clear()
            self._artist_index.clear()

            gc.collect()
            logger.info("Shared music catalog cleaned up")

    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            "service": "SharedMusicCatalogService",
            "catalog_loaded": bool(self._catalog),
            "catalog_size": len(self._catalog),
            "memory_usage": self.get_memory_usage(),
            "status": "healthy" if self._catalog else "not_loaded"
        }


# Global singleton instance
_global_catalog_service = None
_catalog_lock = Lock()

def get_shared_music_catalog() -> SharedMusicCatalogService:
    """Get global shared music catalog service instance."""
    global _global_catalog_service

    if _global_catalog_service is None:
        with _catalog_lock:
            if _global_catalog_service is None:
                _global_catalog_service = SharedMusicCatalogService()
                logger.info("Global shared music catalog service initialized")

    return _global_catalog_service

def cleanup_shared_music_catalog():
    """Cleanup global shared music catalog service."""
    global _global_catalog_service

    with _catalog_lock:
        if _global_catalog_service is not None:
            _global_catalog_service.cleanup()
            _global_catalog_service = None
            logger.info("Global shared music catalog service cleaned up")