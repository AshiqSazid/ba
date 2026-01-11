"""
Adapter that bridges the modern FastAPI CLI with the legacy category builder
implemented inside python.py. Instead of booting the heavy TheraMuse server
(which expects an active MySQL instance), this module instantiates the therapy
classes directly and runs them in-process so we can keep returning the detailed
Birthplace/Instrument/Nostalgia groupings.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import os
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time as _time

import structlog

from app.utils.helpers import make_json_safe

logger = structlog.get_logger(__name__)


class _CSVNostalgiaStore:
    """Minimal subset of DatabaseManager that only serves nostalgia CSV rows."""

    DEFAULT_PATH = "./data/d.xlsx"

    def __init__(self) -> None:
        raw_path = os.getenv("NOSTALGIA_CSV_PATH") or self.DEFAULT_PATH
        self.csv_path = Path(raw_path).expanduser()
        self._dataset = self._load_dataset()
        self._warned_location = False

    def _parse_release(self, raw_value: str) -> Tuple[Optional[str], Optional[int]]:
        raw_value = (raw_value or "").strip()
        if not raw_value:
            return None, None

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
                    return f"{parsed.year}-01-01", parsed.year
                return parsed.strftime("%Y-%m-%d"), parsed.year
            except ValueError:
                continue

        digits = "".join(ch for ch in raw_value if ch.isdigit())
        if len(digits) == 4:
            year = int(digits)
            return f"{year}-01-01", year
        return None, None

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not self.csv_path.exists():
            logger.warning("Nostalgia file not found", path=str(self.csv_path))
            return []

        dataset: List[Dict[str, Any]] = []
        seen: set = set()

        # Handle both CSV and Excel files
        if self.csv_path.suffix.lower() == '.xlsx':
            import pandas as pd
            df = pd.read_excel(self.csv_path)
            for _, row in df.iterrows():
                song_name = str(row.get("song_name", "")).strip()
                if not song_name:
                    continue
                artist = str(row.get("singer", "")).strip()
                key = (song_name.lower(), artist.lower())
                if key in seen:
                    continue
                seen.add(key)
                release_value, release_year = self._parse_release(str(row.get("released_date", "")))
                dataset.append(
                    {
                        "song_name": song_name,
                        "artist_name": artist or None,
                        "release_value": release_value,
                        "release_year": release_year,
                        "genre": str(row.get("genre", "")).strip() or None,
                    }
                )
        else:
            # Original CSV logic
            with self.csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    song_name = (row.get("song_name") or "").strip()
                    if not song_name:
                        continue
                    artist = (row.get("singer") or "").strip()
                    key = (song_name.lower(), artist.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    release_value, release_year = self._parse_release(row.get("released_date") or "")
                    dataset.append(
                        {
                            "song_name": song_name,
                            "artist_name": artist or None,
                            "release_value": release_value,
                            "release_year": release_year,
                            "genre": (row.get("genre") or "").strip() or None,
                        }
                    )
        return dataset

    def get_songs_for_nostalgia_window(
        self,
        city: Optional[str],
        start_year: Optional[int],
        end_year: Optional[int],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if not self._dataset:
            return []

        if city and not self._warned_location:
            logger.info("CSV fallback dataset lacks city metadata; returning general nostalgia songs", city=city)
            self._warned_location = True

        start: Optional[int] = None
        end: Optional[int] = None
        if start_year is not None and end_year is not None:
            start, end = sorted([start_year, end_year])

        rows: List[Dict[str, Any]] = []
        for item in self._dataset:
            release_year = item.get("release_year")
            if (
                start is not None
                and end is not None
                and release_year is not None
                and not (start <= release_year <= end)
            ):
                continue
            rows.append(
                {
                    "song_name": item.get("song_name"),
                    "artist_name": item.get("artist_name"),
                    "release_value": item.get("release_value"),
                    "genre": item.get("genre"),
                }
            )
            if len(rows) >= limit:
                break
        return rows


@contextmanager
def _fast_sleep():
    """
    The legacy therapies sprinkle time.sleep(0.5) calls to throttle network APIs.
    We run entirely offline, so overriding sleep keeps CLI latency reasonable.
    """
    original_sleep = _time.sleep

    def _noop(_: float) -> None:
        return None

    _time.sleep = _noop  # type: ignore[assignment]
    try:
        yield
    finally:
        _time.sleep = original_sleep  # type: ignore[assignment]


class LegacyTheraMuseAdapter:
    """Lazy loader for the legacy recommendation therapies."""

    def __init__(self) -> None:
        self._module: Any = None
        self._dementia_service: Any = None
        self._adhd_service: Any = None
        self._down_service: Any = None
        self._lock = threading.Lock()
        self._script_path = Path(__file__).resolve().parents[2] / "python.py"
        self._csv_store = _CSVNostalgiaStore()

    def _ensure_services(self) -> bool:
        if self._dementia_service:
            return True

        with self._lock:
            if self._dementia_service:
                return True
            if not self._script_path.exists():
                logger.warning("Legacy TheraMuse script not found", path=str(self._script_path))
                return False

            spec = importlib.util.spec_from_file_location("theramuse_legacy", self._script_path)
            if not spec or not spec.loader:
                logger.error("Failed to build import spec for legacy engine", path=str(self._script_path))
                return False

            module = importlib.util.module_from_spec(spec)
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    spec.loader.exec_module(module)  # type: ignore[call-arg]
                    dementia = module.DementiaTherapy(db=self._csv_store)
                    adhd = module.ADHDTherapy()
                    down = module.DownSyndromeTherapy()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Unable to initialise legacy therapy services", error=str(exc))
                return False

            self._module = module
            self._dementia_service = dementia
            self._adhd_service = adhd
            self._down_service = down
            logger.info("Legacy therapy services initialised")
            return True

    def generate_recommendations(
        self,
        patient_info: Dict[str, Any],
        condition: str,
        patient_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run the condition-specific therapy workflow from python.py.
        """
        if not self._ensure_services():
            return None

        condition_key = condition.lower()
        try:
            with _fast_sleep(), contextlib.redirect_stdout(sys.stderr):
                if condition_key == "dementia":
                    payload = self._dementia_service.get_dementia_recommendations(patient_info)
                elif condition_key == "adhd":
                    payload = self._adhd_service.get_adhd_recommendations(patient_info)
                elif condition_key == "down_syndrome":
                    payload = self._down_service.get_down_syndrome_recommendations(patient_info)
                else:
                    logger.warning("Legacy adapter does not support condition", condition=condition_key)
                    return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Legacy recommendation generation failed", error=str(exc), condition=condition_key)
            return None

        normalized = self._finalize_payload(payload, patient_id, condition_key, patient_info)
        return make_json_safe(normalized)

    def record_feedback(
        self,
        patient_id: str,
        session_id: str,
        condition: str,
        song: Dict[str, Any],
        feedback_type: str,
        patient_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        The legacy scripts expect a running database to store feedback. Since we
        bypass that stack, this method currently no-ops and lets the modern ML
        service handle feedback tracking.
        """
        logger.debug(
            "Legacy adapter feedback passthrough",
            patient_id=patient_id,
            session_id=session_id,
            condition=condition,
            feedback_type=feedback_type,
        )
        return False

    def get_analytics(self) -> Optional[Dict[str, Any]]:
        """
        Analytics require the MySQL-backed persistence layer, which is not
        available in the CLI environment. Returning None triggers the existing
        lightweight analytics fallback.
        """
        return None

    def _finalize_payload(
        self,
        payload: Optional[Dict[str, Any]],
        patient_id: Optional[str],
        condition: str,
        patient_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        data = dict(payload or {})
        categories = data.get("categories") or {}
        total_songs = data.get("total_songs")
        if total_songs is None:
            total_songs = sum(len(category.get("songs") or []) for category in categories.values())
            data["total_songs"] = total_songs

        data.setdefault("session_id", f"legacy_{uuid.uuid4()}")
        if patient_id:
            data["patient_id"] = patient_id
        data.setdefault("condition", condition)
        data.setdefault("generated_at", datetime.utcnow().isoformat())
        data.setdefault("method", "legacy_theramuse_engine")
        data.setdefault(
            "bandit_stats",
            {"n_interactions": 0, "avg_reward": 0.0, "exploration_rate": 0.0},
        )
        self._ensure_birthplace_categories(data, patient_info)
        return data

    def _ensure_birthplace_categories(self, payload: Dict[str, Any], patient_info: Dict[str, Any], target: int = 20) -> None:
        if not self._csv_store:
            return

        categories = payload.setdefault("categories", {})
        birth_year = patient_info.get("birth_year") or patient_info.get("birthYear")
        start_year = end_year = None
        if isinstance(birth_year, int):
            start_year = birth_year + 10
            end_year = birth_year + 30

        city = patient_info.get("birthplace_city") or patient_info.get("birthplaceCity")
        country = patient_info.get("birthplace_country") or patient_info.get("birthplaceCountry")

        def ensure_category(key: str, location: Optional[str]) -> None:
            if not location:
                return
            entry = categories.setdefault(
                key,
                {
                    "query": f"CSV nostalgia lookup {location}",
                    "songs": [],
                    "count": 0,
                    "query_source": "csv_nostalgia_lookup",
                },
            )
            songs = list(entry.get("songs") or [])
            deficit = target - len(songs)
            if deficit <= 0:
                entry["songs"] = songs[:target]
                entry["count"] = len(entry["songs"])
                return

            rows = self._csv_store.get_songs_for_nostalgia_window(
                location,
                start_year,
                end_year,
                limit=target * 2,
            )
            added = 0
            for row in rows:
                if len(songs) >= target:
                    break
                songs.append(
                    {
                        "title": row.get("song_name"),
                        "artist": row.get("artist_name") or "Unknown Artist",
                        "description": f"Nostalgia CSV match for {row.get('song_name')}",
                        "source": "csv_nostalgia_lookup",
                        "metadata": {
                            "release_value": row.get("release_value"),
                            "source": "csv_nostalgia_lookup",
                        },
                    }
                )
                added += 1

            if added and "query_source" not in entry:
                entry["query_source"] = "csv_nostalgia_lookup"
            entry["songs"] = songs[:target]
            entry["count"] = len(entry["songs"])

        ensure_category("birthplace_city", city)
        ensure_category("birthplace_country", country)
