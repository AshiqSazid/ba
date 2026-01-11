"""
Database manager service for TheraMuse application.

Handles database connections, table creation, and data access operations.
Includes CSV fallback functionality for offline operation.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.utils.helpers import get_mysql_connection, get_mysql_database_name


class DatabaseManager:
    """
    Manages database operations for TheraMuse application.

    Provides database connection management, table creation,
    and fallback to CSV data when database is unavailable.
    """

    def __init__(self, database: Optional[str] = None):
        self.database = database or get_mysql_database_name()
        self.conn = None
        self._table_columns_cache: Dict[str, Set[str]] = {}
        self._csv_fallback_cache: Optional[List[Dict[str, Any]]] = None
        self._csv_path = os.getenv(
            "NOSTALGIA_CSV_PATH",
            "/home/spectre-rosamund/CVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV/d.csv",
        )
        self._csv_warned_location = False
        self.connect()
        self.create_tables()

    def connect(self) -> None:
        """Establish database connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass

        try:
            self.conn = get_mysql_connection()
            self.conn.autocommit = False
        except Exception as e:
            print(f"⚠️  Failed to connect to database: {e}")
            self.conn = None

    def create_tables(self) -> None:
        """Create database tables if they don't exist."""
        if not self.conn:
            print("⚠️  No database connection available, skipping table creation")
            return

        cursor = self.conn.cursor()

        try:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS therapy_sessions (
                    session_id VARCHAR(64) PRIMARY KEY,
                    patient_id VARCHAR(64) NOT NULL,
                    `condition` VARCHAR(64) NOT NULL,
                    therapy_method VARCHAR(255),
                    total_songs INT,
                    exploration_rate DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_sessions_patient (patient_id),
                    INDEX idx_sessions_condition (`condition`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS therapy_recommendations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(64) NOT NULL,
                    patient_id VARCHAR(64) NOT NULL,
                    category VARCHAR(255) NOT NULL,
                    query TEXT,
                    song_title VARCHAR(255),
                    video_id VARCHAR(128),
                    channel VARCHAR(255),
                    description TEXT,
                    `rank` INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_recs_session (session_id),
                    INDEX idx_recs_patient (patient_id),
                    CONSTRAINT fk_recs_session FOREIGN KEY (session_id)
                        REFERENCES therapy_sessions(session_id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS therapy_feedback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id VARCHAR(64) NOT NULL,
                    session_id VARCHAR(64),
                    `condition` VARCHAR(64) NOT NULL,
                    song_title VARCHAR(255),
                    video_id VARCHAR(128),
                    reward DOUBLE NOT NULL,
                    feedback_type VARCHAR(64),
                    context_features TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_feedback_patient (patient_id),
                    INDEX idx_feedback_session (session_id),
                    INDEX idx_feedback_condition (`condition`),
                    CONSTRAINT fk_feedback_session FOREIGN KEY (session_id)
                        REFERENCES therapy_sessions(session_id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit_stats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    `condition` VARCHAR(64) NOT NULL,
                    n_interactions INT,
                    total_reward DOUBLE,
                    avg_reward DOUBLE,
                    exploration_rate DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_bandit_condition (`condition`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id VARCHAR(64) PRIMARY KEY,
                    name VARCHAR(255),
                    age INT,
                    birth_year INT,
                    `condition` VARCHAR(64),
                    patient_info TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS big5_scores (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id VARCHAR(64),
                    session_id VARCHAR(64),
                    openness DOUBLE,
                    conscientiousness DOUBLE,
                    extraversion DOUBLE,
                    agreeableness DOUBLE,
                    neuroticism DOUBLE,
                    reinforcement_learning INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_big5_patient (patient_id),
                    INDEX idx_big5_session (session_id),
                    CONSTRAINT fk_big5_patient FOREIGN KEY (patient_id)
                        REFERENCES patients(patient_id) ON DELETE SET NULL,
                    CONSTRAINT fk_big5_session FOREIGN KEY (session_id)
                        REFERENCES therapy_sessions(session_id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            self.conn.commit()
            print("✅ Database tables created successfully")

        except Exception as e:
            print(f"⚠️  Failed to create tables: {e}")
            self.conn.rollback()
        finally:
            cursor.close()

    def _parse_csv_release(self, raw_value: str) -> Optional[str]:
        """Parse release date from various formats."""
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

    def _load_csv_fallback(self) -> List[Dict[str, Any]]:
        """Load CSV fallback data for offline operation."""
        if self._csv_fallback_cache is not None:
            return self._csv_fallback_cache

        csv_path = (self._csv_path or "").strip()
        if not csv_path:
            self._csv_fallback_cache = []
            return self._csv_fallback_cache

        csv_file = Path(csv_path).expanduser()
        if not csv_file.exists():
            print(f"ℹ️  Nostalgia CSV fallback not found at {csv_file}")
            self._csv_fallback_cache = []
            return self._csv_fallback_cache

        dataset: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()

        try:
            with csv_file.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    song_name = (row.get("song_name") or "").strip()
                    if not song_name:
                        continue

                    artist_name = (row.get("singer") or "").strip()
                    dedupe_key = (song_name.lower(), artist_name.lower())

                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)

                    release_value = self._parse_csv_release(row.get("released_date") or "")
                    release_year = None
                    if release_value:
                        try:
                            release_year = int(release_value[:4])
                        except ValueError:
                            release_year = None

                    dataset.append(
                        {
                            "song_name": song_name,
                            "artist_name": artist_name or None,
                            "release_value": release_value,
                            "release_year": release_year,
                            "genre": (row.get("genre") or "").strip() or None,
                            "language": (row.get("language") or "").strip() or None,
                        }
                    )

            print(f"ℹ️  Loaded {len(dataset)} fallback songs from CSV at {csv_file}")
        except Exception as exc:
            print(f"⚠️  Failed to load nostalgia CSV fallback: {exc}")
            dataset = []

        self._csv_fallback_cache = dataset
        return dataset

    def _get_csv_fallback_rows(
        self,
        city: Optional[str],
        start_year: Optional[int],
        end_year: Optional[int],
        limit: int,
    ) -> List[Dict]:
        """Get rows from CSV fallback matching criteria."""
        dataset = self._load_csv_fallback()
        if not dataset:
            return []

        if city and not self._csv_warned_location:
            print("ℹ️  CSV fallback dataset lacks city metadata; returning general songs.")
            self._csv_warned_location = True

        results: List[Dict[str, Any]] = []
        start: Optional[int] = None
        end: Optional[int] = None
        if start_year is not None and end_year is not None:
            start, end = sorted([start_year, end_year])

        for row in dataset:
            release_year = row.get("release_year")
            if (
                start is not None
                and end is not None
                and release_year is not None
                and not (start <= release_year <= end)
            ):
                continue

            results.append(
                {
                    "song_name": row.get("song_name"),
                    "artist_name": row.get("artist_name"),
                    "release_value": row.get("release_value"),
                    "genre": row.get("genre"),
                }
            )

            if len(results) >= limit:
                break

        return results

    def _table_exists(self, table: str) -> bool:
        """Check if table exists in database."""
        if not self.conn:
            return False

        cache_key = f"table_exists::{table}"
        if cache_key in self._table_columns_cache:
            return True

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
                """,
                (table,),
            )
            exists = cursor.fetchone()[0] > 0
            if exists:
                self._table_columns_cache[cache_key] = set()
            return exists
        except Exception:
            return False
        finally:
            cursor.close()

    def _get_table_columns(self, table: str) -> Set[str]:
        """Get column names for a table."""
        if not self.conn:
            return set()

        if table in self._table_columns_cache:
            return self._table_columns_cache[table]

        if not self._table_exists(table):
            return set()

        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SHOW COLUMNS FROM `{table}`")
            columns = {row[0] for row in cursor.fetchall()}
            self._table_columns_cache[table] = columns
            return columns
        except Exception:
            return set()
        finally:
            cursor.close()

    def get_songs_for_nostalgia_window(
        self,
        city: Optional[str],
        start_year: Optional[int],
        end_year: Optional[int],
        limit: int = 20,
    ) -> List[Dict]:
        """Get songs for nostalgia therapy with year and city filtering."""
        fallback = lambda: self._get_csv_fallback_rows(city, start_year, end_year, limit)

        if not self.conn or not self._table_exists("songs"):
            return fallback()

        song_columns = self._get_table_columns("songs")
        if not song_columns:
            return fallback()

        select_parts = ["s.song_name AS song_name"]
        join_clause = ""
        singer_columns: Set[str] = set()

        if "singer_id" in song_columns and self._table_exists("singers"):
            singer_columns = self._get_table_columns("singers")
            select_parts.append("sg.singer_name AS artist_name")
            join_clause = "LEFT JOIN singers sg ON sg.singer_id = s.singer_id"
        elif "singer" in song_columns:
            select_parts.append("s.singer AS artist_name")
        else:
            select_parts.append("NULL AS artist_name")

        release_column = None
        release_year_column = None
        for candidate in ("released_date", "release_date"):
            if candidate in song_columns:
                release_column = candidate
                break
        if not release_column and "release_year" in song_columns:
            release_year_column = "release_year"

        if release_column:
            select_parts.append(f"s.{release_column} AS release_value")
        elif release_year_column:
            select_parts.append(f"s.{release_year_column} AS release_value")
        else:
            select_parts.append("NULL AS release_value")

        conditions = []
        params: List[Any] = []

        if start_year and end_year and (release_column or release_year_column):
            start, end = sorted([start_year, end_year])
            if release_column:
                conditions.append(f"s.{release_column} BETWEEN %s AND %s")
                params.extend([f"{start}-01-01", f"{end}-12-31"])
            else:
                conditions.append(f"s.{release_year_column} BETWEEN %s AND %s")
                params.extend([start, end])

        if city:
            city_filters = []
            like_value = f"%{city}%"
            for column in song_columns:
                if "city" in column.lower():
                    city_filters.append(f"s.`{column}` LIKE %s")
            for column in singer_columns:
                lowered = column.lower()
                if "city" in lowered or "hometown" in lowered or "birthplace" in lowered:
                    city_filters.append(f"sg.`{column}` LIKE %s")
            if city_filters:
                conditions.append("(" + " OR ".join(city_filters) + ")")
                params.extend([like_value] * len(city_filters))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        order_clause = "s.song_name ASC"
        if release_column:
            order_clause = f"s.{release_column} DESC"
        elif release_year_column:
            order_clause = f"s.{release_year_column} DESC"

        query = f"""
            SELECT {', '.join(select_parts)}
            FROM songs s
            {join_clause}
            {where_clause}
            ORDER BY {order_clause}
            LIMIT %s
        """
        params.append(limit)

        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            results = [
                {
                    "song_name": row[0],
                    "artist_name": row[1],
                    "release_value": row[2].strftime("%Y-%m-%d") if row[2] else None,
                }
                for row in cursor.fetchall()
            ]
            return results
        except Exception as e:
            print(f"⚠️  Database query failed, using CSV fallback: {e}")
            return fallback()
        finally:
            cursor.close()

    def get_api_health_status(self) -> Dict[str, Any]:
        """Get database health status."""
        return {
            "service": "DatabaseManager",
            "database": self.database,
            "connected": self.conn is not None,
            "tables_created": bool(self.conn and self._table_exists("therapy_sessions")),
            "csv_fallback_available": len(self._load_csv_fallback()) > 0
        }