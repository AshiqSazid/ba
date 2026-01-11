"""
Utility helper functions for TheraMuse application.
"""

from __future__ import annotations

import os
from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np


def make_json_safe(obj: Any) -> Any:
    """
    Convert Python objects to JSON-safe format.

    Args:
        obj: Any Python object

    Returns:
        JSON-safe version of the object
    """
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(item) for item in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def get_mysql_connection():
    """
    Get MySQL database connection with fallback configuration.

    Returns:
        MySQL connection object
    """
    try:
        from db_config import connect_to_mysql
        return connect_to_mysql()
    except ModuleNotFoundError:
        try:
            from backend.db_config import connect_to_mysql
            return connect_to_mysql()
        except ModuleNotFoundError:
            # Fallback implementation
            import mysql.connector
            return mysql.connector.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DATABASE", "theramuse")
            )


def get_mysql_database_name() -> str:
    """
    Get MySQL database name with fallback.

    Returns:
        Database name string
    """
    try:
        from db_config import MYSQL_DATABASE
        return MYSQL_DATABASE
    except ModuleNotFoundError:
        try:
            from backend.db_config import MYSQL_DATABASE
            return MYSQL_DATABASE
        except ModuleNotFoundError:
            return os.getenv("MYSQL_DATABASE", "theramuse")