import os
import sys
from typing import Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from db_config import connect_to_mysql, MYSQL_DATABASE
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from db_config import connect_to_mysql, MYSQL_DATABASE


CSV_PATH = os.getenv(
    "CSV_PATH", "/home/spectre-rosamund/Videos/Screencasts/th.xlsx"
)


def slugify(text: str) -> str:
    slug = "".join(
        ch.lower() if ch.isalnum() else "-"
        for ch in text.strip()
    )
    slug = "-".join(filter(None, slug.split("-")))
    return slug or "song"


def parse_duration(value: object) -> Optional[int]:
    if pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    if raw.isdigit():
        return int(raw)

    parts = raw.split(":")
    if all(part.isdigit() for part in parts):
        parts = [int(part) for part in parts]
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds

    return None


def parse_release_date(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    for dayfirst in (False, True):
        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst)
        if not pd.isna(parsed):
            return parsed.strftime("%Y-%m-%d")

    return None


def main() -> None:
    print(f"Importing CSV data into MySQL database '{MYSQL_DATABASE}'")
    conn = connect_to_mysql()
    cursor = conn.cursor()

    df = pd.read_csv(CSV_PATH)
    df.columns = [col.strip() for col in df.columns]

    limit = os.getenv("CSV_LIMIT")
    if limit:
        try:
            df = df.head(int(limit))
        except ValueError:
            raise ValueError("CSV_LIMIT must be an integer") from None

    required_columns = {"singer", "song_name"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in CSV: {', '.join(sorted(missing))}")

    for _, row in df.iterrows():
        singer_raw = row.get("singer")
        if pd.isna(singer_raw):
            print("Skipped row with NaN singer value")
            continue
        singer_name = str(singer_raw).strip()
        if not singer_name or singer_name.lower() == "nan":
            print("Skipped row with empty singer name")
            continue

        cursor.execute(
            "INSERT IGNORE INTO singers (singer_name) VALUES (%s)",
            (singer_name,),
        )

        cursor.execute(
            "SELECT singer_id FROM singers WHERE singer_name = %s",
            (singer_name,),
        )
        singer_row = cursor.fetchone()
        if not singer_row:
            print(f"Skipped row; failed to resolve singer_id for '{singer_name}'")
            continue
        singer_id = singer_row[0]

        song_raw = row.get("song_name")
        if pd.isna(song_raw):
            print(f"Skipped row with NaN song name (singer: {singer_name})")
            continue
        song_name = str(song_raw).strip()
        if not song_name or song_name.lower() == "nan":
            print(f"Skipped row with empty song name (singer: {singer_name})")
            continue

        song_url = row.get("song_url")
        if pd.isna(song_url) or not str(song_url).strip():
            slug = slugify(song_name)
            song_url = f"https://example.com/{slug}"

        released_date = parse_release_date(row.get("released_date"))
        duration_seconds = parse_duration(row.get("duration"))
        genre = row.get("genre")
        if pd.isna(genre):
            genre = None
        elif isinstance(genre, str):
            genre = genre.strip() or None

        cursor.execute(
            "SELECT song_id FROM songs WHERE singer_id = %s AND song_name = %s",
            (singer_id, song_name),
        )
        if cursor.fetchone():
            print(f"Skipped duplicate: {song_name} by {singer_name}")
            continue

        cursor.execute(
            """
            INSERT INTO songs (singer_id, song_name, song_url, genre, released_date, duration)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                singer_id,
                song_name,
                song_url,
                genre,
                released_date,
                duration_seconds,
            ),
        )

        print(f"Imported: {song_name} by {singer_name}")

    conn.commit()
    cursor.close()
    conn.close()
    print(" CSV imported successfully!")


if __name__ == "__main__":
    main()
