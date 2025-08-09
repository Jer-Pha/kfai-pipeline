import sqlite3
from datetime import date, datetime
from json import dump
from os import makedirs, path
from typing import Iterable, Optional

import googleapiclient.discovery as ytapi
import mysql.connector
from googleapiclient.errors import HttpError
from isodate import parse_duration
from isodate.duration import Duration
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import (
    FetchedTranscriptSnippet,
    YouTubeTranscriptApi,
)

from common.types import CompleteVideoRecord, TranscriptChunk
from extract.utils.config import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_USER,
    SQLITE_DB_PATH,
    YOUTUBE_API_KEY,
)
from extract.utils.types import (
    MySQLConfig,
    RawVideoRecord,
    TranscriptChunk,
    TranscriptSnippet,
    VideoMetadata,
)


def yt_datetime_to_epoch(data: str) -> int:
    """Converts YouTube API ISO datetime to Unix epoch timestamp."""
    if not data:
        return 0
    return int(datetime.fromisoformat(data.replace("Z", "+00:00")).timestamp())


def duration_to_seconds(duration: Duration) -> int:
    """Converts YouTube API ISO duration to total seconds."""
    return int(parse_duration(duration).total_seconds())


def _export_mysql_to_sqlite(
    mysql_config: MySQLConfig, sqlite_db_path: str
) -> None:
    """Exports relevant data from a MySQL database to an SQLite database."""
    mysql_conn, sqlite_conn = None, None
    try:
        # Connect to MySQL
        mysql_conn = mysql.connector.connect(**mysql_config)
        mysql_cursor = mysql_conn.cursor(dictionary=True)

        # Connect to SQLite
        sqlite_conn = sqlite3.connect(sqlite_db_path)
        sqlite_cursor = sqlite_conn.cursor()

        # Create tables in SQLite if they don't exist
        sqlite_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS videos_video (
                id INTEGER PRIMARY KEY,
                video_id TEXT,
                show_id INTEGER,
                producer_id INTEGER
            )
        """
        )
        sqlite_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS shows_show (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """
        )
        sqlite_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS hosts_host (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """
        )
        sqlite_cursor.execute(
            """
          CREATE TABLE IF NOT EXISTS videos_video_hosts(
            video_id INTEGER,
            host_id INTEGER
          )
        """
        )

        # Fetch from MySQL and insert to SQLite (videos_video)
        mysql_cursor.execute(
            "SELECT id, video_id, show_id, producer_id FROM videos_video WHERE channel_id < 3"
        )
        videos = mysql_cursor.fetchall()
        sqlite_cursor.executemany(
            "INSERT INTO videos_video VALUES (:id, :video_id, :show_id, :producer_id)",
            videos,
        )

        # Fetch from MySQL and insert to SQLite (shows_show)
        mysql_cursor.execute("SELECT id, name FROM shows_show")
        shows = mysql_cursor.fetchall()
        sqlite_cursor.executemany(
            "INSERT INTO shows_show VALUES (:id, :name)", shows
        )

        # Fetch from MySQL and insert to SQLite (hosts_host)
        mysql_cursor.execute("SELECT id, name FROM hosts_host")
        hosts = mysql_cursor.fetchall()
        sqlite_cursor.executemany(
            "INSERT INTO hosts_host VALUES (:id, :name)", hosts
        )

        # Fetch from MySQL and insert to SQLite (videos_video_hosts)
        mysql_cursor.execute(
            "SELECT video_id, host_id FROM videos_video_hosts"
        )
        video_hosts = mysql_cursor.fetchall()
        sqlite_cursor.executemany(
            "INSERT INTO videos_video_hosts VALUES (:video_id, :host_id)",
            video_hosts,
        )

        # Commit changes
        sqlite_conn.commit()
        print("Data exported from MySQL to SQLite successfully.")

    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
    except sqlite3.Error as err:
        print(f"Error connecting to SQLite: {err}")
    finally:
        # Close connections
        if mysql_conn and mysql_conn.is_connected():
            mysql_conn.close()
        if sqlite_conn:
            sqlite_conn.close()


def create_local_sqlite_db() -> None:
    # MySQL config
    mysql_config: MySQLConfig = {
        "host": MYSQL_HOST,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "database": MYSQL_DATABASE,
    }

    # SQLite config
    sqlite_db_path = SQLITE_DB_PATH

    _export_mysql_to_sqlite(mysql_config, sqlite_db_path)


def get_video_db_data(
    sqlite_db: str,
    video_ids: Optional[list[str]] = None,
) -> list[RawVideoRecord]:
    """
    Fetches video metadata from the SQLite database.
    If video_ids is provided, fetches data only for those IDs.
    Otherwise, fetches all video data.
    """

    # Get database data from local database
    conn = sqlite3.connect(sqlite_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT
            vv.id, vv.video_id, ss.name AS show_name,
            GROUP_CONCAT(hh.name) AS hosts
        FROM videos_video vv
        JOIN shows_show ss ON vv.show_id = ss.id
        LEFT JOIN videos_video_hosts vvh ON vv.id = vvh.video_id
        LEFT JOIN hosts_host hh ON vvh.host_id = hh.id
    """

    # If specific IDs are requested, add a WHERE clause
    if video_ids:
        # Use placeholders to prevent SQL injection
        placeholders = ",".join("?" for _ in video_ids)
        query += f" WHERE vv.video_id IN ({placeholders})"
        query += " GROUP BY vv.id"
        cursor.execute(query, video_ids)
    else:
        query += " GROUP BY vv.id"
        cursor.execute(query)

    rows = cursor.fetchall()
    conn.close()

    # Return raw database data
    video_data: list[RawVideoRecord] = []
    for row in rows:
        video_data.append(
            {
                "id": row[0],
                "video_id": row[1],
                "show_name": row[2],
                "hosts": (row[3].split(",") if row[3] else []),
            }
        )

    return video_data


def get_youtube_data(video_ids: list[str]) -> dict[str, VideoMetadata] | None:
    """Fetches video data using the YouTube API, handling the 50 ID limit."""
    yt_api = ytapi.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    all_video_data: dict[str, VideoMetadata] = {}

    try:
        for i in range(0, len(video_ids), 50):
            chunk_ids = video_ids[i : i + 50]
            video_request = yt_api.videos().list(
                part="snippet,contentDetails", id=",".join(chunk_ids)
            )
            video_response = video_request.execute()

            if video_response.get("items"):
                for item in video_response["items"]:
                    video_id = item.get("id", "NO ID FOUND>")
                    snippet = item.get("snippet", {})
                    all_video_data[video_id] = {
                        "title": snippet.get("title", "<NO TITLE FOUND>"),
                        "description": snippet.get(
                            "description", "<NO DESCRIPTION FOUND>"
                        ),
                        "published_at": yt_datetime_to_epoch(
                            snippet.get("publishedAt", "")
                        ),
                        "duration": duration_to_seconds(
                            item["contentDetails"].get("duration")
                        ),
                    }
        return all_video_data

    except HttpError as e:
        print(f"Error fetching YouTube data: {e}")
        return None
    except KeyError as e:
        print(f"Error accessing a missing key in YouTube data: {e}")
        return None


def process_video(video: CompleteVideoRecord, output_dir: str) -> bool:
    """Processes a single video, saves to JSON."""
    video_id = video["video_id"]
    published_at = float(video.get("published_at", 0))

    if published_at:
        date_obj = date.fromtimestamp(published_at)
        year = str(date_obj.year)
        month = f"{date_obj.month:02d}"
    else:
        year = "unknown"
        month = "unknown"

    subdir_path = path.join(output_dir, year, month)
    makedirs(subdir_path, exist_ok=True)
    output_path = path.join(subdir_path, f"{video_id}.json")

    if path.exists(output_path):
        return False  # Video already processed

    # Fetch the raw transcript data
    raw_transcript_data = get_raw_transcript_data(video_id)

    if raw_transcript_data == video_id:
        return True  # Skip next time
    elif isinstance(raw_transcript_data, list):
        video["transcript_chunks"] = chunk_transcript_with_overlap(
            raw_transcript_data
        )
        if not video["transcript_chunks"]:
            print(
                f"Warning: Transcript for {video_id} was empty after chunking."
            )
            return False  # Skip if chunking resulted in nothing
    else:
        return False

    with open(output_path, "w", encoding="utf-8") as outfile:
        dump(video, outfile, indent=4)

    return False


def _normalize_transcript(
    snippets: Iterable[FetchedTranscriptSnippet],
) -> list[TranscriptSnippet]:
    return [
        {
            "text": snippet.text,
            "start": snippet.start,
            "duration": snippet.duration,
        }
        for snippet in snippets
    ]


def get_raw_transcript_data(
    video_id: str,
) -> list[TranscriptSnippet] | str | None:
    """
    Fetches the raw transcript data from the YouTube Transcript API.
    Returns a list of snippet dictionaries, each with 'text', 'start',
    and 'duration'.
    """
    try:
        yt_transcript_api = YouTubeTranscriptApi()
        fetched = yt_transcript_api.fetch(video_id=video_id, languages=["en"])
        return _normalize_transcript(fetched)
    except Exception as e:
        error = str(e)
        if (
            "Subtitles are disabled for this video" in error
            or "This video is age-restricted" in error
        ):
            return video_id
        elif (
            "No transcripts were found for any of the requested language"
            " codes: ['en']"
        ) in error:
            try:
                print(
                    "  ...Non-English subtitles found, attempting workaround."
                )
                # Get the list of all available transcripts
                yt_transcript_api = YouTubeTranscriptApi()
                new_transcript_list = yt_transcript_api.list(video_id)

                # Find a transcript that is translatable to English
                for transcript in new_transcript_list:
                    if transcript.is_translatable:
                        print(
                            "  -> Found a translatable transcript"
                            f" in '{transcript.language_code}'."
                            " Translating to English."
                        )

                        trans_snippets = transcript.translate("en").fetch()
                        response = _normalize_transcript(trans_snippets)
                        print("  -> Translation and normalization successful.")
                        return response

                # If no translatable transcripts are found after checking
                print(
                    f"  -> No translatable transcripts found for {video_id}"
                    " - adding to skip list."
                )
                return video_id

            except Exception as e:
                print(
                    f"  !! An error occurred during translation attempt for {video_id}: {e}"
                )
        else:
            print(f"Could not retrieve transcript for {video_id}")
        return None


def chunk_transcript_with_overlap(
    transcript_data: list[TranscriptSnippet],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TranscriptChunk]:
    """
    Chunks a transcript into overlapping, semantically aware pieces,
    while preserving the start timestamp for each chunk.

    Args:
        transcript_data: The raw list of dicts from youtube_transcript_api.
        chunk_size: The target size of each text chunk (in characters).
        chunk_overlap: The amount of overlap between chunks (in characters).

    Returns:
        A list of dictionaries, where each dict is a chunk with 'text'
        and 'start'.
    """
    if not transcript_data:
        return []

    # 1. Combine the transcript into a single text block and create a time map.
    full_text = ""
    # This list will store tuples of (character_index, timestamp)
    char_to_time_map = []

    for snippet in transcript_data:
        start_time = snippet["start"]
        text = snippet.get("text", "").strip() + " "  # Add space for joining

        # Store the start time for the beginning of this snippet's text
        char_to_time_map.append((len(full_text), start_time))
        full_text += text

    # 2. Use a robust text splitter to create overlapping chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(full_text)

    # 3. Re-associate each chunk with its original start time.
    final_chunks: list[TranscriptChunk] = []
    current_search_position = 0  # <-- Add this variable

    for chunk_text in text_chunks:
        # Find the start index of the chunk, starting from our last position
        chunk_start_char_index = full_text.find(
            chunk_text, current_search_position
        )

        if chunk_start_char_index == -1:
            # This should rarely happen, but as a fallback, search from the beginning
            chunk_start_char_index = full_text.find(chunk_text)

        # Find the closest timestamp in our map
        chunk_start_time = None
        for char_index, timestamp in char_to_time_map:
            if char_index <= chunk_start_char_index:
                chunk_start_time = timestamp
            else:
                break

        if chunk_start_time is not None:
            final_chunks.append(
                {
                    "text": " ".join(chunk_text.split()),
                    "start": round(chunk_start_time, 2),
                }
            )

        # Update our search position to prevent re-finding the same text
        if chunk_start_char_index != -1:
            current_search_position = chunk_start_char_index + 1

    return final_chunks
