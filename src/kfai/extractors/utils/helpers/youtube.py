import math
from datetime import datetime
from pathlib import Path

import googleapiclient.discovery as ytapi
from googleapiclient.errors import HttpError
from isodate import parse_duration
from isodate.duration import Duration
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func

from kfai.extractors.utils.config import (
    BASE_YT_DLP_OPTIONS,
    CHUNK_THRESHOLD_SECONDS,
    TEMP_DATA_DIR,
    YOUTUBE_API_KEY,
    YT_COOKIE_FILE,
)
from kfai.extractors.utils.types import VideoMetadata


def yt_datetime_to_epoch(data: str) -> int:
    """Converts YouTube API ISO datetime to Unix epoch timestamp."""
    if not data:
        return 0
    return int(datetime.fromisoformat(data.replace("Z", "+00:00")).timestamp())


def duration_to_seconds(duration: Duration) -> int:
    """Converts YouTube API ISO duration to total seconds."""
    return int(parse_duration(duration).total_seconds())


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


def download_audio_handler(video_id: str, duration: int) -> list[Path] | None:
    """
    Downloads audio for a video in one or more chunks, using a single, unified logic.
    - Videos are always processed in chunks, even if there's only one.
    - Handles authentication using a cookie file for all requests.
    Returns a tuple: (list_of_audio_paths, chunk_duration_in_seconds)
    """
    if not duration:
        print(f"  !! Could not find duration for {video_id}. Cannot download.")
        return None

    base_options = BASE_YT_DLP_OPTIONS.copy()

    if YT_COOKIE_FILE.exists():
        base_options["cookiefile"] = YT_COOKIE_FILE

    chunk_paths = []
    num_chunks = math.ceil(duration / CHUNK_THRESHOLD_SECONDS)

    for i in range(num_chunks):
        start_time = i * CHUNK_THRESHOLD_SECONDS
        end_time = start_time + CHUNK_THRESHOLD_SECONDS
        chunk_path = TEMP_DATA_DIR / f"{video_id}_chunk_{i+1}.m4a"

        chunk_paths.append(chunk_path)

        if chunk_path.exists():
            print(
                f"  -> Chunk {i+1}/{num_chunks} for {video_id} already exists."
                " Skipping download."
            )
            continue

        print(f"  -> Downloading chunk {i+1}/{num_chunks}...")
        options = base_options.copy()
        options["outtmpl"] = chunk_path
        options["download_ranges"] = download_range_func(
            None, [(start_time, end_time)]
        )

        try:
            with YoutubeDL(options) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as e:
            print(f"  !! Error downloading chunk {i+1} for {video_id}: {e}")
            if chunk_path.exists():
                chunk_path.unlink()
            return None

    return chunk_paths
