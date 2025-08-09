from datetime import datetime

import googleapiclient.discovery as ytapi
from googleapiclient.errors import HttpError
from isodate import parse_duration
from isodate.duration import Duration

from extract.utils.config import YOUTUBE_API_KEY
from extract.utils.types import VideoMetadata


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
