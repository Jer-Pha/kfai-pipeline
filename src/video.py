import googleapiclient.discovery as ytapi
from googleapiclient.errors import HttpError
from json import dump
from os import makedirs, path
from datetime import date

import config
from transcript import chunk_transcript_with_overlap, get_raw_transcript_data
from utils import date_to_timestamp, duration_to_seconds


def get_youtube_data(video_ids):
    """Fetches video data using the YouTube API, handling the 50 ID limit."""
    youtube = ytapi.build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY)

    all_video_data = {}

    try:
        for i in range(0, len(video_ids), 50):
            chunk_ids = video_ids[i : i + 50]
            video_request = youtube.videos().list(
                part="snippet,contentDetails", id=",".join(chunk_ids)
            )
            video_response = video_request.execute()

            if video_response.get("items"):
                for item in video_response["items"]:
                    video_id = item.get("id")
                    all_video_data[video_id] = {
                        "title": item["snippet"].get("title"),
                        "description": item["snippet"].get("description"),
                        "published_at": date_to_timestamp(
                            item["snippet"].get("publishedAt")
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


def process_video(video, output_dir):
    """Processes a single video, saves to JSON."""
    video_id = video["video_id"]
    published_at = video.get("published_at", "")

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
    elif raw_transcript_data:
        # Chunk it using the new, improved method
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

    # ... (code to save the JSON file is perfect) ...
    # I changed the key to "transcript_chunks" to be more descriptive
    with open(output_path, "w", encoding="utf-8") as outfile:
        dump(video, outfile, indent=4)

    return False
