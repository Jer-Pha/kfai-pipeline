from datetime import date
from json import dump
from os import makedirs, path

from common.types import CompleteVideoRecord


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
