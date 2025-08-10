from datetime import date
from json import dump

from kfai.core.paths import RAW_JSON_DIR
from kfai.core.types import CompleteVideoRecord
from kfai.extractors.utils.helpers.transcript import (
    chunk_transcript_with_overlap,
    get_raw_transcript_data,
)


def process_video(video_record: CompleteVideoRecord) -> bool:
    """Processes a single video, saves to JSON."""
    video_id = video_record["video_id"]
    published_at = float(video_record.get("published_at", 0))

    if published_at:
        date_obj = date.fromtimestamp(published_at)
        year = str(date_obj.year)
        month = f"{date_obj.month:02d}"
    else:
        year = "unknown"
        month = "unknown"

    subdir_path = RAW_JSON_DIR / year / month
    subdir_path.mkdir(parents=True, exist_ok=True)
    output_path = subdir_path / f"{video_id}.json"

    if output_path.exists():
        return False  # Video already processed

    # Fetch the raw transcript data
    raw_transcript_data = get_raw_transcript_data(video_id)

    if raw_transcript_data == video_id:
        return True  # Skip next time
    elif isinstance(raw_transcript_data, list):
        video_record["transcript_chunks"] = chunk_transcript_with_overlap(
            raw_transcript_data
        )
        if not video_record["transcript_chunks"]:
            print(
                f"Warning: Transcript for {video_id} was empty after chunking."
            )
            return False  # Skip if chunking resulted in nothing
    else:
        return False

    with output_path.open("w", encoding="utf-8") as f:
        dump(video_record, f, indent=4)

    return False
