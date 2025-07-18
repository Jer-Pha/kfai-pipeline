import json

import kfai_helpers.config as config
from kfai_helpers.db import get_video_db_data
from kfai_helpers.video import get_youtube_data

VIDEOS_TO_SKIP_FILE = "skipped_videos.json"
videos_ids_to_skip = tuple()

if __name__ == "__main__":
    try:
        with open(VIDEOS_TO_SKIP_FILE, "r") as f:
            video_list = json.load(f)
            failed_ids = tuple(video_list)
    except (json.JSONDecodeError, IOError) as e:
        print(
            f"-> Warning: Could not read or parse {VIDEOS_TO_SKIP_FILE}."
            f" Error: {e}"
        )

    # 1. Get the base metadata from your local database
    db_metadata = get_video_db_data(
        config.SQLITE_DB_PATH, video_ids=failed_ids
    )

    # 2. Enrich with metadata from the YouTube API
    youtube_api_data = get_youtube_data(failed_ids)

    # 3. Combine the two data sources into a final list
    enriched_metadata = []
    for video in db_metadata:
        video_id = video["video_id"]
        if video_id in youtube_api_data:
            # Merge the DB data with the YouTube API data
            video.update(youtube_api_data[video_id])
            enriched_metadata.append(video)
        else:
            print(
                "Warning: Could not find YouTube API data for failed video ID:",
                video_id,
            )

    # 4. Save the complete, enriched metadata
    with open("failures_to_transcribe.json", "w") as f:
        json.dump(enriched_metadata, f, indent=4)

    print(
        "Created failures_to_transcribe.json with enriched data for"
        f" {len(enriched_metadata)} videos."
    )
