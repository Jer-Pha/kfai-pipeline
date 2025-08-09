import json
from random import uniform
from time import sleep
from typing import cast

from common.config import RAW_JSON_DIR
from common.types import CompleteVideoRecord
from extract.utils.config import SQLITE_DB_PATH, VIDEOS_TO_SKIP_FILE
from extract.utils.helpers.database import (
    create_local_sqlite_db,
    get_video_db_data,
)
from extract.utils.helpers.processing import process_video
from extract.utils.helpers.youtube import get_youtube_data


def run():
    videos_ids_to_skip = set()

    if VIDEOS_TO_SKIP_FILE.exists():
        try:
            with open(VIDEOS_TO_SKIP_FILE, "r") as f:
                video_list = json.load(f)
                videos_ids_to_skip = set(video_list)
            print(
                f"-> Found and loaded {len(videos_ids_to_skip)} previously"
                " processed video IDs to skip this run."
            )
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"-> Warning: Could not read or parse {VIDEOS_TO_SKIP_FILE}."
                f" Starting with an empty set. Error: {e}"
            )
            videos_ids_to_skip = set()
    else:
        print(
            f"-> {VIDEOS_TO_SKIP_FILE} not found. Will create it after the"
            " first successful video processing."
        )

    # Check prevents re-exporting from MySQL on every run.
    if not SQLITE_DB_PATH.exists():
        print(f"Local SQLite database not found at {SQLITE_DB_PATH}.")
        print("Creating it now from the MySQL source...")
        create_local_sqlite_db()
        print("Local SQLite database created successfully.")
    else:
        print(f"Found local SQLite database at {SQLITE_DB_PATH}.")

    print("Finding new videos to process...")

    # Get all video IDs from the source database
    all_db_videos = get_video_db_data(SQLITE_DB_PATH)
    db_video_ids = {v["video_id"] for v in all_db_videos}

    # Scan the output directory to see which videos have been processed
    output_directory = RAW_JSON_DIR
    output_directory.mkdir(parents=True, exist_ok=True)
    processed_video_ids = set()

    for file_path in output_directory.rglob("*.json"):
        video_id = file_path.stem
        processed_video_ids.add(video_id)

    # Find the difference and convert it to a tuple
    new_video_ids = list(
        db_video_ids.difference(processed_video_ids.union(videos_ids_to_skip))
    )

    if not new_video_ids:
        print("No new videos to process. All up to date.")
    else:
        print(f"Found {len(new_video_ids)} new videos to process.")

        # Fetch metadata from the DB for the new videos
        new_video_metadata = get_video_db_data(
            SQLITE_DB_PATH, video_ids=new_video_ids
        )

        # Enrich with data from the YouTube API
        youtube_api_data = get_youtube_data(new_video_ids)

        if youtube_api_data is not None:
            # Process and save
            for video in new_video_metadata:
                video_id = video["video_id"]

                if video_id in videos_ids_to_skip:
                    continue

                if video_id in youtube_api_data:
                    # Merge the DB data with the YouTube API data
                    video_record = cast(
                        CompleteVideoRecord,
                        dict(video) | youtube_api_data[video_id],
                    )

                    # Rate limiting
                    sleep_duration = uniform(2, 5)  # Wait 2 to 5 seconds
                    print(
                        f"   ...waiting for {sleep_duration:.2f} seconds to avoid"
                        " rate-limiting."
                    )
                    sleep(sleep_duration)

                    # Process video
                    print(f"Processing video: {video_id}")
                    skip_next_run = process_video(
                        video_record, output_directory
                    )
                    if skip_next_run:
                        videos_ids_to_skip.add(video_id)
                        try:
                            with open(VIDEOS_TO_SKIP_FILE, "w") as f:
                                # Convert the set to a list to make it JSON-serializable
                                json.dump(
                                    list(videos_ids_to_skip), f, indent=4
                                )
                        except IOError as e:
                            print(
                                "FATAL: Could not write to log"
                                f" file {VIDEOS_TO_SKIP_FILE}. Error: {e}"
                            )
                else:
                    print(
                        "Warning: Could not find YouTube API data for new video"
                        f" ID: {video_id}"
                    )

    print("Processing complete.")
