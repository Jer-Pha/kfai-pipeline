import json
from datetime import date
from pathlib import Path

import whisper

from kfai.core.paths import RAW_JSON_DIR
from kfai.core.types import CompleteVideoRecord
from kfai.extractors.utils.config import (
    CHUNK_THRESHOLD_SECONDS,
    FAILED_VIDEOS_FILE,
    SKIP_LIST,
    WHISPER_MODEL,
    WHISPER_OUTPUT_DIR,
)
from kfai.extractors.utils.helpers.transcript import (
    chunk_transcript_with_overlap,
    transcribe_with_whisper,
)
from kfai.extractors.utils.helpers.youtube import download_audio_handler


# --- Main Execution ---
def run() -> None:
    WHISPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FAILED_VIDEOS_FILE.exists():
        print(f"Error: Input file '{FAILED_VIDEOS_FILE}' not found.")
        print("Please create it with a list of video metadata to process.")
    else:
        print("Loading Whisper model (this may take a moment)...")
        try:
            whisper_model = whisper.load_model(WHISPER_MODEL)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Fatal: Could not load Whisper model. Error: {e}")
            whisper_model = None

        with FAILED_VIDEOS_FILE.open("r", encoding="utf-8") as f:
            videos_to_process: list[CompleteVideoRecord] = json.load(f)

        size = len(videos_to_process)

        print(f"Found {size} videos to transcribe with Whisper.")

        for i, video_metadata in enumerate(videos_to_process):
            video_id = video_metadata["video_id"]

            if video_id in SKIP_LIST:
                continue

            # 1. Check if video has already been transcribed
            published_at = video_metadata["published_at"]
            date_obj = date.fromtimestamp(published_at)
            year = str(date_obj.year)
            month = f"{date_obj.month:02d}"

            subdir_path = RAW_JSON_DIR / year / month
            subdir_path.mkdir(parents=True, exist_ok=True)
            output_path = subdir_path / f"{video_id}.json"

            if output_path.exists():
                continue

            print(f"\n--- Processing {i+1}/{size}: {video_id} ---")

            raw_transcript_data = []
            audio_files_to_cleanup: list[Path] = []

            # 2. Download audio chunks
            duration = video_metadata["duration"]
            chunk_paths = download_audio_handler(video_id, duration)
            if not chunk_paths:
                print(
                    f"  !! Failed to download chunks for {video_id}. Skipping."
                )
                continue

            audio_files_to_cleanup.extend(chunk_paths)

            # 3. Transcribe chunks
            for idx, chunk_path in enumerate(chunk_paths):
                print(f"  -> Transcribing chunk {idx+1}/{len(chunk_paths)}...")
                segments = transcribe_with_whisper(chunk_path, whisper_model)
                if segments:
                    time_offset = idx * CHUNK_THRESHOLD_SECONDS
                    for seg in segments:
                        seg["start"] += time_offset
                    raw_transcript_data.extend(segments)

            if not raw_transcript_data:
                print("  !! No transcript data generated... skipping.")
                # Clean up any downloaded files before skipping
                for file_path in audio_files_to_cleanup:
                    if file_path.exists():
                        file_path.unlink()
                continue

            # 4. Chunk the transcript
            final_chunks = chunk_transcript_with_overlap(raw_transcript_data)

            # 5. Construct the final JSON object in the correct format
            final_json_output = video_metadata.copy()
            final_json_output["transcript_chunks"] = final_chunks

            # 6. Save the output file, creating the directory if necessary
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(final_json_output, f, indent=4)

            print(f"  -> Successfully transcribed and saved to {output_path}")

            # 7. Clean up the audio file to save space
            print("  -> Cleaning up audio files...")
            for file_path in audio_files_to_cleanup:
                if file_path.exists():
                    file_path.unlink()

        print("\nWhisper transcription process complete.")
