from traceback import format_exc

from langchain_ollama import OllamaLLM

from common.config import CLEANED_JSON_DIR, LOGS_DIR, RAW_JSON_DIR
from transform.utils.cleaning import clean_transcript
from transform.utils.config import CLEANING_MODEL
from transform.utils.helpers import (
    check_data_integrity,
    load_raw_data,
    save_cleaned_data,
)
from transform.utils.logger_config import setup_logging

logger = setup_logging()


def run():
    llm = OllamaLLM(
        model=CLEANING_MODEL,
        temperature=0.1,
        top_p=0.92,
        top_k=40,
        keep_alive=60,
        reasoning=False,
        verbose=False,
    )

    CLEANED_JSON_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting local cleaning process. Raw source: '{RAW_JSON_DIR}',"
        f" Cleaned destination: '{CLEANED_JSON_DIR}'"
    )
    _clean_transcript = clean_transcript

    try:
        for file_path in RAW_JSON_DIR.rglob("*.json"):
            relative_path = file_path.relative_to(RAW_JSON_DIR)
            cleaned_path = CLEANED_JSON_DIR / relative_path

            # Skip videos that have already been cleaned
            if cleaned_path.exists():
                continue

            print("\n" + "=" * 50)
            print(f"--- Processing {relative_path} ---")

            # Load video metadata and transcripts into dict
            video_data = load_raw_data(file_path)

            # Skip videos that don't have a transcript
            if not video_data or not video_data.get("transcript_chunks"):
                logger.warning(f"{file_path} does not have a transcript.")
                continue

            # Clean the transcript
            cleaned_video_data = _clean_transcript(
                video_data, relative_path, llm
            )
            assert cleaned_video_data is not None

            # Verify integrity of the cleaned data
            data_is_valid = check_data_integrity(
                video_data, cleaned_video_data, relative_path
            )
            if not data_is_valid:
                continue

            # Save cleaned data to JSON file
            save_cleaned_data(cleaned_path, cleaned_video_data)

        else:
            print("\nCleaning process complete.")

    except:
        logger.critical(
            "A critical, unhandled error occurred in the main execution loop."
        )
        logger.critical(format_exc())
        print(f"\n!! A critical error occurred. See {LOGS_DIR} for details.")
        raise
