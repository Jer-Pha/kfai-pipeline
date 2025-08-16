from traceback import format_exc

from langchain_ollama import OllamaLLM

from kfai.core.paths import CLEANED_JSON_DIR, LOGS_DIR, RAW_JSON_DIR
from kfai.transformers.utils.cleaning import clean_transcript
from kfai.transformers.utils.config import CLEANING_MODEL
from kfai.transformers.utils.helpers import (
    check_data_integrity,
    load_raw_data,
    save_cleaned_data,
)
from kfai.transformers.utils.logger_config import setup_logging

logger = setup_logging()


def run() -> None:
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

    # Reduce import calls
    _load_raw_data = load_raw_data
    _clean_transcript = clean_transcript
    _check_data_integrity = check_data_integrity
    _save_cleaned_data = save_cleaned_data

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
            video_data = _load_raw_data(file_path)

            # Skip videos that don't have a transcript
            if not video_data or not video_data.get("transcript_chunks"):
                logger.warning(f"{file_path} does not have a transcript.")
                continue

            # Clean the transcript
            cleaned_video_data = _clean_transcript(
                video_data, relative_path, llm
            )

            if cleaned_video_data is None:
                continue

            # Verify integrity of the cleaned data
            data_is_valid = _check_data_integrity(
                video_data, cleaned_video_data, relative_path
            )
            if not data_is_valid:
                continue

            # Save cleaned data to JSON file
            _save_cleaned_data(cleaned_path, cleaned_video_data)

        else:
            print("\nCleaning process complete.")

    except:
        logger.critical(
            "A critical, unhandled error occurred in the main execution loop."
        )
        logger.critical(format_exc())
        print(f"\n!! A critical error occurred. See {LOGS_DIR} for details.")
        raise
