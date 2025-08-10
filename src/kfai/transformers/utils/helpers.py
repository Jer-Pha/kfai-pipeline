import json
import logging
import re
from pathlib import Path
from traceback import format_exc

from kfai.core.types import CompleteVideoRecord, TranscriptChunk

logger = logging.getLogger()

# -- GLOBAL REGEX COMPILERS ---
_compile = re.compile
_sub_profanity = _compile(r"\[\u00a0__\u00a0\]").sub
_sub_bracket_tags = _compile(r"\[\s*[^]]*?\s*\]").sub
_sub_whitespace = _compile(r"\s+").sub
_sub_chunk = _compile(r"</?CHUNK>").sub
_sub_squotes = _compile(r"[‘’]").sub
_sub_dquotes = _compile(r"[“”]").sub


def load_raw_data(file_path: Path) -> CompleteVideoRecord | None:
    """Loads and returns the JSON data from a given file path."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            video_data: CompleteVideoRecord = json.load(f)
            return video_data
    except (json.JSONDecodeError, IOError):
        logger.error(f"Failed to load or parse source file: {file_path}")
        logger.error(format_exc())
        return None


def check_data_integrity(
    raw_data: CompleteVideoRecord,
    cleaned_data: CompleteVideoRecord,
    relative_path: Path,
) -> bool:
    """Performs data integrity checks and returns True if all pass."""
    if not cleaned_data or set(cleaned_data.keys()) != set(raw_data.keys()):
        logger.warning(
            f"Data integrity check failed for {relative_path}: Key mismatch"
            " or empty data."
        )
        return False

    assert raw_data["transcript_chunks"] is not None
    raw_transcript_chunks: list[TranscriptChunk] = raw_data[
        "transcript_chunks"
    ]
    raw_chunk_count = len(raw_transcript_chunks)
    cleaned_chunk_count = len(raw_transcript_chunks)

    if cleaned_chunk_count != raw_chunk_count:
        error_msg = (
            f"Data integrity error: {raw_chunk_count} chunks sent,"
            f" but received {cleaned_chunk_count} back."
        )
        logger.error(f"In {relative_path}: {error_msg}")
        return False

    return True


def save_cleaned_data(
    cleaned_path: Path, cleaned_video_data: CompleteVideoRecord
) -> bool:
    """Saves the cleaned data to a JSON file, creating directories if needed."""
    try:
        cleaned_dir = cleaned_path.parent
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        with cleaned_path.open("w", encoding="utf-8") as f:
            json.dump(cleaned_video_data, f, indent=4)
        print(f"  -> Successfully cleaned and saved to {cleaned_path}")
        return True
    except Exception:
        logger.error(f"Failed to save cleaned file: {cleaned_path}")
        logger.error(format_exc())
        return False


def clean_text_chunk(text: str) -> str:
    # Fix transcript profanity reference
    text = _sub_profanity("****", text)

    # Remove filler
    text = text.replace("\u200b", "").replace("\xa0", " ")
    text = text.replace(">>", "")

    # Regex cleanup
    text = _sub_bracket_tags("", text)
    text = _sub_whitespace(" ", text).strip()

    return text


def clean_response(response: str) -> str:
    """Clean common LLM inconsistencies in the response."""
    response = response.split("Here is the cleaned chunk:")[-1]
    response = response.split("Here's the cleaned chunk:")[-1]
    response = response.split("</think>")[-1]
    response = _sub_chunk("", response)
    response = _sub_squotes("'", response)
    response = _sub_dquotes('"', response)
    return response
