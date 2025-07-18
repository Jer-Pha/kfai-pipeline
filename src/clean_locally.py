import os
import json
import logging
import re
import time
from langchain_ollama import OllamaLLM
from traceback import format_exc

from kfai_helpers.utils import format_duration


# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
OLLAMA_MODEL = "llama3.1:8b-instruct-q8_0"
RAW_JSON_DIR = os.path.join(BASE_DIR, "videos")
CLEANED_DIR_NAME = f"videos_cleaned_local-{OLLAMA_MODEL.split(":")[0]}"
CLEANED_JSON_DIR = os.path.join(BASE_DIR, CLEANED_DIR_NAME)
os.makedirs(CLEANED_JSON_DIR, exist_ok=True)

# --- LLM SETUP ---
llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0.1,
    top_p=0.92,
    top_k=40,
    keep_alive=60,
    reasoning=False,
    verbose=False,
)

# --- LOGGING SETUP ---
LOG_LEVEL = logging.WARNING
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_file = os.path.join(LOGS_DIR, "cleaning_process.log")
file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(LOG_LEVEL)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(LOG_LEVEL)
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# -- GLOBAL REGEX COMPILERS ---
_compile = re.compile
_sub_profanity = _compile(r"\[\u00a0__\u00a0\]").sub
_sub_bracket_tags = _compile(r"\[\s*[^]]*?\s*\]").sub
_sub_whitespace = _compile(r"\s+").sub
_sub_chunk = _compile(r"</?CHUNK>").sub
_sub_squotes = _compile(r"[‘’]").sub
_sub_dquotes = _compile(r"[“”]").sub

# --- PROMPT TEMPLATES ---

# Used across all LLM calls
SYSTEM_PROMPT = """
- You are an expert transcription editor for the 'Kinda Funny' and 'Kinda Funny Games' YouTube channels.
- You have been provided a video's metadata and a chunk of the auto-generated transcript.
- Your task is to process the chunk per the FULL INSTRUCTIONS and return the processed string.
- Use the METADATA to bias your corrections.
- You are expected to confidently fix obvious phonetic or name errors using common knowledge, especially when they relate to context found in the title, description, or host names.

FULL INSTRUCTIONS:
Use your contextual knowledge of the hosts, show names, common topics (video games, movies, comics, etc.), and the video's metadata to accomplish the following items while cleaning the chunk's text:
  - Correct phonetic mistakes and spelling errors, especially for names, pop culture references, and brands, even if they’re only approximate matches.
  - Capitalize proper nouns like names, games, and show titles.
  - Do **NOT** change the original meaning, slang, or grammar. Do **NOT** remove filler text. Only correct clear errors.
  - If a word or phrase is ambiguous, leave it as is.
  - **CRITICAL RULE**: Do NOT discard or omit any text, even if it appears to be an incomplete sentence (fragment). If it is just a fragment, clean it for spelling and return the cleaned fragment. Do not try to make it a full sentence.
  - The RESPONSE should be the cleaned chunk and nothing else — do **NOT** include thoughts, explanations, or commentary.

EXAMPLES OF POSSIBLE CHANGES (INPUT → CLEANED):
  - "Tim Geddes" → "Tim Gettys"
  - "wing grety" → "Wayne Gretzky"
  - "final fantasy versus 13" → "Final Fantasy Versus XIII"
  - "game over greggy" → "GameOverGreggy"
"""

# Used with new data each LLM call
USER_PROMPT = """
METADATA CONTEXT:
{metadata}

RAW CHUNK:
{chunk}

RESPONSE:
"""


# --- HELPER FUNCTIONS ---
def _get_file_paths(root: str, filename: str) -> tuple[str, str, str]:
    """Constructs and returns the raw and cleaned file paths."""
    raw_path = os.path.join(root, filename)
    relative_path = os.path.relpath(raw_path, RAW_JSON_DIR)
    cleaned_path = os.path.join(CLEANED_JSON_DIR, relative_path)
    return raw_path, relative_path, cleaned_path


def _load_raw_data(raw_path: str) -> dict:
    """Loads and returns the JSON data from a given file path."""
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            return dict(json.load(f))
    except (json.JSONDecodeError, IOError):
        logger.error(f"Failed to load or parse source file: {raw_path}")
        logger.error(format_exc())
        return {}


def _check_data_integrity(
    raw_data: dict, cleaned_data: dict, relative_path: str
) -> bool:
    """Performs data integrity checks and returns True if all pass."""
    if not cleaned_data or set(cleaned_data.keys()) != set(raw_data.keys()):
        logger.warning(
            f"Data integrity check failed for {relative_path}: Key mismatch"
            " or empty data."
        )
        return False

    raw_chunk_count = len(raw_data["transcript_chunks"])
    cleaned_chunk_count = len(cleaned_data["transcript_chunks"])

    if cleaned_chunk_count != raw_chunk_count:
        error_msg = (
            f"Data integrity error: {raw_chunk_count} chunks sent,"
            f" but received {cleaned_chunk_count} back."
        )
        logger.error(f"In {relative_path}: {error_msg}")
        return False

    return True


def _save_cleaned_data(cleaned_path: str, cleaned_video_data: dict) -> bool:
    """Saves the cleaned data to a JSON file, creating directories if needed."""
    try:
        cleaned_dir = os.path.dirname(cleaned_path)
        os.makedirs(cleaned_dir, exist_ok=True)
        with open(cleaned_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_video_data, f, indent=4)
        print(f"  -> Successfully cleaned and saved to {cleaned_path}")
        return True
    except Exception:
        logger.error(f"Failed to save cleaned file: {cleaned_path}")
        logger.error(format_exc())
        return False


def _clean_text_chunk(text):
    # Fix transcript profanity reference
    text = _sub_profanity("****", text)

    # Remove filler
    text = text.replace("\u200b", "").replace("\xa0", " ")
    text = text.replace(">>", "")

    # Regex cleanup
    text = _sub_bracket_tags("", text)
    text = _sub_whitespace(" ", text).strip()

    return text


def _clean_response(response: str) -> str:
    """Clean common LLM inconsistencies in the response."""
    response = response.split("Here is the cleaned chunk:")[-1]
    response = response.split("Here's the cleaned chunk:")[-1]
    response = response.split("</think>")[-1]
    response = _sub_chunk("", response)
    response = _sub_squotes("'", response)
    response = _sub_dquotes('"', response)
    return response


# --- CORE FUNCTION ---
def _clean_transcript(video_data: dict, relative_path: str) -> dict | None:
    """Cleans a video's transcript with Ollama, one chunk at a time."""
    try:
        profile_start = time.time()

        chunk_count = len(video_data["transcript_chunks"])
        pluralization = "s" if chunk_count != 1 else ""
        print(f"  >> {chunk_count} chunk{pluralization} found", end="")
        cleaned_video_data = {
            k: v for k, v in video_data.items() if k != "transcript_chunks"
        }
        metadata = json.dumps(cleaned_video_data)
        cleaned_video_data["transcript_chunks"] = []

        _invoke_llm = llm.invoke

        _clean = _clean_response
        _format_duration = format_duration

        for chunk in video_data["transcript_chunks"]:
            text = _clean_text_chunk(chunk["text"])

            user_prompt = USER_PROMPT.format(
                metadata=metadata,
                chunk=text,
            )

            try:
                response = _invoke_llm(
                    [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": user_prompt},
                    ]
                )

                response = _clean(response)

                cleaned_video_data["transcript_chunks"].append(
                    {
                        "text": response.strip(),
                        "start": chunk["start"],
                    }
                )
            except:
                logger.error(
                    f"LLM call failed on chunk in {relative_path} starting "
                    f"at {chunk['start']}s."
                )
                logger.error(format_exc())
                print(
                    f"  !! LLM call failed. See {LOGS_DIR} for details."
                    " Skipping video."
                )

                return None

        profile_end = time.time()
        print(f"processed in {_format_duration(profile_end - profile_start)}.")

        return cleaned_video_data
    except:
        logger.error(
            "An unexpected error occurred in _clean_transcript"
            f" for {relative_path}."
        )
        logger.error(format_exc())
        print(
            f"  !! An unexpected error occurred. See {LOGS_DIR} for"
            " details. Skipping video."
        )
        return None


# --- MAIN LOGIC ---
if __name__ == "__main__":
    print(
        f"Starting local cleaning process. Raw source: '{RAW_JSON_DIR}',"
        f" Cleaned destination: '{CLEANED_JSON_DIR}'"
    )

    try:
        # 1. Loop through raw video directories
        for root, _, files in os.walk(RAW_JSON_DIR):
            # 2. Loop through files in each directory
            for filename in files:
                # Ignore non-JSON files
                if not filename.endswith(".json"):
                    continue

                # 3. Get file paths
                raw_path, relative_path, cleaned_path = _get_file_paths(
                    root, filename
                )

                # Skip videos that have already been cleaned
                if os.path.exists(cleaned_path):
                    continue

                print("\n" + "=" * 50)
                print(f"--- Processing {relative_path} ---")

                # 4. Load video metadata and transcripts into dict
                video_data = _load_raw_data(raw_path)

                # Skip videos that don't have a transcript
                if not video_data.get("transcript_chunks"):
                    logger.warning(f"{raw_path} does not have a transcript.")
                    continue

                # 5. Clean the transcript (CORE FUNCTION)
                cleaned_video_data = _clean_transcript(
                    video_data, relative_path
                )

                # 6. Verify integrity of the cleaned data
                data_is_valid = _check_data_integrity(
                    video_data, cleaned_video_data, relative_path
                )
                if not data_is_valid:
                    continue

                # 7. Save cleaned data to JSON file
                _save_cleaned_data(cleaned_path, cleaned_video_data)

        else:
            # 8. Finish
            print("\nCleaning process complete.")

    except:
        logger.critical(
            "A critical, unhandled error occurred in the main execution loop."
        )
        logger.critical(format_exc())
        print(f"\n!! A critical error occurred. See {LOGS_DIR} for details.")
        raise
