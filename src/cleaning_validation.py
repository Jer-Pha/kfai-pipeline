import os
import json
import google.generativeai as genai
from time import sleep
from traceback import format_exc

from config import GEMINI_API_KEY

# --- Configuration ---
RAW_JSON_DIR = "videos"
CLEANED_JSON_DIR = "videos_cleaned"
RESULTS_FILE = "cleaning_validation_results.json"
GEMMA_API_MODEL = "gemma-3-12b-it"
GEMINI_API_MODEL = "gemini-2.5-flash-lite-preview-06-17"
GEMMA_SLEEP_DURATION = 2.1
GEMINI_SLEEP_DURATION = 4.1 - GEMMA_SLEEP_DURATION

# --- General Setup ---
metadata_cache_id = ""
metadata_cache = {}

# --- Setup Models ---
genai.configure(api_key=GEMINI_API_KEY)
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- Setup Gemma Model ---
gemma_model = genai.GenerativeModel(GEMMA_API_MODEL)

GEMMA_PROMPT = """
You are a meticulous quality assurance editor for the YouTube channels 'Kinda Funny' and 'Kinda Funny Games'.
Your task is to compare a "Raw Text" snippet with a "Cleaned Text" snippet and determine if the cleaning process was valid.

A "VALID" change corrects spelling, punctuation, or capitalization without changing the core meaning, intent, or important contextual information (like brand names or inside jokes).
A "FLAGGED" change alters the meaning, removes important context, or "over-corrects" informal language into something generic and less accurate.

Analyze the following two text snippets:

---
RAW TEXT:
"{raw_text}"
---
CLEANED TEXT:
"{cleaned_text}"
---

Based on the rules above, is the change from Raw Text to Cleaned Text valid?
Respond with ONLY the word "VALID" or "FLAGGED".
"""

# --- Setup Gemini Model ---
gemini_model = genai.GenerativeModel(GEMINI_API_MODEL)

GEMINI_PROMPT = """
You are the final arbiter of quality for a YouTube transcription cleaning process.
You previously cleaned a "Raw Text" to produce a "Cleaned Text". A less advanced model has flagged this change as potentially problematic.
Your task is to make a final judgment.

Use your full contextual knowledge of the 'Kinda Funny' and 'Kinda Funny Games' channels as well as the provided video metadata.

- If the cleaning was a valid and good correction (fixing spelling, phonetics, or names), respond with ONLY the word "VALID".
- If the cleaning was an error and changed the meaning or removed important context, respond with ONLY the word "FLAGGED".

---
METADATA CONTEXT:
Show Name: {show_name}
Video Title: {title}
Hosts: {hosts}
---
RAW TEXT:
"{raw_text}"
---
CLEANED TEXT:
"{cleaned_text}"
---

Final Judgment (VALID or FLAGGED):
"""


# --- Helper Functions ---
def load_results():
    """Loads existing results, creating the file if it doesn't exist."""
    if not os.path.exists(RESULTS_FILE):
        # Create a new file with the correct structure
        initial_data = {
            "VALID": [],
            "FLAGGED": [],
            "ERROR": [],
            "MISMATCHED": [],
        }
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=4)
        return initial_data

    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {RESULTS_FILE}. Exiting...")
        exit()  # Exit to prevent overwriting a corrupted file


def save_results(results_data):
    """Saves the results data to the JSON file."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4)


def get_video_metadata(video_id):
    """A helper to load the full metadata for a video."""
    global metadata_cache_id, metadata_cache

    if metadata_cache_id != video_id:
        metadata_cache_id = video_id

        for root, _, files in os.walk(RAW_JSON_DIR):
            if f"{video_id}.json" in files:
                with open(
                    os.path.join(root, f"{video_id}.json"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    metadata_cache = json.load(f)
                    break
        else:
            metadata_cache = {}

    return metadata_cache


def get_validated_ids(results_data):
    """Gets a set of all video IDs that have been processed, regardless of status."""
    valid_ids = set(results_data.get("VALID", []))
    # For FLAGGED, we need to extract the video_id from each dictionary
    flagged_ids = {
        item["video_id"] for item in results_data.get("FLAGGED", [])
    }
    error_ids = set(results_data.get("ERROR", []))
    mismatched_ids = set(results_data.get("MISMATCHED", []))
    return valid_ids.union(flagged_ids, error_ids, mismatched_ids)


def validate_chunk(raw_text, cleaned_text):
    """Makes the API call to Gemma to validate a single chunk."""
    if raw_text == cleaned_text:
        return "VALID"

    prompt = GEMMA_PROMPT.format(raw_text=raw_text, cleaned_text=cleaned_text)

    try:
        response = gemma_model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(temperature=0.0),
            safety_settings=safety_settings,
        )
        result_text = response.text.strip().upper()
        if result_text in ["VALID", "FLAGGED"]:
            return result_text
        else:
            # If the model returns something unexpected, count it as an error.
            print(f"  -> Unexpected API response: '{result_text}'")
            return "ERROR"
    except:
        error = format_exc()
        if "PROHIBITED_CONTENT" in error:
            return "FLAGGED"
        print(f"  !! API Error during validation: {error}")
        return "ERROR"


def audit_flagged_chunk(video_id, raw_text, cleaned_text):
    metadata = get_video_metadata(video_id)
    if not metadata:
        print(
            "  !! Could not find metadata for audit prompt. Defaulting to ERROR."
        )
        return "ERROR"

    prompt = GEMINI_PROMPT.format(
        show_name=metadata.get("show_name", "N/A"),
        title=metadata.get("title", "N/A"),
        hosts=", ".join(metadata.get("hosts", [])),
        raw_text=raw_text,
        cleaned_text=cleaned_text,
    )

    try:
        response = gemini_model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(temperature=0.0),
            safety_settings=safety_settings,
        )
        result_text = response.text.strip().upper()
        if result_text in ["VALID", "FLAGGED"]:
            return result_text
        else:
            print(f"  -> Unexpected Gemini response: '{result_text}'")
            return "ERROR"
    except:
        error = format_exc()
        if "PROHIBITED_CONTENT" in error:
            return "FLAGGED"
        print(f"  !! API Error during Gemini audit: {error}")
        return "ERROR"


# --- Main Execution Logic ---
if __name__ == "__main__":
    results = load_results()
    validated_ids = get_validated_ids(results)
    print(f"Loaded {len(validated_ids)} previously validated video IDs.")

    for root, _, files in os.walk(RAW_JSON_DIR):
        for filename in files:
            if not filename.endswith(".json"):
                continue

            video_id = filename.split(".json")[0]

            if video_id in validated_ids:
                continue

            print(f"\n--- Validating video: {video_id} ---")

            raw_path = os.path.join(root, filename)
            relative_path = os.path.relpath(raw_path, RAW_JSON_DIR)
            cleaned_path = os.path.join(CLEANED_JSON_DIR, relative_path)

            if not os.path.exists(cleaned_path):
                print(
                    f"  -> Skipping: Cleaned file not found at {cleaned_path}"
                )
                continue

            with open(raw_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            with open(cleaned_path, "r", encoding="utf-8") as f:
                cleaned_data = json.load(f)

            raw_chunks = raw_data.get("transcript_chunks", [])
            cleaned_chunks = cleaned_data.get("transcript_chunks", [])

            raw_size = len(raw_chunks)

            if raw_size != len(cleaned_chunks):
                print(
                    "  -> ERROR: Mismatch in chunk count. Saving as MISMATCHED."
                )
                results["MISMATCHED"].append(video_id)
                save_results(results)
                validated_ids.add(video_id)
                continue

            is_flagged_or_error = False
            for i in range(raw_size):
                raw_text = raw_chunks[i]["text"]
                cleaned_text = cleaned_chunks[i]["text"]

                chunk_status = validate_chunk(raw_text, cleaned_text)

                if chunk_status == "FLAGGED":
                    print("* ", end="")
                    chunk_status = audit_flagged_chunk(
                        video_id, raw_text, cleaned_text
                    )

                    sleep(GEMINI_SLEEP_DURATION)

                print(f"  Chunk {i+1}/{raw_size} status: {chunk_status}")

                if chunk_status == "FLAGGED":
                    is_flagged_or_error = True
                    flag_data = {
                        "video_id": video_id,
                        "chunk_index": i,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_text,
                    }
                    results["FLAGGED"].append(flag_data)
                    print(f"  -> FLAGGED chunk found. Logging details.")
                elif chunk_status == "ERROR":
                    is_flagged_or_error = True
                    results["ERROR"].append(video_id)
                    print(
                        f"  -> API error found. Marking video '{video_id}' as ERROR."
                    )
                    break

                sleep(GEMMA_SLEEP_DURATION)

            # After checking all chunks, if none were flagged or errored, mark as VALID
            if not is_flagged_or_error:
                results["VALID"].append(video_id)
                print(f"-> Finished validation for {video_id}. Status: VALID.")

            # Save the results and update the set of processed IDs
            save_results(results)
            validated_ids.add(video_id)

    print("\nValidation process complete.")
