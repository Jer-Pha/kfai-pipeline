import os
import json
import google.generativeai as genai
import re
from copy import deepcopy
from time import sleep
from traceback import format_exc

from config import GEMINI_API_KEY

# --- Configuration ---
RAW_JSON_DIR = "videos"
CLEANED_JSON_DIR = "videos_cleaned"
API_CHUNK_SIZE = 25
SLEEP_DURATION = 6.1
QUOTA_MULTIPLIER = 0.4
GEMINI_API_MODEL, QUOTA_LIMIT = "gemini-2.0-flash", 200
# GEMINI_API_MODEL, QUOTA_LIMIT = "gemini-2.5-flash-preview-05-20", 250
os.makedirs(CLEANED_JSON_DIR, exist_ok=True)
genai.configure(api_key=GEMINI_API_KEY)

# --- Set up the Gemini Model ---
model = genai.GenerativeModel(GEMINI_API_MODEL)
MODEL_CONFIG = genai.types.GenerationConfig(
    temperature=0.0,
    response_mime_type="application/json",
    response_schema=list[str],
)
MODEL_SAFETY = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

CLEANING_PROMPT = """
METADATA:
{metadata}

TRANSCRIPT CHUNKS:
{transcript_chunks}

TASK DESCRIPTION:
- You are an expert transcription editor for the 'Kinda Funny' and 'Kinda Funny Games' YouTube channels.
- You have been provided JSON data for single video's metadata and transcript chunks.
- Your task is to process EACH chunk per the FULL INSTRUCTIONS and provide all processed chunks as a Python List of strings.
- You MUST return the EXACT SAME number of chunks as you were given. If you are given 25 chunks, you must return a list containing 25 strings.

FULL INSTRUCTIONS:
Use your contextual knowledge of the hosts, show names, common topics (video games, movies, comics, etc.), and the video's metadata to accomplish the following items for EACH chunk:
  - Correct common phonetic mistakes and spelling errors.
  - Capitalize proper nouns like names, games, and show titles.
  - **CRITICAL RULE: Do NOT discard or omit any chunks, even if they appear to be incomplete sentences or fragments. If a chunk is just a fragment, clean it for spelling and return the cleaned fragment. Do not try to make it a full sentence.**
  - Do NOT change the original meaning, slang, or grammar. Only correct clear errors.
  - Do NOT remove filler text. Only correct clear errors.
  - If a word or phrase is ambiguous, leave it as is.
  - You must ONLY provide the list of corrected text snippets.
  - Take a breath and relax, you're doing great!
"""


def clean_video_transcript(video_data, first_attempt=True):
    """Sends a single video's data to Gemini for cleaning."""
    try:
        start_times = [
            chunk["start"] for chunk in video_data["transcript_chunks"]
        ]
        chunk_count = len(start_times)
        expected_calls = (chunk_count - 1) // API_CHUNK_SIZE + 1
        pluralization = "s" if expected_calls != 1 else ""
        print(
            f"  {chunk_count} chunks found, expecting {expected_calls}"
            f" API call{pluralization}:"
        )
        cleaned_video_data = deepcopy(video_data)
        cleaned_video_data["transcript_chunks"] = []
        metadata = {
            k: v for k, v in video_data.items() if k != "transcript_chunks"
        }

        text_chunks = []

        for chunk in video_data["transcript_chunks"]:
            # Fix transcript profanity reference
            text = re.sub(r"\[\u00a0__\u00a0\]", "****", chunk["text"])

            # Remove filler
            text = text.replace(">>", "")
            text = re.sub(r"\[\s*[^]]*?\s*\]", "", text)
            text = re.sub(r"[\s{2,}\n]", " ", text)
            start_time = chunk["start"]
            text_chunks.append(text)

        i = 0
        while i < chunk_count:
            batch_counter = 0
            transcript_chunks = ""
            upper_limit = min(i + API_CHUNK_SIZE, chunk_count)

            for j in range(i, upper_limit):
                batch_counter += 1
                transcript_chunks += f"CHUNK: `{text_chunks[j]}`\n"

            prompt = CLEANING_PROMPT.format(
                metadata=metadata, transcript_chunks=transcript_chunks
            )

            print("  -> Calling Gemini API...")
            response = model.generate_content(
                prompt,
                generation_config=MODEL_CONFIG,
                safety_settings=MODEL_SAFETY,
            )

            json_match = re.search(r"\[.*\]", response.text, re.DOTALL)

            if not json_match:
                print(
                    f"  !! ERROR: Could not find a valid JSON list in"
                    " the API response. Skipping video. Response text:"
                )
                print(response.text)
                return None

            response_text = json_match.group(0)

            response_list = json.loads(response_text)
            response_size = len(response_list)

            if response_size != batch_counter:
                print(
                    f" !! ERROR: {batch_counter} chunks sent to API,"
                    f" received {response_size} chunks. Skipping this video..."
                )
                return None

            for j in range(i, upper_limit):
                start_time = start_times[j]
                response_list_index = j - i

                cleaned_video_data["transcript_chunks"].append(
                    {
                        "text": response_list[response_list_index].strip(),
                        "start": start_time,
                    }
                )

            i += API_CHUNK_SIZE
            sleep(SLEEP_DURATION)
        return cleaned_video_data

    except Exception:
        error = format_exc()
        print("  !! An error occurred with the API call:")

        if (
            "#finishreason) is 8." in error
            or "block_reason: PROHIBITED_CONTENT" in error
        ):
            print(
                "Gemini refused due to PROHIBITED_CONTENT. Keeping"
                " original transcript."
            )
            return video_data
        elif "#finishreason) is 2." in error:
            print("Gemini refused due to MAX_TOKENS. Skipping...")
            return video_data
        elif (
            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier" in error
            and first_attempt
        ):
            print(
                "GenerateRequestsPerMinutePerProjectPerModel flagged once"
                " Waiting 120 seconds then attempting again."
            )
            sleep(120)
            return clean_video_transcript(video_data, first_attempt=False)

        print(error)

        return None


# --- Main Execution Logic ---
if __name__ == "__main__":
    print(
        f"Starting cleaning process. Raw source: '{RAW_JSON_DIR}',"
        f" Cleaned destination: '{CLEANED_JSON_DIR}'"
    )
    processed_count = 0

    try:
        for root, _, files in os.walk(RAW_JSON_DIR):
            for filename in files:
                if not filename.endswith(".json"):
                    continue

                # Construct the full path for the raw source file
                raw_path = os.path.join(root, filename)

                relative_path = os.path.relpath(raw_path, RAW_JSON_DIR)
                cleaned_path = os.path.join(CLEANED_JSON_DIR, relative_path)

                # Check if the cleaned version already exists
                if os.path.exists(cleaned_path):
                    continue

                processed_count += 1

                if processed_count >= QUOTA_LIMIT * QUOTA_MULTIPLIER:
                    print("  ...Approaching quota limit, stopping for now...")
                    break

                print("\n" + "=" * 50)
                print(f"--- Processing {processed_count}: {relative_path} ---")

                # Get the directory part of the cleaned path
                cleaned_dir = os.path.dirname(cleaned_path)

                # Ensure the destination subdirectory exists before writing
                os.makedirs(cleaned_dir, exist_ok=True)

                # Load the raw JSON data
                with open(raw_path, "r", encoding="utf-8") as f:
                    video_data = json.load(f)

                # Make the API call to clean the data
                cleaned_video_data = clean_video_transcript(video_data)

                if cleaned_video_data:
                    # Save the new, cleaned file to its correct subdirectory
                    with open(cleaned_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_video_data, f, indent=4)
                    print(
                        f"  -> Successfully cleaned and saved to {cleaned_path}"
                    )
            else:
                continue
            break
        else:
            print("\nCleaning process complete.")

    except:
        print("\n" + ("=" * 50))
        print("=" * 50)
        print("  !! An error occurred with the API call:")
        print(format_exc())
        print("=" * 50)
        print("=" * 50)
        raise
