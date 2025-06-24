import os
import json
import google.generativeai as genai
from collections import defaultdict, deque
from copy import deepcopy
from re import sub
from traceback import format_exc

from config import GEMINI_API_KEY

# --- Configuration ---
RAW_JSON_DIR = "videos"
CLEANED_JSON_DIR = "videos_cleaned"
# GEMINI_API_MODEL, QUOTA_LIMIT = "gemini-2.0-flash", 200
GEMINI_API_MODEL, QUOTA_LIMIT = "gemini-2.5-flash-preview-05-20", 250
os.makedirs(CLEANED_JSON_DIR, exist_ok=True)
genai.configure(api_key=GEMINI_API_KEY)

# --- Set up the Gemini Model ---
model = genai.GenerativeModel(GEMINI_API_MODEL)
generation_config = genai.types.GenerationConfig(temperature=0.0)
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- The Master Prompt ---
SYSTEM_PROMPT = """
METADATA:
{metadata}

TRANSCRIPT CHUNKS:
{transcript_chunks}

TASK DESCRIPTION:
You are an expert transcription editor for the 'Kinda Funny' and 'Kinda Funny Games' YouTube channels.
You have been provided JSON data for single video's metadata and transcript chunks.
Your task is to process EACH chunk per the following instructions and stream back a corrected version of that entire chunk as text.

FULL INSTRUCTIONS:
Use your contextual knowledge of the hosts, show names, common topics (video games, movies, comics, etc.), and the video's metadata to accomplish the following items for EACH chunk:
- Correct common phonetic mistakes and spelling errors.
- Capitalize proper nouns like names, games, and show titles.
- Do NOT change the original meaning, slang, or grammar. Only correct clear errors.
- Do NOT remove filler text. Only correct clear errors.
- If a word or phrase is ambiguous, leave it as is.
- You must ONLY provide the corrected text snippet and nothing else.
- Each transcript chunk should be a single response and should not be split between multiple streams.
"""


def clean_video_transcript(video_data):
    """Sends a single video's data to Gemini for cleaning."""
    try:
        start_times = deque(
            [chunk["start"] for chunk in video_data["transcript_chunks"]]
        )
        cleaned_video_data = deepcopy(video_data)
        cleaned_video_data["transcript_chunks"] = []
        metadata = {
            k: v for k, v in video_data.items() if k != "transcript_chunks"
        }

        transcript_dict = defaultdict(str)

        for chunk in video_data["transcript_chunks"]:
            text = sub(r"\[\u00a0__\u00a0\]", "****", chunk["text"])
            start_time = chunk["start"]
            transcript_dict[start_time] = text

        while start_times:
            transcript_chunks = ""

            for text in transcript_dict.values():
                transcript_chunks += f"- {text}\n"

            prompt = SYSTEM_PROMPT.format(
                metadata=metadata, transcript_chunks=transcript_chunks
            )

            response_stream = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )

            buffer = ""

            for chunk in response_stream:
                text = chunk.text
                if "\n" in text:
                    split_text = text.split("\n")
                    buffer += " " + split_text[0].strip()
                    buffer = sub(r" *' *", "'", buffer)
                    start_time = start_times.popleft()

                    if buffer.startswith("- ") and len(buffer) > 2:
                        buffer = buffer[2:]
                    cleaned_video_data["transcript_chunks"].append(
                        {
                            "text": buffer.strip(),
                            "start": start_time,
                        }
                    )
                    buffer = split_text[1].strip()
                    del transcript_dict[start_time]
                else:
                    buffer += " " + text.strip()
                    buffer = sub(r" *' *", "'", buffer)

        return cleaned_video_data

    except Exception:
        print("\n" + "=" * 50)
        print("  !! An error occurred with the API call:")
        print(format_exc())
        print("=" * 50)
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

                if processed_count > QUOTA_LIMIT * 0.6:
                    print("  ...Approaching quota limit, stopping for now...")
                    break

                print(
                    f"--- Processing file {processed_count}: {relative_path} ---"
                )

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

    # Catch any OTHER general errors
    except Exception as e:
        print("\n" + "=" * 50)
        print("  !! An error occurred with the API call:")
        print(format_exc())
        print("=" * 50)
        raise
