import os
import json
import google.generativeai as genai
import re
import time
from copy import deepcopy
from langchain_ollama import OllamaLLM
from time import sleep
from traceback import format_exc

from config import GEMINI_API_KEY

# --- Configuration ---
RAW_JSON_DIR = "videos"
CLEANED_JSON_DIR = "videos_cleaned_local"
CHUNK_SIZE = 5
SLEEP_DURATION = 0.5
OLLAMA_MODEL = "mixtral:8x7b-instruct-v0.1-q5_K_M"
os.makedirs(CLEANED_JSON_DIR, exist_ok=True)
genai.configure(api_key=GEMINI_API_KEY)
llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.0, format="json")

CLEANING_PROMPT = """
METADATA:
{metadata}

TRANSCRIPT CHUNKS:
{transcript_chunks}

TASK DESCRIPTION:
- You are an expert transcription editor for the 'Kinda Funny' and 'Kinda Funny Games' YouTube channels.
- You have been provided JSON data for single video's metadata and transcript chunks.
- Your task is to process EACH chunk per the FULL INSTRUCTIONS and provide all processed chunks as a JSON array of strings, one for each chunk.
- You MUST return the EXACT SAME number of chunks as you were given. If you are given {chunk_size} chunks, you must return an array containing {chunk_size} strings.

FULL INSTRUCTIONS:
Use your contextual knowledge of the hosts, show names, common topics (video games, movies, comics, etc.), and the video's metadata to accomplish the following items for EACH chunk:
  - Correct common phonetic mistakes and spelling errors.
  - Capitalize proper nouns like names, games, and show titles.
  - **CRITICAL RULE: Do NOT discard or omit any chunks, even if they appear to be incomplete sentences or fragments. If a chunk is just a fragment, clean it for spelling and return the cleaned fragment. Do not try to make it a full sentence.**
  - Do NOT change the original meaning, slang, or grammar. Only correct clear errors.
  - Do NOT remove filler text. Only correct clear errors.
  - If a word or phrase is ambiguous, leave it as is.
  - You must ONLY provide the array of corrected text snippets.
  - Take a breath and relax, you're doing great!
"""


def clean_video_transcript(video_data, first_attempt=True):
    """Sends a single video's data to Gemini for cleaning."""
    try:
        start_times = [
            chunk["start"] for chunk in video_data["transcript_chunks"]
        ]
        chunk_count = len(start_times)
        expected_calls = (chunk_count - 1) // CHUNK_SIZE + 1
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
            text = text.replace("\u200b", "").replace("\xa0", " ")
            text = text.replace(">>", "")
            text = re.sub(r"\[\s*[^]]*?\s*\]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            start_time = chunk["start"]
            text_chunks.append(text)

        i = 0
        while i < chunk_count:
            batch_counter = 0
            transcript_chunks = ""
            upper_limit = min(i + CHUNK_SIZE, chunk_count)

            for j in range(i, upper_limit):
                batch_counter += 1
                transcript_chunks += f"CHUNK: `{text_chunks[j]}`\n"

            prompt = CLEANING_PROMPT.format(
                metadata=metadata,
                transcript_chunks=transcript_chunks,
                chunk_size=CHUNK_SIZE,
            )

            try:
                print(
                    f"  -> Calling local Ollama model for batch {i//CHUNK_SIZE + 1}..."
                )
                start_time = time.time()

                response_text = llm.invoke(prompt)

                end_time = time.time()
                print(
                    f"    ...batch processed in {end_time - start_time:.2f} seconds."
                )

                try:
                    response_list = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract a valid JSON array if possible
                    match = re.search(r"\[.*\]", response_text, re.DOTALL)
                    if match:
                        response_list = json.loads(match.group(0))
                    else:
                        print("!! ERROR: No valid JSON array found.")
                        return None
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
            except Exception as e:
                print(
                    f"!! ERROR: An error occurred with the local LLM call: {e}"
                )
                return None

            i += CHUNK_SIZE
            sleep(SLEEP_DURATION)
        return cleaned_video_data

    except Exception:
        error = format_exc()
        print("  !! An error occurred with the API call:")
        print(error)

        return None


# --- Main Execution Logic ---
if __name__ == "__main__":
    print(
        f"Starting local cleaning process. Raw source: '{RAW_JSON_DIR}',"
        f" Cleaned destination: '{CLEANED_JSON_DIR}'"
    )

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

                print("\n" + "=" * 50)
                print(f"--- Processing {relative_path} ---")

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
