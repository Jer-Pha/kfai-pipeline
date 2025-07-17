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
CHUNK_SIZE = 1
SLEEP_DURATION = 0.5
OLLAMA_MODEL = "llama3.1:8b-instruct-q8_0"
CLEANED_JSON_DIR = f"videos_cleaned_local-{OLLAMA_MODEL.split(":")[0]}"

# Global compilers
_compile = re.compile
_sub_profanity = _compile(r"\[\u00a0__\u00a0\]").sub
_sub_bracket_tags = _compile(r"\[\s*[^]]*?\s*\]").sub
_sub_whitespace = _compile(r"\s+").sub
_sub_chunk = _compile(r"</?CHUNK>").sub
_sub_squotes = _compile(r"[‘’]").sub
_sub_dquotes = _compile(r"[“”]").sub

os.makedirs(CLEANED_JSON_DIR, exist_ok=True)
genai.configure(api_key=GEMINI_API_KEY)
llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0.1,
    top_p=0.92,
    top_k=40,
    keep_alive=60,
    reasoning=False,
    verbose=False,
)

SYSTEM_PROMPT_SINGLE = """
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

# This is a template for the specific data for each API call.
USER_PROMPT_SINGLE = """
METADATA CONTEXT:
{metadata}

RAW CHUNK:
{chunk}

RESPONSE:
"""

SYSTEM_PROMPT_MULTI = f"""
- You are an expert transcription editor for the 'Kinda Funny' and 'Kinda Funny Games' YouTube channels.
- You have been provided JSON data for single video's metadata and auto-generated transcript chunks.
- Your task is to process EACH chunk per the FULL INSTRUCTIONS and provide all processed chunks as a JSON array of strings, one for each chunk.
- Each chunk is enclosed in <CHUNK> ... </CHUNK>. You must return a cleaned string for each one.
- You MUST return the EXACT SAME number of chunks as you were given. If you are given {CHUNK_SIZE} chunks, you must return an array containing {CHUNK_SIZE} strings.
- Use the METADATA to bias your corrections.
- You are expected to confidently fix obvious phonetic or name errors using common knowledge, especially when they relate to context found in the title, description, or host names.

FULL INSTRUCTIONS:
Use your contextual knowledge of the hosts, show names, common topics (video games, movies, comics, etc.), and the video's metadata to accomplish the following items for EACH chunk:
  - Correct phonetic mistakes and spelling errors, especially for names, pop culture references, and brands, even if they’re only approximate matches.
  - Capitalize proper nouns like names, games, and show titles.
  - **CRITICAL RULE: Do NOT discard or omit any chunks, even if they appear to be incomplete sentences or fragments. If a chunk is just a fragment, clean it for spelling and return the cleaned fragment. Do not try to make it a full sentence.**
  - Do NOT change the original meaning, slang, or grammar. Only correct clear errors.
  - Do NOT remove filler text. Only correct clear errors.
  - If a word or phrase is ambiguous, leave it as is.
  - The RESPONSE should be a valid JSON object and nothing else — do **NOT** include thoughts, explanations, or commentary.

EXAMPLES OF POSSIBLE CHANGES (INPUT → CLEANED):
  - "Tim Geddes" → "Tim Gettys"
  - "wing grety" → "Wayne Gretzky"
  - "final fantasy versus 13" → "Final Fantasy Versus XIII"
  - "game over greggy" → "GameOverGreggy"

RESPONSE FORMAT:
{{
    "transcript_chunks": [
        "string",
        "..."
    ]
}}
"""

# This is a template for the specific data for each API call.
USER_PROMPT_MULTI = """
METADATA CONTEXT:
{metadata}

RAW TRANSCRIPT CHUNKS (JSON Array):
{transcript_chunks_json}

RESPONSE:
"""


def _clean_response(response):
    """Clean common LLM inconsistencies in the response."""
    response = response.split("Here is the cleaned chunk:")[-1]
    response = response.split("Here's the cleaned chunk:")[-1]
    response = response.split("</think>")[-1]
    response = _sub_chunk("", response)
    response = _sub_squotes("'", response)
    response = _sub_dquotes('"', response)
    return response


def _clean_single_chunk(video_data):
    """Cleans a single video's transcript with Ollama, one chunk at a time."""
    try:
        profile_start = time.time()

        chunk_count = len(video_data["transcript_chunks"])
        pluralization = "s" if chunk_count != 1 else ""
        print(f"  >> {chunk_count} chunk{pluralization} found")
        cleaned_video_data = {
            k: v for k, v in video_data.items() if k != "transcript_chunks"
        }
        metadata = json.dumps(cleaned_video_data)
        cleaned_video_data["transcript_chunks"] = []

        _invoke_llm = llm.invoke

        _clean = _clean_response

        for chunk in video_data["transcript_chunks"]:
            text = chunk["text"]

            # Fix transcript profanity reference
            text = _sub_profanity("****", text)

            # Remove filler
            text = text.replace("\u200b", "").replace("\xa0", " ")
            text = text.replace(">>", "")

            # Regex cleanup
            text = _sub_bracket_tags("", text)
            text = _sub_whitespace(" ", text).strip()

            user_prompt = USER_PROMPT_SINGLE.format(
                metadata=metadata,
                chunk=text,
            )

            try:
                response = _invoke_llm(
                    [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT_SINGLE,
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
            except Exception as e:
                error = format_exc()
                print(
                    f"!! ERROR: An error occurred with the local LLM:\n{error}"
                )
                raise Exception(e)

        profile_end = time.time()
        print(f"processed in {profile_end - profile_start:.2f} seconds.")

        return cleaned_video_data

    except Exception as e:
        error = format_exc()
        print(f"  !! An unexpected error occured:\n{error}")

        raise Exception(e)


def _clean_multi_chunk(video_data):
    """Cleans a single video's data with Ollama."""
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
        metadata_str = json.dumps(metadata, indent=4)

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

        for i in range(0, chunk_count, CHUNK_SIZE):
            batch_counter = 0
            transcript_chunks = ""
            upper_limit = min(i + CHUNK_SIZE, chunk_count)

            for j in range(i, upper_limit):
                batch_counter += 1
                transcript_chunks += f"\n<CHUNK>{text_chunks[j]}</CHUNK>\n"

            user_prompt = USER_PROMPT_MULTI.format(
                metadata=metadata_str,
                transcript_chunks_json=json.dumps(transcript_chunks, indent=4),
            )

            try:
                print(
                    f"  -> Calling local Ollama model for batch {i//CHUNK_SIZE + 1}..."
                )
                start_time = time.time()

                response_text = llm.invoke(
                    [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT_MULTI,
                        },
                        {"role": "user", "content": user_prompt},
                    ]
                )

                response_text = (
                    response_text.replace("<CHUNK>", "")
                    .replace("</CHUNK>", "")
                    .replace("“", '"')
                    .replace("”", '"')
                    .replace("‘", "'")
                    .replace("’", "'")
                )

                end_time = time.time()
                print(
                    f"    ...batch processed in {end_time - start_time:.2f} seconds."
                )

                try:
                    response_json = json.loads(response_text)
                    if isinstance(response_json, dict) and response_json.get(
                        "transcript_chunks", None
                    ):
                        response_list = response_json["transcript_chunks"]
                    elif isinstance(response_json, list):
                        response_list = response_json
                    else:
                        print(type(response_json))
                        raise json.JSONDecodeError("", "", 0)
                except json.JSONDecodeError:
                    # Try to extract a valid JSON array if possible
                    match = re.search(r"\[.*\]", response_text, re.DOTALL)
                    if match:
                        response_list = json.loads(match.group(0))
                    else:
                        print("!! ERROR: No valid JSON array found.")
                        print(response_text)
                        return None
                response_size = len(response_list)

                if response_size != batch_counter:
                    print(
                        f" !! ERROR: {batch_counter} chunks sent to API,"
                        f" received {response_size} chunks. Skipping this video..."
                    )
                    print(response_list)
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

            print(f"    Sleeping for {SLEEP_DURATION} seconds...")
            sleep(SLEEP_DURATION)
        return cleaned_video_data

    except Exception:
        error = format_exc()
        print("  !! An error occurred with the API call:")
        print(error)

        return None


def _clean_video_transcript(video_data):
    if CHUNK_SIZE == 1:
        return _clean_single_chunk(video_data)
    else:
        return _clean_multi_chunk(video_data)


# --- Main Execution Logic ---
if __name__ == "__main__":
    print(
        f"Starting local cleaning process. Raw source: '{RAW_JSON_DIR}',"
        f" Cleaned destination: '{CLEANED_JSON_DIR}'"
    )

    MAX_RUNTIME_SECONDS = 15 * 60  # 15 minutes
    test_start = time.time()

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
                    video_data = dict(json.load(f))

                if not video_data.get("transcript_chunks"):
                    continue

                # Make the local call to clean the data
                cleaned_video_data = _clean_video_transcript(video_data)

                # Ensure cleaned data matches format
                if not cleaned_video_data or set(
                    cleaned_video_data.keys()
                ) != set(video_data.keys()):
                    continue

                # Check that transcript count did not change
                raw_chunk_count = len(video_data["transcript_chunks"])
                cleaned_chunk_count = len(
                    cleaned_video_data["transcript_chunks"]
                )
                if cleaned_chunk_count != raw_chunk_count:
                    print(
                        f" !! ERROR: {raw_chunk_count} chunks sent to LLM,"
                        f" but received {cleaned_chunk_count} back."
                        " Skipping this video..."
                    )
                    continue

                # Save the new, cleaned file to its correct subdirectory
                with open(cleaned_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned_video_data, f, indent=4)
                print(f"  -> Successfully cleaned and saved to {cleaned_path}")
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
