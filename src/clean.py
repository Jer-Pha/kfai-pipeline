import os
import json
import time
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from config import GEMINI_API_KEY  # Assuming you put your key in config.py

# --- Configuration ---
RAW_JSON_DIR = "videos"
CLEANED_JSON_DIR = "videos_cleaned"
os.makedirs(CLEANED_JSON_DIR, exist_ok=True)
genai.configure(api_key=GEMINI_API_KEY)

# --- Set up the Gemini Model ---
# model = genai.GenerativeModel("gemini-2.0-flash")
model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

generation_config = genai.GenerationConfig(
    temperature=0.0,  # Make the output deterministic
    response_mime_type="application/json",  # Ask for JSON output directly
)

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- The Master Prompt ---
SYSTEM_PROMPT = """
You are an expert transcription editor for the YouTube channels 'Kinda Funny' and 'Kinda Funny Games'.
Your task is to correct spelling and phonetic transcription errors in a given JSON object containing a video's transcript from this archive.

You will be given a JSON object with video metadata and a list of transcript chunks.
Your goal is to return a new JSON object with the exact same structure, but with the 'text' fields in the 'transcript_chunks' corrected.

Follow these rules precisely:
1.  Use the top-level metadata (show_name, hosts, title) as the ground truth for spelling names of people and shows.
2.  Use your knowledge of the 'Kinda Funny' and 'Kinda Funny Games' channels to correct common names (e.g., Greg Miller, Colin Moriarty, Nick Scarpino, Tim Gettys, Portillo) and show titles (e.g., Gamescast, The GameOverGreggy Show, PS I Love You XOXO).
3.  Correct common phonetic mistakes (e.g., "game over gregy" should be "GameOverGreggy").
4.  Correct obvious spelling errors.
5.  Do NOT change the meaning, grammar, slang, or original phrasing. Only correct clear errors.
6.  If a word or phrase is ambiguous, leave it as is. It is better to leave a small error than to introduce a new one.
7.  The output MUST be a valid JSON object matching the input structure. Do not add any conversational text before or after the JSON.
"""


def fix_broken_json(broken_text):
    """A fallback function to ask the LLM to fix its own malformed JSON."""
    print("  -> Attempting self-correction for malformed JSON...")
    try:
        # A simpler, more direct prompt
        fix_prompt = f"""
        The following text is not valid JSON, likely due to unescaped quotes within string values or unnecessary trailing commas.
        Please fix the JSON syntax and return ONLY the corrected, valid JSON object.

        BROKEN JSON:
        {broken_text}
        """

        # Use the same model and config to try and fix it
        response = model.generate_content(
            [fix_prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        # Attempt to parse the fixed version
        return json.loads(response.text)
    except Exception as e:
        print(f"  !! Self-correction failed: {e}")
        return None


def clean_video_transcript(video_data):
    """Sends a single video's data to Gemini for cleaning."""
    try:
        user_prompt = json.dumps(video_data, indent=4)

        response = model.generate_content(
            [SYSTEM_PROMPT, user_prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        try:
            cleaned_data = json.loads(response.text)
            return cleaned_data
        except json.JSONDecodeError:
            print("  -> Received malformed JSON. Initiating fallback.")
            cleaned_text = (
                response.text.strip().lstrip("```json").rstrip("```")
            )
            return fix_broken_json(cleaned_text)

    except Exception as e:
        print(f"  !! An error occurred with the API call: {e}")
        return None


# --- Main Execution Logic ---
if __name__ == "__main__":
    print(
        f"Starting cleaning process. Raw source: '{RAW_JSON_DIR}',"
        f" Cleaned destination: '{CLEANED_JSON_DIR}'"
    )

    # Use a counter to track progress
    processed_count = 0
    api_call_count = 0

    try:
        # --- MODIFIED: Use os.walk to traverse the directory tree ---
        for root, _, files in os.walk(RAW_JSON_DIR):
            for filename in files:
                if not filename.endswith(".json"):
                    continue

                processed_count += 1

                # Construct the full path for the raw source file
                raw_path = os.path.join(root, filename)

                relative_path = os.path.relpath(raw_path, RAW_JSON_DIR)
                cleaned_path = os.path.join(CLEANED_JSON_DIR, relative_path)

                # Check if the cleaned version already exists
                if os.path.exists(cleaned_path):
                    continue

                print(
                    f"--- Processing file {processed_count}: {relative_path} ---"
                )

                # Get the directory part of the cleaned path
                cleaned_dir = os.path.dirname(cleaned_path)

                # --- MODIFIED: Ensure the destination subdirectory exists before writing ---
                os.makedirs(cleaned_dir, exist_ok=True)

                # Load the raw JSON data
                with open(raw_path, "r", encoding="utf-8") as f:
                    video_data = json.load(f)

                # Make the API call to clean the data
                cleaned_video_data = clean_video_transcript(video_data)
                api_call_count += 1

                if cleaned_video_data:
                    # Save the new, cleaned file to its correct subdirectory
                    with open(cleaned_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_video_data, f, indent=4)
                    print(
                        f"  -> Successfully cleaned and saved to {cleaned_path}"
                    )
                else:
                    # Save raw data in cleaned folder once cleaning has failed
                    with open(cleaned_path, "w", encoding="utf-8") as f:
                        json.dump(video_data, f, indent=4)
                    print(
                        "  -> Failed to get a clean response from the API."
                        " Adding raw data to cleaned directory."
                    )

                # Wait to respect the per-minute rate limit
                print(
                    f"  ...waiting 1.1 seconds (API calls today: {api_call_count})..."
                )
                time.sleep(1.1)

    # Catch the SPECIFIC quota error first.
    except google_exceptions.ResourceExhausted as e:
        print("\n" + "=" * 50)
        print("!!! DAILY API QUOTA REACHED !!!")
        print("=" * 50)
        print("Stopping the script for today.")
        print(f"Total API calls made in this session: {api_call_count}")
        print(f"Error details: {e}")
        print(
            "\nRun the script again tomorrow to continue processing the remaining files."
        )

    # Catch any OTHER general errors
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"!! An unexpected error occurred: {e}")
        print("=" * 50)
        # We re-raise the error here to get a full traceback for debugging
        raise

    print("\nCleaning process complete or paused.")
