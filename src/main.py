from extract import fetch_raw_data, process_failed_videos
from load import build_vector_store, interactive_qa
from transform import clean_locally

RESPONSE_MAP = {
    "1": fetch_raw_data.run,
    "2": process_failed_videos.run,
    "3": clean_locally.run,
    "4": build_vector_store.run,
    "5": interactive_qa.run,
}

if __name__ == "__main__":
    print(
        "--- Welcome to KFAI ---\n\n"
        "What would you like to do?\n"
        "   1. Load raw data from KFDB\n"
        "   2. Process failed video IDs\n"
        "   3. Clean raw transcripts\n"
        "   4. Update the vector store\n"
        "   5. Interact with the Query Agent\n\n"
        "Enter the number of your choice (or 'q' to quit):"
    )

    while True:
        user_input = input("> ").strip().lower()

        if user_input in {"q", "quit"}:
            print("Exiting.")
            break

        if user_input in RESPONSE_MAP:
            RESPONSE_MAP[user_input]()
            break
        print(f"Please enter a valid number from 1 to {len(RESPONSE_MAP)}:")
