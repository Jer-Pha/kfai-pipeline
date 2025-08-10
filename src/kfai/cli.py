from extractors import fetch_raw_data, process_failed_videos
from loaders import build_vector_store, interactive_qa
from transformers import clean_locally

RESPONSE_MAP = {
    "1": fetch_raw_data.run,
    "2": process_failed_videos.run,
    "3": clean_locally.run,
    "4": build_vector_store.run,
    "5": interactive_qa.run,
}
USER_MENU = """--- Welcome to KFAI ---

What would you like to do?
{options}

Enter the number of your choice (or 'q' to quit):
"""
MENU_OPTIONS = {
    "1": "Load raw data from KFDB",
    "2": "Process failed video IDs",
    "3": "Clean raw transcripts",
    "4": "Update the vector store",
    "5": "Interact with the Query Agent",
}


def main() -> None:
    options = "\n".join(f"  {k}. {v}" for k, v in MENU_OPTIONS.items())
    print(USER_MENU.format(options=options))

    while True:
        user_input = input("> ").strip().lower()

        if user_input in {"q", "quit"}:
            print("Exiting.")
            break

        if user_input in RESPONSE_MAP:
            print(f"[{MENU_OPTIONS[user_input]}] Beginning process...\n\n")
            RESPONSE_MAP[user_input]()
            break
        print(f"Please enter a valid number from 1 to {len(MENU_OPTIONS)}:")


if __name__ == "__main__":
    main()
