from kfai.extractors import (
    fetch_raw_data,
    process_failed_videos,
    transcribe_failures,
)
from kfai.loaders import build_vector_store, interactive_qa
from kfai.transformers import clean_locally

RESPONSE_MAP = {
    "1": fetch_raw_data.run,
    "2": process_failed_videos.run,
    "3": transcribe_failures.run,
    "4": clean_locally.run,
    "5": build_vector_store.run,
    "6": interactive_qa.run,
}
USER_MENU = """--- Welcome to KFAI ---

What would you like to do?
{options}

Enter the number of your choice (or 'q' to quit).
If you want to chain multiple functions, use '>' (e.g. "1>2>3").
"""
MENU_OPTIONS = {
    "1": "Load raw data from KFDB",
    "2": "Process failed video IDs",
    "3": "Transcribe failed videos",
    "4": "Clean raw transcripts",
    "5": "Update the vector store",
    "6": "Interact with the Query Agent",
}


def main() -> None:
    options = "\n".join(f"  {k}. {v}" for k, v in MENU_OPTIONS.items())
    print(USER_MENU.format(options=options), end="")

    while True:
        user_input = input("> ").strip().lower()

        if user_input in {"q", "quit"}:
            print("Exiting.")
            break

        if user_input in RESPONSE_MAP:
            print(f"[{MENU_OPTIONS[user_input]}] Beginning process...\n")
            RESPONSE_MAP[user_input]()
            break
        else:
            commands_to_run = [cmd.strip() for cmd in user_input.split(">")]
            if all(cmd in RESPONSE_MAP for cmd in commands_to_run):
                print("Beginning chained process:")
                print(
                    "\n".join(
                        f"  {MENU_OPTIONS[cmd]}" for cmd in commands_to_run
                    )
                )
                for cmd in commands_to_run:
                    print(f"\n[{MENU_OPTIONS[cmd]}] Beginning process...\n")
                    RESPONSE_MAP[cmd]()
                break
        print(
            f'Please enter a valid number from 1 to {len(MENU_OPTIONS)} or command chain (e.g. "1>2>3"):'
        )


if __name__ == "__main__":
    main()
    main()
