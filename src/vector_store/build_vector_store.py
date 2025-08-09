import json
import os
import time

from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from load.utils.config import EMBEDDING_MODEL, POSTGRES_DB_PATH
from load.utils.helpers import format_duration

# --- Configuration ---
JSON_SOURCE_DIR = "videos"  # Change to cleaned directory later
COLLECTION_NAME = "video_transcript_chunks"
BATCH_SIZE = 256


# --- Helper Function ---
def get_processed_chunk_ids() -> set[tuple[str, float]]:
    """
    Gets a set of already processed chunk IDs (video_id, start_time)
    by querying the LangChain collection's metadata.
    """
    processed_ids = set()
    # This is a bit of a workaround to access the underlying connection
    # as LangChain's PGVector doesn't have a built-in "list all" method.
    try:
        with create_engine(POSTGRES_DB_PATH).connect() as connection:
            # Query the cmetadata column of the embedding table for this collection
            stmt = text(
                f"""
                SELECT cmetadata FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
                )
            """
            )
            results = connection.execute(
                stmt, {"collection_name": COLLECTION_NAME}
            )
            for row in results:
                # The metadata is the first (and only) column
                metadata = dict(row[0])
                if (
                    metadata
                    and "video_id" in metadata
                    and "start_time" in metadata
                ):
                    processed_ids.add(
                        (metadata["video_id"], metadata["start_time"])
                    )
    except Exception as e:
        print(
            f"Could not fetch processed chunks, assuming first run. Error: {e}"
        )
    return processed_ids


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize Connections
    print("Initializing database connection and embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 128,
        },
    )
    vectorstore = PGVector(
        connection=POSTGRES_DB_PATH,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )
    print("Initialization successful.")

    # 2. Get the set of chunks already in the database for resuming
    processed_chunks_set = get_processed_chunk_ids()
    print(
        f"Found {len(processed_chunks_set)} chunks already in the LangChain collection."
    )

    # 3. Walk through the JSON files to find new documents
    print(f"Starting to process files from '{JSON_SOURCE_DIR}'...")
    new_documents_batch = []
    total_added = 0
    total_skipped = 0
    start_time = time.time()

    for root, _, files in os.walk(JSON_SOURCE_DIR):
        for filename in files:
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                video_data = dict(json.load(f))

            video_id = video_data.get("video_id")
            if not video_id:
                continue

            # Prepare the video-level metadata once
            video_metadata = {
                "video_id": video_id,
                "title": video_data.get("title", "<NO TITLE FOUND>"),
                "show_name": video_data.get(
                    "show_name", "<NO SHOW NAME FOUND>"
                ),
                "hosts": ",".join(
                    video_data.get("hosts", [])
                ),  # String for PGVector `$like` filter
                "published_at": video_data.get(
                    "published_at", 1325376000
                ),  # UNIX epoch, default is 2012-01-01T00:00
            }

            # 4. Iterate through chunks and create Document objects for new ones
            for chunk in video_data.get("transcript_chunks", []):
                assert type(chunk) is dict
                chunk_start_time = float(chunk.get("start", 0))

                # Resumability Check
                if (video_id, chunk_start_time) in processed_chunks_set:
                    total_skipped += 1
                    continue

                # This is a new chunk, prepare its full metadata
                chunk_text = str(chunk.get("text", "")).strip()
                chunk_metadata = video_metadata.copy()
                chunk_metadata["start_time"] = float(chunk_start_time)
                chunk_metadata["text"] = chunk_text

                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
                new_documents_batch.append(doc)

                # 5. Add documents to the vector store in batches
                if len(new_documents_batch) >= BATCH_SIZE:
                    print(
                        f" -> Embedding and inserting batch of {len(new_documents_batch)} documents..."
                    )
                    try:
                        vectorstore.add_documents(new_documents_batch)
                        total_added += len(new_documents_batch)
                    except Exception as e:
                        print(f"  !! Failed to insert batch: {e}")
                    new_documents_batch = []  # Reset the batch

    # 6. Insert any final remaining documents
    if new_documents_batch:
        print(
            f" -> Embedding and inserting final batch of {len(new_documents_batch)} documents..."
        )
        vectorstore.add_documents(new_documents_batch)
        total_added += len(new_documents_batch)

    end_time = time.time()
    print("\n" + "=" * 50)
    print("  Data loading process complete.")
    print(f"  - Added {total_added} new documents to the collection.")
    print(f"  - Skipped {total_skipped} documents that already existed.")
    print(
        f"  Total time for this run: {format_duration(end_time - start_time)}."
    )
    print("=" * 50)
