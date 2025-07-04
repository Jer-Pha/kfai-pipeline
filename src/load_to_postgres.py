import os
import json
from sqlalchemy import create_engine, text
from langchain_huggingface import HuggingFaceEmbeddings

import config

# --- Configuration ---
POSTGRES_DB_PATH = config.POSTGRES_DB_PATH
RAW_JSON_DIR = "videos"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIMENSIONS = 384


# --- Helper Functions ---
def get_processed_chunk_ids(engine):
    """Gets a set of already processed chunks (video_id, start_time)
    to allow resuming.
    """
    processed_ids = set()
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT video_id, start_time FROM video_transcript_chunks"
                )
            )
            for row in result:
                processed_ids.add((row.video_id, row.start_time))
    except Exception as e:
        print(f"  !! ERROR: Something went wrong during chunk retrieval:\n{e}")
    return processed_ids


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize Database Connection and Embedding Model
    print("Initializing database connection and embedding model...")
    try:
        engine = create_engine(POSTGRES_DB_PATH)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("Initialization successful.")
    except Exception as e:
        print(f"Error during initialization: {e}")
        exit()

    # 2. Create Table If It Doesn't Exist
    try:
        with engine.connect() as connection:
            print(
                "Creating tables 'videos' and 'video_transcript_chunks'"
                " if they don't exist..."
            )

            # videos
            connection.execute(
                text(
                    """
                        CREATE TABLE IF NOT EXISTS videos (
                            video_id TEXT PRIMARY KEY,
                            title TEXT,
                            show_name TEXT,
                            hosts TEXT[],
                            description TEXT,
                            published_at TIMESTAMP,
                            duration INT
                        );
                    """
                )
            )

            # video_transcript_chunks
            connection.execute(
                text(
                    f"""
                        CREATE TABLE IF NOT EXISTS video_transcript_chunks (
                            id SERIAL PRIMARY KEY,
                            video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
                            chunk_text TEXT,
                            start_time FLOAT NOT NULL,
                            embedding VECTOR({VECTOR_DIMENSIONS}),
                            UNIQUE (video_id, start_time)
                        );
                    """
                )
            )
            connection.commit()
            print("Tables are ready.")
    except Exception as e:
        print(f"Error creating tables:\n{e}")
        exit()

    # 3. Get the set of chunks that are already in the database
    processed_chunks = get_processed_chunk_ids(engine)
    print(f"Found {len(processed_chunks)} chunks already in the database.")

    # 4. Walk through the JSON files
    print(f"Starting to process files from '{RAW_JSON_DIR}'...")
    for root, _, files in os.walk(RAW_JSON_DIR):
        for filename in files:
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                video_data = json.load(f)

            video_id = video_data.get("video_id")
            if not video_id:
                continue

            print(f"\n-- Processing video: {video_id} --")

            # 5. Upsert Video Metadata into the `videos` table
            try:
                with engine.connect() as connection:
                    stmt = text(
                        """
                            INSERT INTO videos (video_id, title, show_name, hosts, description, published_at, duration)
                            VALUES (:video_id, :title, :show_name, :hosts, :description, TO_TIMESTAMP(:published_at), :duration)
                            ON CONFLICT (video_id) DO UPDATE SET
                                title = EXCLUDED.title,
                                show_name = EXCLUDED.show_name,
                                hosts = EXCLUDED.hosts,
                                description = EXCLUDED.description,
                                published_at = EXCLUDED.published_at,
                                duration = EXCLUDED.duration;
                        """
                    )
                    connection.execute(
                        stmt,
                        {
                            "video_id": video_id,
                            "title": video_data.get("title"),
                            "show_name": video_data.get("show_name"),
                            "hosts": video_data.get("hosts", []),
                            "description": video_data.get("description"),
                            "published_at": video_data.get("published_at", 0),
                            "duration": video_data.get("duration"),
                        },
                    )
                    connection.commit()
            except Exception as e:
                print(
                    f"   !! Error upserting video metadata for {video_id}: {e}"
                )
                continue

            # 6. Process and Insert New Chunks
            chunks_to_process = [
                chunk
                for chunk in video_data.get("transcript_chunks", [])
                if (video_id, chunk.get("start")) not in processed_chunks
            ]

            if not chunks_to_process:
                print(
                    "   -> All chunks for this video are already in the"
                    " database. Skipping."
                )
                continue

            print(
                f"   -> Found {len(chunks_to_process)} new chunks to embed"
                " and insert."
            )

            chunk_texts = [
                chunk.get("text", "") for chunk in chunks_to_process
            ]
            chunk_embeddings = embeddings.embed_documents(chunk_texts)

            insert_data = []
            for i, chunk in enumerate(chunks_to_process):
                insert_data.append(
                    {
                        "video_id": video_id,
                        "chunk_text": chunk.get("text"),
                        "start_time": chunk.get("start"),
                        "embedding": chunk_embeddings[i],
                    }
                )

            try:
                with engine.connect() as connection:
                    for item in insert_data:
                        stmt = text(
                            """
                            INSERT INTO video_transcript_chunks (video_id, chunk_text, start_time, embedding)
                            VALUES (:video_id, :chunk_text, :start_time, :embedding)
                            ON CONFLICT (video_id, start_time) DO NOTHING;
                        """
                        )
                        connection.execute(stmt, item)
                    connection.commit()
                print(
                    f"   -> Successfully inserted {len(insert_data)} new chunks."
                )
            except Exception as e:
                print(f"   !! Error inserting chunk data for {video_id}: {e}")

    print("\nData loading process complete.")
