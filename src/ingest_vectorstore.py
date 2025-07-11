import time
from sqlalchemy import create_engine, text
from langchain.schema.document import Document
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

import config  # Make sure this includes POSTGRES_DB_PATH and COLLECTION_TABLE

# --- Configuration ---
POSTGRES_DB_PATH = config.POSTGRES_DB_PATH
COLLECTION_TABLE = (
    "video_transcript_chunks"  # Used as collection name in LangChain
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Main Execution ---
if __name__ == "__main__":
    # --- Init ---
    print("Connecting to DB and initializing embedding model...")
    engine = create_engine(POSTGRES_DB_PATH)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = PGVector(
        connection=POSTGRES_DB_PATH,
        collection_name=COLLECTION_TABLE,
        embeddings=embeddings,
    )

    # --- Get Existing Document IDs in langchain_pg_embedding ---
    print("Checking existing vector documents...")
    existing_docs = set()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT document FROM langchain_pg_embedding")
        )
        for row in result:
            existing_docs.add(row.document.strip())

    # --- Stream chunks from video_transcript_chunks ---
    print("Reading from source table...")
    new_documents = []
    start = time.time()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT video_id, start_time, chunk_text
            FROM video_transcript_chunks
            WHERE chunk_text IS NOT NULL AND chunk_text <> ''
        """
            )
        )
        count = 0
        skipped = 0
        for row in result:
            document_text = row.chunk_text.strip()
            if document_text in existing_docs:
                skipped += 1
                continue

            metadata = {
                "video_id": row.video_id,
                "start_time": float(row.start_time),
            }

            new_documents.append(
                Document(page_content=document_text, metadata=metadata)
            )
            count += 1

            # Insert in batches of 256
            if len(new_documents) >= 256:
                vectorstore.add_documents(new_documents)
                print(f" -> Inserted {len(new_documents)} documents...")
                new_documents = []

        # Insert any remaining documents
        if new_documents:
            vectorstore.add_documents(new_documents)
            print(f" -> Inserted final {len(new_documents)} documents.")

    end = time.time()
    print(
        f"\n✅ Done. Added {count} new documents, skipped {skipped} already existing ones."
    )
    print(f"⏱️ Total time: {end - start:.2f} seconds.")
