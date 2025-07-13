import json
import time
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_postgres import PGVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine

import config
from parse_query import get_filter, get_unique_metadata

import logging

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
POSTGRES_DB_PATH = config.POSTGRES_DB_PATH
COLLECTION_TABLE = "video_transcript_chunks"
EMBEDDING_COLUMN = "embedding"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# QA_LLM = "phi4-mini:3.8b"
# QA_LLM = "mistral:7b-instruct"
QA_LLM = "qwen3:8b"
QA_PROMPT = """
    CONTEXT:
    {context}

    TOPICS:
    {topics}

    INSTRUCTIONS:
    - You are a factual Q&A assistant for the 'Kinda Funny' YouTube channel archive.
    - The context provided above is relevant snippets of direct transcript from episodes.
    - Respond to the USER QUERY (below) based **ONLY** on this CONTEXT.
    - The response should focus on the above TOPICS.
    - The response should an analytical summary and may contain direct quotes from the context.

    IMPORTANT RULES:
    1. Do not use outside knowledge — only what’s in the CONTEXT.
    2. If the CONTEXT lacks the answer, say so directly.
    3. Format your answer as a short, direct paragraph (no lists or bullets unless requested). Preferably between 200-400 words, but this rule is flexible.
    4. Treat the CONTEXT as possibly incomplete or informal (transcript-based).
    5. If you direct quote a piece of context, use the "video_id" metadata as a citation for the quote.

    ONLY OUTPUT THE ANSWER TEXT AND NOTHING ELSE.
    DO NOT INCLUDE THOUGHTS, EXPLANATIONS, OR COMMENTARY.

    USER QUERY:
    {input}

    RESPONSE:
"""


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize embeddings and vector store connection
    print(" -> Connecting to vector store and initializing embedding model...")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PGVector(
        connection=POSTGRES_DB_PATH,
        collection_name=COLLECTION_TABLE,
        embeddings=embeddings,
    )

    # 2. Initialize database connection
    print(" -> Connecting to database...")
    engine = create_engine(POSTGRES_DB_PATH)

    print(" -> Connection successful.")

    # 3. Fetch unique metadata
    print(" -> Fetching unique metadata for retriever context...")
    show_names, hosts = get_unique_metadata(engine)

    # 4. Build QA Agent
    qa_prompt = PromptTemplate(
        template=QA_PROMPT, input_variables=["input", "context"]
    )
    qa_chain = create_stuff_documents_chain(
        llm=OllamaLLM(model=QA_LLM, temperature=0.3, think=False),
        prompt=qa_prompt,
    )

    end_time = time.time()
    print(
        "\n--- KFAI Agent is ready."
        f" Setup took {end_time - start_time:.2f} seconds. ---"
    )

    # 5. Start the Interactive Query Loop
    while True:
        print("\n--- Ask a question, or type 'exit' to quit. ---")
        query = input("\n> ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        start_time = time.time()

        # 6. Build retriever with custom filter
        topics, filter_dict = get_filter(query, show_names, hosts)

        docs = vector_store.similarity_search(topics, k=25, filter=filter_dict)

        print(f"\n[Retrieved {len(docs)} documents]")

        docs_with_metadata = []

        for idx, doc in enumerate(docs, 1):
            docs_with_metadata.append(
                Document(
                    page_content=(
                        f"TRANSCRIPT #{idx} TEXT:\n"
                        f"```{doc.page_content}```\n"
                        f"TRANSCRIPT #{idx} METADATA:\n"
                        f"```{json.dumps(doc.metadata)}```\n\n"
                    ),
                    metadata=doc.metadata,
                )
            )

        # DEBUG PRINT STATEMENTS
        for idx, doc in enumerate(docs_with_metadata, 1):
            print(f"Document #{idx}:")
            print(doc.page_content)

        print("Thinking...")
        result = qa_chain.invoke(
            {
                "input": query,
                "topics": topics,
                "context": docs_with_metadata,
            }
        )

        if result:
            print("\nAnswer:")
            print(result)
            print("\nSources:")
            source_found = False
            for doc in docs:
                metadata = doc.metadata
                video_id = metadata["video_id"]
                if video_id in result:
                    source_found = True
                    start_time = metadata["start_time"]
                    print(
                        f"  - From video ID {video_id} ("
                        f"at ~{int(start_time // 60)}m"
                        f" {int(start_time % 60)}s)"
                    )
                    print(
                        f"    Link: https://www.youtube.com/watch?v"
                        f"={video_id}&t={int(start_time)}s"
                    )
            if not source_found:
                print("  - No sources found.")
        else:
            print("  !!  WARNING: No result.")

        end_time = time.time()
        print(f"\n...response took {end_time - start_time:.2f} seconds.")
