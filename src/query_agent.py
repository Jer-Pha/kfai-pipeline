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
CONTEXT_COUNT = 100
EMBEDDING_COLUMN = "embedding"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QA_MODEL = "qwen3:14b-q4_K_M"
llm = OllamaLLM(
    model=QA_MODEL,
    temperature=0.4,
    top_p=0.95,
    top_k=50,
    reasoning=True,
    verbose=False,
)
QA_PROMPT = """
    CONTEXT:
    {context}

    TOPICS:
    {topics}

    INSTRUCTIONS:
    - You are a factual Q&A assistant for the 'Kinda Funny' YouTube channel archive.
    - The context provided above is relevant snippets of direct transcript from episodes.
    - Your task is to respond to the USER QUERY (below) based **ONLY** on this CONTEXT.
    - Use the "video_id" metadata as a citation for each sentence.
    - Focus your response on the list of TOPICS (above) and the USER QUERY.
    - Do not direct quote the context unless the user asked for a direct quote.

    IMPORTANT RULES:
    1. The response **MUST NOT** include previous knowledge that is not mentioned in the CONTEXT.
    2. If the CONTEXT lacks the answer, say so directly.
    3. Treat the CONTEXT as possibly incomplete or informal (transcript-based chunks).
    4. The user must know which video(s) you are referencing for each sentence — cite your sources!
    5. The response **MUST** be formatted as a paragraph — no lists or bullets unless the user requests them directly.
    6. Do **NOT* stop at the first piece of context that answers the question. Go through the entire CONTEXT then formulate your response. You **SHOULD** reference as many videos as necessary to fully answer the USER QUERY.
    7. Only output the RESPONSE text and nothing else — do **NOT** include thoughts, explanations, or commentary.

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
        llm=llm,
        prompt=qa_prompt,
    )

    end_time = time.time()
    print(
        "\n--- KFAI Agent is ready."
        f" Setup took {end_time - start_time:.2f} seconds. ---"
    )
    print(f"Model: {QA_MODEL}")

    # 5. Start the Interactive Query Loop
    while True:
        print("\n--- Ask a question, or type 'exit' to quit. ---")
        query = input("\n> ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        start = time.time()

        # 6. Build retriever with custom filter
        topics, filter_dict = get_filter(query, show_names, hosts)

        docs = vector_store.similarity_search(
            topics, k=CONTEXT_COUNT, filter=filter_dict
        )
        doc_count = len(docs)

        if not doc_count:
            print(
                "  !!  WARNING: No documents found, skipping this question..."
            )
            continue

        print(
            f"\nRetrieved {len(docs)} documents -"
            f" [upper limit: {CONTEXT_COUNT}]"
        )

        docs_with_metadata = []

        for idx, doc in enumerate(docs, 1):
            metadata = doc.metadata
            metadata.pop("text", None)

            docs_with_metadata.append(
                Document(
                    page_content=(
                        f"TRANSCRIPT #{idx} TEXT:\n"
                        f"```{doc.page_content}```\n"
                        f"TRANSCRIPT #{idx} METADATA:\n"
                        f"```{json.dumps(metadata)}```\n\n"
                    ),
                    metadata=metadata,
                )
            )

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

        end = time.time()
        print(f"\n...response took {(end - start):.2f} seconds.")
