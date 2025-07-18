import os
import json
import time
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration ---
JSON_SOURCE_DIR = "videos"  # Change to "videos_cleaned" later
VECTOR_STORE_PATH = "faiss_index_raw"  # Change to "faiss_index" later
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# OLLAMA_MODEL = "phi4-mini-reasoning"
OLLAMA_MODEL = "deepseek-r1:8b"


# --- Step 1: Document Loading Function ---
def load_docs_from_json():
    all_docs = []
    video_count = 0
    print(f"Loading documents from '{JSON_SOURCE_DIR}'...")

    for root, _, files in os.walk(JSON_SOURCE_DIR):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                video_id = data.get("video_id", "")
                title = data.get("title", "")
                show_name = data.get("show_name", "")
                hosts = ", ".join(data.get("hosts", []))

                # Create a Document for each transcript chunk
                for i, chunk in enumerate(data.get("transcript_chunks", [])):
                    start_time = chunk.get("start", 0)
                    doc_metadata = {
                        "video_id": video_id,
                        "title": title,
                        "show_name": show_name,
                        "hosts": hosts,
                        "start_time_seconds": start_time,
                        "source": (
                            "https://www.youtube.com/"
                            f"watch?v={video_id}&"
                            f"t={int(start_time)}s"
                        ),
                    }
                    doc = Document(
                        page_content=chunk.get("text", ""),
                        metadata=doc_metadata,
                    )
                    all_docs.append(doc)

                video_count += 1
    print(f"Loaded {len(all_docs)} chunks from {video_count} videos.")
    return all_docs


# --- Step 2: Vector Store Creation/Loading Function ---
def get_vector_store(force_rebuild=False):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(VECTOR_STORE_PATH) and not force_rebuild:
        print(f"Loading existing vector store from '{VECTOR_STORE_PATH}'...")
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,  # Needed for FAISS with langchain
        )
    else:
        print("Building new vector store...")
        docs = load_docs_from_json()

        vector_store = FAISS.from_documents(docs, embeddings)
        print(f"Saving vector store to '{VECTOR_STORE_PATH}'...")
        vector_store.save_local(VECTOR_STORE_PATH)

    return vector_store


# --- Step 3: RAG Chain Creation Function ---
def create_qa_chain(vector_store):
    print("Creating RAG chain with Ollama...")
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.0)

    prompt_template = """
        CONTEXT:
        {context}

        QUESTION:
        {question}

        INSTRUCTIONS:
        You are a factual Q&A assistant for the 'Kinda Funny' YouTube channel archive.
        The context provided is a direct transcript from all videos on this channel.
        Your task is to answer the user's QUESTION based ONLY on the provided CONTEXT.

        CRITICAL RULES:
        1.  Your entire response must be based **exclusively** on the text provided in the CONTEXT section.
        2.  Do NOT use any of your outside knowledge. Do not add information that is not explicitly mentioned in the context.
        3.  If the context does not contain the answer, you MUST state that you do not have enough information.
        4.  Your final answer should be a direct, concise paragraph. Do NOT use lists or bullet points unless the question specifically asks for them.
        5.  Do NOT output your thought process, reasoning, or any text other than the final answer.

        ANSWER:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 5},  # Top 5 chunks
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


# --- Main Execution ---
if __name__ == "__main__":

    start_time = time.time()
    vector_store = get_vector_store()
    qa_chain = create_qa_chain(vector_store)
    end_time = time.time()

    print(
        "\n--- AI Agent is ready. Setup"
        f" took {end_time - start_time:.2f} seconds. ---"
    )
    print("--- Ask a question, or type 'exit' to quit. ---")

    while True:
        query = input("\n> ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        start_time = time.time()

        print("\nThinking...")
        result = qa_chain.invoke(query)

        print("\nAnswer:")
        print(result["result"])

        print("\nSources:")
        if result.get("source_documents"):
            for source in result["source_documents"]:
                metadata = source.metadata
                print(
                    f"  - In '{metadata['title']}'"
                    f" (at ~{int(metadata['start_time_seconds'] // 60)}"
                    f"m {int(metadata['start_time_seconds'] % 60)}s)"
                )
                print(f"    Link: {metadata['source']}")
        else:
            print("  - No sources found.")

        end_time = time.time()
        print(f"\n...response took {end_time - start_time:.2f} seconds.")
