from langchain_postgres import PGVector
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine, text
import time

import config

# --- Configuration ---
POSTGRES_DB_PATH = config.POSTGRES_DB_PATH
COLLECTION_TABLE = "video_transcript_chunks"
EMBEDDING_COLUMN = "embedding"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SELF_QUERY_LLM = "deepseek-r1:8b"
QA_LLM = "qwen3:8b"
PRIMARY_HOST_MAP = {
    "Greg": "Greg Miller",
    "Tim": "Tim Gettys",
    "Nick": "Nick Scarpino",
    "Kevin": "Kevin Coello",
    "Joey": "Joey Noelle",
    "Andy": "Andy Cortez",
    "Barrett": "Barrett Courtney",
    "Blessing": "Blessing Adeoye Jr.",
    "Mike": "Mike Howard",
    "SnowBikeMike": "Mike Howard",
    "Roger": "Roger Pokorny",
    "Parris": "Parris Lilly",
    "Paris": "Parris Lilly",
    "Gary": "Gary Whitta",
    "Fran": "Fran Mirabella III",
    "Janet": "Janet Garcia",
    "Andrea": "Andrea Rene",
    "Tamoor": "Tamoor Hussain",
    "Jared": "Jared Petty",
    "Colin": "Colin Moriarty",
}


# -- Helper functions --
def get_unique_metadata(engine):
    """Queries the database to get all unique show names and hosts."""
    show_names = set()
    hosts = set()
    with engine.connect() as connection:
        # Get unique show names
        show_result = connection.execute(
            text("SELECT DISTINCT show_name FROM videos ORDER BY show_name;")
        )
        for row in show_result:
            if row.show_name:
                show_names.add(row.show_name)

        # Get unique hosts
        host_result = connection.execute(
            text(
                "SELECT DISTINCT unnest(hosts) AS host FROM videos ORDER BY host;"
            )
        )
        for row in host_result:
            if row.host:
                hosts.add(row.host)

    return sorted(show_names), sorted(hosts)


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize database connection
    print(" -> Connecting to database...")
    engine = create_engine(POSTGRES_DB_PATH)

    # 2. Fetch unique metadata
    print(" -> Fetching unique metadata for retriever context...")
    show_names, hosts = get_unique_metadata(engine)

    # 3. Initialize embeddings and vector store connection
    print(" -> Connecting to vector store and initializing embedding model...")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PGVector(
        connection=POSTGRES_DB_PATH,
        collection_name=COLLECTION_TABLE,
        embeddings=embeddings,
    )
    print(" -> Connection successful.")

    # 4. Define metadata for the self-querying retriever
    metadata_field_info = [
        AttributeInfo(
            name="show_name",
            description=(
                "The name of the show the transcript chunk is from, e.g."
                ", 'The GameOverGreggy Show' or 'Gamescast'"
            ),
            type="string",
        ),
        AttributeInfo(
            name="hosts",
            description=(
                "A list of hosts or guests present in the video, e.g."
                ", ['Greg Miller', 'Colin Moriarty']"
            ),
            type="list[string]",
        ),
        AttributeInfo(
            name="published_at",
            description="The date the video was published",
            type="string",
        ),
        AttributeInfo(
            name="title",
            description="The title of the YouTube video",
            type="string",
        ),
    ]
    primary_host_instructions = ", ".join(
        [f"'{k}' likely refers to '{v}'" for k, v in PRIMARY_HOST_MAP.items()]
    )
    document_content_description = f"""
        A text chunk from a YouTube video transcript from the 'Kinda Funny' channel.

        INSTRUCTIONS FOR FILTERING:
        1.  When a user mentions a common first name, assume it refers to a primary host. Use this mapping: {primary_host_instructions}.
        2.  For other names, use the full list of guests and hosts provided below.
        3.  When filtering by show name, use one of the valid show names provided below.

        LIST OF ALL GUESTS AND HOSTS:
        {hosts}

        LIST OF ALL SHOW NAMES:
        {show_names}
    """

    # 5. Instantiate the self-querying retriever
    print(" -> Initializing Self-Querying Retriever...")
    llm_for_retriever = OllamaLLM(model=SELF_QUERY_LLM, temperature=0.0)
    retriever = SelfQueryRetriever.from_llm(
        llm=llm_for_retriever,
        vectorstore=vector_store,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True,  # Change to False after testing
        enable_limit=True,
    )
    print(" -> Retriever is ready.")

    # 6. Build the final RAG chain for Q&A
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

        /no_think
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model=QA_LLM, temperature=1.0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
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

        print("\nThinking...")
        result = qa_chain.invoke({"query": query})

        print("\nAnswer:")
        print(result["result"])

        print("\nSources:")
        if result.get("source_documents"):
            for source in result["source_documents"]:
                metadata = source.metadata
                print(
                    f"  - From video ID {metadata['video_id']} ("
                    f"at ~{int(metadata['start_time'] // 60)}m"
                    f" {int(metadata['start_time'] % 60)}s)"
                )
                print(
                    f"    Link: https://www.youtube.com/watch?v"
                    f"={metadata['video_id']}&t={int(metadata['start_time'])}s"
                )
        else:
            print("  - No sources found.")

        end_time = time.time()
        print(f"\n...response took {end_time - start_time:.2f} seconds.")
