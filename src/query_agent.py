from langchain_postgres import PGVector
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains.retrieval_qa.base import RetrievalQA
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

    # SQL injection prevention
    show_names = [s.replace("{", "").replace("}", "") for s in show_names]
    hosts = [h.replace("{", "").replace("}", "") for h in hosts]

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
                ', "The GameOverGreggy Show"'
            ),
            type="string",
        ),
        AttributeInfo(
            name="hosts",
            description=(
                "A list of hosts or guests present in the video, e.g."
                ', ["Greg Miller","Tim Gettys","Nick Scarpino",'
                '"Colin Moriarty","Michael Rosenbaum"]'
            ),
            type="list[string]",
        ),
        AttributeInfo(
            name="published_at",
            description=(
                "The date the video was published, e.g."
                ', "2014-02-03 06:00:04"'
            ),
            type="timestamp",
        ),
    ]
    primary_host_instructions = ", ".join(
        [f"'{k}' likely refers to '{v}'" for k, v in PRIMARY_HOST_MAP.items()]
    )
    retriever_prompt = f"""
        You are a query translator. Your task is to convert a user's natural language query into a structured JSON object in the format defined below.

        << Structured Request Schema >>
        Respond using a markdown code block with a JSON object like this:

        ```json
        {{{{
            "query": "string",  // The user's original input, minus any metadata constraints
            "filter": "string"  // A stringified logical filter expression, or "NO_FILTER"
        }}}}
        ```

        USER QUERY:
        {{query}}

        FILTER FORMAT:
            - Comparison operators: eq, ne, gt, gte, lt, lte, like, in, nin
            - Logical operators: and, or, not
            - Filters must be valid logical expressions using only the attributes listed below
            - Use only attribute names exactly as provided
            - Use "YYYY-MM-DD" format for date comparisons
            - If no metadata constraints are present, return "NO_FILTER"

        AVAILABLE METADATA ATTRIBUTES:
            1. show_name (string)
                - Name of the show, e.g. "The GameOverGreggy Show"
                - Use eq comparator only

            2. hosts (list[string])
                - List of hosts or guests in the video
                - Use in with one or more names
                - If the user mentions a first name, map it using this list:
                {primary_host_instructions}

            3. published_at (timestamp)
                - Full timestamp the video was published (e.g. "2014-02-03 00:00:00")
                - Use gt, lt, gte, lte, or eq
                - Always include the time as midnight ("00:00:00")

        ONLY USE FILTERS IF THEY APPLY.
            - Do not include metadata conditions in the query string.
            - If the user mentions something unrelated to metadata, ignore it for filtering.

        LIST OF SHOW NAMES:
        {show_names}

        LIST OF KNOWN HOSTS AND GUESTS:
        {hosts}

        << Example >>

        User Input:
        Episodes of PS I Love You XOXO with Colin and Greg before 2018

        Structured JSON Response:

        ```json
        {{{{
            "query": "Episodes of PS I Love You XOXO with Colin and Greg before 2018",
            "filter": "and(eq(\\"show_name\\", \\"PS I Love You XOXO\\"), in(\\"hosts\\", [\\"Colin Moriarty\\", \\"Greg Miller\\"]), lt(\\"published_at\\", \\"2018-01-01 00:00:00\\"))"
        }}}}
        ```

        If no filters can be extracted from the user input, return:

        ```json
        {{{{
            "query": "Some natural question",
            "filter": "NO_FILTER"
        }}}}
        ```

        /no_think
    """

    # 5. Instantiate the self-querying retriever
    print(" -> Initializing Self-Querying Retriever...")
    retriever_llm_options = {
        "temperature": 0.3,
        "num_predict": 2048,  # Hard stop after 2048 tokens to prevent infinite loops
    }
    llm_for_retriever = OllamaLLM(
        model=SELF_QUERY_LLM, **retriever_llm_options
    )

    retriever = SelfQueryRetriever.from_llm(
        llm=llm_for_retriever,
        vectorstore=vector_store,
        document_contents=retriever_prompt,
        metadata_field_info=metadata_field_info,
        verbose=True,  # Change to False after testing
        enable_limit=True,
    )
    print(" -> Retriever is ready.")

    # Testing
    print(" -> Submitting test query now...")
    try:
        retriever.invoke(
            input="What is Greg Miller's favorite video game console?",
            kwargs={
                "query": "On PS I Love You, what did Greg and Colin say about Rocket League?"
            },
        )
    except Exception as e:
        print(f"Error directly from retriever: {e}")

    # 6. Build the final RAG chain for Q&A
    qa_prompt = """
        CONTEXT:
        {context}

        USER QUERY:
        {question}

        INSTRUCTIONS:
        - You are a factual Q&A assistant for the 'Kinda Funny' YouTube channel archive.
        - The context provided below is a direct transcript from episodes.
        - Respond to the USER QUERY based **only** on this CONTEXT.

        IMPORTANT RULES:
        1. Do not use outside knowledge — only what’s in the CONTEXT.
        2. If the CONTEXT lacks the answer, say so directly.
        3. Format your answer as a short, direct paragraph (no lists or bullets unless requested).
        4. Do not include your reasoning, thoughts, or any internal process — just the answer.
        5. Do not repeat the user's question.
        6. Treat the CONTEXT as possibly incomplete or informal (transcript-based).

        ANSWER:

        /no_think
    """

    prompt = PromptTemplate(
        template=qa_prompt, input_variables=["context", "question"]
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
