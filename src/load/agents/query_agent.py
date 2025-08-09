import json
import time

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_postgres import PGVector
from sqlalchemy import create_engine

from load.utils.config import (
    COLLECTION_TABLE,
    CONTEXT_COUNT,
    EMBEDDING_MODEL,
    POSTGRES_DB_PATH,
    QA_MODEL,
)
from load.utils.filtering import get_filter
from load.utils.helpers.database import get_unique_metadata
from load.utils.helpers.datetime import format_duration
from load.utils.helpers.llm import clean_llm_response
from load.utils.prompts import QA_PROMPT


class QueryAgent:
    """Manages the process of querying a document collection to answer questions.

    This agent orchestrates the end-to-end process of handling a user's query.
    It connects to a PGVector database to retrieve relevant document chunks,
    formats them with their metadata, and then passes them as context to a
    Large Language Model (LLM) to generate a final, synthesized answer.

    The primary entry point for using an initialized agent is the
    `process_query` method.

    Attributes:
        llm (OllamaLLM): The language model instance used for generating answers.
        embeddings (HuggingFaceEmbeddings): The model used to create vector
            embeddings for documents and queries.
        vector_store (PGVector): The connection to the vector database where
            documents are stored.
        show_names (list[str]): A list of unique show names fetched from the
            database, used for query filtering.
        hosts (list[str]): A list of unique hosts fetched from the database,
            used for query filtering.
        qa_chain: The LangChain chain object that combines the prompt, LLM,
            and document formatting logic.
    """

    def __init__(self, llm: OllamaLLM) -> None:
        start = time.time()
        print(" -> Initializing QueryAgent...")

        self.llm = llm

        # 1. Initialize embeddings and vector store connection
        print(
            " -> Connecting to vector store and initializing embedding model..."
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = PGVector(
            connection=POSTGRES_DB_PATH,
            collection_name=COLLECTION_TABLE,
            embeddings=self.embeddings,
        )

        # 2. Initialize database connection
        print(" -> Connecting to database and fetching metadata...")
        engine = create_engine(POSTGRES_DB_PATH)
        self.show_names, self.hosts = get_unique_metadata(engine)

        # 4. Build QA Agent
        qa_prompt = PromptTemplate(
            template=QA_PROMPT, input_variables=["input", "context", "topics"]
        )
        self.qa_chain = create_stuff_documents_chain(
            llm=self.llm, prompt=qa_prompt
        )

        end = time.time()
        print(
            "\n--- KFAI Agent is ready."
            f" Setup took {format_duration(end - start)}. ---"
        )
        print(f"Model: {QA_MODEL}")

    def _print_sources(self, result: str, docs: list[Document]) -> None:
        """
        Prints the sources for the given result from the list of documents.
        """
        print("\nSources:")
        source_found = False
        for doc in docs:
            metadata = doc.metadata
            video_id = metadata.get("video_id")
            start_time = metadata.get("start_time")
            if (
                video_id
                and start_time
                and video_id in result
                and str(start_time) in result
            ):
                source_found = True
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
            print("  - No direct sources cited in the response.")

    def _retrieve_documents(self, query: str) -> tuple[list[Document], str]:
        """Retrieves relevant docs from the vector store based on the query."""
        topics, filter_dict = get_filter(query, self.show_names, self.hosts)
        docs = self.vector_store.similarity_search(
            topics, k=CONTEXT_COUNT, filter=filter_dict
        )
        doc_count = len(docs)
        if doc_count == 0:
            return [], ""

        print(f"\nRetrieved {doc_count} docs - [upper limit: {CONTEXT_COUNT}]")
        return docs, topics

    def _format_documents_for_context(
        self, docs: list[Document]
    ) -> list[Document]:
        """Formats retrieved documents into a string for the LLM context."""
        docs_with_metadata = []
        for idx, doc in enumerate(docs, 1):
            metadata = doc.metadata
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
        return docs_with_metadata

    def _generate_response(
        self, query: str, context_docs: list[Document], topics: str
    ) -> str:
        """Invokes the LLM chain to generate a response."""
        print("Thinking...")
        response = self.qa_chain.invoke(
            {
                "input": query,
                "topics": topics,
                "context": context_docs,
            }
        )
        return clean_llm_response(response)

    def _present_results(
        self, result: str, docs: list[Document], start: float
    ) -> None:
        """Prints the final answer and its sources."""
        if result:
            print("\nAnswer:")
            print(result)
            self._print_sources(result, docs)
        else:
            print("  !!  WARNING: No result.")

        end = time.time()
        print(f"\n...response took {format_duration(end - start)}.")

    def process_query(self, query: str) -> None:
        """Processes a single user query from retrieval to final output.

        This method serves as the main entry point for the agent. It
        orchestrates the full pipeline:
            1. Retrieves relevant documents and topics from the vector store.
            2. Formats the documents to be used as context.
            3. Generates a response using the LLM.
            4. Prints the final answer and its sources directly to the console.

        Args:
            query (str): The user's question to be answered.

        Returns:
            None: This method prints results to stdout and does not return a value.
        """

        start = time.time()
        docs, topics = self._retrieve_documents(query)

        if not docs:
            print(
                "  !!  WARNING: No documents found, skipping this question..."
            )
            return

        docs_with_metadata = self._format_documents_for_context(docs)
        result = self._generate_response(query, docs_with_metadata, topics)
        self._present_results(result, docs, start)
