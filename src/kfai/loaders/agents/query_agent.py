import json
import time
from copy import deepcopy
from typing import Iterable, cast

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_postgres import PGVector
from sqlalchemy import create_engine

from kfai.loaders.utils.config import (
    COLLECTION_TABLE,
    CONTEXT_COUNT,
    EMBEDDING_MODEL,
    POSTGRES_DB_PATH,
    QA_MODEL,
)
from kfai.loaders.utils.filtering import get_filter
from kfai.loaders.utils.helpers.database import get_unique_metadata
from kfai.loaders.utils.helpers.datetime import format_duration
from kfai.loaders.utils.helpers.llm import clean_llm_response
from kfai.loaders.utils.prompts import QA_PROMPT
from kfai.loaders.utils.types import (
    EmbeddingCMetadata,
    GroupedSourceData,
    PGVectorText,
    TimestampReference,
    VideoDataSource,
)


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

    def _get_structured_sources(
        self, result: str, docs: list[Document]
    ) -> list[VideoDataSource]:
        """
        Gathers, groups, and sorts cited sources into a structured list of
        dictionaries suitable for a front-end to render.
        """

        # Step 1: Gather and group all cited sources by video_id
        grouped_sources: dict[str, GroupedSourceData] = {}

        for doc in docs:
            metadata: EmbeddingCMetadata = cast(
                EmbeddingCMetadata, doc.metadata
            )
            video_id = metadata["video_id"]
            start_time = metadata["start_time"]

            if (
                video_id
                and start_time is not None
                and video_id in result
                and str(int(start_time)) in result
            ):
                if "timestamps" not in grouped_sources[video_id]:
                    grouped_sources[video_id]["timestamps"] = set()
                grouped_sources[video_id]["timestamps"].add(int(start_time))
                if "metadata" not in grouped_sources[video_id]:
                    grouped_sources[video_id]["metadata"] = metadata
                    # Remove chunk-specific text for video-wide metadata
                    grouped_sources[video_id]["metadata"]["text"] = ""

        if not grouped_sources:
            return []

        # Step 2: Convert the dictionary into a list and sort by release date
        sorted_videos: list[GroupedSourceData] = sorted(
            grouped_sources.values(),
            key=lambda video: video["metadata"]["published_at"],
        )

        # Step 3: Build the structured, sorted data
        source_list: list[VideoDataSource] = []
        for video_info in sorted_videos:
            metadata = video_info["metadata"]
            video_id = metadata["video_id"]

            sorted_timestamps = sorted(list(video_info["timestamps"]))
            timestamps: list[TimestampReference] = []

            for total_seconds in sorted_timestamps:
                minutes, seconds = divmod(total_seconds, 60)
                hours, minutes = divmod(minutes, 60)

                if hours > 0:
                    formatted_time = f"{hours}:{minutes:02d}:{seconds:02d}"
                else:
                    formatted_time = f"{minutes}:{seconds:02d}"

                timestamps.append(
                    {
                        "timestamp_sec": total_seconds,
                        "formatted_time": formatted_time,
                        "timestamp_href": (
                            "https://www.youtube.com/"
                            f"watch?v={video_id}&t={total_seconds}s"
                        ),
                    }
                )

            video_data: VideoDataSource = {
                "title": metadata["title"],
                "show_name": metadata["show_name"],
                "video_href": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail_src": (
                    f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"
                ),
                "references": timestamps,
            }
            source_list.append(video_data)

        return source_list

    def _print_sources(self, result: str, docs: list[Document]) -> None:
        """Prints the sources for the given result using a structured format."""
        print("\nSources:")
        structured_sources = self._get_structured_sources(result, docs)

        if not structured_sources:
            print("  - No direct sources cited in the response.")
            return

        for video_data in structured_sources:
            print("\n" + "=" * 50)
            print(f"  Video: {video_data['title']}")
            print(f"  Show:  {video_data['show_name']}")
            print(f"  Link:  {video_data['video_href']}")
            print(f"  Image: {video_data['thumbnail_src']}")
            print(
                f"  Referenced at:",
                ", ".join(
                    ref["formatted_time"] for ref in video_data["references"]
                ),
            )
        else:
            print("\n" + "=" * 50, end="\n\n")

    def _sort_documents(self, docs: list[Document]) -> list[Document]:
        """Sorts documents by their metadata's publish date and start time."""
        return sorted(
            docs,
            key=lambda doc: (
                cast(EmbeddingCMetadata, doc.metadata)["published_at"],
                cast(EmbeddingCMetadata, doc.metadata)["video_id"],
                cast(EmbeddingCMetadata, doc.metadata)["start_time"],
            ),
        )

    def _retrieve_documents(self, query: str) -> tuple[list[Document], str]:
        """Retrieves relevant docs from the vector store based on the query."""
        topics, filter_dict = get_filter(query, self.show_names, self.hosts)

        topic_filters: list[dict[str, str]] = []

        if filter_dict:
            for filter in filter_dict.get("$and", []):
                or_clause = filter.get("$or", dict())
                if or_clause:
                    topic_filters.extend(
                        cast(Iterable[dict[str, str]], or_clause)
                    )
        else:
            return [], ""

        assert filter_dict is not None  # For mypy

        docs = []
        topic_count = len(topic_filters)

        if topic_count < 2:  # One or zero topics
            docs = self.vector_store.similarity_search(
                topics, k=CONTEXT_COUNT, filter=filter_dict
            )
        else:
            # Get docs for each topic filter
            unfiltered_docs: list[tuple[Document, float]] = []
            for topic_filter in topic_filters:
                temp_filter = dict(deepcopy(filter_dict))
                if "$and" in temp_filter:
                    assert isinstance(temp_filter["$and"], list)
                    for item in temp_filter["$and"]:
                        if "$or" in item:
                            temp_filter["$and"].remove(item)
                            temp_filter["$and"].append(
                                cast(
                                    dict[str, list[PGVectorText]], topic_filter
                                )
                            )
                            break
                    unfiltered_docs.extend(
                        self.vector_store.similarity_search_with_relevance_scores(
                            topics, k=CONTEXT_COUNT, filter=temp_filter
                        )
                    )

            # Sort docs by relevance score
            unfiltered_docs.sort(key=lambda x: x[1], reverse=True)

            # Deduplicate the results, limiting to CONTEXT_COUNT
            seen_docs = set()
            for doc, _ in unfiltered_docs:
                doc_metadata: EmbeddingCMetadata = cast(
                    EmbeddingCMetadata, doc.metadata
                )
                doc_id = (
                    f"{doc_metadata['video_id']}"
                    f"-{doc_metadata['start_time']}"
                )
                if doc_id not in seen_docs:
                    docs.append(doc)
                    seen_docs.add(doc_id)

                if len(seen_docs) >= CONTEXT_COUNT:
                    break

        docs = self._sort_documents(docs)
        doc_count = len(docs)
        if doc_count == 0:
            return [], ""

        print(
            f"\nRetrieved {doc_count} docs - [upper limit:"
            f" {CONTEXT_COUNT}]"
        )
        return docs, topics

    def _format_documents_for_context(
        self, docs: list[Document]
    ) -> list[Document]:
        """Formats retrieved documents into a string for the LLM context."""
        docs_with_metadata = []
        for idx, doc in enumerate(docs, 1):
            metadata: EmbeddingCMetadata = cast(
                EmbeddingCMetadata, doc.metadata
            )
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
