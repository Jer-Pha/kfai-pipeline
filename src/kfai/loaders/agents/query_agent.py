from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import TYPE_CHECKING, cast

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

from kfai.loaders.utils.config import (
    COLLECTION_TABLE,
    CONTEXT_COUNT,
    EMBEDDING_MODEL,
    POSTGRES_DB_PATH,
    QA_MODEL,
    TIMESTAMP_BUFFER,
)
from kfai.loaders.utils.filtering import build_filter
from kfai.loaders.utils.helpers.database import get_unique_metadata
from kfai.loaders.utils.helpers.datetime import format_duration
from kfai.loaders.utils.parsing import parse_query
from kfai.loaders.utils.prompts import QA_PROMPT
from kfai.loaders.utils.types import (
    AgentResponse,
    EmbeddingCMetadata,
    GroupedSourceData,
    SourceCitation,
)

if TYPE_CHECKING:
    from langchain_ollama import OllamaLLM

    from kfai.loaders.utils.types import TimestampReference, VideoDataSource


class QueryAgent:
    """Manages the process of querying a document collection to answer
    questions.

    This agent orchestrates the end-to-end process of handling a user's query.
    It connects to a PGVector database to retrieve relevant document chunks,
    formats them with their metadata, and then passes them as context to a
    Large Language Model (LLM) to generate a final, synthesized answer.

    The primary entry point for using an initialized agent is the
    `process_query` method.

    Attributes:
        llm (OllamaLLM): The language model instance used for generating
            answers.
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
            " -> Connecting to vector store and initializing embedding model.."
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = PGVector(
            connection=POSTGRES_DB_PATH,
            collection_name=COLLECTION_TABLE,
            embeddings=self.embeddings,
        )
        self.parser = PydanticOutputParser(pydantic_object=AgentResponse)

        # 2. Initialize database connection
        print(" -> Connecting to database and fetching metadata...")
        engine = create_engine(POSTGRES_DB_PATH)
        self.show_names, self.hosts = get_unique_metadata(engine)

        # 4. Build QA Agent
        qa_prompt = PromptTemplate(
            template=QA_PROMPT,
            input_variables=["input", "context"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.qa_chain = qa_prompt | self.llm | self.parser

        end = time.time()
        print(
            "\n--- KFAI Agent is ready."
            f" Setup took {format_duration(end - start)}. ---"
        )
        print(f"Model: {QA_MODEL}")

    def _get_structured_sources(
        self, sources: list[SourceCitation], docs: list[Document]
    ) -> list[VideoDataSource]:
        """
        Gathers, groups, and sorts cited sources into a structured list of
        dictionaries suitable for a front-end to render.
        """

        # Step 1: Gather and group all cited sources by video_id
        grouped_sources: dict[str, GroupedSourceData] = {}
        response_video_ids = [s.video_id for s in sources]
        response_start_times = [int(s.start_time) for s in sources]

        for doc in docs:
            metadata: EmbeddingCMetadata = cast(
                EmbeddingCMetadata, doc.metadata
            )
            video_id = metadata["video_id"]
            start_time = metadata["start_time"]

            if (
                video_id
                and start_time is not None
                and video_id in response_video_ids
                and int(start_time) in response_start_times
            ):
                if video_id not in grouped_sources:
                    grouped_sources[video_id] = GroupedSourceData(
                        timestamps=set(), metadata=metadata
                    )
                    # Remove chunk-specific text for video-wide metadata
                    grouped_sources[video_id]["metadata"]["text"] = ""

                grouped_sources[video_id]["timestamps"].add(int(start_time))

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

                timestamp_sec = total_seconds + TIMESTAMP_BUFFER
                timestamps.append(
                    {
                        "timestamp_sec": timestamp_sec,
                        "formatted_time": formatted_time,
                        "timestamp_href": (
                            "https://www.youtube.com/"
                            f"watch?v={video_id}&t={timestamp_sec}s"
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

    def _print_sources(
        self, sources: list[SourceCitation], docs: list[Document]
    ) -> None:
        """Prints the sources for the given response using a structured
        format.
        """
        print("\nSources:")
        structured_sources = self._get_structured_sources(sources, docs)

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
                "  Referenced at:",
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

    def _retrieve_documents(self, query: str) -> list[Document] | None:
        """Retrieves relevant docs from the vector store based on the query."""
        parsed_response = parse_query(query, self.show_names, self.hosts)

        if not parsed_response:
            return None

        filter_dict = build_filter(parsed_response)
        topics = parsed_response.topics

        if not filter_dict and not topics:
            return None

        if filter_dict is None:
            filter_dict = {"$and": []}

        unfiltered_docs: list[tuple[Document, float]] = []

        if not topics:
            unfiltered_docs.extend(
                self.vector_store.similarity_search_with_relevance_scores(
                    query, k=CONTEXT_COUNT, filter=filter_dict
                )
            )

        # Get docs for each topic filter
        for topic in topics:
            print(f"  Gathering docs for topic: {topic}")
            temp_filter = dict(deepcopy(filter_dict))

            # Include title in topic search
            hybrid_topic_filter = {
                "$or": [
                    {"title": {"$ilike": f"%{topic}%"}},
                    {"text": {"$ilike": f"%{topic}%"}},
                ]
            }
            temp_filter["$and"].append(hybrid_topic_filter)

            # Build relevant query
            temp_topics = topics.copy()
            temp_topics.remove(topic)
            temp_query = ", ".join(temp_topics) if temp_topics else query

            unfiltered_docs.extend(
                self.vector_store.similarity_search_with_relevance_scores(
                    temp_query, k=CONTEXT_COUNT, filter=temp_filter
                )
            )

        # Sort docs by relevance score
        unfiltered_docs.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate the results, limiting to CONTEXT_COUNT
        docs = []
        seen_docs = set()
        for doc, _ in unfiltered_docs:
            doc_metadata = cast(EmbeddingCMetadata, doc.metadata)
            doc_id = f"{doc_metadata['video_id']}-{doc_metadata['start_time']}"
            if doc_id not in seen_docs:
                docs.append(doc)
                seen_docs.add(doc_id)

            if len(seen_docs) >= CONTEXT_COUNT:
                break

        docs = self._sort_documents(docs)
        doc_count = len(docs)
        if doc_count == 0:
            return None

        print(f"\nRetrieved {doc_count} docs - [upper limit: {CONTEXT_COUNT}]")
        return docs

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
        self, query: str, context_docs: list[Document]
    ) -> AgentResponse:
        """Invokes the LLM chain to generate a response."""
        print("Thinking...")
        response: AgentResponse = self.qa_chain.invoke(
            {
                "input": query,
                "context": context_docs,
            }
        )
        return response

    def process_query(self, query: str, is_gui: bool = False) -> str | None:
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
            None: This method prints results to stdout and does not
                  return a value.
        """

        start = time.time()

        docs = self._retrieve_documents(query)

        if not docs and is_gui:
            return (
                "I could not find any relevant documents to answer your"
                " question. Please try rephrasing."
            )
        elif not docs:
            print(
                "  !!  WARNING: No documents found, skipping this question..."
            )
            return None

        docs_with_metadata = self._format_documents_for_context(docs)

        response = self._generate_response(query, docs_with_metadata)

        if is_gui:
            final_response = self._format_response_for_gui(response, docs)
        else:
            print("\nAnswer:")
            print(response.query_response)
            self._print_sources(response.sources, docs)

        end = time.time()
        print(f"\n...response took {format_duration(end - start)}.")

        return final_response if is_gui else None

    def _format_response_for_gui(
        self, response: AgentResponse, docs: list[Document]
    ) -> str:
        """Formats the final result and sources into a single Markdown
        string.
        """
        structured_sources = self._get_structured_sources(
            response.sources, docs
        )

        response_parts = [response.query_response]

        if structured_sources:
            response_parts.append("\n\n---\n**Sources:**")
            for video_data in structured_sources:
                # Create a clickable Markdown image link
                thumbnail_markdown = (
                    f"[![{video_data['title']}]({video_data['thumbnail_src']})]"
                    f"({video_data['video_href']})"
                )
                # Video Title and Link
                response_parts.append(f"\n\n**{video_data['title']}**")
                response_parts.append(thumbnail_markdown)

                # Show Name
                response_parts.append(
                    f"*   **Show:** {video_data['show_name']}"
                )

                # Timestamps
                time_links = ", ".join(
                    f"[{ref['formatted_time']}]({ref['timestamp_href']})"
                    for ref in video_data["references"]
                )
                response_parts.append(f"*   **Referenced at:** {time_links}")
        else:
            response_parts.append(
                "\n\n---\n**Sources:**\n- No direct sources cited in the"
                " response."
            )

        return "\n".join(response_parts)
