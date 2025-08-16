import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

# The module we are testing
from kfai.loaders.agents.query_agent import QueryAgent

# --- Test Data and Fixtures ---

# A sample document list we can reuse in tests
SAMPLE_DOCS = [
    Document(
        page_content="Content about video 1 at 60s.",
        metadata={
            "video_id": "vid1",
            "start_time": 60,
            "published_at": "2023-01-01T00:00:00Z",
            "title": "Video One",
            "show_name": "Show A",
            "text": "Content about video 1 at 60s.",
        },
    ),
    Document(
        page_content="Content about video 2 at 120s.",
        metadata={
            "video_id": "vid2",
            "start_time": 120,
            "published_at": "2023-02-01T00:00:00Z",
            "title": "Video Two",
            "show_name": "Show B",
            "text": "Content about video 2 at 120s.",
        },
    ),
    # Add a doc with a timestamp over 1 hour to test hour formatting
    Document(
        page_content="Content about video 3 at 3661s.",
        metadata={
            "video_id": "vid3",
            "start_time": 3661,  # 1 hour, 1 minute, 1 second
            "published_at": "2023-03-01T00:00:00Z",
            "title": "Video Three",
            "show_name": "Show C",
            "text": "Content about video 3 at 3661s.",
        },
    ),
]


@pytest.fixture
def mocked_agent(mocker) -> QueryAgent:
    """
    Initializes a QueryAgent with all external dependencies mocked.
    """
    mocker.patch("kfai.loaders.agents.query_agent.HuggingFaceEmbeddings")
    mocker.patch("kfai.loaders.agents.query_agent.PGVector")
    mocker.patch("kfai.loaders.agents.query_agent.create_engine")
    mocker.patch(
        "kfai.loaders.agents.query_agent.get_unique_metadata",
        return_value=(["Show A"], ["Host A"]),
    )
    mock_chain = MagicMock()
    mocker.patch(
        "kfai.loaders.agents.query_agent.create_stuff_documents_chain",
        return_value=mock_chain,
    )
    mock_llm = MagicMock(spec=OllamaLLM)
    agent = QueryAgent(llm=mock_llm)
    agent.mock_chain = mock_chain
    # Mock the vector_store attribute directly on the instance for retrieval tests
    agent.vector_store = MagicMock()
    return agent


# --- Tests for Helper Methods ---


def test_get_structured_sources(mocked_agent):
    """
    Tests structuring, sorting, and time formatting (including hours).
    """
    llm_result = "vid1 at 60s, vid2 at 120s, and vid3 at 3661s."
    structured_sources = mocked_agent._get_structured_sources(
        llm_result, SAMPLE_DOCS
    )
    assert len(structured_sources) == 3
    assert structured_sources[0]["title"] == "Video One"
    assert structured_sources[2]["title"] == "Video Three"
    assert structured_sources[0]["references"][0]["formatted_time"] == "1:00"
    assert (
        structured_sources[2]["references"][0]["formatted_time"] == "1:01:01"
    )  # Covers line 153


def test_get_structured_sources_no_match(mocked_agent):
    llm_result = "This result does not cite any sources."
    structured_sources = mocked_agent._get_structured_sources(
        llm_result, SAMPLE_DOCS
    )
    assert structured_sources == []


def test_sort_documents(mocked_agent):
    unsorted_docs = list(reversed(SAMPLE_DOCS))
    sorted_docs = mocked_agent._sort_documents(unsorted_docs)
    assert sorted_docs[0].metadata["video_id"] == "vid1"


def test_retrieve_documents_single_topic(mocker, mocked_agent):
    """Covers the topic_count < 2 path in _retrieve_documents."""
    mocker.patch(
        "kfai.loaders.agents.query_agent.get_filter",
        return_value=("topics", {"$and": []}),
    )
    mocked_agent.vector_store.similarity_search.return_value = SAMPLE_DOCS
    docs, topics = mocked_agent._retrieve_documents("query")
    assert docs == mocked_agent._sort_documents(SAMPLE_DOCS)
    assert topics == "topics"
    mocked_agent.vector_store.similarity_search.assert_called_once()


def test_retrieve_documents_multiple_topics(mocker, mocked_agent):
    """Covers the topic_count >= 2 path, including deduplication."""
    # Simulate a filter with multiple topics
    multi_topic_filter = {"$and": [{"$or": [{"text": "A"}, {"text": "B"}]}]}
    mocker.patch(
        "kfai.loaders.agents.query_agent.get_filter",
        return_value=("topics", multi_topic_filter),
    )
    # Simulate the vector store returning duplicate docs with scores
    docs_with_scores = [
        (SAMPLE_DOCS[0], 0.9),
        (SAMPLE_DOCS[1], 0.8),
        (SAMPLE_DOCS[0], 0.7),
    ]
    mocked_agent.vector_store.similarity_search_with_relevance_scores.return_value = (
        docs_with_scores
    )

    docs, _ = mocked_agent._retrieve_documents("query")

    # Assert that deduplication worked
    assert len(docs) == 2
    # Assert that the call was made twice (once for each topic)
    assert (
        mocked_agent.vector_store.similarity_search_with_relevance_scores.call_count
        == 2
    )


def test_retrieve_documents_no_filter(mocker, mocked_agent):
    """Covers the path where get_filter returns no filter_dict."""
    mocker.patch(
        "kfai.loaders.agents.query_agent.get_filter",
        return_value=("topics", None),
    )
    docs, topics = mocked_agent._retrieve_documents("query")
    assert docs == []
    assert topics == ""


def test_retrieve_documents_no_docs_found(mocker, mocked_agent):
    """Covers the path where the vector store returns no documents."""
    mocker.patch(
        "kfai.loaders.agents.query_agent.get_filter",
        return_value=("topics", {"$and": []}),
    )
    mocked_agent.vector_store.similarity_search.return_value = []
    docs, topics = mocked_agent._retrieve_documents("query")
    assert docs == []
    assert topics == ""


def test_retrieve_documents_multi_topic_breaks_at_context_limit(
    mocker, mocked_agent
):
    """
    Covers the break statement when CONTEXT_COUNT is reached during deduplication.
    """
    # 1. Arrange
    # Patch the constant to a small number just for this test.
    mocker.patch("kfai.loaders.agents.query_agent.CONTEXT_COUNT", 2)

    # Simulate a filter with multiple topics to trigger the correct code path.
    multi_topic_filter = {"$and": [{"$or": [{"text": "A"}, {"text": "B"}]}]}
    mocker.patch(
        "kfai.loaders.agents.query_agent.get_filter",
        return_value=("topics", multi_topic_filter),
    )

    # Provide more unique docs (3) than the mocked CONTEXT_COUNT (2).
    # The SAMPLE_DOCS fixture has 3 unique documents.
    docs_with_scores = [
        (SAMPLE_DOCS[0], 0.9),
        (SAMPLE_DOCS[1], 0.8),
        (
            SAMPLE_DOCS[2],
            0.7,
        ),  # This doc should be ignored because the loop breaks.
    ]
    mocked_agent.vector_store.similarity_search_with_relevance_scores.return_value = (
        docs_with_scores
    )

    # 2. Act
    docs, _ = mocked_agent._retrieve_documents("query")

    # 3. Assert
    # The length of the final docs list should be exactly the mocked CONTEXT_COUNT,
    # proving that the loop correctly broke after finding 2 documents.
    assert len(docs) == 2


def test_format_documents_for_context(mocked_agent):
    formatted_docs = mocked_agent._format_documents_for_context(
        [SAMPLE_DOCS[0]]
    )
    assert "TRANSCRIPT #1 TEXT" in formatted_docs[0].page_content
    assert "TRANSCRIPT #1 METADATA" in formatted_docs[0].page_content


def test_generate_response(mocker, mocked_agent):
    """Covers _generate_response."""
    mocker.patch(
        "kfai.loaders.agents.query_agent.clean_llm_response",
        return_value="Cleaned Answer",
    )
    result = mocked_agent._generate_response("query", SAMPLE_DOCS, "topics")
    assert result == "Cleaned Answer"
    mocked_agent.mock_chain.invoke.assert_called_once_with(
        {
            "input": "query",
            "topics": "topics",
            "context": SAMPLE_DOCS,
        }
    )


def test_present_results(mocker, mocked_agent):
    """Covers _present_results and its call to _print_sources."""
    mock_print = mocker.patch("builtins.print")
    mock_print_sources = mocker.patch.object(mocked_agent, "_print_sources")

    # Test with a valid result
    mocked_agent._present_results("An answer", SAMPLE_DOCS, time.time())
    mock_print.assert_any_call("\nAnswer:")
    mock_print_sources.assert_called_once_with("An answer", SAMPLE_DOCS)

    # Test with an empty result
    mocked_agent._present_results("", SAMPLE_DOCS, time.time())
    mock_print.assert_any_call("  !!  WARNING: No result.")


def test_print_sources(mocker, mocked_agent):
    """Covers the _print_sources method."""
    mock_print = mocker.patch("builtins.print")

    # Test with sources
    mocked_agent._print_sources("vid1 at 60s", SAMPLE_DOCS)
    mock_print.assert_any_call("  Video: Video One")

    # Test with no sources
    mocked_agent._print_sources("no sources here", SAMPLE_DOCS)
    mock_print.assert_any_call("  - No direct sources cited in the response.")


def test_format_response_for_gui(mocked_agent):
    """Covers _format_response_for_gui."""
    # Test with sources
    result_with_sources = mocked_agent._format_response_for_gui(
        "vid1 at 60s", SAMPLE_DOCS
    )
    assert "**Sources:**" in result_with_sources
    assert (
        "[Video One](https://www.youtube.com/watch?v=vid1)"
        in result_with_sources
    )

    # Test without sources
    result_no_sources = mocked_agent._format_response_for_gui(
        "no sources", SAMPLE_DOCS
    )
    assert "No direct sources cited" in result_no_sources


# --- Tests for Main Orchestration Method: process_query ---


def test_process_query_cli_happy_path(mocker, mocked_agent):
    mocker.patch.object(
        mocked_agent,
        "_retrieve_documents",
        return_value=(SAMPLE_DOCS, "topics"),
    )
    mocker.patch.object(
        mocked_agent, "_generate_response", return_value="Final LLM Answer"
    )
    mock_present_results = mocker.patch.object(
        mocked_agent, "_present_results"
    )
    result = mocked_agent.process_query("some query", is_gui=False)
    assert result is None
    mock_present_results.assert_called_once()
    call_args = mock_present_results.call_args[0]
    assert call_args[0] == "Final LLM Answer"
    assert call_args[1] == SAMPLE_DOCS


def test_process_query_gui_happy_path(mocker, mocked_agent):
    mocker.patch.object(
        mocked_agent,
        "_retrieve_documents",
        return_value=(SAMPLE_DOCS, "topics"),
    )
    mocker.patch.object(
        mocked_agent, "_generate_response", return_value="Final LLM Answer"
    )
    mocker.patch.object(
        mocked_agent,
        "_format_response_for_gui",
        return_value="<Formatted Markdown>",
    )
    result = mocked_agent.process_query("some query", is_gui=True)
    assert result == "<Formatted Markdown>"


def test_process_query_no_docs_found_cli(mocker, mocked_agent):
    mocker.patch.object(
        mocked_agent, "_retrieve_documents", return_value=([], "")
    )
    mock_generate_response = mocker.patch.object(
        mocked_agent, "_generate_response"
    )
    result = mocked_agent.process_query("query with no results", is_gui=False)
    assert result is None
    mock_generate_response.assert_not_called()


def test_process_query_no_docs_found_gui(mocker, mocked_agent):
    mocker.patch.object(
        mocked_agent, "_retrieve_documents", return_value=([], "")
    )
    result = mocked_agent.process_query("query with no results", is_gui=True)
    assert "could not find any relevant documents" in result


def test_process_query_no_llm_result_gui(mocker, mocked_agent):
    mocker.patch.object(
        mocked_agent,
        "_retrieve_documents",
        return_value=(SAMPLE_DOCS, "topics"),
    )
    mocker.patch.object(mocked_agent, "_generate_response", return_value="")
    result = mocked_agent.process_query("some query", is_gui=True)
    assert "model did not generate a response" in result
