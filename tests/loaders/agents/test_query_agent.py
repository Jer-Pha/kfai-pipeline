from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

# The module we are testing
from kfai.loaders.agents.query_agent import QueryAgent
from kfai.loaders.utils.types import (
    AgentResponse,
    QueryParseResponse,
    SourceCitation,
)

# --- Test Data and Fixtures ---
SAMPLE_DOCS = [
    Document(
        page_content="Doc 2",
        metadata={
            "video_id": "v2",
            "start_time": 20.0,
            "published_at": 2,
            "title": "T2",
            "show_name": "S2",
        },
    ),
    Document(
        page_content="Doc 1",
        metadata={
            "video_id": "v1",
            "start_time": 10.0,
            "published_at": 1,
            "title": "T1",
            "show_name": "S1",
        },
    ),
]


@pytest.fixture
def mocked_agent(mocker):
    """Initializes a QueryAgent with all external dependencies mocked."""
    mocker.patch("kfai.loaders.agents.query_agent.HuggingFaceEmbeddings")
    mocker.patch("kfai.loaders.agents.query_agent.PGVector")
    mocker.patch("kfai.loaders.agents.query_agent.create_engine")
    mocker.patch(
        "kfai.loaders.agents.query_agent.get_unique_metadata",
        return_value=([], []),
    )
    mock_chain = MagicMock()
    mocker.patch("kfai.loaders.agents.query_agent.PydanticOutputParser")
    mocker.patch(
        "kfai.loaders.agents.query_agent.PromptTemplate"
    ).return_value.__or__.return_value.__or__.return_value = mock_chain

    agent = QueryAgent(llm=MagicMock())
    agent.vector_store = MagicMock()
    agent.qa_chain = mock_chain
    return agent


# --- Test Suite ---


def test_retrieve_documents_happy_path(mocker, mocked_agent):
    mocker.patch(
        "kfai.loaders.agents.query_agent.parse_query",
        return_value=QueryParseResponse(topics=["topic1"]),
    )
    mocker.patch(
        "kfai.loaders.agents.query_agent.build_filter",
        return_value={"$and": []},
    )
    docs_with_scores = [
        (SAMPLE_DOCS[0], 0.9),
        (SAMPLE_DOCS[1], 0.8),
        (SAMPLE_DOCS[0], 0.7),
    ]
    mocked_agent.vector_store.similarity_search_with_relevance_scores.return_value = docs_with_scores  # noqa: E501
    docs = mocked_agent._retrieve_documents("query")
    assert docs is not None
    assert len(docs) == 2


@pytest.mark.parametrize(
    "parse_return, build_return",
    [
        (None, None),
        (QueryParseResponse(topics=[]), None),
    ],
)
def test_retrieve_documents_returns_none(
    mocker, mocked_agent, parse_return, build_return
):
    mocker.patch(
        "kfai.loaders.agents.query_agent.parse_query",
        return_value=parse_return,
    )
    mocker.patch(
        "kfai.loaders.agents.query_agent.build_filter",
        return_value=build_return,
    )
    assert mocked_agent._retrieve_documents("query") is None


def test_retrieve_documents_no_docs_found(mocker, mocked_agent):
    """Covers the path where the vector store returns no documents."""
    mocker.patch(
        "kfai.loaders.agents.query_agent.parse_query",
        return_value=QueryParseResponse(topics=["topic1"]),
    )
    mocker.patch(
        "kfai.loaders.agents.query_agent.build_filter",
        return_value={"$and": []},
    )
    mocked_agent.vector_store.similarity_search_with_relevance_scores.return_value = []  # noqa: E501
    assert mocked_agent._retrieve_documents("query") is None


def test_sort_documents(mocked_agent):
    """Tests the document sorting logic."""
    sorted_docs = mocked_agent._sort_documents(SAMPLE_DOCS)
    assert sorted_docs[0].metadata["video_id"] == "v1"
    assert sorted_docs[1].metadata["video_id"] == "v2"


def test_format_documents_for_context(mocked_agent):
    """Tests the context formatting logic."""
    formatted_docs = mocked_agent._format_documents_for_context(SAMPLE_DOCS)
    assert "TRANSCRIPT #1 TEXT" in formatted_docs[0].page_content
    assert "TRANSCRIPT #2 TEXT" in formatted_docs[1].page_content


def test_generate_response(mocked_agent):
    """Tests the response generation call."""
    mocked_agent._generate_response("query", SAMPLE_DOCS)
    mocked_agent.qa_chain.invoke.assert_called_once_with(
        {
            "input": "query",
            "context": SAMPLE_DOCS,
        }
    )


def test_get_structured_sources(mocked_agent):
    sources = [SourceCitation(video_id="v1", start_time=10.0)]
    structured = mocked_agent._get_structured_sources(sources, SAMPLE_DOCS)
    assert len(structured) == 1
    assert structured[0]["title"] == "T1"


def test_print_sources(mocker, mocked_agent):
    """Tests the CLI source printing logic."""
    mock_print = mocker.patch("builtins.print")
    # Test with sources
    sources = [SourceCitation(video_id="v1", start_time=10.0)]
    mocked_agent._print_sources(sources, SAMPLE_DOCS)
    mock_print.assert_any_call("  Video: T1")
    # Test with no sources
    mocked_agent._print_sources([], SAMPLE_DOCS)
    mock_print.assert_any_call("  - No direct sources cited in the response.")


def test_format_response_for_gui(mocked_agent):
    """Tests the GUI response formatting logic."""
    response = AgentResponse(
        query_response="Answer",
        sources=[SourceCitation(video_id="v1", start_time=10.0)],
    )
    formatted_str = mocked_agent._format_response_for_gui(
        response, SAMPLE_DOCS
    )
    assert "**Sources:**" in formatted_str
    assert "![T1](https://i.ytimg.com/vi/v1/mqdefault.jpg)" in formatted_str
    # Test with no sources
    response_no_sources = AgentResponse(query_response="Answer", sources=[])
    formatted_str_no_sources = mocked_agent._format_response_for_gui(
        response_no_sources, SAMPLE_DOCS
    )
    assert "No direct sources cited" in formatted_str_no_sources


def test_process_query_happy_path_cli(mocker, mocked_agent):
    mocker.patch.object(
        mocked_agent, "_retrieve_documents", return_value=SAMPLE_DOCS
    )
    mock_response_obj = AgentResponse(query_response="Test Answer", sources=[])
    mocker.patch.object(
        mocked_agent, "_generate_response", return_value=mock_response_obj
    )
    result = mocked_agent.process_query("query", is_gui=False)
    assert result is None


def test_process_query_happy_path_gui(mocker, mocked_agent):
    """Tests the successful GUI flow."""
    mocker.patch.object(
        mocked_agent, "_retrieve_documents", return_value=SAMPLE_DOCS
    )
    mock_response_obj = AgentResponse(query_response="Test Answer", sources=[])
    mocker.patch.object(
        mocked_agent, "_generate_response", return_value=mock_response_obj
    )
    mock_format_gui = mocker.patch.object(
        mocked_agent,
        "_format_response_for_gui",
        return_value="Formatted GUI String",
    )

    result = mocked_agent.process_query("query", is_gui=True)

    assert result == "Formatted GUI String"
    mock_format_gui.assert_called_once_with(mock_response_obj, SAMPLE_DOCS)


def test_process_query_no_docs_cli(mocked_agent):
    """Tests the CLI flow when no documents are found."""
    mocked_agent._retrieve_documents = MagicMock(return_value=None)
    result = mocked_agent.process_query("query", is_gui=False)
    assert result is None


def test_process_query_no_docs_gui(mocked_agent):
    mocked_agent._retrieve_documents = MagicMock(return_value=None)
    result = mocked_agent.process_query("query", is_gui=True)
    assert "could not find any relevant documents" in result


# Add this test to cover the hour formatting in _get_structured_sources
def test_get_structured_sources_formats_hours(mocked_agent):
    """Tests that timestamps over an hour are formatted correctly."""
    # Arrange
    # Create a doc with a start_time > 3600 seconds
    doc_with_hour = Document(
        page_content="Doc 3",
        metadata={
            "video_id": "v3",
            "start_time": 3661.0,
            "published_at": 3,
            "title": "T3",
            "show_name": "S3",
        },
    )
    sources = [SourceCitation(video_id="v3", start_time=3661.0)]

    # Act
    structured = mocked_agent._get_structured_sources(sources, [doc_with_hour])

    # Assert
    assert len(structured) == 1
    assert structured[0]["references"][0]["formatted_time"] == "1:01:01"


# Add this test to cover the `filter_dict is None` branch in
# _retrieve_documents
def test_retrieve_documents_no_filter_dict(mocker, mocked_agent):
    """
    Tests the retrieval path when build_filter returns None but topics exist.
    """
    # Arrange
    mocker.patch(
        "kfai.loaders.agents.query_agent.parse_query",
        return_value=QueryParseResponse(topics=["topic1"]),
    )
    mocker.patch(
        "kfai.loaders.agents.query_agent.build_filter", return_value=None
    )
    mocked_agent.vector_store.similarity_search_with_relevance_scores.return_value = []  # noqa: E501

    # Act
    mocked_agent._retrieve_documents("query")

    # Assert
    # --- CORRECTED ASSERTION ---
    # The vector store should have been called once (for the one topic).
    mocked_agent.vector_store.similarity_search_with_relevance_scores.assert_called_once()  # noqa: E501

    # Get the arguments passed to the call
    call_kwargs = mocked_agent.vector_store.similarity_search_with_relevance_scores.call_args.kwargs  # noqa: E501

    # Verify the filter was constructed correctly from an empty base
    expected_filter = {"$and": [{"text": {"$ilike": "%topic1%"}}]}
    assert "filter" in call_kwargs
    assert call_kwargs["filter"] == expected_filter


# Add this test to cover the context limit break in _retrieve_documents
def test_retrieve_documents_breaks_at_context_limit(mocker, mocked_agent):
    """Covers the break statement when CONTEXT_COUNT is reached during
    deduplication.
    """
    # Arrange
    mocker.patch(
        "kfai.loaders.agents.query_agent.CONTEXT_COUNT", 1
    )  # Set a low limit
    mocker.patch(
        "kfai.loaders.agents.query_agent.parse_query",
        return_value=QueryParseResponse(topics=[]),
    )
    mocker.patch(
        "kfai.loaders.agents.query_agent.build_filter",
        return_value={"$and": []},
    )
    # Provide more unique docs (2) than the mocked CONTEXT_COUNT (1)
    docs_with_scores = [(SAMPLE_DOCS[0], 0.9), (SAMPLE_DOCS[1], 0.8)]
    mocked_agent.vector_store.similarity_search_with_relevance_scores.return_value = docs_with_scores  # noqa: E501

    # Act
    docs = mocked_agent._retrieve_documents("query")

    # Assert
    # The length of the final docs list should be exactly the mocked
    # CONTEXT_COUNT
    assert len(docs) == 1


def test_process_query_cli_prints_duration(mocker, mocked_agent):
    """Tests that the final duration is printed in the CLI flow."""
    mocker.patch.object(
        mocked_agent, "_retrieve_documents", return_value=SAMPLE_DOCS
    )
    mock_response_obj = AgentResponse(query_response="Test Answer", sources=[])
    mocker.patch.object(
        mocked_agent, "_generate_response", return_value=mock_response_obj
    )
    mock_print = mocker.patch("builtins.print")
    mocker.patch(
        "time.time", side_effect=[100.0, 105.5]
    )  # Mock start and end time

    mocked_agent.process_query("query", is_gui=False)

    mock_print.assert_any_call("\n...response took 5.50 seconds.")
