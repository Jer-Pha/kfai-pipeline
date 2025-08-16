from unittest.mock import MagicMock

import pytest
from langchain_ollama import OllamaLLM

# The module we are testing
from kfai.loaders.utils import parsing as parsing_utils
from kfai.loaders.utils.helpers.datetime import iso_string_to_epoch

# --- Test Setup ---


# By using a pytest fixture, we can create a reusable mock LLM object
# for all our tests in this file.
@pytest.fixture
def mock_llm(mocker) -> MagicMock:
    """Creates a mock of the OllamaLLM object."""
    # We use mocker.MagicMock to create a flexible mock object.
    # spec=OllamaLLM ensures the mock behaves like a real OllamaLLM instance,
    # raising an error if we try to access a non-existent attribute.
    return mocker.MagicMock(spec=OllamaLLM)


# --- Tests for parse_shows ---


def test_parse_shows_happy_path(mock_llm):
    """
    Tests the normal, successful execution of parse_shows.
    """
    # 1. Arrange: Set up the inputs and the mock's behavior
    query = "Tell me about The Daily Show and Last Week Tonight"
    show_names = ["The Daily Show", "Last Week Tonight", "The Colbert Report"]

    # This is the simulated JSON string we expect the LLM to return
    mock_response_json = '{"shows": ["The Daily Show", "Last Week Tonight"]}'
    mock_llm.invoke.return_value = mock_response_json

    # 2. Act: Call the function with the mock LLM
    result = parsing_utils.parse_shows(query, show_names, mock_llm)

    # 3. Assert: Verify the outcome
    assert result == ["The Daily Show", "Last Week Tonight"]
    mock_llm.invoke.assert_called_once()  # Verifies the LLM was called exactly once


def test_parse_shows_handles_bad_json(mock_llm):
    """
    Tests that parse_shows returns an empty list if the LLM returns invalid JSON.
    """
    # 1. Arrange: Configure the mock to return a malformed string
    query = "any query"
    show_names = ["any show"]
    mock_llm.invoke.return_value = '{"shows": ["Missing closing bracket"]'

    # 2. Act: Call the function
    result = parsing_utils.parse_shows(query, show_names, mock_llm)

    # 3. Assert: The function should handle the error gracefully
    assert result == []


def test_parse_shows_handles_llm_exception(mock_llm):
    """
    Tests that parse_shows returns an empty list if the llm.invoke call fails.
    """
    # 1. Arrange: Configure the mock to raise an exception when called
    query = "any query"
    show_names = ["any show"]
    mock_llm.invoke.side_effect = Exception("LLM service unavailable")

    # 2. Act: Call the function
    result = parsing_utils.parse_shows(query, show_names, mock_llm)

    # 3. Assert: The function's try/except block should catch the error
    assert result == []


# --- Tests for parse_hosts ---


def test_parse_hosts_happy_path(mock_llm):
    """
    Tests the normal, successful execution of parse_hosts.
    """
    # 1. Arrange
    query = "Who is Jon Stewart?"
    hosts = ["Jon Stewart", "John Oliver"]
    mock_response_json = '{"hosts": ["Jon Stewart"]}'
    mock_llm.invoke.return_value = mock_response_json

    # 2. Act
    result = parsing_utils.parse_hosts(query, hosts, mock_llm)

    # 3. Assert
    assert result == ["Jon Stewart"]
    mock_llm.invoke.assert_called_once()


def test_parse_hosts_handles_bad_json(mock_llm):
    """
    Tests that parse_hosts returns an empty list for malformed LLM responses.
    """
    # 1. Arrange
    mock_llm.invoke.return_value = '{"hosts": not valid json}'

    # 2. Act
    result = parsing_utils.parse_hosts("any query", ["any host"], mock_llm)

    # 3. Assert
    assert result == []


# --- Tests for parse_year_range ---


# pytest.mark.parametrize allows us to run the same test function with different inputs.
# This is highly efficient for testing multiple conditions in the same function.
@pytest.mark.parametrize(
    "llm_response, expected_filters, expected_years",
    [
        # Case 1: Exact Year
        (
            '{"exact_year": "2023"}',
            [
                {
                    "published_at": {
                        "$gte": iso_string_to_epoch("2023-01-01T00:00:00")
                    }
                },
                {
                    "published_at": {
                        "$lte": iso_string_to_epoch("2023-12-31T23:59:59")
                    }
                },
            ],
            ["2023"],
        ),
        # Case 2: Year Range
        (
            '{"year_range": "2020-2022"}',
            [
                {
                    "published_at": {
                        "$gte": iso_string_to_epoch("2020-01-01T00:00:00")
                    }
                },
                {
                    "published_at": {
                        "$lte": iso_string_to_epoch("2022-12-31T23:59:59")
                    }
                },
            ],
            ["2020", "2022"],
        ),
        # Case 3: Before Year
        (
            '{"before_year": "2019"}',
            [
                {"published_at": {"$gte": 1325376000}},
                {
                    "published_at": {
                        "$lte": iso_string_to_epoch("2018-12-31T23:59:59")
                    }
                },
            ],
            ["2018"],
        ),
        # --- NEW TEST CASE ADDED HERE ---
        # Case 4: After Year
        (
            '{"after_year": "2022"}',
            [
                {
                    "published_at": {
                        "$gte": iso_string_to_epoch("2023-01-01T00:00:00")
                    }
                },
                # The upper bound is based on our mocked 'current_year' of 2024
                {
                    "published_at": {
                        "$lte": iso_string_to_epoch("2024-12-31T23:59:59")
                    }
                },
            ],
            ["2023"],
        ),
        # Case 5: No year found in response
        (
            '{"exact_year": "NOT_FOUND"}',
            [],
            [],
        ),
        # Case 6: Empty response from LLM
        (
            "",
            [],
            [],
        ),
    ],
)
def test_parse_year_range_scenarios(
    mocker, mock_llm, llm_response, expected_filters, expected_years
):
    """
    Tests various successful scenarios for parse_year_range.
    """
    # 1. Arrange
    # Mock datetime.now() to return a fixed year (e.g., 2024)
    # This makes the test deterministic and future-proof.
    mock_datetime = mocker.patch("kfai.loaders.utils.parsing.datetime")
    mock_datetime.now.return_value.year = 2024

    mock_llm.invoke.return_value = llm_response

    # 2. Act
    filters, years = parsing_utils.parse_year_range("any query", mock_llm)

    # 3. Assert
    assert filters == expected_filters
    assert years == expected_years


def test_parse_year_range_handles_bad_json(mock_llm):
    """
    Tests that parse_year_range handles malformed JSON gracefully.
    """
    # 1. Arrange
    mock_llm.invoke.return_value = '{"exact_year": 2023'  # Invalid JSON

    # 2. Act
    filters, years = parsing_utils.parse_year_range("any query", mock_llm)

    # 3. Assert
    assert filters == []
    assert years == []


# --- Tests for parse_topics ---


def test_parse_topics_happy_path(mock_llm):
    """
    Tests the normal, successful execution of parse_topics.
    """
    # 1. Arrange
    query = "Tell me about AI and open source"
    show_filter = ["Some Show"]
    hosts_filter = ["Some Host"]
    years = ["2023"]

    # Note: The function filters out topics that are already in the metadata
    mock_response_json = '{"topics": ["AI", "open source", "Some Show"]}'
    mock_llm.invoke.return_value = mock_response_json

    # 2. Act
    result = parsing_utils.parse_topics(
        query, show_filter, hosts_filter, years, mock_llm
    )

    # 3. Assert
    assert result == [
        "AI",
        "open source",
    ]  # "Some Show" should be filtered out
    mock_llm.invoke.assert_called_once()


def test_parse_topics_handles_llm_exception(mock_llm):
    """
    Tests that parse_topics returns an empty list on LLM failure.
    """
    # 1. Arrange: Configure the mock to raise an exception
    mock_llm.invoke.side_effect = Exception("LLM service unavailable")

    # 2. Act: Call the function with minimal valid inputs
    result = parsing_utils.parse_topics(
        query="any query",
        show_filter=[],
        hosts_filter=[],
        years=[],
        llm=mock_llm,
    )

    # 3. Assert: The function should catch the error and return an empty list
    assert result == []
