import pytest

# The module we are testing
from kfai.loaders.utils import filtering as filtering_utils

# --- Tests for the helper function: _build_filter ---


# We use parametrize to efficiently test many combinations of inputs
@pytest.mark.parametrize(
    "inputs, expected_output",
    [
        # Case 1: All inputs are provided
        (
            {
                "shows_list": ["Show A"],
                "hosts_list": ["Host B"],
                "year_filter": [{"published_at": {"$gte": 123}}],
                "topics_list": ["Topic C", "Topic D"],
            },
            {
                "$and": [
                    {"show_name": {"$in": ["Show A"]}},
                    {"hosts": {"$like": "%Host B%"}},
                    {"published_at": {"$gte": 123}},
                    {
                        "$or": [
                            {"text": {"$ilike": "%Topic C%"}},
                            {"text": {"$ilike": "%Topic D%"}},
                        ]
                    },
                ]
            },
        ),
        # Case 2: No inputs provided, should return None
        (
            {
                "shows_list": [],
                "hosts_list": [],
                "year_filter": [],
                "topics_list": [],
            },
            None,
        ),
        # Case 3: Only shows and topics
        (
            {
                "shows_list": ["Show A", "Show B"],
                "hosts_list": [],
                "year_filter": [],
                "topics_list": ["Topic C"],
            },
            {
                "$and": [
                    {"show_name": {"$in": ["Show A", "Show B"]}},
                    {"$or": [{"text": {"$ilike": "%Topic C%"}}]},
                ]
            },
        ),
        # Case 4: Host with special SQL characters that need escaping
        (
            {
                "shows_list": [],
                "hosts_list": ["Host_With%Chars"],
                "year_filter": [],
                "topics_list": [],
            },
            {"$and": [{"hosts": {"$like": "%Host\\_With\\%Chars%"}}]},
        ),
        # Case 5: Topic with leading/trailing whitespace should be ignored
        (
            {
                "shows_list": [],
                "hosts_list": [],
                "year_filter": [],
                "topics_list": ["Valid Topic", "   ", ""],
            },
            {"$and": [{"$or": [{"text": {"$ilike": "%Valid Topic%"}}]}]},
        ),
    ],
)
def test_build_filter(inputs, expected_output):
    """
    Tests the _build_filter function with various combinations of inputs.
    """
    # The function is private, but we can test it directly in Python
    # by calling it with the _ prefix.
    result = filtering_utils._build_filter(**inputs)
    assert result == expected_output


# --- Tests for the main function: get_filter ---


def test_get_filter_happy_path(mocker):
    """
    Tests the get_filter function by mocking its parsing dependencies.
    This test verifies the orchestration logic.
    """
    # 1. Arrange: Mock all dependencies of get_filter
    # We don't need the LLM to run, so we mock its class
    mocker.patch("kfai.loaders.utils.filtering.OllamaLLM")

    # Mock the return values of the parsing functions
    mock_parse_shows = mocker.patch(
        "kfai.loaders.utils.filtering.parse_shows",
        return_value=["Kinda Funny Games Daily"],
    )
    mock_parse_hosts = mocker.patch(
        "kfai.loaders.utils.filtering.parse_hosts",
        return_value=["Greg Miller"],
    )
    mock_parse_year = mocker.patch(
        "kfai.loaders.utils.filtering.parse_year_range",
        return_value=([{"published_at": {"$gte": 123}}], ["2023"]),
    )
    mock_parse_topics = mocker.patch(
        "kfai.loaders.utils.filtering.parse_topics",
        return_value=["PlayStation", "Xbox"],
    )

    query = "Tell me what Greg Miller said about PlayStation and Xbox in 2023"
    show_names = ["Kinda Funny Games Daily"]
    hosts = ["Greg Miller"]

    # 2. Act: Call the function
    topics_str, filter_dict = filtering_utils.get_filter(
        query, show_names, hosts
    )

    # 3. Assert
    # Assert the final topics string is correct
    assert topics_str == "PlayStation, Xbox"

    # Assert the final filter dictionary is correctly assembled
    expected_filter = {
        "$and": [
            {"show_name": {"$in": ["Kinda Funny Games Daily"]}},
            {"hosts": {"$like": "%Greg Miller%"}},
            {"published_at": {"$gte": 123}},
            {
                "$or": [
                    {"text": {"$ilike": "%PlayStation%"}},
                    {"text": {"$ilike": "%Xbox%"}},
                ]
            },
        ]
    }
    assert filter_dict == expected_filter

    # Assert that the parsing functions were called with the correct arguments
    mock_parse_shows.assert_called_once()
    mock_parse_hosts.assert_called_once()
    mock_parse_year.assert_called_once()
    mock_parse_topics.assert_called_once()


def test_get_filter_when_no_topics_are_found(mocker):
    """
    Tests that the returned topics string falls back to the original query
    if the parsing function returns no topics.
    """
    # 1. Arrange
    mocker.patch("kfai.loaders.utils.filtering.OllamaLLM")
    mocker.patch("kfai.loaders.utils.filtering.parse_shows", return_value=[])
    mocker.patch("kfai.loaders.utils.filtering.parse_hosts", return_value=[])
    mocker.patch(
        "kfai.loaders.utils.filtering.parse_year_range", return_value=([], [])
    )
    # This is the key part of this test: parse_topics returns an empty list
    mocker.patch("kfai.loaders.utils.filtering.parse_topics", return_value=[])

    query = "A very generic query"

    # 2. Act
    topics_str, filter_dict = filtering_utils.get_filter(query, [], [])

    # 3. Assert
    # The topics string should be the original query
    assert topics_str == query
    # Since no filters were found, the dictionary should be None
    assert filter_dict is None
