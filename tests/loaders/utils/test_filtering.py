import pytest

# The module we are testing
from kfai.loaders.utils import filtering as filtering_utils
from kfai.loaders.utils.helpers.datetime import iso_string_to_epoch
from kfai.loaders.utils.types import QueryParseResponse

# --- Test Suite for build_filter ---


@pytest.mark.parametrize(
    "response_data, expected_filter_part",
    [
        # Case 1: Only shows
        ({"shows": ["Show A"]}, {"show_name": {"$in": ["Show A"]}}),
        # Case 2: Only hosts (with character escaping)
        ({"hosts": ["Host_B"]}, {"hosts": {"$like": "%Host\\_B%"}}),
        # Case 3: Exact year
        (
            {"exact_year": "2023"},
            {
                "published_at": {
                    "$gte": iso_string_to_epoch("2023-01-01T00:00:00")
                }
            },
        ),
        # Case 4: Year range
        (
            {"year_range": "2020-2022"},
            {
                "published_at": {
                    "$lte": iso_string_to_epoch("2022-12-31T23:59:59")
                }
            },
        ),
        # Case 5: Before year
        (
            {"before_year": "2019"},
            {
                "published_at": {
                    "$lte": iso_string_to_epoch("2018-12-31T23:59:59")
                }
            },
        ),
        # Case 6: After year (mocking datetime.now)
        (
            {"after_year": "2022"},
            {
                "published_at": {
                    "$gte": iso_string_to_epoch("2023-01-01T00:00:00")
                }
            },
        ),
    ],
)
def test_build_filter_individual_conditions(
    mocker, response_data, expected_filter_part
):
    """Tests that each condition correctly adds its part to the filter."""
    # Arrange
    # Mock datetime.now for the 'after_year' test to make it deterministic
    if "after_year" in response_data:
        mock_datetime = mocker.patch("kfai.loaders.utils.filtering.datetime")
        mock_datetime.now.return_value.year = 2024

    # Create a Pydantic object from the test data
    parsed_response = QueryParseResponse(**response_data)

    # Act
    result = filtering_utils.build_filter(parsed_response)

    # Assert
    assert result is not None
    assert "$and" in result
    # Check that the expected dictionary is a subset of one of the conditions
    assert any(
        all(item in condition.items() for item in expected_filter_part.items())
        for condition in result["$and"]
    )


def test_build_filter_all_conditions():
    """Tests that a response with all possible data builds a complete
    filter.
    """
    # Arrange
    parsed_response = QueryParseResponse(
        shows=["Show A"],
        hosts=["Host B"],
        exact_year="2023",
        topics=[
            "Topic C"
        ],  # Topics are ignored by this function, which is correct
    )

    # Act
    result = filtering_utils.build_filter(parsed_response)

    # Assert
    assert result is not None
    # Should contain 4 conditions: show, host, and year (gte) ad year (lte)
    assert len(result["$and"]) == 4


def test_build_filter_no_conditions():
    """Tests that the function returns None if no filterable data is
    provided.
    """
    # Arrange
    parsed_response = QueryParseResponse(topics=["Topic C"])  # Only topics

    # Act
    result = filtering_utils.build_filter(parsed_response)

    # Assert
    assert result is None
