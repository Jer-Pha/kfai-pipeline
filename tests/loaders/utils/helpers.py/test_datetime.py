import pytest

from kfai.loaders.utils.helpers import datetime as datetime_utils

# --- Tests for iso_string_to_epoch ---


@pytest.mark.parametrize(
    "input_iso, expected_epoch",
    [
        # Case 1: A standard date and time
        ("2012-01-01T00:00:00", 1325376000),
        # Case 2: A more recent date
        ("2024-07-23T12:30:00", 1721737800),
        # Case 3: A date with timezone information
        ("2024-07-23T12:30:00+05:00", 1721719800),
        # Edge Case 4: Empty string input
        ("", 0),
    ],
)
def test_iso_string_to_epoch(input_iso, expected_epoch):
    """
    Tests the conversion of ISO 8601 strings to Unix epoch timestamps.
    """
    assert datetime_utils.iso_string_to_epoch(input_iso) == expected_epoch


# --- Tests for format_duration ---


@pytest.mark.parametrize(
    "input_seconds, expected_string",
    [
        # Case 1: Only seconds
        (5.12345, "5.12 seconds"),
        # Case 2: Seconds and minutes
        (125.678, "2 minutes, 5.68 seconds"),
        # Case 3: Plural minutes
        (180, "3 minutes, 0.00 seconds"),
        # Case 4: Seconds, minutes, and hours
        (3725.5, "1 hour, 2 minutes, 5.50 seconds"),
        # Case 5: Plural hours
        (7320, "2 hours, 2 minutes, 0.00 seconds"),
        # Edge Case 6: Zero seconds
        (0, "0.00 seconds"),
    ],
)
def test_format_duration(input_seconds, expected_string):
    """
    Tests the formatting of a duration in seconds into a human-readable string.
    """
    assert datetime_utils.format_duration(input_seconds) == expected_string
