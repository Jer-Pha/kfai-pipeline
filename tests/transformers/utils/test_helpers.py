import json
from unittest.mock import ANY, MagicMock

import pytest

# The module we are testing
from kfai.transformers.utils import helpers as helpers_utils

# --- Tests for load_raw_data ---


def test_load_raw_data_happy_path(mocker):
    """Tests successful loading and parsing of a JSON file."""
    # Arrange
    mock_path = MagicMock()
    json_data = '{"video_id": "vid1", "title": "Test"}'
    mocker.patch.object(
        mock_path, "open", mocker.mock_open(read_data=json_data)
    )

    # Act
    data = helpers_utils.load_raw_data(mock_path)

    # Assert
    assert data["video_id"] == "vid1"
    assert data["title"] == "Test"


@pytest.mark.parametrize(
    "error", [json.JSONDecodeError("msg", "doc", 0), IOError("msg")]
)
def test_load_raw_data_handles_errors(mocker, error):
    """Tests that file read or JSON parsing errors are caught and logged."""
    # Arrange
    mock_path = MagicMock()
    mocker.patch.object(mock_path, "open", side_effect=error)
    mock_logger = mocker.patch("kfai.transformers.utils.helpers.logger")
    mocker.patch("traceback.format_exc")  # Mock traceback to keep output clean

    # Act
    data = helpers_utils.load_raw_data(mock_path)

    # Assert
    assert data is None
    assert mock_logger.error.call_count == 2
    mock_logger.error.assert_any_call(
        f"Failed to load or parse source file: {mock_path}"
    )


# --- Tests for check_data_integrity ---


@pytest.fixture
def sample_data():
    """Provides sample raw and cleaned data for integrity checks."""
    raw = {
        "video_id": "v1",
        "transcript_chunks": [{"text": "a"}, {"text": "b"}],
    }
    cleaned = {
        "video_id": "v1",
        "transcript_chunks": [{"text": "c"}, {"text": "d"}],
    }
    return raw, cleaned


def test_check_data_integrity_happy_path(sample_data):
    """Tests that integrity check passes with valid data."""
    raw, cleaned = sample_data
    assert (
        helpers_utils.check_data_integrity(raw, cleaned, MagicMock()) is True
    )


def test_check_data_integrity_fails_on_empty_data(mocker, sample_data):
    """Tests failure when cleaned_data is None or empty."""
    raw, _ = sample_data
    mock_logger = mocker.patch("kfai.transformers.utils.helpers.logger")
    assert helpers_utils.check_data_integrity(raw, None, MagicMock()) is False
    mock_logger.warning.assert_called_once()


def test_check_data_integrity_fails_on_key_mismatch(mocker, sample_data):
    """Tests failure when keys don't match between raw and cleaned data."""
    raw, _ = sample_data
    cleaned_mismatch = {"video_id": "v1"}  # Missing transcript_chunks key
    mock_logger = mocker.patch("kfai.transformers.utils.helpers.logger")
    assert (
        helpers_utils.check_data_integrity(raw, cleaned_mismatch, MagicMock())
        is False
    )
    mock_logger.warning.assert_called_once()


def test_check_data_integrity_fails_on_chunk_count_mismatch(
    mocker, sample_data
):
    """Tests failure when the number of transcript chunks is different."""
    raw, cleaned = sample_data
    cleaned["transcript_chunks"] = [
        {"text": "only one chunk"}
    ]  # Mismatch in count
    mock_logger = mocker.patch("kfai.transformers.utils.helpers.logger")
    assert (
        helpers_utils.check_data_integrity(raw, cleaned, MagicMock()) is False
    )
    mock_logger.error.assert_called_once()


# --- Tests for save_cleaned_data ---


def test_save_cleaned_data_happy_path(mocker):
    """Tests successful saving of a JSON file."""
    # Arrange
    mock_path = MagicMock()
    mock_parent_dir = MagicMock()
    mock_path.parent = mock_parent_dir
    mock_open = mocker.mock_open()
    mocker.patch.object(mock_path, "open", mock_open)
    mock_dump = mocker.patch("json.dump")

    # Act
    result = helpers_utils.save_cleaned_data(mock_path, {"video_id": "v1"})

    # Assert
    assert result is True
    mock_parent_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_dump.assert_called_once_with({"video_id": "v1"}, ANY, indent=4)


def test_save_cleaned_data_handles_exception(mocker):
    """Tests that an exception during file write is caught and logged."""
    # Arrange
    mock_path = MagicMock()
    mocker.patch.object(mock_path, "open", side_effect=IOError("Disk full"))
    mock_logger = mocker.patch("kfai.transformers.utils.helpers.logger")
    mocker.patch("traceback.format_exc")

    # Act
    result = helpers_utils.save_cleaned_data(mock_path, {})

    # Assert
    assert result is False
    assert mock_logger.error.call_count == 2
    mock_logger.error.assert_any_call(
        f"Failed to save cleaned file: {mock_path}"
    )


# --- Tests for clean_text_chunk (Pure Function) ---


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        (
            "Text with profanity reference [\u00a0__\u00a0]",
            "Text with profanity reference ****",
        ),
        (
            "Text with\u200bzero\u200bwidth\xa0space",
            "Text withzerowidth space",
        ),
        # CORRECTED: The extra space is correctly removed by the whitespace sub
        ("Text with >> arrows", "Text with arrows"),
        # CORRECTED: The extra space is correctly removed by the whitespace sub
        ("Text [with bracket tags] and content", "Text and content"),
        ("Text with   multiple   spaces", "Text with multiple spaces"),
    ],
)
def test_clean_text_chunk(input_text, expected_output):
    assert helpers_utils.clean_text_chunk(input_text) == expected_output


# --- Tests for clean_response (Pure Function) ---


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Here is the cleaned chunk:Cleaned text", "Cleaned text"),
        ("Here's the cleaned chunk:Cleaned text", "Cleaned text"),
        ("<think>Thought process</think>Cleaned text", "Cleaned text"),
        ("<CHUNK>Cleaned text</CHUNK>", "Cleaned text"),
        (
            "It’s a test with ‘curly’ quotes.",
            "It's a test with 'curly' quotes.",
        ),
        ("He said, “Hello world”.", 'He said, "Hello world".'),
    ],
)
def test_clean_response(input_text, expected_output):
    assert helpers_utils.clean_response(input_text) == expected_output
