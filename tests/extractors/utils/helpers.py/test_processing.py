from unittest.mock import MagicMock, call

import pytest

# The module we are testing
from kfai.extractors.utils.helpers import processing as processing_utils

# --- Comprehensive Fixture for Mocking Dependencies ---


@pytest.fixture
def mock_dependencies(mocker):
    """A single fixture to mock all external dependencies of the process_video function."""
    # Mock the helper functions that are dependencies
    mock_get_transcript = mocker.patch(
        "kfai.extractors.utils.helpers.processing.get_raw_transcript_data"
    )
    mock_chunk_transcript = mocker.patch(
        "kfai.extractors.utils.helpers.processing.chunk_transcript_with_overlap"
    )

    # Mock file system interactions, which are complex due to chained calls
    mock_raw_json_dir = mocker.patch(
        "kfai.extractors.utils.helpers.processing.RAW_JSON_DIR"
    )
    # The result of RAW_JSON_DIR / year
    mock_year_dir = MagicMock()
    # The result of RAW_JSON_DIR / year / month
    mock_month_dir = MagicMock()
    # The result of ... / f"{video_id}.json"
    mock_output_path = MagicMock()

    mock_raw_json_dir.__truediv__.return_value = mock_year_dir
    mock_year_dir.__truediv__.return_value = mock_month_dir
    mock_month_dir.__truediv__.return_value = mock_output_path

    # Mock the file writer
    mock_dump = mocker.patch("kfai.extractors.utils.helpers.processing.dump")
    mock_print = mocker.patch("builtins.print")

    return {
        "get_transcript": mock_get_transcript,
        "chunk_transcript": mock_chunk_transcript,
        "raw_json_dir": mock_raw_json_dir,
        "output_path": mock_output_path,
        "subdir_path": mock_month_dir,
        "dump": mock_dump,
        "print": mock_print,
    }


# A sample video record for reuse in tests
SAMPLE_VIDEO_RECORD = {
    "video_id": "vid1",
    "published_at": 1672531200,  # 2023-01-01
    "title": "Test Video",
}

# --- Test Suite ---


def test_process_video_happy_path(mock_dependencies):
    """
    Tests the main success path: transcript is found, chunked, and file is written.
    """
    # 1. Arrange
    mock_dependencies["output_path"].exists.return_value = False
    mock_dependencies["get_transcript"].return_value = [
        {"text": "hello world"}
    ]
    mock_dependencies["chunk_transcript"].return_value = [
        {"start": 0, "text": "chunk1"}
    ]

    # 2. Act
    result = processing_utils.process_video(SAMPLE_VIDEO_RECORD)

    # 3. Assert
    assert result is False  # Should not be skipped next run
    mock_dependencies["subdir_path"].mkdir.assert_called_once_with(
        parents=True, exist_ok=True
    )

    # Verify the final record was written to file
    mock_dependencies["dump"].assert_called_once()
    # Check that the 'transcript_chunks' key was added to the record before dumping
    dumped_record = mock_dependencies["dump"].call_args[0][0]
    assert "transcript_chunks" in dumped_record
    assert dumped_record["transcript_chunks"] == [
        {"start": 0, "text": "chunk1"}
    ]


def test_process_video_already_exists(mock_dependencies):
    """Tests that the function exits early if the output file already exists."""
    # 1. Arrange
    mock_dependencies["output_path"].exists.return_value = True

    # 2. Act
    result = processing_utils.process_video(SAMPLE_VIDEO_RECORD)

    # 3. Assert
    assert result is False
    # The key assertion: none of the processing helpers should be called
    mock_dependencies["get_transcript"].assert_not_called()
    mock_dependencies["dump"].assert_not_called()


def test_process_video_transcription_fails(mock_dependencies):
    """
    Tests the case where transcription fails and returns the video_id,
    indicating it should be skipped in the future.
    """
    # 1. Arrange
    mock_dependencies["output_path"].exists.return_value = False
    mock_dependencies["get_transcript"].return_value = (
        "vid1"  # The failure signal
    )

    # 2. Act
    result = processing_utils.process_video(SAMPLE_VIDEO_RECORD)

    # 3. Assert
    assert result is True  # Should be skipped next run
    mock_dependencies["dump"].assert_not_called()


def test_process_video_empty_chunks(mock_dependencies):
    """
    Tests that if chunking results in an empty list, a warning is printed
    and the file is not written.
    """
    # 1. Arrange
    mock_dependencies["output_path"].exists.return_value = False
    mock_dependencies["get_transcript"].return_value = [{"text": "some text"}]
    mock_dependencies["chunk_transcript"].return_value = (
        []
    )  # Empty list after chunking

    # 2. Act
    result = processing_utils.process_video(SAMPLE_VIDEO_RECORD)

    # 3. Assert
    assert result is False
    mock_dependencies["print"].assert_called_once_with(
        "Warning: Transcript for vid1 was empty after chunking."
    )
    mock_dependencies["dump"].assert_not_called()


def test_process_video_unknown_transcript_result(mock_dependencies):
    """
    Tests the final 'else' case where the transcript result is neither a
    list nor the video_id string (e.g., None).
    """
    # 1. Arrange
    mock_dependencies["output_path"].exists.return_value = False
    mock_dependencies["get_transcript"].return_value = None

    # 2. Act
    result = processing_utils.process_video(SAMPLE_VIDEO_RECORD)

    # 3. Assert
    assert result is False
    mock_dependencies["dump"].assert_not_called()


def test_process_video_creates_unknown_date_path(mock_dependencies):
    """
    Tests that if 'published_at' is missing, it creates 'unknown/unknown' dirs.
    """
    # 1. Arrange
    mock_dependencies["output_path"].exists.return_value = False
    mock_dependencies["get_transcript"].return_value = []
    video_record_no_date = {"video_id": "vid1", "title": "Test Video"}

    # 2. Act
    processing_utils.process_video(video_record_no_date)

    # 3. Assert
    # --- FINAL CORRECTED ASSERTION ---
    # Directly access the mock for RAW_JSON_DIR from the fixture
    raw_json_dir_mock = mock_dependencies["raw_json_dir"]

    # Assert that the '/' operator was called on it with "unknown"
    raw_json_dir_mock.__truediv__.assert_called_with("unknown")

    # Assert that the next '/' operator was also called with "unknown"
    year_dir_mock = raw_json_dir_mock.__truediv__.return_value
    year_dir_mock.__truediv__.assert_called_with("unknown")
