import json
from unittest.mock import ANY, MagicMock, call

import pytest

# The module we are testing
from kfai.transformers.utils import cleaning as cleaning_utils


# --- Corrected Fixture ---
@pytest.fixture
def mock_deps(mocker):
    """A fixture to mock all external dependencies of clean_transcript."""
    # --- KEY CHANGE: Mock the logger object directly ---
    mock_logger = mocker.patch("kfai.transformers.utils.cleaning.logger")

    # Mock helper functions
    mock_clean_response = mocker.patch(
        "kfai.transformers.utils.cleaning.clean_response"
    )
    mock_clean_text_chunk = mocker.patch(
        "kfai.transformers.utils.cleaning.clean_text_chunk"
    )

    # Mock LLM and its invoke method
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock()

    # Mock progress bar
    mock_tqdm = mocker.patch("kfai.transformers.utils.cleaning.tqdm")
    mock_progress_bar = mock_tqdm.return_value

    # Mock prompts and json.dumps
    mocker.patch(
        "kfai.transformers.utils.cleaning.SYSTEM_PROMPT", "System prompt."
    )
    mocker.patch(
        "kfai.transformers.utils.cleaning.USER_PROMPT",
        "User prompt: {metadata} {chunk}",
    )
    mocker.patch("json.dumps", return_value="{}")

    # Mock print for console output
    mock_print = mocker.patch("builtins.print")

    return {
        "clean_response": mock_clean_response,
        "clean_text_chunk": mock_clean_text_chunk,
        "llm": mock_llm,
        "logger": mock_logger,  # This is now the direct mock of the logger
        "progress_bar": mock_progress_bar,
        "print": mock_print,
    }


# --- Test Data ---
SAMPLE_VIDEO_RECORD = {
    "id": 1,
    "video_id": "vid1",
    "show_name": "Show A",
    "hosts": ["Host A"],
    "title": "Video Title",
    "description": "Description",
    "published_at": 123,
    "duration": 456,
    "transcript_chunks": [
        {"text": "chunk 1 raw", "start": 10.0},
        {"text": "chunk 2 raw", "start": 20.0},
    ],
}

# --- Test Suite ---


def test_clean_transcript_happy_path(mock_deps):
    mock_deps["clean_text_chunk"].side_effect = [
        "chunk 1 clean",
        "chunk 2 clean",
    ]
    mock_deps["llm"].invoke.side_effect = ["llm response 1", "llm response 2"]
    mock_deps["clean_response"].side_effect = [
        "cleaned response 1",
        "cleaned response 2",
    ]

    cleaned_data = cleaning_utils.clean_transcript(
        SAMPLE_VIDEO_RECORD, MagicMock(), mock_deps["llm"]
    )

    assert cleaned_data is not None
    assert len(cleaned_data["transcript_chunks"]) == 2
    assert cleaned_data["transcript_chunks"][0]["text"] == "cleaned response 1"
    assert mock_deps["progress_bar"].update.call_count == 2
    mock_deps["progress_bar"].close.assert_called_once()


def test_clean_transcript_llm_call_failure(mock_deps):
    """
    Tests that the function handles an LLM invocation error gracefully.
    """
    # 1. Arrange
    mock_deps["llm"].invoke.side_effect = Exception("LLM connection error")
    relative_path_mock = MagicMock()
    relative_path_mock.__str__.return_value = "path/to/video.json"

    # 2. Act
    result = cleaning_utils.clean_transcript(
        SAMPLE_VIDEO_RECORD, relative_path_mock, mock_deps["llm"]
    )

    # 3. Assert
    assert result is None

    # --- CORRECTED ASSERTIONS ---
    # Verify that error was logged twice
    assert mock_deps["logger"].error.call_count == 2

    # Check the content of the first error log call
    first_error_call = mock_deps["logger"].error.call_args_list[0]
    expected_message = (
        f"LLM call failed on chunk in {relative_path_mock} starting at 10.0s."
    )
    assert first_error_call.args[0] == expected_message

    # Check that the second call (the traceback) was made with ANY string
    second_error_call = mock_deps["logger"].error.call_args_list[1]
    assert second_error_call == call(ANY)

    mock_deps["progress_bar"].close.assert_called_once()


def test_clean_transcript_general_failure(mock_deps):
    """
    Tests that the function handles unexpected errors (e.g., missing data in video_data).
    """
    # 1. Arrange
    malformed_record = {"video_id": "vid1", "transcript_chunks": None}
    relative_path_mock = MagicMock()
    relative_path_mock.__str__.return_value = "path/to/video.json"

    # 2. Act
    result = cleaning_utils.clean_transcript(
        malformed_record, relative_path_mock, mock_deps["llm"]
    )

    # 3. Assert
    assert result is None

    # --- CORRECTED ASSERTIONS ---
    # Verify that error was logged twice
    assert mock_deps["logger"].error.call_count == 2

    # Check the content of the first error log call
    first_error_call = mock_deps["logger"].error.call_args_list[0]
    expected_message = f"An unexpected error occurred in clean_transcript() for {relative_path_mock}."
    assert first_error_call.args[0] == expected_message

    # Check that the second call (the traceback) was made with ANY string
    second_error_call = mock_deps["logger"].error.call_args_list[1]
    assert second_error_call == call(ANY)

    mock_deps["progress_bar"].close.assert_not_called()
