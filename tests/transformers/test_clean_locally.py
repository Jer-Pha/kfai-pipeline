from unittest.mock import MagicMock

import pytest

from kfai.transformers import clean_locally


# --- Final Fixture ---
@pytest.fixture
def mock_deps(mocker):
    """A single fixture to mock all external dependencies of the run script."""
    # Mock classes and constants
    mocker.patch("kfai.transformers.clean_locally.OllamaLLM")
    mock_cleaned_json_dir = mocker.patch(
        "kfai.transformers.clean_locally.CLEANED_JSON_DIR"
    )
    mock_raw_json_dir = mocker.patch(
        "kfai.transformers.clean_locally.RAW_JSON_DIR"
    )
    mock_logs_dir = mocker.patch("kfai.transformers.clean_locally.LOGS_DIR")

    # Function names match imports
    mocks = {
        "load_raw_data": mocker.patch(
            "kfai.transformers.clean_locally.load_raw_data"
        ),
        "clean_transcript": mocker.patch(
            "kfai.transformers.clean_locally.clean_transcript"
        ),
        "check_data_integrity": mocker.patch(
            "kfai.transformers.clean_locally.check_data_integrity"
        ),
        "save_cleaned_data": mocker.patch(
            "kfai.transformers.clean_locally.save_cleaned_data"
        ),
        "logger": mocker.patch("kfai.transformers.clean_locally.logger"),
        "print": mocker.patch("builtins.print"),
        "raw_json_dir": mock_raw_json_dir,
        "cleaned_json_dir": mock_cleaned_json_dir,
        "logs_dir": mock_logs_dir,
    }
    return mocks


# --- Test Data ---
SAMPLE_VIDEO_DATA = {
    "video_id": "vid1",
    "transcript_chunks": [{"text": "raw"}],
}

# --- Test Suite ---


def test_run_happy_path(mock_deps):
    """Tests the main success path for processing a single new file."""
    # Arrange
    mock_file_path = MagicMock()
    mock_deps["raw_json_dir"].rglob.return_value = [mock_file_path]

    mock_cleaned_path = MagicMock()
    mock_cleaned_path.exists.return_value = False
    mock_deps["cleaned_json_dir"].__truediv__.return_value = mock_cleaned_path

    mock_deps["load_raw_data"].return_value = SAMPLE_VIDEO_DATA
    mock_deps["clean_transcript"].return_value = {
        "video_id": "vid1",
        "transcript_chunks": [{"text": "clean"}],
    }
    mock_deps["check_data_integrity"].return_value = True
    mock_deps["save_cleaned_data"].return_value = True

    # Act
    clean_locally.run()

    # Assert
    mock_deps["load_raw_data"].assert_called_once_with(mock_file_path)
    mock_deps["clean_transcript"].assert_called_once()
    mock_deps["check_data_integrity"].assert_called_once()
    mock_deps["save_cleaned_data"].assert_called_once()
    mock_deps["print"].assert_any_call("\nCleaning process complete.")


@pytest.mark.parametrize(
    "reason, mock_setup",
    [
        ("already cleaned", {"cleaned_exists": True}),
        ("load fails", {"load_return": None}),
        (
            "no transcript",
            {"load_return": {"video_id": "v1", "transcript_chunks": []}},
        ),
        ("cleaning fails", {"clean_return": None}),
        ("integrity fails", {"integrity_return": False}),
    ],
)
def test_run_skips_videos_for_various_reasons(mock_deps, reason, mock_setup):
    # Arrange
    mock_file_path = MagicMock()
    mock_deps["raw_json_dir"].rglob.return_value = [mock_file_path]

    mock_cleaned_path = MagicMock()
    mock_cleaned_path.exists.return_value = mock_setup.get(
        "cleaned_exists", False
    )
    mock_deps["cleaned_json_dir"].__truediv__.return_value = mock_cleaned_path

    mock_deps["load_raw_data"].return_value = mock_setup.get(
        "load_return", SAMPLE_VIDEO_DATA
    )
    mock_deps["clean_transcript"].return_value = mock_setup.get(
        "clean_return", {"video_id": "v1"}
    )
    mock_deps["check_data_integrity"].return_value = mock_setup.get(
        "integrity_return", True
    )

    # Act
    clean_locally.run()

    # Assert
    mock_deps["save_cleaned_data"].assert_not_called()
    mock_deps["print"].assert_any_call("\nCleaning process complete.")


def test_run_handles_critical_exception(mocker, mock_deps):
    """Tests the main try/except block that wraps the loop."""
    # Arrange
    mock_deps["raw_json_dir"].rglob.side_effect = Exception("Disk read error")
    mocker.patch("traceback.format_exc")

    # Act & Assert
    with pytest.raises(Exception, match="Disk read error"):
        clean_locally.run()

    # Verify that the critical error was logged
    mock_deps["logger"].critical.assert_called()
    mock_deps["print"].assert_any_call(
        f"\n!! A critical error occurred. See {mock_deps['logs_dir']} for"
        " details."
    )
