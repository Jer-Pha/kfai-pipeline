import json
from unittest.mock import MagicMock

import pytest

from kfai.extractors import transcribe_failures


# --- Fixture for Mocking Dependencies ---
@pytest.fixture
def mock_deps(mocker):
    """A comprehensive fixture to mock all external dependencies."""
    return {
        "download_audio": mocker.patch(
            "kfai.extractors.transcribe_failures.download_audio_handler"
        ),
        "transcribe_whisper": mocker.patch(
            "kfai.extractors.transcribe_failures.transcribe_with_whisper"
        ),
        "chunk_transcript": mocker.patch(
            "kfai.extractors.transcribe_failures.chunk_transcript_with_overlap"
        ),
        "dump": mocker.patch("json.dump"),
        "print": mocker.patch("builtins.print"),
        "whisper_load": mocker.patch(
            "whisper.load_model", return_value=MagicMock()
        ),
        "failed_file_path": mocker.patch(
            "kfai.extractors.transcribe_failures.FAILED_VIDEOS_FILE"
        ),
        "raw_json_dir_path": mocker.patch(
            "kfai.extractors.transcribe_failures.RAW_JSON_DIR"
        ),
        "temp_data_dir": mocker.patch(
            "kfai.extractors.transcribe_failures.TEMP_DATA_DIR"
        ),
        "skip_list": mocker.patch(
            "kfai.extractors.transcribe_failures.SKIP_LIST",
            ["vid_in_skiplist"],
        ),
        "chunk_threshold": mocker.patch(
            "kfai.extractors.transcribe_failures.CHUNK_THRESHOLD_SECONDS", 100
        ),
    }


# --- Test Data ---
SAMPLE_VIDEO = {
    "video_id": "vid1",
    "published_at": 1672531200,
    "duration": 150,
    "title": "Test",
}


# --- Helper for Mocking Paths ---
def setup_path_mocks(mocker, mock_deps, output_exists=False):
    """
    Helper to correctly configure the three-level chained path mock.
    RAW_JSON_DIR / year / month / filename
    """
    mock_output_path = MagicMock()
    mock_output_path.exists.return_value = output_exists
    # This correctly mocks the three '/' operations
    mock_deps[
        "raw_json_dir_path"
    ].__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = (  # noqa: E501
        mock_output_path
    )
    mocker.patch.object(mock_output_path, "open", mocker.mock_open())


# --- Test Suite ---


def test_run_happy_path(mocker, mock_deps):
    """Tests the main success path: downloads, transcribes, chunks,
    saves, and cleans up.
    """
    # Arrange
    mock_deps["failed_file_path"].exists.return_value = True
    mocker.patch.object(
        mock_deps["failed_file_path"],
        "open",
        mocker.mock_open(read_data=json.dumps([SAMPLE_VIDEO])),
    )
    setup_path_mocks(
        mocker, mock_deps, output_exists=False
    )  # Critical: output does NOT exist
    mock_chunk_path1, mock_chunk_path2 = MagicMock(), MagicMock()
    mock_deps["download_audio"].return_value = [
        mock_chunk_path1,
        mock_chunk_path2,
    ]
    mock_deps["transcribe_whisper"].side_effect = [
        [{"start": 5.0}],
        [{"start": 10.0}],
    ]
    mock_deps["chunk_transcript"].return_value = ["final_chunk"]

    # Act
    transcribe_failures.run()

    # Assert
    mock_deps["download_audio"].assert_called_once_with("vid1", 150)
    assert mock_deps["transcribe_whisper"].call_count == 2
    final_raw_transcript = mock_deps["chunk_transcript"].call_args[0][0]
    assert final_raw_transcript[1]["start"] == 110.0
    mock_deps["dump"].assert_called_once()
    mock_chunk_path1.unlink.assert_called_once()


def test_run_input_file_not_found(mock_deps):
    mock_deps["failed_file_path"].exists.return_value = False
    transcribe_failures.run()
    mock_deps["print"].assert_any_call(
        f"Error: Input file '{mock_deps['failed_file_path']}' not found."
    )
    mock_deps["whisper_load"].assert_not_called()


def test_run_whisper_model_fails_to_load(mock_deps):
    mock_deps["failed_file_path"].exists.return_value = True
    mock_deps["whisper_load"].side_effect = Exception("CUDA error")
    transcribe_failures.run()
    mock_deps["print"].assert_any_call(
        "Fatal: Could not load Whisper model. Error: CUDA error"
    )
    mock_deps["failed_file_path"].open.assert_not_called()


@pytest.mark.parametrize(
    "video_id, output_exists",
    [("vid_in_skiplist", False), ("vid_processed", True)],
)
def test_run_skips_videos(mocker, mock_deps, video_id, output_exists):
    video_record = {**SAMPLE_VIDEO, "video_id": video_id}
    mock_deps["failed_file_path"].exists.return_value = True
    mocker.patch.object(
        mock_deps["failed_file_path"],
        "open",
        mocker.mock_open(read_data=json.dumps([video_record])),
    )
    if output_exists:
        setup_path_mocks(mocker, mock_deps, output_exists=True)

    transcribe_failures.run()
    mock_deps["download_audio"].assert_not_called()


def test_run_handles_download_failure(mocker, mock_deps):
    mock_deps["failed_file_path"].exists.return_value = True
    mocker.patch.object(
        mock_deps["failed_file_path"],
        "open",
        mocker.mock_open(read_data=json.dumps([SAMPLE_VIDEO])),
    )
    setup_path_mocks(mocker, mock_deps, output_exists=False)
    mock_deps["download_audio"].return_value = None

    transcribe_failures.run()
    mock_deps["print"].assert_any_call(
        "  !! Failed to download chunks for vid1. Skipping."
    )
    mock_deps["transcribe_whisper"].assert_not_called()


def test_run_handles_transcription_failure(mocker, mock_deps):
    mock_deps["failed_file_path"].exists.return_value = True
    mocker.patch.object(
        mock_deps["failed_file_path"],
        "open",
        mocker.mock_open(read_data=json.dumps([SAMPLE_VIDEO])),
    )
    setup_path_mocks(mocker, mock_deps, output_exists=False)
    mock_chunk_path = MagicMock()
    mock_deps["download_audio"].return_value = [mock_chunk_path]
    mock_deps["transcribe_whisper"].return_value = None

    transcribe_failures.run()
    mock_deps["print"].assert_any_call(
        "  !! No transcript data generated... skipping."
    )
    mock_chunk_path.unlink.assert_called_once()
    mock_deps["dump"].assert_not_called()
