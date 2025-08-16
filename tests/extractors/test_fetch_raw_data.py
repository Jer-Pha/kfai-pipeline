import json
from unittest.mock import MagicMock

import pytest

# The module we are testing
from kfai.extractors import fetch_raw_data

# --- Corrected Comprehensive Fixture ---


@pytest.fixture
def mock_dependencies(mocker):
    """A fixture to mock all external dependencies EXCEPT file I/O."""
    # Mock file system paths and their methods
    mock_skip_file = mocker.patch(
        "kfai.extractors.fetch_raw_data.VIDEOS_TO_SKIP_FILE"
    )
    mock_sqlite_path = mocker.patch(
        "kfai.extractors.fetch_raw_data.SQLITE_DB_PATH"
    )
    mock_raw_json_dir = mocker.patch(
        "kfai.extractors.fetch_raw_data.RAW_JSON_DIR"
    )

    # Mock helper function dependencies
    mocks = {
        "create_sqlite": mocker.patch(
            "kfai.extractors.fetch_raw_data.create_local_sqlite_db"
        ),
        "get_db_data": mocker.patch(
            "kfai.extractors.fetch_raw_data.get_video_db_data"
        ),
        "get_yt_data": mocker.patch(
            "kfai.extractors.fetch_raw_data.get_youtube_data"
        ),
        "process_video": mocker.patch(
            "kfai.extractors.fetch_raw_data.process_video"
        ),
        "sleep": mocker.patch("kfai.extractors.fetch_raw_data.sleep"),
        "print": mocker.patch("builtins.print"),
        "skip_file_path": mock_skip_file,
        "sqlite_path": mock_sqlite_path,
        "raw_json_dir": mock_raw_json_dir,
    }
    return mocks


# --- Corrected Test Suite ---


def test_run_happy_path(mocker, mock_dependencies):
    """
    Tests the main success path: loads a skip file, finds new videos,
    processes them, and updates the skip file.
    """
    # 1. Arrange
    mock_dependencies["skip_file_path"].exists.return_value = True
    mock_dependencies["sqlite_path"].exists.return_value = True
    mock_processed_file = MagicMock()
    mock_processed_file.stem = "vid2"
    mock_dependencies["raw_json_dir"].rglob.return_value = [
        mock_processed_file
    ]

    mock_file_open = mocker.mock_open(read_data='["vid1"]')
    mocker.patch.object(
        mock_dependencies["skip_file_path"], "open", mock_file_open
    )

    mock_dependencies["get_db_data"].side_effect = [
        [{"video_id": "vid1"}, {"video_id": "vid2"}, {"video_id": "vid3"}],
        [{"video_id": "vid3", "other_data": "db_value"}],
    ]
    mock_dependencies["get_yt_data"].return_value = {
        "vid3": {"yt_data": "api_value"}
    }
    mock_dependencies["process_video"].return_value = True

    # 2. Act
    fetch_raw_data.run()

    # 3. Assert
    mock_dependencies["get_yt_data"].assert_called_once_with(["vid3"])
    expected_record = {
        "video_id": "vid3",
        "other_data": "db_value",
        "yt_data": "api_value",
    }
    mock_dependencies["process_video"].assert_called_once_with(expected_record)

    # --- CORRECTED ASSERTION ---
    # Get the mock file handle
    handle = mock_file_open()
    # Join all the calls to write() into a single string
    written_content = "".join(c.args[0] for c in handle.write.call_args_list)
    # Assert that the complete written content matches the expected JSON
    assert sorted(json.loads(written_content)) == ["vid1", "vid3"]


# Add this new test to cover the final missing line
def test_run_skips_video_in_final_loop(mock_dependencies):
    """
    Covers the 'continue' statement for a video that is in the skip set
    but also makes it into the final processing loop.
    """
    # Arrange
    mock_dependencies["skip_file_path"].exists.return_value = True
    mock_dependencies["sqlite_path"].exists.return_value = True
    # No previously processed JSON files
    mock_dependencies["raw_json_dir"].rglob.return_value = []

    # Mock the skip file to contain 'vid1'
    mock_file_open = mock_dependencies["skip_file_path"].open
    mock_file_open.return_value.__enter__.return_value.read.return_value = (
        '["vid1"]'
    )

    # DB returns two "new" videos, one of which is in the skip file
    mock_dependencies["get_db_data"].side_effect = [
        [{"video_id": "vid1"}, {"video_id": "vid2"}],
        [{"video_id": "vid1"}, {"video_id": "vid2"}],
    ]
    # YouTube API has data for both
    mock_dependencies["get_yt_data"].return_value = {"vid1": {}, "vid2": {}}

    # Act
    fetch_raw_data.run()

    # Assert
    # The key assertion is that process_video was only called for 'vid2'
    mock_dependencies["process_video"].assert_called_once()
    assert (
        mock_dependencies["process_video"].call_args[0][0]["video_id"]
        == "vid2"
    )


def test_run_no_new_videos(mock_dependencies):
    # This doesn't need to mock file I/O because the skip file doesn't exist
    mock_dependencies["skip_file_path"].exists.return_value = False
    mock_dependencies["sqlite_path"].exists.return_value = True
    mock_dependencies["get_db_data"].return_value = [{"video_id": "vid1"}]
    mock_dependencies["raw_json_dir"].rglob.return_value = [
        MagicMock(stem="vid1")
    ]

    fetch_raw_data.run()

    mock_dependencies["get_yt_data"].assert_not_called()
    mock_dependencies["process_video"].assert_not_called()
    mock_dependencies["print"].assert_any_call(
        "No new videos to process. All up to date."
    )


def test_run_creates_sqlite_db_if_not_exists(mock_dependencies):
    # This doesn't need to mock file I/O because the skip file doesn't exist
    mock_dependencies["skip_file_path"].exists.return_value = False
    mock_dependencies["sqlite_path"].exists.return_value = False
    mock_dependencies["get_db_data"].return_value = []

    fetch_raw_data.run()

    mock_dependencies["create_sqlite"].assert_called_once()


@pytest.mark.parametrize(
    "error", [json.JSONDecodeError("msg", "doc", 0), OSError("msg")]
)
def test_run_handles_corrupt_skip_file(mocker, mock_dependencies, error):
    mock_dependencies["skip_file_path"].exists.return_value = True
    # Mock the .open() method to raise an error on read
    mock_file_open = mocker.mock_open()
    mock_file_open.return_value.read.side_effect = error
    mocker.patch.object(
        mock_dependencies["skip_file_path"], "open", mock_file_open
    )

    mock_dependencies["get_db_data"].return_value = []

    fetch_raw_data.run()

    mock_dependencies["print"].assert_any_call(
        "-> Warning: Could not read or parse"
        f" {mock_dependencies['skip_file_path']}."
        f" Starting with an empty set. Error: {error}"
    )


def test_run_handles_missing_youtube_data(mock_dependencies):
    # This doesn't need to mock file I/O because the skip file doesn't exist
    mock_dependencies["skip_file_path"].exists.return_value = False
    mock_dependencies["sqlite_path"].exists.return_value = True
    mock_dependencies["get_db_data"].side_effect = [
        [{"video_id": "vid1"}],
        [{"video_id": "vid1"}],
    ]
    mock_dependencies["get_yt_data"].return_value = {}

    fetch_raw_data.run()

    mock_dependencies["process_video"].assert_not_called()
    mock_dependencies["print"].assert_any_call(
        "Warning: Could not find YouTube API data for new video ID: vid1"
    )


def test_run_handles_failed_skip_file_write(mocker, mock_dependencies):
    mock_dependencies["skip_file_path"].exists.return_value = False
    mock_dependencies["sqlite_path"].exists.return_value = True
    mock_dependencies["get_db_data"].side_effect = [
        [{"video_id": "vid1"}],
        [{"video_id": "vid1"}],
    ]
    mock_dependencies["get_yt_data"].return_value = {"vid1": {}}
    mock_dependencies["process_video"].return_value = True

    # Mock the .open() method to raise an error on write
    mock_file_open = mocker.mock_open()
    mock_file_open.return_value.write.side_effect = OSError("Disk full")
    mocker.patch.object(
        mock_dependencies["skip_file_path"], "open", mock_file_open
    )

    fetch_raw_data.run()

    mock_dependencies["print"].assert_any_call(
        "FATAL: Could not write to log file"
        f" {mock_dependencies['skip_file_path']}. Error: Disk full"
    )
