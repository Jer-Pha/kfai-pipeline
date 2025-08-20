import pytest

from kfai.extractors import process_failed_videos

# --- Comprehensive Fixture for Mocking Dependencies ---


@pytest.fixture
def mock_dependencies(mocker):
    """A single fixture to mock all external dependencies of the run script."""
    # Mock file system paths and their .open() methods
    mock_skip_file = mocker.patch(
        "kfai.extractors.process_failed_videos.VIDEOS_TO_SKIP_FILE"
    )
    mock_failed_file = mocker.patch(
        "kfai.extractors.process_failed_videos.FAILED_VIDEOS_FILE"
    )

    # Mock helper function dependencies
    mocks = {
        "get_db_data": mocker.patch(
            "kfai.extractors.process_failed_videos.get_video_db_data"
        ),
        "get_yt_data": mocker.patch(
            "kfai.extractors.process_failed_videos.get_youtube_data"
        ),
        "dump": mocker.patch("json.dump"),
        "print": mocker.patch("builtins.print"),
        "skip_file_path": mock_skip_file,
        "failed_file_path": mock_failed_file,
    }
    return mocks


# --- Test Suite ---


def test_run_happy_path(mocker, mock_dependencies):
    """
    Tests the main success path: reads failed IDs, enriches them, and
    writes the output.
    """
    # 1. Arrange
    # Mock the reading of the skip file
    mocker.patch.object(
        mock_dependencies["skip_file_path"],
        "open",
        mocker.mock_open(read_data='["vid1", "vid2"]'),
    )

    # Mock the return values of the data fetching helpers
    mock_dependencies["get_db_data"].return_value = [
        {"video_id": "vid1", "db_data": "value1"},
        {"video_id": "vid2", "db_data": "value2"},
    ]
    mock_dependencies["get_yt_data"].return_value = {
        "vid1": {"yt_data": "api_value1"},
        "vid2": {"yt_data": "api_value2"},
    }

    # Mock the writing of the final output file
    mock_output_open = mocker.mock_open()
    mocker.patch.object(
        mock_dependencies["failed_file_path"], "open", mock_output_open
    )

    # 2. Act
    process_failed_videos.run()

    # 3. Assert
    # Verify helpers were called with the correct IDs
    mock_dependencies["get_db_data"].assert_called_once_with(
        video_ids=["vid1", "vid2"]
    )
    mock_dependencies["get_yt_data"].assert_called_once_with(["vid1", "vid2"])

    # Verify the final JSON was dumped with correctly merged data
    mock_dependencies["dump"].assert_called_once()
    dumped_data = mock_dependencies["dump"].call_args[0][0]
    assert len(dumped_data) == 2
    assert dumped_data[0] == {
        "video_id": "vid1",
        "db_data": "value1",
        "yt_data": "api_value1",
    }


def test_run_handles_corrupt_skip_file(mocker, mock_dependencies):
    """
    Tests that a corrupt skip file is handled gracefully and the script
    continues.
    """
    # 1. Arrange
    # Mock the skip file read to raise an error
    mock_skip_open = mocker.mock_open()
    mock_skip_open.side_effect = OSError("File not found")
    mocker.patch.object(
        mock_dependencies["skip_file_path"], "open", mock_skip_open
    )

    # 2. Act
    process_failed_videos.run()

    # 3. Assert
    # A warning should be printed
    mock_dependencies["print"].assert_any_call(
        "-> Warning: Could not read or parse"
        f" {mock_dependencies['skip_file_path']}. Error: File not found"
    )
    # The script should continue and call the helpers with an empty list
    mock_dependencies["get_db_data"].assert_called_once_with(video_ids=[])


def test_run_handles_youtube_api_error(mocker, mock_dependencies):
    """
    Tests that if the YouTube API fails (returns None), the script
    does not write an output file.
    """
    mocker.patch.object(
        mock_dependencies["skip_file_path"],
        "open",
        mocker.mock_open(read_data='["vid1"]'),  # Provide some valid JSON
    )

    # Arrange: get_yt_data returns None
    mock_dependencies["get_yt_data"].return_value = None

    # Act
    process_failed_videos.run()

    # Assert: The final dump should never be called
    mock_dependencies["dump"].assert_not_called()


def test_run_handles_partial_youtube_data(mocker, mock_dependencies):
    """
    Tests that if some videos are missing from the API response, they
    are excluded from the final output.
    """
    # Arrange
    mocker.patch.object(
        mock_dependencies["skip_file_path"],
        "open",
        mocker.mock_open(read_data='["vid1", "vid2"]'),
    )
    mock_dependencies["get_db_data"].return_value = [
        {"video_id": "vid1", "db_data": "value1"},
        {"video_id": "vid2", "db_data": "value2"},
    ]
    # API only returns data for vid1
    mock_dependencies["get_yt_data"].return_value = {
        "vid1": {"yt_data": "api_value1"}
    }
    mocker.patch.object(
        mock_dependencies["failed_file_path"], "open", mocker.mock_open()
    )

    # Act
    process_failed_videos.run()

    # Assert
    # A warning should be printed for the missing video
    video_id = "vid2"
    mock_dependencies["print"].assert_any_call(
        "Warning: Could not find YouTube API data for failed video"
        f" ID: {video_id}"
    )
    # The final dumped data should only contain the one successful video
    dumped_data = mock_dependencies["dump"].call_args[0][0]
    assert len(dumped_data) == 1
    assert dumped_data[0]["video_id"] == "vid1"
