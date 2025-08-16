from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError

# The module we are testing
from kfai.extractors.utils.helpers import youtube as youtube_utils

# --- Tests for simple helper functions ---


@pytest.mark.parametrize(
    "input_str, expected_epoch",
    [
        ("2023-01-01T12:00:00Z", 1672574400),
        ("2024-07-22T20:30:00Z", 1721680200),
        ("", 0),  # Edge case: empty string
    ],
)
def test_yt_datetime_to_epoch(input_str, expected_epoch):
    assert youtube_utils.yt_datetime_to_epoch(input_str) == expected_epoch


@pytest.mark.parametrize(
    "input_duration, expected_seconds",
    [
        ("PT1H2M3S", 3723),  # 1 hour, 2 minutes, 3 seconds
        ("PT15M50S", 950),  # 15 minutes, 50 seconds
        ("PT5S", 5),
    ],
)
def test_duration_to_seconds(input_duration, expected_seconds):
    # isodate.parse_duration expects a Duration object, but the YouTube API
    # provides a string. The function under test handles this.
    assert (
        youtube_utils.duration_to_seconds(input_duration) == expected_seconds
    )


# --- Tests for get_youtube_data ---


@pytest.fixture
def mock_yt_api(mocker):
    """Fixture to mock the entire googleapiclient chain."""
    mock_ytapi_module = mocker.patch(
        "kfai.extractors.utils.helpers.youtube.ytapi"
    )
    mock_service = MagicMock()
    mock_videos_resource = MagicMock()
    mock_list_request = MagicMock()

    mock_ytapi_module.build.return_value = mock_service
    mock_service.videos.return_value = mock_videos_resource
    mock_videos_resource.list.return_value = mock_list_request

    # Return the final object in the chain, whose .execute() we will control
    return mock_list_request.execute


def test_get_youtube_data_single_batch(mock_yt_api):
    """Tests fetching data for a number of videos under the 50 ID limit."""
    # Arrange: A fake API response
    fake_response = {
        "items": [
            {
                "id": "vid1",
                "snippet": {
                    "title": "Title 1",
                    "publishedAt": "2023-01-01T00:00:00Z",
                },
                "contentDetails": {"duration": "PT10M"},
            }
        ]
    }
    mock_yt_api.return_value = fake_response

    # Act
    data = youtube_utils.get_youtube_data(["vid1"])

    # Assert
    assert "vid1" in data
    assert data["vid1"]["title"] == "Title 1"
    assert data["vid1"]["duration"] == 600
    mock_yt_api.assert_called_once()


def test_get_youtube_data_multiple_batches(mock_yt_api):
    """Tests that the function correctly chunks requests for over 50 IDs."""
    video_ids = [f"vid{i}" for i in range(51)]  # 51 video IDs
    mock_yt_api.return_value = {
        "items": []
    }  # Response content doesn't matter here

    # Act
    youtube_utils.get_youtube_data(video_ids)

    # Assert: API should have been called twice (1 batch of 50, 1 batch of 1)
    assert mock_yt_api.call_count == 2


def test_get_youtube_data_http_error(mock_yt_api):
    """Tests that the function returns None on an HttpError."""
    mock_yt_api.side_effect = HttpError(MagicMock(), b"API error")
    data = youtube_utils.get_youtube_data(["vid1"])
    assert data is None


def test_get_youtube_data_key_error(mock_yt_api):
    """Tests that the function returns None if the response is malformed."""
    mock_yt_api.return_value = {
        "items": [{"id": "vid1"}]
    }  # Missing snippet/contentDetails
    data = youtube_utils.get_youtube_data(["vid1"])
    assert data is None


# --- Tests for download_audio_handler ---


@pytest.fixture
def mock_downloader(mocker):
    """Fixture to mock YoutubeDL and file system interactions."""
    # Mock YoutubeDL and its context manager
    mock_ydl_class = mocker.patch(
        "kfai.extractors.utils.helpers.youtube.YoutubeDL"
    )
    mock_ydl_instance = mock_ydl_class.return_value.__enter__.return_value

    # --- CORRECTED PATH MOCKING ---
    # Mock the TEMP_DATA_DIR constant directly.
    # This is the object the '/' is called on.
    mock_temp_dir = mocker.patch(
        "kfai.extractors.utils.helpers.youtube.TEMP_DATA_DIR"
    )

    # The result of the '/' operation is what we need to control.
    # This mock will represent the 'chunk_path' variable inside the loop.
    mock_chunk_path = mock_temp_dir.__truediv__.return_value

    # Mock the constant for the chunk threshold to make tests deterministic
    mocker.patch(
        "kfai.extractors.utils.helpers.youtube.CHUNK_THRESHOLD_SECONDS", 100
    )

    # Return a dictionary of mocks to use in tests
    return {
        "ydl_class": mock_ydl_class,
        "ydl_instance": mock_ydl_instance,
        # This is now the correct mock object to configure in our tests
        "path_instance": mock_chunk_path,
    }


def test_download_audio_handler_single_chunk(mock_downloader):
    """Tests downloading a short video that fits in one chunk."""
    mock_downloader[
        "path_instance"
    ].exists.return_value = False  # File doesn't exist

    paths = youtube_utils.download_audio_handler("vid1", duration=90)

    assert len(paths) == 1
    mock_downloader["ydl_instance"].download.assert_called_once()


def test_download_audio_handler_multiple_chunks(mock_downloader):
    """Tests downloading a long video that requires multiple chunks."""
    mock_downloader["path_instance"].exists.return_value = False

    paths = youtube_utils.download_audio_handler(
        "vid1", duration=250
    )  # Requires 3 chunks

    assert len(paths) == 3
    assert mock_downloader["ydl_instance"].download.call_count == 3


def test_download_audio_handler_skips_existing_chunks(mock_downloader):
    """Tests that existing chunks are not re-downloaded."""
    mock_downloader[
        "path_instance"
    ].exists.return_value = True  # File *does* exist

    paths = youtube_utils.download_audio_handler("vid1", duration=150)

    assert len(paths) == 2
    # The key assertion: download should never be called
    mock_downloader["ydl_instance"].download.assert_not_called()


def test_download_audio_handler_download_error(mock_downloader):
    """Tests that an error during download cleans up and returns None."""
    # --- CORRECTED MOCK BEHAVIOR ---
    # Use side_effect to provide a sequence of return values.
    # 1st call to .exists() returns False (triggers download).
    # 2nd call to .exists() returns True (simulates partial file for cleanup).
    mock_downloader["path_instance"].exists.side_effect = [False, True]

    # Simulate an error when .download() is called
    mock_downloader["ydl_instance"].download.side_effect = Exception(
        "Download failed"
    )

    paths = youtube_utils.download_audio_handler("vid1", duration=150)

    assert paths is None
    # Verify that the failed chunk was deleted
    mock_downloader["path_instance"].unlink.assert_called_once()


def test_download_audio_handler_no_duration(mock_downloader):
    """Tests the initial guard clause for zero duration."""
    paths = youtube_utils.download_audio_handler("vid1", duration=0)
    assert paths is None
    mock_downloader["ydl_instance"].download.assert_not_called()
