import pytest

from kfai.extractors.utils.helpers.database import get_video_db_data
from kfai.extractors.utils.helpers.processing import process_video
from kfai.extractors.utils.helpers.youtube import get_youtube_data


def test_youtube_data_fetch():
    video_ids = ["test_video_id"]  # Use a known test video
    result = get_youtube_data(video_ids)

    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == 1


def test_db_data_fetch():
    result = get_video_db_data(limit=5)

    assert isinstance(result, list)
    assert len(result) <= 5
    for video in result:
        assert "video_id" in video
        assert "title" in video
        assert "show_name" in video
