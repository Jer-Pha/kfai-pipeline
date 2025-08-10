import os
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_raw_data():
    return {
        "video_id": "test123",
        "title": "Test Video",
        "show_name": "Kinda Funny Games Daily",
        "hosts": ["Greg Miller", "Tim Gettys"],
        "published_at": 1640995200,  # 2022-01-01
        "transcript_chunks": [
            {
                "text": "hey what's up everybody this is greg miller",
                "start": 0.0,
                "duration": 2.5,
            }
        ],
    }


@pytest.fixture
def test_paths(tmp_path):
    return {
        "raw": tmp_path / "raw",
        "cleaned": tmp_path / "cleaned",
        "vectors": tmp_path / "vectors",
    }
