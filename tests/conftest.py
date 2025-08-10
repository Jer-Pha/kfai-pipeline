import sys
from pathlib import Path

import pytest

# Add src directory to Python path before any kfai imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# Now we can import from kfai
from kfai.core.paths import (
    CLEANED_JSON_DIR,
    DATA_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    RAW_JSON_DIR,
    VIDEO_DATA_DIR,
)


@pytest.fixture
def sample_raw_data():
    return {
        "id": 1,
        "video_id": "test123",
        "title": "Test Video",
        "show_name": "Kinda Funny Games Daily",
        "hosts": ["Greg Miller", "Tim Gettys"],
        "published_at": 1640995200,
        "transcript_chunks": [
            {
                "text": "hey what's up everybody this is greg miller",
                "start": 0.0,
                "duration": 2.5,
            }
        ],
    }


@pytest.fixture
def temp_test_dirs(tmp_path):
    """Create temporary test directories that mirror the project structure"""
    test_raw = tmp_path / "raw"
    test_cleaned = tmp_path / "cleaned"

    test_raw.mkdir()
    test_cleaned.mkdir()

    return {"raw": test_raw, "cleaned": test_cleaned}
