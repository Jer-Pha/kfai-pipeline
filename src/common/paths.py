from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
VIDEO_DATA_DIR = PROJECT_ROOT / "video_data"
RAW_JSON_DIR = VIDEO_DATA_DIR / "raw"
CLEANED_JSON_DIR = VIDEO_DATA_DIR / "cleaned"
