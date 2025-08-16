from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DATA_DIR = DATA_DIR / "video_data"
RAW_JSON_DIR = VIDEO_DATA_DIR / "raw"
CLEANED_JSON_DIR = VIDEO_DATA_DIR / "cleaned"
