from os import getenv

from dotenv import load_dotenv

from kfai.core.paths import DATA_DIR

load_dotenv()

# YouTube
YOUTUBE_API_KEY = getenv("YOUTUBE_API_KEY", default="")

# Remote
MYSQL_HOST = getenv("MYSQL_HOST", default="")
MYSQL_USER = getenv("MYSQL_USER", default="")
MYSQL_PASSWORD = getenv("MYSQL_PASSWORD", default="")
MYSQL_DATABASE = getenv("MYSQL_DATABASE", default="")

# Local
SQLITE_DB_NAME = getenv("SQLITE_DB_NAME", default=".sqlite")


# Paths
SQLITE_DB_PATH = DATA_DIR / SQLITE_DB_NAME
VIDEOS_TO_SKIP_FILE = DATA_DIR / "skipped_videos.json"
FAILED_VIDEOS_FILE = DATA_DIR / "failures_to_transcribe.json"
WHISPER_OUTPUT_DIR = DATA_DIR / "whisper_output"
YT_COOKIE_FILE = DATA_DIR / "www.youtube.com_cookies.txt"

# Whisper / YoutubeDL
WHISPER_MODEL = "medium.en"
CHUNK_THRESHOLD_SECONDS = 7200  # 2 hour
BASE_YT_DLP_OPTIONS = {
    "format": "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio",
    "nopart": True,
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    },
}
SKIP_LIST = {
    "ahLoo444NXk",  # Deleted
    "VsW8wQ9wOeY",  # No dialog
}
