from os import getenv

from core.paths import DATA_DIR
from dotenv import load_dotenv

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
SQLITE_DB_PATH = DATA_DIR / SQLITE_DB_NAME


# Other
VIDEOS_TO_SKIP_FILE = DATA_DIR / "skipped_videos.json"
FAILED_VIDEOS_FILE = DATA_DIR / "failures_to_transcribe.json"
