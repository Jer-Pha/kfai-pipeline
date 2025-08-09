from os import getenv

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
SQLITE_DB_PATH = getenv("SQLITE_DB_PATH", default="")
