from dotenv import load_dotenv
from os import getenv

load_dotenv()

DEBUG = getenv("DEBUG", default="True").lower() == "true"
YOUTUBE_API_KEY = getenv("YOUTUBE_API_KEY", default="")

DB_HOST = getenv("DB_HOST", default="")
DB_USER = getenv("DB_USER", default="")
DB_PASSWORD = getenv("DB_PASSWORD", default="")
DB_DATABASE = getenv("DB_DATABASE", default="")

SQLITE_DB_PATH = getenv("SQLITE_DB_PATH", default="")
POSTGRES_DB_PATH = getenv("POSTGRES_DB_PATH", default="")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
CLEANING_MODEL = "llama3.1:8b-instruct-q8_0"
PARSING_MODEL = "qwen3:14b-q4_K_M"
QA_MODEL = "qwen3:14b-q4_K_M"
