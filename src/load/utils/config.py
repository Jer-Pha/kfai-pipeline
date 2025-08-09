from os import getenv

from dotenv import load_dotenv

from common.config import CLEANED_JSON_DIR, RAW_JSON_DIR

load_dotenv()


POSTGRES_DB_PATH = getenv("POSTGRES_DB_PATH", default="")
COLLECTION_NAME = "video_transcript_chunks"
BATCH_SIZE = 256
COLLECTION_TABLE = "video_transcript_chunks"
CONTEXT_COUNT = 100

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
PARSING_MODEL = "qwen3:14b-q4_K_M"
QA_MODEL = "qwen3:14b-q4_K_M"

JSON_SOURCE_DIR = RAW_JSON_DIR  # Change to CLEANED_JSON_DIR later
