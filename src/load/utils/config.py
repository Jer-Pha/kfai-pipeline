from os import getenv

from dotenv import load_dotenv

load_dotenv()

# PostgreSQL
POSTGRES_DB_PATH = getenv("POSTGRES_DB_PATH", default="")

# Models
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
PARSING_MODEL = "qwen3:14b-q4_K_M"
QA_MODEL = "qwen3:14b-q4_K_M"
