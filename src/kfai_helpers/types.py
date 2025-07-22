from typing import Optional, TypedDict, Union


# --- parse_query.py ---
class PGVectorText(TypedDict):
    text: dict[str, str]


class PGVectorShowName(TypedDict):
    show_name: dict[str, list[str]]


class PGVectorHosts(TypedDict):
    hosts: dict[str, str]


class PGVectorPublishedAt(TypedDict):
    published_at: dict[str, int]


# --- db.py ---
class MySQLConfig(TypedDict):
    host: str
    user: str
    password: str
    database: str


class RawVideoRecord(TypedDict):
    id: int
    video_id: str
    show_name: str
    hosts: list[str]


# --- transcript.py ---
class TranscriptChunk(TypedDict):
    text: str
    start: float


class TranscriptSnippet(TypedDict):
    text: str
    start: float
    duration: float


# --- video.py ---
class VideoMetadata(TypedDict):
    title: str
    description: str
    published_at: int
    duration: int


class CompleteVideoRecord(RawVideoRecord, VideoMetadata):
    transcript_chunks: Optional[list[TranscriptChunk]]
