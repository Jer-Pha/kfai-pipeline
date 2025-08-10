from typing import TypedDict


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


class TranscriptSnippet(TypedDict):
    text: str
    start: float
    duration: float


class VideoMetadata(TypedDict):
    title: str
    description: str
    published_at: int
    duration: int
