from typing import Optional, TypedDict


class PGVectorText(TypedDict):
    text: dict[str, str]


class PGVectorShowName(TypedDict):
    show_name: dict[str, list[str]]


class PGVectorHosts(TypedDict):
    hosts: dict[str, str]


class PGVectorPublishedAt(TypedDict):
    published_at: dict[str, int]


class EmbeddingCMetadata(TypedDict):
    hosts: str
    title: str
    video_id: str
    show_name: str
    start_time: float
    published_at: int
    text: str


class GroupedSourceData(TypedDict):
    timestamps: set[int]
    metadata: EmbeddingCMetadata


class TimestampReference(TypedDict):
    timestamp_sec: int
    formatted_time: str
    timestamp_href: str


class VideoDataSource(TypedDict):
    title: str
    show_name: str
    video_href: str
    thumbnail_src: str
    references: list[TimestampReference]
