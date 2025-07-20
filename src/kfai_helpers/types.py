from typing import Optional, TypedDict, Union


# --- parse_query.py ---
class PGVectorFilterGte(TypedDict):
    __annotations__ = {
        "$gte": str,
    }


class PGVectorFilterLte(TypedDict):
    __annotations__ = {
        "$lte": str,
    }


class PGVectorFilterIn(TypedDict):
    __annotations__ = {
        "$in": list[str],
    }


class PGVectorFilterLike(TypedDict):
    __annotations__ = {
        "$like": str,
    }


class PGVectorFilterILike(TypedDict):
    __annotations__ = {
        "$ilike": str,
    }


class PGVectorText(TypedDict):
    text: PGVectorFilterILike


class PGVectorShowName(TypedDict):
    show_name: PGVectorFilterIn


class PGVectorHosts(TypedDict):
    hosts: PGVectorFilterLike


class PGVectorPublishedAt(TypedDict):
    published_at: PGVectorFilterGte | PGVectorFilterLte


class PGVectorFilterOr(TypedDict):
    __annotations__ = {
        "$or": list[PGVectorText],
    }


class PGVectorFilter(TypedDict):
    __annotations__ = {
        "$and": list[
            Union[
                PGVectorShowName,
                PGVectorHosts,
                PGVectorPublishedAt,
                PGVectorFilterOr,
            ]
        ],
    }


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


class CompleteVideoRecord(TypedDict):
    id: int
    video_id: str
    show_name: str
    hosts: list[str]
    title: str
    description: str
    published_at: int
    duration: int
    transcript_chunks: Optional[list[TranscriptChunk]]
