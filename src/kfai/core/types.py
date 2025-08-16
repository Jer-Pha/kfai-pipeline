from typing import TypedDict

from kfai.extractors.utils.types import RawVideoRecord, VideoMetadata


class TranscriptChunk(TypedDict):
    text: str
    start: float


class CompleteVideoRecord(RawVideoRecord, VideoMetadata):
    transcript_chunks: list[TranscriptChunk] | None
