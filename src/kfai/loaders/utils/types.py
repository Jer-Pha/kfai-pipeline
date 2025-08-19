from typing import TypedDict

from pydantic import BaseModel, Field


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


class QueryParseResponse(BaseModel):
    """A Pydantic model to structure the parsed output from a user query."""

    shows: list[str] = Field(
        default=[],
        description="A list of show names mentioned in the user's query.",
    )
    hosts: list[str] = Field(
        default=[],
        description="A list of host names mentioned in the user's query.",
    )
    topics: list[str] = Field(
        default=[],
        description=(
            "A list of the core semantic topics from the user's query,"
            " excluding any identified shows or hosts."
        ),
    )
    exact_year: str | None = Field(
        default=None,
        description="The specific year if mentioned (e.g., '2015').",
    )
    year_range: str | None = Field(
        default=None,
        description="A year range if mentioned (e.g., '2015-2017').",
    )
    before_year: str | None = Field(
        default=None,
        description=(
            "A year if the query specifies a time 'before' it (e.g.,"
            " 'before 2020')."
        ),
    )
    after_year: str | None = Field(
        default=None,
        description=(
            "A year if the query specifies a time 'after' it (e.g.,"
            " 'after 2018')."
        ),
    )


class SourceCitation(BaseModel):
    video_id: str = Field(
        description="The unique video_id of the source document."
    )
    start_time: float = Field(
        description=(
            "The start time in seconds of the relevant transcript chunk."
        )
    )


class AgentResponse(BaseModel):
    query_response: str = Field(
        description="The final, comprehensive answer to the user's query."
    )
    sources: list[SourceCitation] = Field(
        description=(
            "A list of all the source documents used to generate the answer."
        )
    )
