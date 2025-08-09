from typing import Optional, TypedDict


class PGVectorText(TypedDict):
    text: dict[str, str]


class PGVectorShowName(TypedDict):
    show_name: dict[str, list[str]]


class PGVectorHosts(TypedDict):
    hosts: dict[str, str]


class PGVectorPublishedAt(TypedDict):
    published_at: dict[str, int]
