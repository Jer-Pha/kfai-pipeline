import re

from langchain_ollama import OllamaLLM

from kfai.loaders.utils.config import PARSING_MODEL
from kfai.loaders.utils.parsing import (
    parse_hosts,
    parse_shows,
    parse_topics,
    parse_year_range,
)
from kfai.loaders.utils.types import (
    PGVectorHosts,
    PGVectorPublishedAt,
    PGVectorShowName,
    PGVectorText,
)


def _build_filter(
    shows_list: list[str],
    hosts_list: list[str],
    year_filter: list[PGVectorPublishedAt],
    topics_list: list[str],
) -> (
    dict[
        str,
        list[
            PGVectorShowName
            | PGVectorHosts
            | PGVectorPublishedAt
            | dict[str, list[PGVectorText]]
        ],
    ]
    | None
):
    """Convert to filter for PGVector retriever"""
    print("  -> Combining filter...")
    filter_conditions: list[
        PGVectorShowName
        | PGVectorHosts
        | PGVectorPublishedAt
        | dict[str, list[PGVectorText]]
    ] = []

    if shows_list:
        show_filter: PGVectorShowName = {"show_name": {"$in": shows_list}}
        filter_conditions.append(show_filter)

    for host in hosts_list:
        host = re.sub(r"([%_])", r"\\\1", host)
        host_filter: PGVectorHosts = {"hosts": {"$like": f"%{host}%"}}
        filter_conditions.append(host_filter)

    for filter in year_filter:
        if filter:
            filter_conditions.append(filter)

    topic_filters: list[PGVectorText] = []
    for topic in topics_list:
        if not topic.strip():
            continue
        topic = re.sub(r"([%_])", r"\\\1", topic)
        topic_filters.append({"text": {"$ilike": f"%{topic}%"}})
    if topic_filters:
        filter_conditions.append({"$or": topic_filters})

    if filter_conditions:
        filter_dict = {"$and": filter_conditions}
        print("    Final filter:\n", filter_dict)
        return filter_dict

    return None


def get_filter(
    query: str,
    show_names: list[str],
    hosts: list[str],
) -> tuple[
    str,
    dict[
        str,
        list[
            PGVectorShowName
            | PGVectorHosts
            | PGVectorPublishedAt
            | dict[str, list[PGVectorText]]
        ],
    ]
    | None,
]:
    llm = OllamaLLM(
        model=PARSING_MODEL,
        temperature=0.1,
        top_p=0.92,
        top_k=25,
        reasoning=True,
        verbose=False,
        keep_alive=300,
    )

    print("Building filter...")
    print(f"  Model: {PARSING_MODEL}")

    shows_list = parse_shows(query, show_names, llm)
    hosts_list = parse_hosts(query, hosts, llm)
    year_filter, years = parse_year_range(query, llm)
    topics_list = parse_topics(query, shows_list, hosts_list, years, llm)

    filter_dict = _build_filter(
        shows_list, hosts_list, year_filter, topics_list
    )

    if topics_list:
        topics = ", ".join(topics_list)
    else:
        topics = query

    return topics, filter_dict
