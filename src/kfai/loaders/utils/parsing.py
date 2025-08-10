import json
from datetime import datetime
from traceback import format_exc

from langchain_ollama import OllamaLLM
from loaders.utils.constants import PRIMARY_HOST_MAP
from loaders.utils.helpers.datetime import iso_string_to_epoch
from loaders.utils.helpers.llm import clean_llm_response
from loaders.utils.prompts import (
    GET_HOSTS_PROMPT,
    GET_SHOWS_PROMPT,
    GET_TOPICS_PROMPT,
    GET_YEARS_PROMPT,
)
from loaders.utils.types import PGVectorPublishedAt

PRIMARY_HOSTS = ", ".join(
    [f"'{k}' likely refers to '{v}'" for k, v in PRIMARY_HOST_MAP.items()]
)


def parse_shows(
    query: str, show_names: list[str], llm: OllamaLLM
) -> list[str]:
    try:
        get_shows_response = llm.invoke(
            GET_SHOWS_PROMPT.format(
                query=query,
                show_names=", ".join(show_names),
            )
        )
        get_shows_response = clean_llm_response(get_shows_response)
        print("    Shows found:\n", get_shows_response)
        get_shows_data = dict(json.loads(get_shows_response))
        shows_data: list[str] = get_shows_data.get("shows", [])
        return shows_data
    except Exception as e:
        print(f" !! Error while parsing shows:\n{e}\n")

    return []


def parse_hosts(query: str, hosts: list[str], llm: OllamaLLM) -> list[str]:
    try:
        get_hosts_response = llm.invoke(
            GET_HOSTS_PROMPT.format(
                query=query,
                hosts=", ".join(hosts),
                primary_hosts=PRIMARY_HOSTS,
            )
        )
        get_hosts_response = clean_llm_response(get_hosts_response)
        print("    Hosts found:\n", get_hosts_response)
        get_hosts_data = dict(json.loads(get_hosts_response))
        hosts_data: list[str] = get_hosts_data.get("hosts", [])
        return hosts_data
    except Exception as e:
        print(f" !! Error while parsing hosts:\n{e}\n")

    return []


def parse_year_range(
    query: str, llm: OllamaLLM
) -> tuple[list[PGVectorPublishedAt], list[str]]:
    try:
        current_year = datetime.now().year
        get_year_response = llm.invoke(
            GET_YEARS_PROMPT.format(
                query=query,
                year=current_year,
            )
        )

        get_year_response = clean_llm_response(get_year_response)

        if not get_year_response:
            print("no year found")
            return [], []

        parsed_data = dict(json.loads(get_year_response))

        filter_gte: dict[str, int] = {"$gte": 0}
        filter_lte: dict[str, int] = {"$lte": 0}
        years: list[str] = []

        if parsed_data.get("exact_year", None) != "NOT_FOUND":
            print("exact year found:", parsed_data["exact_year"])
            year = parsed_data["exact_year"]
            filter_gte["$gte"] = iso_string_to_epoch(f"{year}-01-01T00:00:00")
            filter_lte["$lte"] = iso_string_to_epoch(f"{year}-12-31T23:59:59")
            years.append(year)
        elif parsed_data.get("year_range", None) != "NOT_FOUND":
            print("year range found:", parsed_data["year_range"])
            _range = parsed_data["year_range"].split("-")
            start = _range[0]
            end = _range[1]
            filter_gte["$gte"] = iso_string_to_epoch(f"{start}-01-01T00:00:00")
            filter_lte["$lte"] = iso_string_to_epoch(f"{end}-12-31T23:59:59")
            years.append(start)
            years.append(end)
        elif parsed_data.get("before_year", None) != "NOT_FOUND":
            print("before year found:", parsed_data["before_year"])
            year = str(int(parsed_data["before_year"]) - 1)
            filter_gte["$gte"] = 1325376000
            filter_lte["$lte"] = iso_string_to_epoch(f"{year}-12-31T23:59:59")
            years.append(year)
        elif parsed_data.get("after_year", None) != "NOT_FOUND":
            print("after year found:", parsed_data["after_year"])
            year = str(int(parsed_data["after_year"]) + 1)
            filter_gte["$gte"] = iso_string_to_epoch(f"{year}-01-01T00:00:00")
            filter_lte["$lte"] = iso_string_to_epoch(
                f"{current_year}-12-31T23:59:59"
            )
            years.append(year)
        else:
            print("no year found")
            return [], []

        return [{"published_at": filter_gte}, {"published_at": filter_lte}], [
            *years
        ]

    except Exception as e:
        print(f"\n !! Error while parsing year range:\n{e}\n")
        return [], []


def parse_topics(
    query: str,
    show_filter: list[str],
    hosts_filter: list[str],
    years: list[str],
    llm: OllamaLLM,
) -> list[str]:
    try:
        metadata = show_filter + hosts_filter + years
        get_topics_response = llm.invoke(
            GET_TOPICS_PROMPT.format(
                query=query,
                metadata=", ".join(metadata),
            )
        )
        get_topics_response = clean_llm_response(get_topics_response)
        print("    Topics found:\n", get_topics_response)
        get_topics_data = json.loads(get_topics_response)
        topics = get_topics_data["topics"]
        return [t for t in topics if t not in metadata]

    except:
        print(f" !! Error while parsing topics:")
        print(format_exc(), end="\n\n")

    return []
