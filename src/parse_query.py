import json

from datetime import datetime
from langchain_ollama import OllamaLLM
from pprint import pprint
from sqlalchemy import create_engine, text, Engine
from typing import Optional

import config

# --- Configuration ---
POSTGRES_DB_PATH = config.POSTGRES_DB_PATH
MODEL = "phi4-mini:3.8b"
# PARSE_MODEL = "mistral:7b-instruct"
STRING_LLM = OllamaLLM(
    model=MODEL,
    temperature=0.5,
    verbose=True,
)
JSON_LLM = OllamaLLM(
    model=MODEL,
    temperature=0,
    verbose=True,
    format="json",
)

# --- Prompt Config ---
PRIMARY_HOST_MAP = {
    "Greg": "Greg Miller",
    "Tim": "Tim Gettys",
    "Nick": "Nick Scarpino",
    "Kevin": "Kevin Coello",
    "Joey": "Joey Noelle",
    "Andy": "Andy Cortez",
    "Barrett": "Barrett Courtney",
    "Blessing": "Blessing Adeoye Jr.",
    "Mike": "Mike Howard",
    "SnowBikeMike": "Mike Howard",
    "Roger": "Roger Pokorny",
    "Parris": "Parris Lilly",
    "Paris": "Parris Lilly",
    "Gary": "Gary Whitta",
    "Fran": "Fran Mirabella III",
    "Janet": "Janet Garcia",
    "Andrea": "Andrea Rene",
    "Tamoor": "Tamoor Hussain",
    "Jared": "Jared Petty",
    "Colin": "Colin Moriarty",
}
PRIMARY_HOSTS = ", ".join(
    [f"'{k}' likely refers to '{v}'" for k, v in PRIMARY_HOST_MAP.items()]
)
GET_SHOWS_PROMPT = """
    KNOWN SHOW NAMES:
    {show_names}

    INSTRUCTIONS
    - You are a meticulous string analyzer that has been given a USER QUERY (below)
    - Your task is to analyze the provided USER QUERY and look for any show name strings that are reasonably similar, or exact, to those in the SHOW NAMES master list (above), and convert them to the correct name from the master list then return the corrected string(s)
    - Show names would be reasonably similar for several reasons, including:
        - Punctuation
        - Spelling
        - Capitalization
        - Missing words (partial name)
        - Obvious initialization
    - Some examples of possible conversions:
        - "ps i love you" converts to "PS I Love You"
        - "KF podcast" converts to "Kinda Funny Podcast"
        - "Gamecast" converts to "Gamescast"
    - Return a JSON object that matches the below formatting
    - Do **NOT** use markdown formatting. Do **NOT** include explanations. Do **NOT** answer the user question
    - If no matches are found, return an empty array as the value: []

    RESPONSE FORMAT (JSON):
    {{
      "shows": ["string", ...]
    }}

    EXAMPLE #1 QUERY:
    What did Greg say about The Witcher 3 on PS I Luv you?

    EXAMPLE #1 RESPONSE:
    {{
      "shows": ["PS I Love You XOXO"]
    }}

    EXAMPLE #2 QUERY:
    I remember them discussing the 2019 World Series, was it on the KF Podcast or game over greggy show?

    EXAMPLE #2 RESPONSE:
    {{
      "shows": ["Kinda Funny Podcast", "The GameOverGreggy Show"]
    }}

    EXAMPLE #3 QUERY:
    Which podcast did they talk about Greg's chicken wings?

    EXAMPLE #3 RESPONSE:
    {{
      "shows": [] // No known shows in query
    }}

    USER QUERY:
    {query}

    RESPONSE:
"""
GET_HOSTS_PROMPT = """
    HOST NAMES:
    {hosts}

    PRIMARY HOST MAP:
    {primary_hosts}

    INSTRUCTIONS
    - You are a meticulous string analyzer that has been given a USER QUERY (below)
    - Your task is to analyze the provided USER QUERY and look for any host name strings that are reasonably similar, or exact, to those in the HOST NAMES master list (above), and convert them to the correct name from the master list then return the corrected string(s)
    - If a name in the string is ambiguous, such as only being a first name, use the PRIMARY HOST MAP to see if it can be converted
    - Host names would be reasonably similar for several reasons, including:
        - Punctuation
        - Spelling
        - Capitalization
        - Missing words (partial name)
        - Obvious initialization
    - Some examples of possible conversions:
        - "gregg miller" converts to "Greg Miller"
        - "Tim geddes" converts to "Tim Gettys"
        - "Paris Lily" converts to "Parris Lilly"
    - Return a JSON object that matches the below formatting
    - Do **NOT** use markdown formatting. Do **NOT** include explanations. Do **NOT** answer the user question
    - If no matches are found, return an empty array as the value: []

    RESPONSE FORMAT (JSON):
    {{
      "hosts": ["string", ...]
    }}

    EXAMPLE #1 QUERY:
    What did joeynoelle say about The Witcher 3?

    EXAMPLE #1 RESPONSE:
    {{
      "hosts": ["Joey Noelle"]
    }}

    EXAMPLE #2 QUERY:
    What city did Greg, colin, and Christine Stimer live in?

    EXAMPLE #2 RESPONSE:
    {{
      "hosts": ["Greg Miller", "Colin Moriarty", "Kristine Steimer"]
    }}

    EXAMPLE #3 QUERY:
    Which podcast did they talk about the Olympics?

    EXAMPLE #3 RESPONSE:
    {{
      "hosts": [] // No known hosts in query
    }}

    USER QUERY:
    {query}

    RESPONSE:
"""
GET_YEARS_PROMPT = """
    INSTRUCTIONS
    You are a meticulous string analyzer. You have been provided a USER QUERY below. Your task is to **PARSE** this query for the following information and return a JSON object that matches the below formatting. You should **NOT** answer the user question, you **ONLY** need to parse the required information.

    PARSE ITEMS:
        1. exact_year (string)
            - Identify if the user is requesting information about an exact year
            - If no year is given or if the user gave a window of before, after, or range of years: return `"NOT_FOUND"`
            - Treat years before 2012 and after {year} as mistakes and return `"NOT_FOUND"`
            - If found, should be in the format: "YYYY"

        2. before_year (string)
            - Identify if the user is requesting information **before** a specific year
            - Only return a year if the user is clearly asking for information before a certain year, else: return `"NOT_FOUND"`
            - Treat years before 2013 as mistakes and return `"NOT_FOUND"`
            - If found, should be in the format: "YYYY"

        3. after_year (string)
            - Identify if the user is requesting information **after** a specific year
            - Only return a year if the user is clearly asking for information after a certain year, else: return `"NOT_FOUND"`
            - Treat years after {year} as mistakes and return `"NOT_FOUND"`
            - If found, should be in the format: "YYYY"

        4. year_range (string)
            - Identify if the user is requesting information in the range of multiple years
            - Treat years before 2012 as mistakes and return 2012 as the start of the range
            - Treat years after {year} as mistakes and return {year} as the end of the range
            - If no range is given: return `"NOT_FOUND"`
            - If found, should be in the format: "YYYY-YYYY"

    RESPONSE FORMAT (JSON):
    {{
      "exact_year": "string | NOT_FOUND",
      "before_year": "string | NOT_FOUND",
      "after_year": "string | NOT_FOUND",
      "year_range": "string | NOT_FOUND"
    }}

    EXAMPLE #1 QUERY:
    What were the most popular topics on the podcast in 2017?

    EXAMPLE #1 RESPONSE:
    {{
      "exact_year": "2017",
      "before_year": "NOT_FOUND",
      "after_year": "NOT_FOUND",
      "year_range": "NOT_FOUND"
    }}

    EXAMPLE #2 QUERY:
    Did they talk about this topic before 2020?

    EXAMPLE #2 RESPONSE:
    {{
      "exact_year": "NOT_FOUND",
      "before_year": "2020",
      "after_year": "NOT_FOUND",
      "year_range": "NOT_FOUND"
    }}

    EXAMPLE #3 QUERY:
    Have they mentioned his name since 2019?

    EXAMPLE #3 RESPONSE:
    {{
      "exact_year": "NOT_FOUND",
      "before_year": "NOT_FOUND",
      "after_year": "2019",
      "year_range": "NOT_FOUND"
    }}

    EXAMPLE #4 QUERY:
    Who were the main hosts between 2006 and 2018?

    EXAMPLE #4 RESPONSE:
    {{
      "exact_year": "NOT_FOUND",
      "before_year": "NOT_FOUND",
      "after_year": "NOT_FOUND",
      "year_range": "2012-2018"
    }}

    EXAMPLE #5 QUERY:
    Where did they record the episodes from 2023 to 2026?

    EXAMPLE #5 RESPONSE:
    {{
      "exact_year": "NOT_FOUND",
      "before_year": "NOT_FOUND",
      "after_year": "NOT_FOUND",
      "year_range": "2023-2025"
    }}

    USER QUERY:
    {query}

    RESPONSE:
"""
GET_TOPICS_PROMPT = """
    KNOWN SHOW NAMES:
    {show_names}

    INSTRUCTIONS:
    - You are a meticulous string analyzer and have been provided a USER QUERY below
    - Your task is to **PARSE** this query for the main **topics** and return them in a comma-separated string
    - If there is no obvious main topic, return the original query (see EXAMPLE #3)
    - Do **NOT** include topics that are similar to KNOWN SHOW NAMES (above)
    - You should **NOT** answer the user question, you **ONLY** need to parse the required information.
    - Respond with **ONLY** the string of comma-separated topics. Do **NOT** use markdown formatting. Do **NOT** include explanations.

    EXAMPLE #1 QUERY:
    What did Tamoor and Ben Starr think about Cyberpunk 2077?

    EXAMPLE #1 RESPONSE:
    Cyberpunk 2077

    EXAMPLE #2 QUERY:
    What episode of the Gamescast did they talk about Spider-Man?

    EXAMPLE #2 RESPONSE:
    Spider-Man

    EXAMPLE #3 QUERY:
    What did Joey Noelle and Andrea talk about on the Kinda Funny Morning Show?

    EXAMPLE #3 RESPONSE:
    What did Joey Noelle and Andrea talk about on the Kinda Funny Morning Show?

    USER QUERY:
    {query}

    RESPONSE:
"""


def get_unique_metadata(engine: Engine) -> tuple[list, list]:
    """Queries the database to get all unique show names and hosts."""
    show_names = set()
    hosts = set()
    with engine.connect() as connection:
        # Get unique show names
        show_result = connection.execute(
            text("SELECT DISTINCT show_name FROM videos;")
        )
        for row in show_result:
            if row.show_name:
                show_names.add(row.show_name)

        # Get unique hosts
        host_result = connection.execute(
            text(
                """
                SELECT host
                FROM (
                    SELECT unnest(hosts) AS host
                    FROM videos
                ) AS unnested_hosts
                GROUP BY host
                HAVING COUNT(*) >= 5;
                """
            )
        )
        for row in host_result:
            if row.host:
                hosts.add(row.host)

    print("Host count:", len(hosts))

    return sorted(show_names), sorted(hosts)


def _parse_shows(query: str, show_names: list) -> Optional[list]:
    try:
        get_shows_response = JSON_LLM.invoke(
            GET_SHOWS_PROMPT.format(
                query=query, show_names=", ".join(show_names)
            )
        )
        print("    Shows found:\n", get_shows_response)

        get_shows_data = json.loads(get_shows_response)

        if get_shows_data.get("shows", []):
            return get_shows_data["shows"]

    except Exception as e:
        print(f" !! Error while parsing shows:\n{e}\n")

    return None


def _parse_hosts(query: str, hosts: list) -> Optional[list]:
    try:
        get_hosts_response = JSON_LLM.invoke(
            GET_HOSTS_PROMPT.format(
                query=query,
                hosts=", ".join(hosts),
                primary_hosts=PRIMARY_HOSTS,
            )
        )
        print("    Hosts found:\n", get_hosts_response)

        get_hosts_data = json.loads(get_hosts_response)

        host_filters = []

        for host in get_hosts_data.get("hosts", []):
            host_filters.append({"hosts": {"$like": f"%{host}%"}})

        return host_filters
    except Exception as e:
        print(f" !! Error while parsing hosts:\n{e}\n")

    return None


def _parse_year_range(query: str) -> Optional[tuple[dict, dict]]:
    try:
        current_year = datetime.now().year
        response = JSON_LLM.invoke(
            GET_YEARS_PROMPT.format(
                query=query,
                year=current_year,
            )
        )

        parsed_data = json.loads(response)

        if parsed_data["exact_year"] != "NOT_FOUND":
            print("  ...exact year found", parsed_data["exact_year"])
            year = int(parsed_data["exact_year"])
            return (
                {"published_at": {"$gte": f"{year}-01-01T00:00:00"}},
                {"published_at": {"$lte": f"{year}-12-31T23:59:59"}},
            )

        if parsed_data["year_range"] != "NOT_FOUND":
            print("  ...year range found:\n", parsed_data["year_range"])
            _range = parsed_data["year_range"].split("-")
            start = int(_range[0])
            end = int(_range[1])
            return (
                {"published_at": {"$gte": f"{start}-01-01T00:00:00"}},
                {"published_at": {"$lte": f"{end}-12-31T23:59:59"}},
            )

        if parsed_data["before_year"] != "NOT_FOUND":
            print("  ...before year found", parsed_data["before_year"])
            year = int(parsed_data["before_year"]) - 1
            return (
                {"published_at": {"$gte": "2012-01-01T00:00:00"}},
                {"published_at": {"$lte": f"{year}-12-31T23:59:59"}},
            )

        if parsed_data["after_year"] != "NOT_FOUND":
            print("  ...after year found", parsed_data["after_year"])
            year = int(parsed_data["after_year"]) + 1
            return (
                {"published_at": {"$gte": f"{year}-01-01T00:00:00"}},
                {"published_at": {"$lte": f"{current_year}-12-31T23:59:59"}},
            )
    except Exception as e:
        print(f" !! Error while parsing year range:\n{e}\n")

    return None


def _parse_topics(query: str, show_names: list) -> str:
    topics = STRING_LLM.invoke(
        GET_TOPICS_PROMPT.format(
            query=query,
            show_names=show_names,
        )
    )
    print("    Topics found:\n", topics)
    return topics


def _build_filter(query: str, show_names: list, hosts: list) -> Optional[dict]:
    """Convert to filter for PGVector retriever"""
    print("Building filter...")
    filter_conditions = []

    print(" -> Parsing shows...")
    show_filter = _parse_shows(query, show_names)
    if show_filter:
        filter_conditions.append({"show_name": {"$in": show_filter}})

    print(" -> Parsing hosts...")
    hosts_filter = _parse_hosts(query, hosts)
    if hosts_filter:
        for condition in hosts_filter:
            filter_conditions.append(condition)

    print(" -> Parsing year range...")
    year_filter = _parse_year_range(query)
    if year_filter is not None:
        for filter in year_filter:
            filter_conditions.append(filter)

    if filter_conditions:
        filter_dict = {"$and": filter_conditions}
        print("Filter built:\n", filter_dict)
        return filter_dict

    return None


def get_filter(query: str, show_names: list, hosts: list):
    filter_dict = _build_filter(query, show_names, hosts)

    print(" -> Parsing topics...")
    topics = _parse_topics(query, show_names)
    return filter_dict, topics
