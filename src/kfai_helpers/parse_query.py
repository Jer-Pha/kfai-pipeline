import json
import re
from datetime import datetime
from langchain_ollama import OllamaLLM
from sqlalchemy import text, Engine
from traceback import format_exc
from typing import Union

from .config import PARSING_MODEL, POSTGRES_DB_PATH
from .types import (
    PGVectorHosts,
    PGVectorPublishedAt,
    PGVectorShowName,
    PGVectorText,
)
from .utils import iso_string_to_epoch

# --- CONFIGURATION ---
POSTGRES_DB_PATH = POSTGRES_DB_PATH
llm = OllamaLLM(
    model=PARSING_MODEL,
    temperature=0.1,
    top_p=0.92,
    top_k=25,
    reasoning=True,
    verbose=False,
    keep_alive=300,
)

# -- GLOBAL REGEX COMPILERS ---
_compile = re.compile
_sub_squotes = _compile(r"[‘’]").sub
_sub_dquotes = _compile(r"[“”]").sub

# --- PROMPT SETUP ---
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

# --- PROMPT TEMPLATES ---
GET_SHOWS_PROMPT = """
    KNOWN SHOW NAMES:
    {show_names}

    INSTRUCTIONS
    - You are a meticulous string analyzer that has been given a USER QUERY (below)
    - Your task is to analyze the provided USER QUERY and look for any show name strings that are reasonably similar, or exact, to those in the SHOW NAMES master list (above), and convert them to the correct name from the master list then return the corrected string(s)
    - Show names would be **reasonably similar** for several reasons, including:
        - Punctuation
        - Spelling
        - Capitalization
        - Missing words (partial name)
        - Obvious initialization
    - *CRITICAL* do not use single names, such as Colin, Greg, or Blessing, to extrapolte into a show name. **Reasonably similar** requires at least two words from the known shows.
    - Some examples of possible conversions:
        - "ps i love you" converts to "PS I Love You"
        - "KF podcast" converts to "Kinda Funny Podcast"
        - "Gamecast" converts to "Gamescast"
        - "KFGD" converts to "Kinda Funny Games Daily"
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
    I remember Colin and Nick discussing the 2019 World Series, was it on the KF Podcast or game over greggy show?

    EXAMPLE #2 RESPONSE:
    {{
      "shows": ["Kinda Funny Podcast", "The GameOverGreggy Show"] // Colin is a regular host, his name should not immediately add "A Conversation With Colin" into your response
    }}

    EXAMPLE #3 QUERY:
    Which podcast did they talk about Greg's chicken wings?

    EXAMPLE #3 RESPONSE:
    {{
      "shows": [] // No known shows in query. Greg is a regular host, his name should not immediately add "The GameOverGreggy Show" into your response
    }}

    EXAMPLE #3 QUERY:
    On KFGD, has Blessing ever mentioned Donkey Kong?

    EXAMPLE #3 RESPONSE:
    {{
      "shows": ["Kinda Funny Games Daily"] // Blessing is a regular host, his name should not immediately add "The Blessing Show" into your response
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
    METADATA:
    {metadata}

    INSTRUCTIONS
    - You are a meticulous string analyzer that has been given a USER QUERY (below)
    - Your task is to **PARSE** this query for its main **topics**
    - Do **NOT** include any topics that are similar, or identical, to the METADATA (above)
    - Similar topics might be missing words or have different capitalization, for example:
        - "PS I love you" might refer to "PS I Love You XOXO"
        - "gameovergreggy show" might refer to "The GameOverGreggy Show"
        - "KF pod cast" might refer to "Kinda Funny Podcast"
    - Consider any phrases with two or more words that **surrounded by quotes** as a topic, such as "boom goes the dynamite"
    - Topics should typically be nouns or proper nouns, not phrases unless the user surrounds it in quotes
    - If the user misspelled a topic, return it with the correct spelling
    - Return a JSON object that matches the below formatting
    - Do **NOT** use markdown formatting. Do **NOT** include explanations. Do **NOT** answer the user question.
    - If no matches are found, return an empty array as the value: []
    - **CRITICAL** If a topic has Roman numerals, return topics for both the Roman and standard numbers; for example:
        - "Final Fantasy Versus XIII" as a topic should return both "Final Fantasy Versus XIII" and "Final Fantasy Versus 13"
        - "Rocky IV" as a topic should return both "Rocky IV" and "Rocky 4"

    RESPONSE FORMAT (JSON):
    {{
      "topics": ["string", ...]
    }}

    EXAMPLE #1 QUERY:
    What did Greg say about The Witcher III on P.S. I love you?

    EXAMPLE #1 METADATA:
    Greg Miller, PS I Love You XOXO

    EXAMPLE #1 RESPONSE:
    {{
      "topics": ["The Witcher III", "The Witcher 3"]
    }}

    EXAMPLE #2 QUERY:
    I remember them discussing the 2019 World Series and hotdog eatin contests, was it on the KF Podcast or game over greggy show?

    EXAMPLE #2 METADATA:
    Kinda Funny Podcast, The GameOverGreggy Show

    EXAMPLE #2 RESPONSE:
    {{
      "topics": ["2019 World Series", "hotdog eating contests"]
    }}

    EXAMPLE #3 QUERY:
    What did joeynoelle and Andrea talk about on the Morning Show?

    EXAMPLE #3 METADATA:
    Joey Noelle, Andrea Rene, Kinda Funny Morning Show

    EXAMPLE #3 RESPONSE:
    {{
      "topics": [] // No topics found
    }}

    USER QUERY:
    {query}

    RESPONSE:
"""


# --- HELPER FUNCTIONS ---
def get_unique_metadata(engine: Engine) -> tuple[list[str], list[str]]:
    """Queries the database to get all unique show names and hosts."""
    show_names = set()
    hosts = set()
    with engine.connect() as connection:
        # Get unique show names
        show_query = text(
            """
            SELECT DISTINCT (cmetadata ->> 'show_name') AS show_name
            FROM langchain_pg_embedding
            WHERE (cmetadata ->> 'show_name') IS NOT NULL;
        """
        )
        show_result = connection.execute(show_query)
        for row in show_result:
            if row[0]:  # row[0] is the show_name
                show_names.add(row[0])

        # Get unique hosts
        host_query = text(
            """
            SELECT host
            FROM (
                SELECT
                    TRIM(regexp_split_to_table(cmetadata ->> 'hosts', ',')) AS host,
                    (cmetadata ->> 'video_id') AS video_id
                FROM langchain_pg_embedding
            ) AS unnested_hosts
            WHERE host <> ''
            GROUP BY host
            HAVING COUNT(DISTINCT video_id) >= 5
            ORDER BY host;
        """
        )
        host_result = connection.execute(host_query)
        for row in host_result:
            if row[0]:  # row[0] is the host
                hosts.add(row[0])

    print("Host count:", len(hosts))

    return sorted(show_names), sorted(hosts)


def clean_llm_response(response: str) -> str:
    """Clean common LLM inconsistencies in the response."""
    response = response.split("</think>")[-1]
    response = _sub_squotes("'", response)
    response = _sub_dquotes('"', response)
    return response.strip()


# --- CORE FUNCTIONS ---
def _parse_shows(query: str, show_names: list[str]) -> list[str]:
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


def _parse_hosts(query: str, hosts: list[str]) -> list[str]:
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


def _parse_year_range(
    query: str,
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


def _parse_topics(
    query: str,
    show_filter: list[str],
    hosts_filter: list[str],
    years: list[str],
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


def _build_filter(
    shows_list: list[str],
    hosts_list: list[str],
    year_filter: list[PGVectorPublishedAt],
    topics_list: list[str],
) -> (
    dict[
        str,
        list[
            Union[
                PGVectorShowName,
                PGVectorHosts,
                PGVectorPublishedAt,
                dict[str, list[PGVectorText]],
            ]
        ],
    ]
    | None
):
    """Convert to filter for PGVector retriever"""
    print("  -> Combining filter...")
    filter_conditions: list[
        Union[
            PGVectorShowName,
            PGVectorHosts,
            PGVectorPublishedAt,
            dict[str, list[PGVectorText]],
        ]
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


def get_filter(query: str, show_names: list[str], hosts: list[str]) -> tuple[
    str,
    dict[
        str,
        list[
            Union[
                PGVectorShowName,
                PGVectorHosts,
                PGVectorPublishedAt,
                dict[str, list[PGVectorText]],
            ]
        ],
    ]
    | None,
]:
    print("Building filter...")
    print(f"  Model: {PARSING_MODEL}")

    print(" -> Parsing shows...")
    shows_list = _parse_shows(query, show_names)

    print(" -> Parsing hosts...")
    hosts_list = _parse_hosts(query, hosts)

    print(" -> Parsing year range...", end="")
    year_filter, years = _parse_year_range(query)

    print(" -> Parsing topics...")
    topics_list = _parse_topics(query, shows_list, hosts_list, years)

    filter_dict = _build_filter(
        shows_list, hosts_list, year_filter, topics_list
    )

    if topics_list:
        topics = ", ".join(topics_list)
    else:
        topics = query

    return topics, filter_dict
