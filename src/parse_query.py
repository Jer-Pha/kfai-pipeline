import json

from langchain_ollama import OllamaLLM
from pprint import pprint
from sqlalchemy import create_engine, text, Engine

import config

# --- Configuration ---
POSTGRES_DB_PATH = config.POSTGRES_DB_PATH
# CLEAN_MODEL = "phi4-mini:3.8b"
CLEAN_MODEL = "mistral:7b-instruct"
PARSE_MODEL = "phi4-mini:3.8b"
# PARSE_MODEL = "mistral:7b-instruct"
CLEAN_LLM = OllamaLLM(
    model=CLEAN_MODEL,
    temperature=0,
    verbose=False,
)
PARSE_LLM = OllamaLLM(
    model=PARSE_MODEL,
    temperature=0,
    verbose=False,
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
CLEAN_SHOW_PROMPT = """
    SHOW NAMES:
    {show_names}

    INSTRUCTIONS
    - You are a meticulous string analyzer. You have been given a USER STRING (below).
    - Your task is to analyze the provided USER STRING and look for any show name strings that are reasonably similar, but not exact, to those in the SHOW NAMES master list (above), and convert them to the correct name from the master list then return the corrected string.
    - Show names would be reasonably similar for several reasons, including:
        - Punctuation
        - Spelling
        - Capitalization
        - Missing words (partial name)
    - Some examples of possible conversions:
        - "ps i love you" converts to "PS I Love You"
        - "KF podcast" converts to "Kinda Funny Podcast"
        - "Gamecast" converts to "Gamescast"
    - Respond with **ONLY** the corrected user string. Do **NOT** use markdown formatting. Do **NOT** include explanations.

    EXAMPLE STRING:
    What did Greg say about The Witcher 3 on PS I Luv you?

    EXAMPLE RESPONSE:
    What did Greg say about The Witcher 3 on PS I Love You XOXO?

    USER STRING:
    {query}

    RESPONSE:
"""
CLEAN_HOSTS_PROMPT = """
    HOST NAMES:
    {hosts}

    PRIMARY HOST MAP:
    {primary_hosts}

    INSTRUCTIONS
    - You are a meticulous string analyzer. You have been given a USER STRING (below).
    - Your task is to analyze the provided USER STRING and look for any host name strings that are reasonably similar, but not exact, to those in the HOST NAMES master list (above), and convert them to the correct name from the master list then return the corrected string.
    - If only a first name is given, map it using the PRIMARY HOST MAP to match the correct host name for conversion.
    - The converted names **MUST** match those in the HOST NAMES list.
    - Host names would be reasonably similar for several reasons, including:
        - Punctuation
        - Spelling
        - Capitalization
        - Missing words (partial name)
    - Some examples of possible conversions:
        - "gregg miller" converts to "Greg Miller"
        - "Tim geddes" converts to "Tim Gettys"
        - "Paris Lily" converts to "Parris Lilly"
    - Respond with **ONLY** the corrected user string. Do **NOT** use markdown formatting. Do **NOT** include explanations.

    EXAMPLE STRING:
    What city did Greg, colin, and Christine Stimer live in?

    EXAMPLE RESPONSE:
    What city did Greg Miller, Colin Moriarty, and Kristine Steimer live in?

    USER STRING:
    {query}

    RESPONSE:
"""
PARSE_PROMPT = """
    KNOWN SHOW NAMES:
    {show_names}

    KNOWN HOSTS:
    {hosts}

    PRIMARY HOST MAP:
    {primary_hosts}

    INSTRUCTIONS:
    You have been provided a USER QUERY below. Your task is to **PARSE** this query for the following information and return a JSON object that matches the below formatting. You should **NOT** answer the user question, you **ONLY** need to parse the required information.

    PARSE ITEMS:
        1. show_name (string)
            - You must **ONLY** use known values, provided above (KNOWN SHOW NAMES)
            - If there is more than one known show name found, choose the best one based on the context of the query
            - If no known value is found in the query, return `"NOT_FOUND"`

        2. hosts (list[string])
            - You must **ONLY** use known values, provided above (KNOWN HOSTS)
            - If the user mentions a first name, map it using the provided PRIMARY HOST MAP
            - If no known value is found in the query, return `["NOT_FOUND"]`

        3. topic (string)
            - Any keywords in the user query that do not match the `show_name` or `hosts`
            - Do not include common words, e.g. 'the', 'a', 'an', 'and', etc.
            - Do not return an empty string, use the original user query as a fallback

    RESPONSE FORMAT (JSON):
    {{
      "show_name": "string | NOT_FOUND",
      "hosts": ["string", ...] | ["NOT_FOUND"],
      "topic": "string"
    }}

    EXAMPLE #1 QUERY:
    What did Tamoor and Ben Starr think about Cyberpunk 2077?

    EXAMPLE #1 RESPONSE:
    {{
      "show_name": "NOT_FOUND", // No known show mentioned in the query
      "hosts": ["Tamoor Hussain", "Ben Starr"],
      "topic": "Cyberpunk 2077"
    }}

    EXAMPLE #2 QUERY:
    What did Joey Noelle and Andrea talk about on the Kinda Funny Morning Show?

    EXAMPLE #2 RESPONSE:
    {{
      "show_name": "Kinda Funny Morning Show",
      "hosts": ["Joey Noelle", "Andrea Rene"],
      "topic": "What did Joey Noelle and Andrea talk about on the Kinda Funny Morning Show?" // Fallback to original query
    }}

    EXAMPLE #3 QUERY:
    What episode of the Gamescast did they talk about Spider-Man?

    EXAMPLE #3 RESPONSE:
    {{
      "show_name": "Gamescast",
      "hosts": ["NOT_FOUND"], // No known hosts mentioned in the query
      "topic": "Spider-Man"
    }}

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
            text("SELECT DISTINCT unnest(hosts) AS host FROM videos;")
        )
        for row in host_result:
            if row.host:
                hosts.add(row.host)

    return sorted(show_names), sorted(hosts)


def build_filter(parsed_data: dict, query: str) -> dict:
    """Convert to filter for PGVector retriever"""
    filter_conditions = []

    if parsed_data["show_name"] != "NOT_FOUND":
        filter_conditions.append(
            {"show_name": {"$eq": parsed_data["show_name"]}}
        )

    if parsed_data["hosts"] and parsed_data["hosts"] != ["NOT_FOUND"]:
        hosts_filter = {
            "$and": [
                {"hosts": {"$in": [host]}} for host in parsed_data["hosts"]
            ]
        }
        filter_conditions.append(hosts_filter)

    filter_dict = {"$and": filter_conditions}

    topics = parsed_data.get("topic", query)

    print(filter_dict)

    return filter_dict, topics


def parse_query(query: str, show_names: list, hosts: list) -> dict:
    # 1. Clean up user query before parsing
    print(" -> Cleaning query show names...")
    cleaned_query_show_name = CLEAN_LLM.invoke(
        CLEAN_SHOW_PROMPT.format(
            query=query,
            show_names=", ".join(show_names),
        )
    )
    print(" -> Cleaning query host names...")
    cleaned_query_hosts = CLEAN_LLM.invoke(
        CLEAN_HOSTS_PROMPT.format(
            query=cleaned_query_show_name,
            hosts=", ".join(hosts),
            primary_hosts=PRIMARY_HOSTS,
        )
    )
    print(" -> Cleaning complete...")
    print("    Raw:", query)
    print("    Cleaned:", cleaned_query_hosts)

    # 2. Send query to LLM for parsing
    print(" -> Parsing query to dictionary...")
    response = PARSE_LLM.invoke(
        PARSE_PROMPT.format(
            query=cleaned_query_hosts,
            show_names=", ".join(show_names),
            hosts=", ".join(hosts),
            primary_hosts=PRIMARY_HOSTS,
        )
    )
    parsed_data = json.loads(response)
    print("    Data:")
    pprint(parsed_data, indent=2)

    return parsed_data


def get_filter(query: str, show_names: list, hosts: list):
    parsed_data = parse_query(query, show_names, hosts)
    filter_dict, topics = build_filter(parsed_data, query)
    return filter_dict, topics
