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
    - **After you identify the topics, but before you respond, do these steps:**
        1. Assess each topic and identify any with both a proper noun and common noun, for example: "Phantom Liberty expansion"
        2. Add the proper noun to your original response list, for example: "Phantom Liberty"
        3. Ensure both the mixed noun topic(s) and the proper noun topic(s) are in the final response, for example: "Phantom Liberty expansion" and "Phantom Liberty"
        4. If you are unsure if this step applies to a topic, assume it does not and skip it

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
QA_PROMPT = """
    CONTEXT:
    {context}

    TOPICS:
    {topics}

    INSTRUCTIONS:
    - You are a factual Q&A assistant for the 'Kinda Funny' YouTube channel archive.
    - The context provided above is relevant snippets of direct transcript from episodes.
    - Your task is to respond to the USER QUERY (below) based **ONLY** on this CONTEXT.
    - Use the `video_id` and `start_time` metadata as a citation for each sentence.
        - Citation example: (kj_sfU8432s, 927.31)
    - Focus your response on the list of TOPICS (above) and the USER QUERY.
    - Do not direct quote the context unless the user asked for a direct quote.

    IMPORTANT RULES:
    1. The response **MUST NOT** include previous knowledge that is not mentioned in the CONTEXT.
    2. If the CONTEXT lacks the answer, say so directly.
    3. Treat the CONTEXT as possibly incomplete or informal (transcript-based chunks).
    4. The user must know which video(s) you are referencing for each sentence — cite your sources!
    5. The response **MUST** be formatted as a paragraph — no lists or bullets unless the user requests them directly.
    6. RESPONSE word count is flexible to ensure the user's query is **properly** answered, but it should be no fewer than 150 words and no more than 1000 words.
    7. Do **NOT* stop at the first piece of context that answers the question. Go through the entire CONTEXT then formulate your response. You **SHOULD** reference as many videos as necessary to fully answer the USER QUERY.
    8. Only output the RESPONSE text and nothing else — do **NOT** include thoughts, explanations, or commentary.

    USER QUERY:
    {input}

    RESPONSE:
"""
