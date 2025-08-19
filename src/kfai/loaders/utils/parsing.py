from datetime import datetime

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from kfai.loaders.utils.config import PARSING_MODEL
from kfai.loaders.utils.constants import PRIMARY_HOST_MAP
from kfai.loaders.utils.prompts import PARSER_PROMPT
from kfai.loaders.utils.types import QueryParseResponse

PRIMARY_HOSTS = ", ".join(
    [f"'{k}' likely refers to '{v}'" for k, v in PRIMARY_HOST_MAP.items()]
)


def parse_query(
    query: str, show_names: list[str], hosts: list[str]
) -> QueryParseResponse | None:
    """
    Parses a user query in a single LLM call to extract shows, hosts, topics,
    and date information, returning a formatted topics string and a filter
    dictionary.
    """
    llm = OllamaLLM(
        model=PARSING_MODEL,
        temperature=0.1,
        top_p=0.92,
        top_k=25,
        reasoning=True,
        verbose=False,
    )

    parser = PydanticOutputParser(pydantic_object=QueryParseResponse)

    prompt = PromptTemplate(
        template=PARSER_PROMPT,
        input_variables=[
            "query",
            "show_names",
            "hosts",
            "primary_hosts",
            "year",
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    chain = prompt | llm | parser

    try:
        response: QueryParseResponse = chain.invoke(
            {
                "query": query,
                "show_names": ", ".join(show_names),
                "hosts": ", ".join(hosts),
                "primary_hosts": PRIMARY_HOSTS,
                "year": datetime.now().year,
            }
        )
        return response

    except Exception as e:
        print(f" !! Error during single-call parsing: {e}")
        return None
