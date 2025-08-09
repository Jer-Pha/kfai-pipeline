import re


def clean_llm_response(response: str) -> str:
    """Clean common LLM inconsistencies in the response."""
    response = response.split("</think>")[-1]
    response = re.sub(r"[‘’]", "'", response)
    response = re.sub(r"[“”]", '"', response)
    return response.strip()
