from unittest.mock import MagicMock

import pytest

from kfai.loaders.utils import parsing as parsing_utils
from kfai.loaders.utils.types import QueryParseResponse


# --- Fixture for Mocking Dependencies ---
@pytest.fixture
def mock_deps(mocker):
    """Mocks the LangChain components used in parse_query."""
    # Mock the classes
    mocker.patch("kfai.loaders.utils.parsing.OllamaLLM")
    mock_parser_class = mocker.patch(
        "kfai.loaders.utils.parsing.PydanticOutputParser"
    )
    mock_prompt_template = mocker.patch(
        "kfai.loaders.utils.parsing.PromptTemplate"
    )

    # Mock the instances and the final chain
    mock_parser_instance = mock_parser_class.return_value
    mock_chain = MagicMock()
    # Simulate the `prompt | llm | parser` chain construction
    mock_prompt_template.return_value.__or__.return_value.__or__.return_value = mock_chain  # noqa: E501

    return {
        "chain": mock_chain,
        "parser": mock_parser_instance,
    }


# --- Test Suite ---
def test_parse_query_happy_path(mock_deps):
    """Tests that parse_query successfully invokes the chain and
    returns a Pydantic object.
    """
    # Arrange
    # Simulate the chain returning a valid Pydantic object
    expected_response = QueryParseResponse(
        shows=["Show A"], topics=["Topic B"]
    )
    mock_deps["chain"].invoke.return_value = expected_response

    # Act
    result = parsing_utils.parse_query("query", ["Show A"], ["Host C"])

    # Assert
    assert result == expected_response
    mock_deps["chain"].invoke.assert_called_once()
    # Check that the invoke dictionary contains the correct data
    invoke_args = mock_deps["chain"].invoke.call_args[0][0]
    assert invoke_args["query"] == "query"
    assert "Show A" in invoke_args["show_names"]


def test_parse_query_handles_exception(mock_deps):
    """Tests that the function returns None if the LangChain chain fails."""
    # Arrange
    mock_deps["chain"].invoke.side_effect = Exception("LLM call failed")

    # Act
    result = parsing_utils.parse_query("query", [], [])

    # Assert
    assert result is None
