import pytest

# The module we are testing
from kfai.loaders import interactive_qa

# --- Fixture for Mocking Dependencies ---


@pytest.fixture
def mock_deps(mocker):
    """A fixture to mock the dependencies of the run script."""
    # Mock the classes that are instantiated
    mocker.patch("kfai.loaders.interactive_qa.OllamaLLM")
    mock_query_agent_class = mocker.patch(
        "kfai.loaders.interactive_qa.QueryAgent"
    )

    # Get a handle to the instance that will be created
    mock_query_agent_instance = mock_query_agent_class.return_value

    return {
        "QueryAgent": mock_query_agent_class,
        "query_agent_instance": mock_query_agent_instance,
    }


# --- Test Suite ---


def test_run_happy_path(mocker, capsys, mock_deps):
    """
    Tests the main success path: a user asks a question, then exits.
    """
    # 1. Arrange: Simulate user typing a question, then "exit"
    mocker.patch(
        "builtins.input", side_effect=["What is Kinda Funny?", "exit"]
    )

    # 2. Act
    interactive_qa.run()

    # 3. Assert
    # Verify the agent was initialized
    mock_deps["QueryAgent"].assert_called_once()

    # Verify the agent's process_query method was called with the correct query
    mock_deps["query_agent_instance"].process_query.assert_called_once_with(
        "What is Kinda Funny?"
    )

    # Verify the user prompts were displayed
    captured = capsys.readouterr()
    assert "--- Ask a question, or type 'exit' to quit. ---" in captured.out
    assert "Exiting..." in captured.out


def test_run_skips_empty_and_whitespace_input(mocker, mock_deps):
    """
    Tests that the script ignores empty or whitespace-only input.
    """
    # 1. Arrange: Simulate user pressing enter, then spaces, then "exit"
    mocker.patch("builtins.input", side_effect=["", "   ", "exit"])

    # 2. Act
    interactive_qa.run()

    # 3. Assert
    # The key assertion is that process_query was never called
    mock_deps["query_agent_instance"].process_query.assert_not_called()


def test_run_handles_input_exception(mocker, capsys, mock_deps):
    """
    Tests that the script gracefully exits if input() raises an exception.
    """
    # 1. Arrange: Simulate an EOFError from the input function
    mocker.patch("builtins.input", side_effect=EOFError("Simulated EOF"))

    # 2. Act
    interactive_qa.run()

    # 3. Assert
    # Verify an error message was printed
    captured = capsys.readouterr()
    assert "Exiting due to unknown error" in captured.out
    assert "Simulated EOF" in captured.out

    # Verify that the agent was not called
    mock_deps["query_agent_instance"].process_query.assert_not_called()


def test_run_exits_immediately(mocker, mock_deps):
    """
    Tests that the script exits immediately if the first input is 'exit'.
    """
    # 1. Arrange: Simulate user typing "exit" right away
    mocker.patch("builtins.input", return_value="exit")

    # 2. Act
    interactive_qa.run()

    # 3. Assert
    # Verify the agent was not called
    mock_deps["query_agent_instance"].process_query.assert_not_called()
