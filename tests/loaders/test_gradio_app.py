from unittest.mock import MagicMock

import pytest

# The module we are testing
from kfai.loaders import gradio_app

# --- Comprehensive Fixture for Mocking Dependencies ---


@pytest.fixture
def mock_deps(mocker):
    """A single fixture to mock all external dependencies of the run script."""
    # Mock the classes that are instantiated
    mock_ollama = mocker.patch("kfai.loaders.gradio_app.OllamaLLM")
    mock_query_agent_class = mocker.patch("kfai.loaders.gradio_app.QueryAgent")
    mock_chat_interface_class = mocker.patch(
        "kfai.loaders.gradio_app.gr.ChatInterface"
    )

    # Mock the instances that are created
    mock_query_agent_instance = mock_query_agent_class.return_value
    mock_chat_interface_instance = mock_chat_interface_class.return_value

    # --- CRITICAL: Mock the .launch() method to prevent the server from starting ---
    mocker.patch.object(mock_chat_interface_instance, "launch")

    return {
        "OllamaLLM": mock_ollama,
        "QueryAgent": mock_query_agent_class,
        "ChatInterface": mock_chat_interface_class,
        "query_agent_instance": mock_query_agent_instance,
        "chat_interface_instance": mock_chat_interface_instance,
    }


# --- Test Suite ---


def test_run_initializes_and_launches_correctly(mock_deps):
    """
    Tests that the run function correctly initializes all components and
    configures the Gradio ChatInterface with the right parameters.
    """
    # 1. Act
    gradio_app.run()

    # 2. Assert
    # Verify that the LLM and Agent were initialized
    mock_deps["OllamaLLM"].assert_called_once()
    mock_deps["QueryAgent"].assert_called_once()

    # Verify that the ChatInterface was initialized
    mock_deps["ChatInterface"].assert_called_once()

    # Check the keyword arguments passed to ChatInterface
    chat_interface_kwargs = mock_deps["ChatInterface"].call_args.kwargs
    assert chat_interface_kwargs["title"] == "KF/AI"
    assert "fn" in chat_interface_kwargs  # Check that the function was passed

    # Verify that the launch method was called, preventing the server from starting
    mock_deps["chat_interface_instance"].launch.assert_called_once_with(
        share=False
    )


def test_chat_with_agent_bridge_function(mock_deps):
    """
    Tests the internal 'chat_with_agent' function to ensure it correctly
    calls the query agent.
    """
    # 1. Arrange
    # Run the main function to define the inner chat_with_agent function
    gradio_app.run()

    # Extract the actual function that was passed to the ChatInterface mock
    chat_function = mock_deps["ChatInterface"].call_args.kwargs["fn"]

    # Configure the mock agent to return a specific response
    mock_deps["query_agent_instance"].process_query.return_value = (
        "Test response from agent"
    )

    # 2. Act
    # Call the extracted function as Gradio would
    response = chat_function("Test message", history=[])

    # 3. Assert
    # Verify the agent's process_query method was called correctly
    mock_deps["query_agent_instance"].process_query.assert_called_once_with(
        "Test message", True
    )

    # Verify the function returns the agent's response
    assert response == "Test response from agent"


def test_chat_with_agent_handles_none_response(mock_deps):
    """
    Tests that the chat_with_agent function raises an AssertionError if the
    agent returns None, as per the code's logic.
    """
    # 1. Arrange
    gradio_app.run()
    chat_function = mock_deps["ChatInterface"].call_args.kwargs["fn"]
    # Configure the mock agent to return None
    mock_deps["query_agent_instance"].process_query.return_value = None

    # 2. Act & Assert
    # Use pytest.raises to confirm that an AssertionError is thrown
    with pytest.raises(AssertionError):
        chat_function("A message that causes a None response", history=[])
