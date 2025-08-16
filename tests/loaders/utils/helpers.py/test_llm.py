import pytest

# The module we are testing
from kfai.loaders.utils.helpers import llm as llm_utils


# Use parametrize to efficiently test multiple scenarios with one function
@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        # Case 1: Test removal of <think> tag
        (
            "<think>I am thinking about the answer.</think>Here is the final"
            " response.",
            "Here is the final response.",
        ),
        # Case 2: Test replacement of curly single quotes
        (
            "It’s a test with ‘special’ quotes.",
            "It's a test with 'special' quotes.",
        ),
        # Case 3: Test replacement of curly double quotes
        ("He said, “Hello world”.", 'He said, "Hello world".'),
        # Case 4: Test stripping of leading/trailing whitespace
        ("   Some text with spaces.   ", "Some text with spaces."),
        # Case 5: Test a combination of all operations
        (
            "  <think>Thinking...</think>  Here’s a “test”.  ",
            'Here\'s a "test".',
        ),
        # Case 6: Test a string that is already clean
        (
            "This is a clean string with 'quotes'.",
            "This is a clean string with 'quotes'.",
        ),
        # Case 7: Test an empty string input
        ("", ""),
        # Case 8: Test a string that only contains a think block
        ("<think>This should be removed.</think>", ""),
    ],
)
def test_clean_llm_response(input_string, expected_output):
    """Tests the clean_llm_response function with various inputs to
    ensure it correctly removes thought tags, normalizes quotes, and
    strips whitespace.
    """
    assert llm_utils.clean_llm_response(input_string) == expected_output
