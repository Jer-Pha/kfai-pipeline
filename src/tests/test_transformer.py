import pytest

from kfai.transformers.utils.cleaning import clean_transcript
from kfai.transformers.utils.helpers import check_data_integrity


def test_transcript_cleaning(sample_raw_data):
    # Mock LLM for testing
    class MockLLM:
        def invoke(self, prompt):
            return "Hey what's up everybody, this is Greg Miller."

    cleaned_data = clean_transcript(sample_raw_data, "test.json", MockLLM())

    assert cleaned_data is not None
    assert cleaned_data["video_id"] == sample_raw_data["video_id"]
    assert isinstance(cleaned_data["transcript_chunks"], list)

    # Test proper capitalization and names
    cleaned_text = cleaned_data["transcript_chunks"][0]["text"]
    assert cleaned_text.startswith("Hey")
    assert "Greg Miller" in cleaned_text


def test_data_integrity(sample_raw_data):
    cleaned_data = sample_raw_data.copy()
    cleaned_data["transcript_chunks"][0][
        "text"
    ] = "Hey what's up everybody, this is Greg Miller."

    is_valid = check_data_integrity(sample_raw_data, cleaned_data, "test.json")
    assert is_valid == True
