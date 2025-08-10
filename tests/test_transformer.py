import pytest

from kfai.transformers.utils.cleaning import clean_transcript
from kfai.transformers.utils.helpers import check_data_integrity


class MockLLM:
    def invoke(self, prompt: str) -> str:
        if "greg miller" in prompt.lower():
            return "Hey what's up everybody, this is Greg Miller."
        if "andy tim" in prompt.lower():
            return "Andy Cortez and Tim Gettys were discussing PlayStation."
        return prompt.capitalize() + "."


def test_transcript_cleaning_basic(sample_raw_data):
    """Test basic transcript cleaning functionality"""
    cleaned_data = clean_transcript(sample_raw_data, "test.json", MockLLM())

    assert cleaned_data is not None
    assert cleaned_data["id"] == 1  # Check integer ID
    assert cleaned_data["video_id"] == "test123"  # Check string video_id
    assert isinstance(cleaned_data["transcript_chunks"], list)

    # Test proper capitalization and names
    cleaned_text = cleaned_data["transcript_chunks"][0]["text"]
    assert cleaned_text.startswith("Hey")
    assert "Greg Miller" in cleaned_text


def test_transcript_cleaning_multiple_chunks(sample_raw_data):
    """Test cleaning with multiple transcript chunks"""
    sample_raw_data["transcript_chunks"].append(
        {"text": "andy tim were talking about playstation", "start": 2.5}
    )

    cleaned_data = clean_transcript(sample_raw_data, "test.json", MockLLM())
    assert cleaned_data is not None
    assert len(cleaned_data["transcript_chunks"]) == 2
    assert "Andy Cortez" in cleaned_data["transcript_chunks"][1]["text"]


@pytest.mark.parametrize("field", ["id", "video_id", "title", "show_name"])
def test_required_fields_present(sample_raw_data, field):
    """Test that required fields are preserved during cleaning"""
    cleaned_data = clean_transcript(sample_raw_data, "test.json", MockLLM())
    assert cleaned_data is not None
    assert field in cleaned_data
    assert cleaned_data[field] == sample_raw_data[field]

    # Check field types
    if field == "id":
        assert isinstance(cleaned_data["id"], int)
    elif field == "video_id":
        assert isinstance(cleaned_data["video_id"], str)
