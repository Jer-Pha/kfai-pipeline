from typing import TypeGuard

from kfai.core.types import CompleteVideoRecord, TranscriptChunk
from kfai.transformers.utils.cleaning import clean_transcript


class MockLLM:
    """Mock LLM for testing transcript cleaning"""

    def invoke(self, prompt_list: list[dict[str, str]]) -> str:
        user_prompt = prompt_list[1]["content"]
        if "everybody this is greg miller" in user_prompt.lower():
            return "Hey what's up everybody, this is Greg Miller."
        if "andy tim were talking" in user_prompt.lower():
            return "Andy Cortez and Tim Gettys were discussing PlayStation."
        return user_prompt.capitalize() + "."


def is_valid_complete_video_record(
    data: dict,
) -> TypeGuard[CompleteVideoRecord]:
    """Validate that a dict matches CompleteVideoRecord structure"""
    required_fields = {
        "id": int,
        "video_id": str,
        "title": str,
        "show_name": str,
        "hosts": list,
        "published_at": int,
        "duration": int,
        "description": str,
        "transcript_chunks": list,
    }

    return all(
        isinstance(data.get(field), type_)
        for field, type_ in required_fields.items()
    )


def is_valid_transcript_chunk(
    data: dict,
) -> TypeGuard[TranscriptChunk]:
    """Validate that a dict matches CompleteVideoRecord structure"""
    required_fields = {"text": str, "start": float}

    return all(
        isinstance(data.get(field), type_)
        for field, type_ in required_fields.items()
    )


def test_transcript_cleaning_basic(
    sample_raw_data: CompleteVideoRecord,
) -> None:
    """Test basic transcript cleaning functionality and type safety"""
    cleaned_data = clean_transcript(sample_raw_data, "test.json", MockLLM())

    # Validate data and types
    assert cleaned_data is not None
    assert is_valid_complete_video_record(cleaned_data)
    assert all(
        is_valid_transcript_chunk(chunk)
        for chunk in cleaned_data["transcript_chunks"]
    )

    # Test field values are preserved
    for field in CompleteVideoRecord.__annotations__:
        if field != "transcript_chunks":
            assert field in cleaned_data
            assert cleaned_data[field] == sample_raw_data[field]
            assert isinstance(
                cleaned_data[field], type(sample_raw_data[field])
            )
            if field == "hosts":
                assert all(
                    isinstance(host, str) for host in cleaned_data[field]
                )
        else:
            assert isinstance(cleaned_data["transcript_chunks"], list)

    # Test cleaning functionality
    cleaned_text = cleaned_data["transcript_chunks"][0]["text"]
    assert cleaned_text.startswith("Hey")
    assert "Greg Miller" in cleaned_text


def test_transcript_cleaning_multiple_chunks(
    sample_raw_data: CompleteVideoRecord,
) -> None:
    """Test cleaning with multiple transcript chunks"""
    # Add a second chunk
    sample_raw_data["transcript_chunks"].append(
        TranscriptChunk(
            text="andy tim were talking about playstation",
            start=2.5,
        )
    )
    raw_data_length = len(sample_raw_data["transcript_chunks"])

    # Clean and validate
    cleaned_data = clean_transcript(sample_raw_data, "test.json", MockLLM())
    assert cleaned_data is not None
    assert is_valid_complete_video_record(cleaned_data)
    assert len(cleaned_data["transcript_chunks"]) == raw_data_length

    # Check cleaning of second chunk
    print(cleaned_data["transcript_chunks"])
    second_chunk = cleaned_data["transcript_chunks"][1]
    assert is_valid_transcript_chunk(second_chunk)
    assert "Andy Cortez" in second_chunk["text"]
    assert "Tim Gettys" in second_chunk["text"]
