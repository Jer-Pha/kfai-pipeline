from unittest.mock import MagicMock

import pytest

from kfai.extractors.utils.helpers import transcript as transcript_utils

# --- Tests for get_raw_transcript_data ---


@pytest.fixture
def mock_yt_transcript_api(mocker):
    """Fixture to mock the YouTubeTranscriptApi class and its instances."""
    mock_api_class = mocker.patch(
        "kfai.extractors.utils.helpers.transcript.YouTubeTranscriptApi"
    )
    mock_api_instance = mock_api_class.return_value
    return mock_api_instance


def test_get_raw_transcript_data_happy_path(mock_yt_transcript_api):
    """Tests the successful fetching and normalization of an English
    transcript.
    """
    # Arrange: Simulate the API returning transcript snippets
    mock_snippet = MagicMock()
    mock_snippet.text = "Hello world"
    mock_snippet.start = 1.0
    mock_snippet.duration = 2.5
    mock_yt_transcript_api.fetch.return_value = [mock_snippet]

    # Act
    result = transcript_utils.get_raw_transcript_data("vid1")

    # Assert
    assert result == [{"text": "Hello world", "start": 1.0, "duration": 2.5}]
    mock_yt_transcript_api.fetch.assert_called_once_with(
        video_id="vid1", languages=["en"]
    )


def test_get_raw_transcript_data_subtitles_disabled(mock_yt_transcript_api):
    """Tests that the video_id is returned if subtitles are disabled."""
    mock_yt_transcript_api.fetch.side_effect = Exception(
        "Subtitles are disabled for this video"
    )
    result = transcript_utils.get_raw_transcript_data("vid1")
    assert result == "vid1"


def test_get_raw_transcript_data_translation_success(mock_yt_transcript_api):
    """Tests the successful translation of a non-English transcript."""
    # Arrange: First fetch fails, then list and translate succeed
    mock_translatable_transcript = MagicMock()
    mock_translatable_transcript.is_translatable = True
    mock_translatable_transcript.language_code = "de"

    mock_translated_snippet = MagicMock(
        text="Translated text", start=5.0, duration=3.0
    )
    mock_translatable_transcript.translate.return_value.fetch.return_value = [
        mock_translated_snippet
    ]

    mock_yt_transcript_api.fetch.side_effect = Exception(
        "No transcripts were found"
    )
    mock_yt_transcript_api.list.return_value = [mock_translatable_transcript]

    # Act
    result = transcript_utils.get_raw_transcript_data("vid1")

    # Assert
    assert result == [
        {"text": "Translated text", "start": 5.0, "duration": 3.0}
    ]
    mock_yt_transcript_api.list.assert_called_once_with("vid1")
    mock_translatable_transcript.translate.assert_called_once_with("en")


def test_get_raw_transcript_data_no_translatable_found(mock_yt_transcript_api):
    """Tests the case where no translatable transcripts are found."""
    mock_non_translatable = MagicMock(is_translatable=False)
    mock_yt_transcript_api.fetch.side_effect = Exception(
        "No transcripts were found"
    )
    mock_yt_transcript_api.list.return_value = [mock_non_translatable]

    result = transcript_utils.get_raw_transcript_data("vid1")
    assert result == "vid1"


def test_get_raw_transcript_data_translation_fails(mock_yt_transcript_api):
    """Tests that an error during the translation attempt is handled."""
    mock_translatable_transcript = MagicMock(is_translatable=True)
    mock_translatable_transcript.translate.side_effect = Exception(
        "Translation API failed"
    )
    mock_yt_transcript_api.fetch.side_effect = Exception(
        "No transcripts were found"
    )
    mock_yt_transcript_api.list.return_value = [mock_translatable_transcript]

    # Act & Assert: Function shouldn't crash and should return None or video_id
    # In this case, it will fall through to the generic error after the loop
    assert transcript_utils.get_raw_transcript_data("vid1") is None


def test_get_raw_transcript_data_generic_error(mock_yt_transcript_api):
    """Tests that a generic, unhandled API error returns None."""
    mock_yt_transcript_api.fetch.side_effect = Exception("A generic API error")
    result = transcript_utils.get_raw_transcript_data("vid1")
    assert result is None


# --- Tests for chunk_transcript_with_overlap ---


def test_chunk_transcript_with_overlap():
    """Tests the core logic of chunking and timestamp re-association."""
    # Arrange: A sample transcript that will be split
    transcript_data = [
        {"text": "This is the first sentence.", "start": 0.0, "duration": 2.0},
        {
            "text": "This is the second sentence that provides overlap.",
            "start": 2.0,
            "duration": 3.0,
        },
        {
            "text": "This is the third and final sentence.",
            "start": 5.0,
            "duration": 2.5,
        },
    ]

    # Act: Use a small chunk size to force a split
    chunks = transcript_utils.chunk_transcript_with_overlap(
        transcript_data, chunk_size=50, chunk_overlap=20
    )

    # Assert
    # The number of chunks can vary, so test for a reasonable number
    assert len(chunks) > 1
    # The first chunk must start at the beginning
    assert chunks[0]["start"] == 0.0
    # The last chunk's text must contain the end of the original text
    assert "final sentence" in chunks[-1]["text"]
    # The start time of the last chunk should correspond to its content
    assert chunks[-1]["start"] >= 2.0


def test_chunk_transcript_empty_input():
    """Tests that an empty transcript results in an empty list of chunks."""
    assert transcript_utils.chunk_transcript_with_overlap([]) == []


def test_chunk_transcript_fallback_search(mocker):
    """
    Tests the fallback logic where a chunk is not found from the
    current search position.
    This is an edge case for text splitters that might modify text.
    """
    # Arrange
    transcript_data = [
        {"text": "abc", "start": 0.0},
        {"text": "def", "start": 1.0},
    ]
    # Mock split_text to return a chunk that appears earlier in the text
    mocker.patch(
        "langchain.text_splitter.RecursiveCharacterTextSplitter.split_text",
        return_value=["def ", "abc "],  # Return chunks out of order
    )

    # Act
    chunks = transcript_utils.chunk_transcript_with_overlap(transcript_data)

    # Assert
    assert len(chunks) == 2
    assert chunks[0]["start"] == 1.0  # 'def'
    assert chunks[1]["start"] == 0.0  # 'abc'


# --- Tests for transcribe_with_whisper ---


def test_transcribe_with_whisper_happy_path():
    """Tests the successful transcription and reformatting of data."""
    # Arrange
    mock_whisper_model = MagicMock()
    mock_whisper_model.transcribe.return_value = {
        "segments": [
            {"text": " Hello ", "start": 1.0, "end": 3.5},
            {"text": " world. ", "start": 4.0, "end": 5.0},
        ]
    }
    mock_audio_path = MagicMock()
    mock_audio_path.__str__.return_value = "/fake/path.m4a"

    # Act
    result = transcript_utils.transcribe_with_whisper(
        mock_audio_path, mock_whisper_model
    )

    # Assert
    assert result == [
        {"text": "Hello", "start": 1.0, "duration": 2.5},
        {"text": "world.", "start": 4.0, "duration": 1.0},
    ]
    mock_whisper_model.transcribe.assert_called_once_with(
        "/fake/path.m4a", verbose=False, language="en", fp16=False
    )


def test_transcribe_with_whisper_no_model():
    """Tests the guard clause for a missing whisper model."""
    assert transcript_utils.transcribe_with_whisper(MagicMock(), None) is None


def test_transcribe_with_whisper_transcription_error():
    """Tests that an exception during transcription returns None."""
    mock_whisper_model = MagicMock()
    mock_whisper_model.transcribe.side_effect = Exception("Whisper error")
    assert (
        transcript_utils.transcribe_with_whisper(
            MagicMock(), mock_whisper_model
        )
        is None
    )
