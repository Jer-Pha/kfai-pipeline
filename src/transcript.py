from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


def get_raw_transcript_data(video_id):
    """
    Fetches the raw transcript data from the YouTube Transcript API.
    Returns a list of snippet dictionaries, each with 'text', 'start', and 'duration'.
    """
    try:
        # fetch() is deprecated, get_transcript() is the current method
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en"]
        )
        return transcript_list
    except Exception as e:
        print(f"Could not retrieve transcript for {video_id}: {e}")
        return None


def chunk_transcript_with_overlap(
    transcript_data, chunk_size=1000, chunk_overlap=200
):
    """
    Chunks a transcript into overlapping, semantically aware pieces,
    while preserving the start timestamp for each chunk.

    Args:
        transcript_data: The raw list of dicts from youtube_transcript_api.
        chunk_size: The target size of each text chunk (in characters).
        chunk_overlap: The amount of overlap between chunks (in characters).

    Returns:
        A list of dictionaries, where each dict is a chunk with 'text', 'start', and 'end'.
    """
    if not transcript_data:
        return []

    # 1. Combine the transcript into a single text block and create a time map.
    full_text = ""
    # This list will store tuples of (character_index, timestamp)
    char_to_time_map = []

    for snippet in transcript_data:
        start_time = snippet["start"]
        text = snippet["text"].strip() + " "  # Add space for joining

        # Store the start time for the beginning of this snippet's text
        char_to_time_map.append((len(full_text), start_time))
        full_text += text

    # 2. Use a robust text splitter to create overlapping chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(full_text)

    # 3. Re-associate each chunk with its original start time.
    final_chunks = []
    current_search_position = 0  # <-- Add this variable

    for chunk_text in text_chunks:
        # Find the start index of the chunk, starting from our last position
        chunk_start_char_index = full_text.find(
            chunk_text, current_search_position
        )

        if chunk_start_char_index == -1:
            # This should rarely happen, but as a fallback, search from the beginning
            chunk_start_char_index = full_text.find(chunk_text)

        # Find the closest timestamp in our map
        chunk_start_time = None
        for char_index, timestamp in char_to_time_map:
            if char_index <= chunk_start_char_index:
                chunk_start_time = timestamp
            else:
                break

        if chunk_start_time is not None:
            final_chunks.append(
                {
                    "text": chunk_text,
                    "start": round(chunk_start_time, 2),
                }
            )

        # Update our search position to prevent re-finding the same text
        if chunk_start_char_index != -1:
            current_search_position = chunk_start_char_index + 1

    return final_chunks
