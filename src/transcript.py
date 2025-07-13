from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


def get_raw_transcript_data(video_id):
    """
    Fetches the raw transcript data from the YouTube Transcript API.
    Returns a list of snippet dictionaries, each with 'text', 'start', and 'duration'.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en"]
        )
        return transcript_list
    except Exception as e:
        e = str(e)
        if (
            "Subtitles are disabled for this video" in e
            or "This video is age-restricted" in e
        ):
            return video_id
        elif (
            "No transcripts were found for any of the requested language"
            " codes: ['en']"
        ) in e:
            try:
                print(
                    "  ...Non-English subtitles found, attempting workaround."
                )
                # Get the list of all available transcripts
                transcript_list = YouTubeTranscriptApi.list_transcripts(
                    video_id
                )

                # Find a transcript that is translatable to English
                for transcript in transcript_list:
                    if transcript.is_translatable:
                        print(
                            "  -> Found a translatable transcript"
                            f" in '{transcript.language_code}'."
                            " Translating to English."
                        )

                        trans_snippets = transcript.translate("en").fetch()

                        # Convert the list of objects to a list of dictionaries
                        normalized_transcript = []
                        for snippet in trans_snippets:
                            normalized_transcript.append(
                                {
                                    "text": snippet.text,
                                    "start": snippet.start,
                                    "duration": snippet.duration,
                                }
                            )

                        print("  -> Translation and normalization successful.")
                        return normalized_transcript

                # If no translatable transcripts are found after checking
                print(
                    f"  -> No translatable transcripts found for {video_id}"
                    " - adding to skip list."
                )
                return video_id

            except Exception as e:
                print(
                    f"  !! An error occurred during translation attempt for {video_id}: {e}"
                )
        else:
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
                    "text": " ".join(chunk_text.split()),
                    "start": round(chunk_start_time, 2),
                }
            )

        # Update our search position to prevent re-finding the same text
        if chunk_start_char_index != -1:
            current_search_position = chunk_start_char_index + 1

    return final_chunks
