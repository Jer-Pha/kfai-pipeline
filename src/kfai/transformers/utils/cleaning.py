import json
from pathlib import Path
from traceback import format_exc

from langchain_ollama import OllamaLLM
from tqdm import tqdm

from kfai.core.paths import LOGS_DIR
from kfai.core.types import CompleteVideoRecord, TranscriptChunk
from kfai.transformers.utils.helpers import clean_response, clean_text_chunk
from kfai.transformers.utils.logger_config import setup_logging
from kfai.transformers.utils.prompts import SYSTEM_PROMPT, USER_PROMPT

logger = setup_logging()


def clean_transcript(
    video_data: CompleteVideoRecord, relative_path: Path, llm: OllamaLLM
) -> CompleteVideoRecord | None:
    """Cleans a video's transcript with Ollama, one chunk at a time."""
    try:
        assert video_data["transcript_chunks"] is not None
        transcript_chunks: list[TranscriptChunk] = video_data[
            "transcript_chunks"
        ]
        chunk_count = len(transcript_chunks)
        cleaned_video_data: CompleteVideoRecord = {
            "id": video_data["id"],
            "video_id": video_data["video_id"],
            "show_name": video_data["show_name"],
            "hosts": video_data["hosts"],
            "title": video_data["title"],
            "description": video_data["description"],
            "published_at": video_data["published_at"],
            "duration": video_data["duration"],
            "transcript_chunks": [],
        }
        assert cleaned_video_data["transcript_chunks"] is not None
        metadata = (
            json.dumps(cleaned_video_data)
            .replace("{", "{{")
            .replace("}", "}}")
        )  # Escape brackets for `user_prompt_template.format(chunk=text)`

        _invoke_llm = llm.invoke
        _clean = clean_response

        progress_bar = tqdm(
            total=chunk_count,
            unit="chunk",
        )

        user_prompt_template = USER_PROMPT.format(
            metadata=metadata, chunk="{chunk}"
        )

        for chunk in transcript_chunks:
            text = clean_text_chunk(chunk["text"])
            user_prompt = user_prompt_template.format(chunk=text)

            try:
                response = _invoke_llm(
                    [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": user_prompt},
                    ]
                )

                response = _clean(response)
                cleaned_chunk: TranscriptChunk = {
                    "text": response.strip(),
                    "start": chunk["start"],
                }
                cleaned_video_data["transcript_chunks"].append(cleaned_chunk)
                progress_bar.update(1)
            except:
                logger.error(
                    f"LLM call failed on chunk in {relative_path} starting "
                    f"at {chunk['start']}s."
                )
                logger.error(format_exc())
                print(
                    f"  !! LLM call failed. See {LOGS_DIR} for details."
                    " Skipping video."
                )
                progress_bar.close()
                return None

        progress_bar.close()

        return cleaned_video_data
    except:
        logger.error(
            "An unexpected error occurred in clean_transcript()"
            f" for {relative_path}."
        )
        logger.error(format_exc())
        print(
            f"  !! An unexpected error occurred. See {LOGS_DIR} for"
            " details. Skipping video."
        )
        return None
