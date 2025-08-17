from typing import Any

import numpy as np
import torch
from whisper.model import Whisper

# This is the function signature for the top-level `transcribe` function.
# We need to define it because the Whisper class uses it.
def transcribe(
    model: Whisper,
    audio: str | np.ndarray | torch.Tensor,
    *,
    verbose: bool | None = ...,
    temperature: float | tuple[float, ...] = ...,
    compression_ratio_threshold: float | None = ...,
    logprob_threshold: float | None = ...,
    no_speech_threshold: float | None = ...,
    condition_on_previous_text: bool = ...,
    initial_prompt: str | None = ...,
    carry_initial_prompt: bool = ...,
    word_timestamps: bool = ...,
    prepend_punctuations: str = ...,
    append_punctuations: str = ...,
    clip_timestamps: str | list[float] = ...,
    hallucination_silence_threshold: float | None = ...,
    **decode_options: Any,
) -> dict[str, Any]: ...
