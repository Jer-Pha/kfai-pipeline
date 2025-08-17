# stubs/whisper/__init__.pyi
from typing import Any

# These types are used in the transcribe function signature.
# We can define them here or import them if they were in other stubs.
# For simplicity, we'll define placeholder types.
class Whisper:
    def transcribe(
        self,
        audio: str,  # Your code passes a string path
        *,
        verbose: bool | None = ...,
        language: str | None = ...,
        fp16: bool | None = ...,
        # We use `**kwargs: Any` to represent all the other possible
        # arguments that your code doesn't use.
        **kwargs: Any,
    ) -> dict[str, Any]: ...

# This is the function your code calls directly from the `whisper` module.
def load_model(name: str) -> Whisper: ...
