from collections.abc import Iterable
from typing import Any

class download_range_func:
    def __init__(
        self,
        chapters: Iterable[str] | None,
        ranges: Iterable[tuple[float, float]] | None,
        from_info: bool = ...,
    ) -> None: ...
    def __call__(
        self, info_dict: dict[str, Any], ydl: Any
    ) -> Iterable[dict[str, Any]]: ...
