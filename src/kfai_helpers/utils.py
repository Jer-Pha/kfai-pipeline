from datetime import datetime
from isodate import parse_duration
from isodate.duration import Duration


def date_to_timestamp(data: str) -> int:
    """Converts YouTube API ISO datetime to Unix epoch timestamp."""
    if not data:
        return 0
    return int(datetime.fromisoformat(data.replace("Z", "+00:00")).timestamp())


def duration_to_seconds(duration: Duration) -> int:
    """Converts YouTube API ISO duration to total seconds."""
    return int(parse_duration(duration).total_seconds())


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds into hours, minutes, and seconds."""
    hours = int(seconds // 3600)  # Integer division for hours
    minutes = int((seconds % 3600)) // 60  # Remaining seconds for minutes
    seconds = seconds % 60  # Remaining seconds - keep as float

    # Build the formatted string
    duration_str = ""
    if hours > 0:
        duration_str += f"{hours} hour{'s' if hours > 1 else ''}, "
    if minutes > 0:
        duration_str += f"{minutes} minute{'s' if minutes > 1 else ''}, "
    duration_str += f"{seconds:.2f} seconds"

    return duration_str
