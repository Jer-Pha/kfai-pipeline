from datetime import datetime, timezone


def iso_string_to_epoch(date_string: str) -> int:
    """Converts an ISO 8601 formatted date string into a Unix epoch timestamp.
    Forces UTC timezone.

    Example:
    "2012-01-01T00:00:00" -> 1325376000
    """
    if not date_string:
        return 0

    dt_object = datetime.fromisoformat(date_string).replace(
        tzinfo=timezone.utc
    )

    return int(dt_object.timestamp())


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
