from datetime import datetime
from isodate import parse_duration
from json import load
from os import path


def load_video_data(json_path):
    """Loads JSON data from a file."""

    if not path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = load(file)
        return data
    except Exception as e:
        print(f"Error loading json {json_path}: {e}")
        return None


def date_to_timestamp(data):
    """Converts YouTube API ISO datetime to Unix epoch timestamp."""
    return int(datetime.fromisoformat(data.replace("Z", "+00:00")).timestamp())


def duration_to_seconds(duration):
    """Converts YouTube API ISO duration to total seconds."""
    return int(parse_duration(duration).total_seconds())


def format_duration(seconds):
    """Formats a duration in seconds into hours, minutes, and seconds."""
    hours = int(seconds // 3600)  # Integer division for hours
    minutes = int((seconds % 3600)) // 60  # Remaining seconds for minutes
    seconds = int(seconds % 60)  # Remaining seconds

    # Build the formatted string
    duration_str = "Execution time: "
    if hours:
        duration_str += f"{hours} hour{'s' if hours > 1 else ''}, "
    if minutes:
        duration_str += f"{minutes} minute{'s' if minutes > 1 else ''}, "
    duration_str += f"{seconds} second{'s' if seconds > 1 else ''}"

    return duration_str
