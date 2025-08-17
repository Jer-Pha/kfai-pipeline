from datetime import timedelta

from isodate.duration import Duration

def parse_duration(
    datestring: str, as_timedelta_if_possible: bool = ...
) -> timedelta | Duration: ...
