from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

from kfai.loaders.utils.helpers.datetime import iso_string_to_epoch
from kfai.loaders.utils.types import QueryParseResponse

if TYPE_CHECKING:
    from kfai.loaders.utils.types import (
        PGVectorHosts,
        PGVectorPublishedAt,
        PGVectorShowName,
        PGVectorText,
    )


def build_filter(
    parsed_response: QueryParseResponse,
) -> (
    dict[
        str,
        list[
            PGVectorShowName
            | PGVectorHosts
            | PGVectorPublishedAt
            | PGVectorText
        ],
    ]
    | None
):
    """Convert to filter for PGVector retriever"""
    print("  -> Building filter...")
    filter_conditions: list[
        PGVectorShowName | PGVectorHosts | PGVectorPublishedAt | PGVectorText
    ] = []

    if parsed_response.exact_year:
        year = parsed_response.exact_year
        filter_conditions.append(
            {
                "published_at": {
                    "$gte": iso_string_to_epoch(f"{year}-01-01T00:00:00")
                }
            }
        )
        filter_conditions.append(
            {
                "published_at": {
                    "$lte": iso_string_to_epoch(f"{year}-12-31T23:59:59")
                }
            }
        )
    elif parsed_response.year_range:
        _range = parsed_response.year_range.split("-")
        start, end = _range[0], _range[1]
        filter_conditions.append(
            {
                "published_at": {
                    "$gte": iso_string_to_epoch(f"{start}-01-01T00:00:00")
                }
            }
        )
        filter_conditions.append(
            {
                "published_at": {
                    "$lte": iso_string_to_epoch(f"{end}-12-31T23:59:59")
                }
            }
        )
    elif parsed_response.before_year:
        year = str(int(parsed_response.before_year) - 1)
        filter_conditions.append({"published_at": {"$gte": 1325376000}})
        filter_conditions.append(
            {
                "published_at": {
                    "$lte": iso_string_to_epoch(f"{year}-12-31T23:59:59")
                }
            }
        )
    elif parsed_response.after_year:
        year = str(int(parsed_response.after_year) + 1)
        filter_conditions.append(
            {
                "published_at": {
                    "$gte": iso_string_to_epoch(f"{year}-01-01T00:00:00")
                }
            }
        )
        filter_conditions.append(
            {
                "published_at": {
                    "$lte": iso_string_to_epoch(
                        f"{datetime.now().year}-12-31T23:59:59"
                    )
                }
            }
        )

    shows_list = parsed_response.shows
    hosts_list = parsed_response.hosts

    if shows_list:
        show_filter: PGVectorShowName = {"show_name": {"$in": shows_list}}
        filter_conditions.append(show_filter)

    for host in hosts_list:
        host = re.sub(r"([%_])", r"\\\1", host)
        host_filter: PGVectorHosts = {"hosts": {"$like": f"%{host}%"}}
        filter_conditions.append(host_filter)

    if filter_conditions:
        filter_dict = {"$and": filter_conditions}
        print("    Final filter:\n", filter_dict)
        return filter_dict

    print("  No filter parsed...")
    return None
