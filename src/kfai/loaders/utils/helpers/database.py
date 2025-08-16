from sqlalchemy import Engine, create_engine, text

from kfai.loaders.utils.config import COLLECTION_NAME, POSTGRES_DB_PATH


def get_processed_chunk_ids() -> set[tuple[str, float]]:
    """
    Gets a set of already processed chunk IDs (video_id, start_time)
    by querying the LangChain collection's metadata.
    """
    processed_ids = set()
    # This is a bit of a workaround to access the underlying connection
    # as LangChain's PGVector doesn't have a built-in "list all" method.
    try:
        with create_engine(POSTGRES_DB_PATH).connect() as connection:
            # Query the cmetadata column of the embedding table for this collection
            stmt = text(
                f"""
                SELECT cmetadata FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
                )
            """
            )
            results = connection.execute(
                stmt, {"collection_name": COLLECTION_NAME}
            )
            for row in results:
                # The metadata is the first (and only) column
                metadata = dict(row[0])
                if (
                    metadata
                    and "video_id" in metadata
                    and "start_time" in metadata
                ):
                    processed_ids.add(
                        (metadata["video_id"], metadata["start_time"])
                    )
    except Exception as e:
        print(
            f"Could not fetch processed chunks, assuming first run. Error: {e}"
        )
    return processed_ids


def get_unique_metadata(engine: Engine) -> tuple[list[str], list[str]]:
    """Queries the database to get all unique show names and hosts."""
    show_names = set()
    hosts = set()
    with engine.connect() as connection:
        # Get unique show names
        show_query = text(
            """
            SELECT DISTINCT (cmetadata ->> 'show_name') AS show_name
            FROM langchain_pg_embedding
            WHERE (cmetadata ->> 'show_name') IS NOT NULL;
        """
        )
        show_result = connection.execute(show_query)
        for row in show_result:
            if row[0]:  # row[0] is the show_name
                show_names.add(row[0])

        # Get unique hosts
        host_query = text(
            """
            SELECT host
            FROM (
                SELECT
                    TRIM(regexp_split_to_table(cmetadata ->> 'hosts', ',')) AS host,
                    (cmetadata ->> 'video_id') AS video_id
                FROM langchain_pg_embedding
            ) AS unnested_hosts
            WHERE host <> ''
            GROUP BY host
            HAVING COUNT(DISTINCT video_id) >= 5
            ORDER BY host;
        """
        )
        host_result = connection.execute(host_query)
        for row in host_result:
            if row[0]:  # row[0] is the host
                hosts.add(row[0])

    print("Host count:", len(hosts))

    return sorted(show_names), sorted(hosts)
