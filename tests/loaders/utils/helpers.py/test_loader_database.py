from unittest.mock import MagicMock

from kfai.loaders.utils.helpers import database as db_utils

# --- Tests for get_processed_chunk_ids ---


def test_get_processed_chunk_ids_happy_path(mocker):
    """Tests that the function correctly fetches and parses metadata
    from a mock DB.
    """
    # 1. Arrange
    # Mock the database rows that will be returned by the query
    mock_rows = [
        # Each row is a tuple, and the first element is the metadata dict
        ({"video_id": "v1", "start_time": 10.0},),
        ({"video_id": "v2", "start_time": 20.0},),
        # This row is missing a required key and should be ignored
        ({"video_id": "v3"},),
    ]

    # Mock the create_engine function and the connection context manager
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_create_engine = mocker.patch(
        "kfai.loaders.utils.helpers.database.create_engine",
        return_value=mock_engine,
    )
    mock_engine.connect.return_value.__enter__.return_value = mock_connection
    mock_connection.execute.return_value = mock_rows

    # 2. Act
    processed_ids = db_utils.get_processed_chunk_ids()

    # 3. Assert
    assert processed_ids == {("v1", 10.0), ("v2", 20.0)}
    mock_create_engine.assert_called_once()
    mock_connection.execute.assert_called_once()


def test_get_processed_chunk_ids_db_error(mocker):
    """
    Tests that the function handles a database connection error gracefully.
    """
    # 1. Arrange: Make the create_engine call raise an exception
    mocker.patch(
        "kfai.loaders.utils.helpers.database.create_engine",
        side_effect=Exception("Connection refused"),
    )

    # 2. Act
    processed_ids = db_utils.get_processed_chunk_ids()

    # 3. Assert: Function should catch the exception and return an empty set
    assert processed_ids == set()


# --- Tests for get_unique_metadata ---


def test_get_unique_metadata_happy_path(mocker):
    """
    Tests that unique, sorted lists of shows and hosts are returned.
    """
    # 1. Arrange
    # Simulate the data returned for the two separate queries
    mock_show_rows = [("Show B",), ("Show A",), ("Show B",)]  # Duplicates
    mock_host_rows = [("Host C",), ("Host A",)]

    # Create a mock engine and connection
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection
    # Use side_effect to return different results for each .execute() call
    mock_connection.execute.side_effect = [mock_show_rows, mock_host_rows]

    # 2. Act
    show_names, hosts = db_utils.get_unique_metadata(mock_engine)

    # 3. Assert
    # Verify that the results are deduplicated and sorted
    assert show_names == ["Show A", "Show B"]
    assert hosts == ["Host A", "Host C"]
    # Verify that two queries were executed
    assert mock_connection.execute.call_count == 2


def test_get_unique_metadata_empty_db(mocker):
    """
    Tests behavior when the database queries return no results.
    """
    # 1. Arrange
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection
    # Simulate both queries returning empty lists
    mock_connection.execute.side_effect = [[], []]

    # 2. Act
    show_names, hosts = db_utils.get_unique_metadata(mock_engine)

    # 3. Assert
    assert show_names == []
    assert hosts == []
