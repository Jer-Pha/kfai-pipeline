import sqlite3
from unittest.mock import MagicMock, call

import pytest

# The module we are testing
from kfai.extractors.utils.helpers import database as db_utils

# --- Tests for create_local_sqlite_db (the wrapper function) ---


def test_create_local_sqlite_db(mocker):
    """
    Tests that the wrapper function correctly calls the main export function
    with a properly constructed config dictionary.
    """
    # 1. Arrange
    # Mock the main export function that this wrapper calls
    mock_export = mocker.patch(
        "kfai.extractors.utils.helpers.database._export_mysql_to_sqlite"
    )
    # Mock the config constants to ensure the test is isolated
    mocker.patch(
        "kfai.extractors.utils.helpers.database.MYSQL_HOST", "testhost"
    )
    mocker.patch(
        "kfai.extractors.utils.helpers.database.MYSQL_USER", "testuser"
    )
    mocker.patch(
        "kfai.extractors.utils.helpers.database.MYSQL_PASSWORD", "testpass"
    )
    mocker.patch(
        "kfai.extractors.utils.helpers.database.MYSQL_DATABASE", "testdb"
    )

    # 2. Act
    db_utils.create_local_sqlite_db()

    # 3. Assert
    # Verify that the export function was called exactly once
    mock_export.assert_called_once()
    # Verify that it was called with the correct config dictionary
    expected_config = {
        "host": "testhost",
        "user": "testuser",
        "password": "testpass",
        "database": "testdb",
    }
    mock_export.assert_called_with(expected_config)


# --- Tests for get_video_db_data (the read function) ---


@pytest.fixture
def mock_sqlite_connect(mocker):
    """Fixture to mock the sqlite3 connection and cursor."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mocker.patch("sqlite3.connect").return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor


def test_get_video_db_data_fetches_all(mock_sqlite_connect):
    """Tests fetching all video data when no specific IDs are provided."""
    # 1. Arrange
    mock_conn, mock_cursor = mock_sqlite_connect
    # Simulate the database returning two rows
    mock_rows = [
        (1, "vid1", "Show A", "Host1,Host2"),
        (2, "vid2", "Show B", "Host3"),
    ]
    mock_cursor.fetchall.return_value = mock_rows

    # 2. Act
    video_data = db_utils.get_video_db_data()

    # 3. Assert
    # Verify the query was executed without a WHERE clause
    assert "WHERE" not in mock_cursor.execute.call_args[0][0]
    # Verify the data is correctly parsed
    assert len(video_data) == 2
    assert video_data[0]["video_id"] == "vid1"
    assert video_data[0]["hosts"] == ["Host1", "Host2"]
    assert video_data[1]["hosts"] == ["Host3"]
    mock_conn.close.assert_called_once()


def test_get_video_db_data_fetches_specific_ids(mock_sqlite_connect):
    """Tests fetching data for a specific list of video IDs."""
    # 1. Arrange
    mock_conn, mock_cursor = mock_sqlite_connect
    mock_cursor.fetchall.return_value = [(1, "vid1", "Show A", "Host1")]

    # 2. Act
    video_data = db_utils.get_video_db_data(video_ids=["vid1", "vid3"])

    # 3. Assert
    # Verify the query was executed WITH a WHERE clause and placeholders
    query_string = mock_cursor.execute.call_args[0][0]
    query_params = mock_cursor.execute.call_args[0][1]
    assert "WHERE vv.video_id IN (?,?)" in query_string
    assert query_params == ["vid1", "vid3"]
    assert len(video_data) == 1


def test_get_video_db_data_handles_null_hosts(mock_sqlite_connect):
    """Tests that a NULL value for hosts is correctly handled."""
    # 1. Arrange
    mock_conn, mock_cursor = mock_sqlite_connect
    # Simulate a row where the GROUP_CONCAT result is None
    mock_cursor.fetchall.return_value = [(1, "vid1", "Show A", None)]

    # 2. Act
    video_data = db_utils.get_video_db_data()

    # 3. Assert
    # The 'hosts' key should be an empty list, not None or an error
    assert video_data[0]["hosts"] == []


# --- Tests for _export_mysql_to_sqlite (the main ETL logic) ---


@pytest.fixture
def mock_db_connections(mocker):
    """Fixture to mock both MySQL and SQLite connections."""
    # Mock MySQL
    mock_mysql_conn = MagicMock()
    mock_mysql_cursor = MagicMock()
    mocker.patch("mysql.connector.connect").return_value = mock_mysql_conn
    mock_mysql_conn.cursor.return_value = mock_mysql_cursor
    # Simulate fetchall returning different data for each query
    mock_mysql_cursor.fetchall.side_effect = [
        [{"id": 1}],  # videos
        [{"id": 2}],  # shows
        [{"id": 3}],  # hosts
        [{"video_id": 4}],  # video_hosts
    ]

    # Mock SQLite
    mock_sqlite_conn = MagicMock()
    mock_sqlite_cursor = MagicMock()
    mocker.patch("sqlite3.connect").return_value = mock_sqlite_conn
    mock_sqlite_conn.cursor.return_value = mock_sqlite_cursor

    return (
        mock_mysql_conn,
        mock_mysql_cursor,
        mock_sqlite_conn,
        mock_sqlite_cursor,
    )


def test_export_mysql_to_sqlite_happy_path(mock_db_connections):
    """Tests the successful export from MySQL to SQLite."""
    # 1. Arrange
    mock_mysql_conn, mysql_cursor, sqlite_conn, sqlite_cursor = (
        mock_db_connections
    )

    # 2. Act
    db_utils._export_mysql_to_sqlite(
        {}
    )  # Config doesn't matter as it's mocked

    # 3. Assert
    # Verify all CREATE TABLE statements were executed
    assert sqlite_cursor.execute.call_count == 4
    # Verify all INSERT statements were executed
    assert sqlite_cursor.executemany.call_count == 4
    # Verify the correct data was passed to an insert
    sqlite_cursor.executemany.assert_any_call(
        "INSERT INTO videos_video VALUES (:id, :video_id, :show_id, :producer_id)",
        [{"id": 1}],
    )
    # Verify commit and close were called
    sqlite_conn.commit.assert_called_once()
    mock_mysql_conn.close.assert_called_once()
    sqlite_conn.close.assert_called_once()


def test_export_mysql_error(mocker):
    """Tests that an error connecting to MySQL is handled gracefully."""
    # 1. Arrange
    # Import the actual Error class to be raised
    from mysql.connector import Error as MySQLError

    # Configure the mock to raise the correct, specific exception type
    mocker.patch("mysql.connector.connect", side_effect=MySQLError)

    mock_sqlite_connect = mocker.patch("sqlite3.connect")
    mock_print = mocker.patch("builtins.print")

    # 2. Act
    db_utils._export_mysql_to_sqlite({})

    # 3. Assert
    # SQLite should never have been touched because the MySQL error happened first
    mock_sqlite_connect.assert_not_called()
    # Verify that the correct error message was printed
    mock_print.assert_any_call("Error connecting to MySQL: Unknown error")


def test_export_sqlite_error(mocker):
    """Tests that an error connecting to SQLite is handled gracefully."""
    # 1. Arrange
    mock_mysql_conn = MagicMock()
    mocker.patch("mysql.connector.connect").return_value = mock_mysql_conn
    mocker.patch("sqlite3.connect", side_effect=sqlite3.Error)

    # 2. Act
    db_utils._export_mysql_to_sqlite({})

    # 3. Assert
    # The MySQL connection should still be closed in the 'finally' block
    mock_mysql_conn.close.assert_called_once()
