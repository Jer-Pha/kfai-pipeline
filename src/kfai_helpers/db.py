import mysql.connector
import sqlite3

from .config import DB_DATABASE, DB_HOST, DB_PASSWORD, DB_USER, SQLITE_DB_PATH


def _export_mysql_to_sqlite(mysql_config, sqlite_db_path):
    """Exports relevant data from a MySQL database to an SQLite database."""
    mysql_conn, sqlite_conn = None, None
    try:
        # Connect to MySQL
        mysql_conn = mysql.connector.connect(**mysql_config)
        mysql_cursor = mysql_conn.cursor(dictionary=True)

        # Connect to SQLite
        sqlite_conn = sqlite3.connect(sqlite_db_path)
        sqlite_cursor = sqlite_conn.cursor()

        # Create tables in SQLite if they don't exist
        sqlite_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS videos_video (
                id INTEGER PRIMARY KEY,
                video_id TEXT,
                show_id INTEGER,
                producer_id INTEGER
            )
        """
        )

        sqlite_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS shows_show (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """
        )

        sqlite_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS hosts_host (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """
        )

        sqlite_cursor.execute(
            """
          CREATE TABLE IF NOT EXISTS videos_video_hosts(
            video_id INTEGER,
            host_id INTEGER
          )
        """
        )

        # Fetch from MySQL and insert to SQLite (videos_video)
        mysql_cursor.execute(
            "SELECT id, video_id, show_id, producer_id FROM videos_video WHERE channel_id < 3"
        )
        videos = mysql_cursor.fetchall()
        sqlite_cursor.executemany(
            "INSERT INTO videos_video VALUES (:id, :video_id, :show_id, :producer_id)",
            videos,
        )

        # Fetch from MySQL and insert to SQLite (shows_show)
        mysql_cursor.execute("SELECT id, name FROM shows_show")
        shows = mysql_cursor.fetchall()
        sqlite_cursor.executemany(
            "INSERT INTO shows_show VALUES (:id, :name)", shows
        )

        # Fetch from MySQL and insert to SQLite (hosts_host)
        mysql_cursor.execute("SELECT id, name FROM hosts_host")
        hosts = mysql_cursor.fetchall()
        sqlite_cursor.executemany(
            "INSERT INTO hosts_host VALUES (:id, :name)", hosts
        )

        # Fetch from MySQL and insert to SQLite (videos_video_hosts)
        mysql_cursor.execute(
            "SELECT video_id, host_id FROM videos_video_hosts"
        )
        video_hosts = mysql_cursor.fetchall()
        sqlite_cursor.executemany(
            "INSERT INTO videos_video_hosts VALUES (:video_id, :host_id)",
            video_hosts,
        )

        # Commit changes
        sqlite_conn.commit()
        print("Data exported from MySQL to SQLite successfully.")

    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
    except sqlite3.Error as err:
        print(f"Error connecting to SQLite: {err}")
    finally:
        # Close connections
        if mysql_conn and mysql_conn.is_connected():
            mysql_conn.close()
        if sqlite_conn:
            sqlite_conn.close()


def create_local_sqlite_db():
    # MySQL config
    mysql_config = {
        "host": DB_HOST,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "database": DB_DATABASE,
    }

    _export_mysql_to_sqlite(mysql_config, SQLITE_DB_PATH)


def get_video_db_data(sqlite_db, video_ids=None):
    """
    Fetches video metadata from the SQLite database.
    If video_ids is provided, fetches data only for those IDs.
    Otherwise, fetches all video data.
    """

    # Get database data from local database
    conn = sqlite3.connect(sqlite_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT
            vv.id, vv.video_id, ss.name AS show_name,
            GROUP_CONCAT(hh.name) AS hosts
        FROM videos_video vv
        JOIN shows_show ss ON vv.show_id = ss.id
        LEFT JOIN videos_video_hosts vvh ON vv.id = vvh.video_id
        LEFT JOIN hosts_host hh ON vvh.host_id = hh.id
    """

    # If specific IDs are requested, add a WHERE clause
    if video_ids:
        # Use placeholders to prevent SQL injection
        placeholders = ",".join("?" for _ in video_ids)
        query += f" WHERE vv.video_id IN ({placeholders})"
        query += " GROUP BY vv.id"
        cursor.execute(query, video_ids)
    else:
        query += " GROUP BY vv.id"
        cursor.execute(query)

    rows = cursor.fetchall()
    conn.close()

    # Return raw database data
    video_data = []
    for row in rows:
        video_data.append(
            {
                "id": row[0],
                "video_id": row[1],
                "show_name": row[2],
                "hosts": (row[3].split(",") if row[3] else []),
            }
        )

    return video_data
