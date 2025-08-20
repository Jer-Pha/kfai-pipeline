import json
from unittest.mock import MagicMock

from kfai.loaders import build_vector_store

# --- Test Data ---

# Create fake JSON data as strings to simulate file contents
FAKE_JSON_DATA_1 = json.dumps(
    {
        "video_id": "vid1",
        "title": "Video One",
        "show_name": "Show A",
        "hosts": ["Host A", "Host B"],
        "published_at": 1672531200,
        "transcript_chunks": [
            {"start": 10.0, "text": "New chunk 1"},
            {"start": 20.0, "text": "New chunk 2"},
        ],
    }
)

FAKE_JSON_DATA_2 = json.dumps(
    {
        "video_id": "vid2",
        "title": "Video Two",
        "show_name": "Show B",
        "hosts": ["Host C"],
        "published_at": 1675209600,
        "transcript_chunks": [
            {"start": 30.0, "text": "Existing chunk"},
            {"start": 40.0, "text": "New chunk 3"},
        ],
    }
)


# --- Test Suite ---


def test_run_happy_path(mocker):
    """
    Tests the main success path: skipping existing chunks, batching new ones,
    and inserting a final partial batch.
    """
    # 1. Arrange: Mock all external dependencies
    # Mock classes and constants
    mocker.patch("kfai.loaders.build_vector_store.HuggingFaceEmbeddings")
    mock_pgvector_class = mocker.patch(
        "kfai.loaders.build_vector_store.PGVector"
    )
    mock_pgvector_instance = mock_pgvector_class.return_value
    mocker.patch(
        "kfai.loaders.build_vector_store.BATCH_SIZE", 2
    )  # Use a small batch size for testing

    # Mock helper functions
    mock_get_processed = mocker.patch(
        "kfai.loaders.build_vector_store.get_processed_chunk_ids",
        return_value={("vid2", 30.0)},  # Simulate one chunk already in the DB
    )

    # Mock file system operations
    mock_json_dir = mocker.patch(
        "kfai.loaders.build_vector_store.JSON_SOURCE_DIR"
    )
    mock_file1 = MagicMock()
    mock_file2 = MagicMock()
    # Use mock_open to simulate reading from the mock files
    mock_file1.open = mocker.mock_open(read_data=FAKE_JSON_DATA_1)
    mock_file2.open = mocker.mock_open(read_data=FAKE_JSON_DATA_2)
    mock_json_dir.rglob.return_value = [mock_file1, mock_file2]

    # Mock print to check the final summary
    mock_print = mocker.patch("builtins.print")

    # 2. Act: Run the script
    build_vector_store.run()

    # 3. Assert
    mock_get_processed.assert_called_once()
    mock_json_dir.rglob.assert_called_with("*.json")

    # Assert that add_documents was called twice:
    # Once for the full batch, once for the final
    assert mock_pgvector_instance.add_documents.call_count == 2

    # Check the contents of the first (full) batch call
    first_call_args = mock_pgvector_instance.add_documents.call_args_list[0]
    batch1 = first_call_args.args[0]
    assert len(batch1) == 2
    assert batch1[0].page_content == "New chunk 1"
    assert batch1[1].page_content == "New chunk 2"

    # Check the contents of the second (final) batch call
    second_call_args = mock_pgvector_instance.add_documents.call_args_list[1]
    batch2 = second_call_args.args[0]
    assert len(batch2) == 1
    assert batch2[0].page_content == "New chunk 3"
    assert batch2[0].metadata["video_id"] == "vid2"

    # Check the final summary printout
    mock_print.assert_any_call("  - Added 3 new documents to the collection.")
    mock_print.assert_any_call("  - Skipped 1 documents that already existed.")


def test_run_db_insertion_fails(mocker):
    """Tests that the script handles an exception during DB insertion
    and continues.
    """
    # 1. Arrange
    mocker.patch("kfai.loaders.build_vector_store.HuggingFaceEmbeddings")
    mock_pgvector_class = mocker.patch(
        "kfai.loaders.build_vector_store.PGVector"
    )
    mock_pgvector_instance = mock_pgvector_class.return_value
    # Simulate a database error on the first attempt
    mock_pgvector_instance.add_documents.side_effect = [
        Exception("DB connection lost"),
        None,
    ]
    mocker.patch("kfai.loaders.build_vector_store.BATCH_SIZE", 1)
    mocker.patch(
        "kfai.loaders.build_vector_store.get_processed_chunk_ids",
        return_value=set(),
    )
    mock_json_dir = mocker.patch(
        "kfai.loaders.build_vector_store.JSON_SOURCE_DIR"
    )
    mock_file = MagicMock()
    mock_file.open = mocker.mock_open(read_data=FAKE_JSON_DATA_1)
    mock_json_dir.rglob.return_value = [mock_file]
    mock_print = mocker.patch("builtins.print")

    # 2. Act
    build_vector_store.run()

    # 3. Assert
    # The script should have tried to add documents twice
    assert mock_pgvector_instance.add_documents.call_count == 2
    # The error message should have been printed
    mock_print.assert_any_call(
        "  !! Failed to insert batch: DB connection lost"
    )
    # The final summary should reflect that only the second batch was added
    mock_print.assert_any_call("  - Added 1 new documents to the collection.")


def test_run_no_new_documents(mocker):
    """
    Tests the case where all documents found are already in the database.
    """
    # 1. Arrange
    mocker.patch("kfai.loaders.build_vector_store.HuggingFaceEmbeddings")
    mock_pgvector_class = mocker.patch(
        "kfai.loaders.build_vector_store.PGVector"
    )
    mock_pgvector_instance = mock_pgvector_class.return_value
    # All chunks are already processed
    mocker.patch(
        "kfai.loaders.build_vector_store.get_processed_chunk_ids",
        return_value={("vid1", 10.0), ("vid1", 20.0)},
    )
    mock_json_dir = mocker.patch(
        "kfai.loaders.build_vector_store.JSON_SOURCE_DIR"
    )
    mock_file = MagicMock()
    mock_file.open = mocker.mock_open(read_data=FAKE_JSON_DATA_1)
    mock_json_dir.rglob.return_value = [mock_file]
    mock_print = mocker.patch("builtins.print")

    # 2. Act
    build_vector_store.run()

    # 3. Assert
    # No documents should have been added to the vector store
    mock_pgvector_instance.add_documents.assert_not_called()
    mock_print.assert_any_call("  - Added 0 new documents to the collection.")
    mock_print.assert_any_call("  - Skipped 2 documents that already existed.")


def test_run_skips_file_with_no_video_id(mocker):
    """
    Tests that a JSON file missing a 'video_id' is skipped gracefully.
    """
    # 1. Arrange
    # Mock all dependencies to prevent side effects
    mocker.patch("kfai.loaders.build_vector_store.HuggingFaceEmbeddings")
    mock_pgvector_class = mocker.patch(
        "kfai.loaders.build_vector_store.PGVector"
    )
    mock_pgvector_instance = mock_pgvector_class.return_value
    mocker.patch(
        "kfai.loaders.build_vector_store.get_processed_chunk_ids",
        return_value=set(),
    )

    # Create a fake JSON string that is missing the 'video_id' key
    json_missing_id = json.dumps(
        {
            "title": "A video with no ID",
            "transcript_chunks": [{"start": 10.0, "text": "Some text"}],
        }
    )

    # Mock the file system to return only this malformed file
    mock_json_dir = mocker.patch(
        "kfai.loaders.build_vector_store.JSON_SOURCE_DIR"
    )
    mock_file = MagicMock()
    mock_file.open = mocker.mock_open(read_data=json_missing_id)
    mock_json_dir.rglob.return_value = [mock_file]
    mock_print = mocker.patch("builtins.print")

    # 2. Act
    build_vector_store.run()

    # 3. Assert
    # The main assertion is that no documents were ever added to the store,
    # because the file was skipped before any chunks were processed.
    mock_pgvector_instance.add_documents.assert_not_called()

    # Check the final summary to confirm nothing was added or skipped
    mock_print.assert_any_call("  - Added 0 new documents to the collection.")
    mock_print.assert_any_call("  - Skipped 0 documents that already existed.")
