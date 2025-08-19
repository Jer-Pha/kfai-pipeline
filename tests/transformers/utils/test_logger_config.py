import logging

from kfai.transformers.utils import logger_config


def test_setup_logging_adds_handlers_only_once(mocker):
    """
    Tests that setup_logging correctly adds handlers to a clean root logger
    and does NOT add them again if they already exist.
    """
    # 1. Arrange: Mock external dependencies and control global logger state
    # Mock the FileHandler to prevent actual file I/O
    mock_file_handler_class = mocker.patch("logging.FileHandler")
    # Mock the StreamHandler for consistency
    mock_stream_handler_class = mocker.patch("logging.StreamHandler")
    # Mock the LOG_FILE path to isolate the test
    mocker.patch(
        "kfai.transformers.utils.logger_config.LOG_FILE", "fake/path/test.log"
    )

    # --- Critical Setup for Testing Global Logger ---
    # Get the root logger, which is a global singleton
    root_logger = logging.getLogger()
    # Store its original handlers to be restored later
    original_handlers = root_logger.handlers.copy()
    # Manually remove all handlers to ensure a clean state for the first call
    for handler in original_handlers:
        root_logger.removeHandler(handler)

    try:
        # 2. Act (First Call): Run the function on a clean logger
        logger1 = logger_config.setup_logging()

        # 3. Assert (First Call)
        # The function should have added exactly two handlers
        assert len(root_logger.handlers) == 2
        assert logger1 is root_logger  # Verify it returns the root logger

        # Verify the handlers were instantiated correctly
        mock_file_handler_class.assert_called_once_with(
            "fake/path/test.log", mode="a", encoding="utf-8"
        )
        mock_stream_handler_class.assert_called_once_with()

        # 4. Act (Second Call): Run function again on the now-configured logger
        logger2 = logger_config.setup_logging()

        # 5. Assert (Second Call)
        # The number of handlers should NOT have changed. This proves the
        # `if not logger.hasHandlers():` block was correctly skipped.
        assert len(root_logger.handlers) == 2
        assert logger2 is root_logger

    finally:
        # --- Restore the global logger state ---
        # Remove the handlers the test added
        for handler in root_logger.handlers.copy():
            root_logger.removeHandler(handler)
        # Add the original handlers back for other tests
        for handler in original_handlers:
            root_logger.addHandler(handler)
