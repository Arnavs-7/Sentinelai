"""Centralized logging configuration for the Sentinel AI project."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path("/tmp/logs")
_LOG_FILE = _LOG_DIR / "sentinel.log"
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def get_logger(name: str) -> logging.Logger:
    """Return a logger with console and rotating-file handlers.

    The logger emits structured records of the form
    ``timestamp | level | name | message``. A console handler and a
    rotating file handler writing to ``logs/sentinel.log`` are attached
    exactly once per logger name. The ``logs/`` directory is created if
    it does not already exist.

    Args:
        name: Name of the logger, typically ``__name__`` of the caller.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=_LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
