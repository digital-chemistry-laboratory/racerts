"""Logging helpers: racerts logs through the "racerts" logger and never prints."""

import logging
import threading
from contextlib import contextmanager, nullcontext

LOGGER_NAME = "racerts"

_lock = threading.Lock()
_requests = []  # levels asked for by the show_messages blocks that are running
_saved = {}  # the logger's own level and the handler we added, while any block runs


@contextmanager
def show_messages(level: int):
    """
    Show racerts messages down to level while the block runs.

    The level of the racerts logger is only ever lowered, never raised, so DEBUG output
    asked for elsewhere stays. Without a configured handler, one that writes to stderr
    is added. Blocks can be nested and run in threads; the logger is restored when the
    last one ends.
    """
    logger = logging.getLogger(LOGGER_NAME)
    with _lock:
        if not _requests:
            _saved["level"] = logger.level
            _saved["effective"] = logger.getEffectiveLevel()
            _saved["handler"] = None
            if not logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter("%(levelname)s %(name)s: %(message)s")
                )
                logger.addHandler(handler)
                _saved["handler"] = handler
        _requests.append(level)
        _apply(logger)
    try:
        yield
    finally:
        with _lock:
            _requests.remove(level)
            _apply(logger)


def _apply(logger: logging.Logger) -> None:
    if _requests:
        wanted = min(_requests)
        logger.setLevel(wanted if wanted < _saved["effective"] else _saved["level"])
        return
    logger.setLevel(_saved["level"])
    if _saved["handler"] is not None:
        logger.removeHandler(_saved["handler"])
    _saved.clear()


def verbose_logging(verbose):
    """Show progress (INFO) while the block runs, if verbose."""
    return show_messages(logging.INFO) if verbose else nullcontext()


def cli_logging(verbosity: int):
    """Command line: warnings, INFO with -v, DEBUG with -vv, on stderr."""
    return show_messages(
        {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    )
