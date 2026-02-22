"""
Logging Configuration — structured logging for MangoMAS Demo.

Provides a centralized, configurable logging setup using Python's standard
``logging`` module. All configuration is driven by environment variables
with sensible defaults — no hard-coded values.

Environment Variables:
    MANGOMAS_LOG_LEVEL: Root log level (default: ``INFO``).
    MANGOMAS_LOG_FORMAT: Output format — ``text`` (default) or ``json``.
"""

from __future__ import annotations

import json as _json
import logging
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Configuration from environment (no hard-coded values)
# ---------------------------------------------------------------------------
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FORMAT = "text"

_ENV_LOG_LEVEL = "MANGOMAS_LOG_LEVEL"
_ENV_LOG_FORMAT = "MANGOMAS_LOG_FORMAT"

_LOGGER_NAME = "mangomas_demo"

_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_TEXT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, _TEXT_DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        # Propagate extra structured fields if present
        for key in ("component", "elapsed_ms", "cell_type", "strategy"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        return _json.dumps(entry)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``mangomas_demo`` namespace.

    Args:
        name: Optional dotted name appended to ``mangomas_demo``.
              If *None*, returns the root package logger.

    Returns:
        A configured :class:`logging.Logger`.
    """
    full_name = f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME
    return logging.getLogger(full_name)


def configure_logging() -> logging.Logger:
    """Configure the root ``mangomas_demo`` logger from environment variables.

    Safe to call multiple times — will not duplicate handlers.

    Returns:
        The configured root :class:`logging.Logger` for the package.
    """
    logger = logging.getLogger(_LOGGER_NAME)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    level_name = os.environ.get(_ENV_LOG_LEVEL, _DEFAULT_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    fmt_choice = os.environ.get(_ENV_LOG_FORMAT, _DEFAULT_LOG_FORMAT).lower()
    if fmt_choice == "json":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT, _TEXT_DATE_FORMAT))

    logger.addHandler(handler)
    return logger
