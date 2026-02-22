"""
Unit Tests — Structured Logging Infrastructure.
"""

from __future__ import annotations

import json
import logging
import os
from unittest import mock

import pytest

from mangomas_demo.logging_config import (
    _ENV_LOG_FORMAT,
    _ENV_LOG_LEVEL,
    _LOGGER_NAME,
    _JSONFormatter,
    configure_logging,
    get_logger,
)


class TestGetLogger:
    """Tests for the get_logger helper."""

    def test_root_logger_name(self) -> None:
        """get_logger() with no args should return the root package logger."""
        logger = get_logger()
        assert logger.name == _LOGGER_NAME

    def test_child_logger_name(self) -> None:
        """get_logger('foo') should return mangomas_demo.foo."""
        logger = get_logger("foo")
        assert logger.name == f"{_LOGGER_NAME}.foo"

    def test_nested_child_logger(self) -> None:
        """get_logger('cells.executor') should return a dotted child."""
        logger = get_logger("cells.executor")
        assert logger.name == f"{_LOGGER_NAME}.cells.executor"

    def test_returns_logging_logger(self) -> None:
        """Should return a standard logging.Logger instance."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)


class TestConfigureLogging:
    """Tests for configure_logging."""

    def _reset_logger(self) -> None:
        """Remove all handlers from the root mangomas logger."""
        logger = logging.getLogger(_LOGGER_NAME)
        logger.handlers.clear()

    def test_creates_handler(self) -> None:
        """Should attach at least one handler."""
        self._reset_logger()
        logger = configure_logging()
        assert len(logger.handlers) >= 1

    def test_idempotent(self) -> None:
        """Calling configure_logging twice should not duplicate handlers."""
        self._reset_logger()
        configure_logging()
        handler_count = len(logging.getLogger(_LOGGER_NAME).handlers)
        configure_logging()
        assert len(logging.getLogger(_LOGGER_NAME).handlers) == handler_count

    def test_default_level_is_info(self) -> None:
        """Default log level should be INFO when env var is unset."""
        self._reset_logger()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV_LOG_LEVEL, None)
            logger = configure_logging()
        assert logger.level == logging.INFO

    def test_env_var_sets_level_debug(self) -> None:
        """MANGOMAS_LOG_LEVEL=DEBUG should set DEBUG level."""
        self._reset_logger()
        with mock.patch.dict(os.environ, {_ENV_LOG_LEVEL: "DEBUG"}):
            logger = configure_logging()
        assert logger.level == logging.DEBUG
        self._reset_logger()

    def test_env_var_sets_level_warning(self) -> None:
        """MANGOMAS_LOG_LEVEL=WARNING should set WARNING level."""
        self._reset_logger()
        with mock.patch.dict(os.environ, {_ENV_LOG_LEVEL: "WARNING"}):
            logger = configure_logging()
        assert logger.level == logging.WARNING
        self._reset_logger()

    def test_env_var_case_insensitive(self) -> None:
        """Log level env var should be case-insensitive."""
        self._reset_logger()
        with mock.patch.dict(os.environ, {_ENV_LOG_LEVEL: "debug"}):
            logger = configure_logging()
        assert logger.level == logging.DEBUG
        self._reset_logger()

    def test_json_format_selected(self) -> None:
        """MANGOMAS_LOG_FORMAT=json should use JSONFormatter."""
        self._reset_logger()
        with mock.patch.dict(os.environ, {_ENV_LOG_FORMAT: "json"}):
            logger = configure_logging()
        formatter = logger.handlers[0].formatter
        assert isinstance(formatter, _JSONFormatter)
        self._reset_logger()

    def test_text_format_default(self) -> None:
        """Default format should be text (not JSON)."""
        self._reset_logger()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV_LOG_FORMAT, None)
            logger = configure_logging()
        formatter = logger.handlers[0].formatter
        assert not isinstance(formatter, _JSONFormatter)
        self._reset_logger()


class TestJSONFormatter:
    """Tests for the structured JSON log formatter."""

    def test_output_is_valid_json(self) -> None:
        """Formatted output should be parseable JSON."""
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="mangomas_demo.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "mangomas_demo.test"

    def test_includes_timestamp(self) -> None:
        """JSON output should include a timestamp."""
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="t.py", lineno=1,
            msg="ts check", args=(), exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert "timestamp" in parsed

    def test_extra_fields_propagated(self) -> None:
        """Extra structured fields should appear in JSON output."""
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="t.py", lineno=1,
            msg="extra test", args=(), exc_info=None,
        )
        record.component = "cells"  # type: ignore[attr-defined]
        record.elapsed_ms = 12.5  # type: ignore[attr-defined]
        parsed = json.loads(formatter.format(record))
        assert parsed["component"] == "cells"
        assert parsed["elapsed_ms"] == 12.5

    def test_exception_included(self) -> None:
        """Exception info should be serialized in JSON output."""
        formatter = _JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="t.py", lineno=1,
            msg="error", args=(), exc_info=exc_info,
        )
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestLoggingIntegration:
    """Integration tests verifying logging works across modules."""

    def test_module_loggers_exist(self) -> None:
        """All core modules should have named loggers."""
        expected = [
            "features",
            "cells.executor",
            "mcts.engine",
            "routing.router",
            "agents.orchestrator",
        ]
        for name in expected:
            logger = get_logger(name)
            assert logger.name == f"{_LOGGER_NAME}.{name}"

    def test_no_warnings_during_normal_operation(self, caplog: pytest.LogCaptureFixture) -> None:
        """Normal cell execution should not produce WARNING+ logs."""
        from mangomas_demo.cells.executor import execute_cell

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            execute_cell("reasoning", "Test input for logging check")
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name.startswith(_LOGGER_NAME)
        ]
        assert len(warning_records) == 0

    def test_debug_suppressed_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """DEBUG messages should not appear when logger is set to INFO level.

        Uses caplog at INFO level to verify DEBUG records are filtered out.
        """
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            from mangomas_demo.features import featurize64
            featurize64("debug suppression test")
        debug_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG and r.name.startswith(_LOGGER_NAME)
        ]
        assert len(debug_records) == 0
