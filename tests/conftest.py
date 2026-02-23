"""
Shared test fixtures for MangoMAS Demo.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_text() -> str:
    """Standard test input text."""
    return "Design a secure API gateway with rate limiting and JWT authentication"


@pytest.fixture
def security_text() -> str:
    """Text containing PII for security tests."""
    return "Contact john.doe@example.com or call 555-123-4567 for SSN 123-45-6789"


@pytest.fixture
def empty_inputs() -> list[str]:
    """Various empty/whitespace inputs."""
    return ["", " ", "\t\n", "\n  "]


@pytest.fixture
def figurative_text() -> str:
    """Text with idioms for FigLiteral cell."""
    return "It's raining cats and dogs outside"


@pytest.fixture
def short_text() -> str:
    """Minimal valid text."""
    return "hello"
