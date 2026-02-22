"""
Security Tests — PII detection, input sanitization, secrets scan.
"""

from __future__ import annotations

import os
import re

from mangomas_demo.cells.executor import execute_cell


class TestPIIDetection:
    """Tests for PII detection and redaction."""

    def test_email_detected(self) -> None:
        """Should detect email addresses as PII."""
        result = execute_cell("ethics", "Send to admin@company.com please")
        assert not result["is_safe"]
        pii_types = [p["type"] for p in result["pii_detected"]]
        assert "email" in pii_types

    def test_phone_detected(self) -> None:
        """Should detect phone numbers."""
        result = execute_cell("ethics", "Call us at 800-555-1234")
        assert not result["is_safe"]
        pii_types = [p["type"] for p in result["pii_detected"]]
        assert "phone" in pii_types

    def test_ssn_detected(self) -> None:
        """Should detect SSN patterns."""
        result = execute_cell("ethics", "SSN is 078-05-1120")
        assert not result["is_safe"]
        pii_types = [p["type"] for p in result["pii_detected"]]
        assert "ssn" in pii_types

    def test_multiple_pii(self, security_text: str) -> None:
        """Should detect multiple PII types simultaneously."""
        result = execute_cell("ethics", security_text)
        assert len(result["pii_detected"]) >= 2

    def test_redaction_removes_pii(self) -> None:
        """Redacted text should not contain original PII."""
        result = execute_cell("ethics", "Email: secret@corp.io")
        assert "secret@corp.io" not in result["redacted_text"]
        assert "[REDACTED]" in result["redacted_text"]


class TestInputSanitization:
    """Tests for input validation and sanitization."""

    def test_empty_string_rejected(self) -> None:
        """Empty strings should be rejected with error."""
        result = execute_cell("reasoning", "")
        assert result["status"] == "error"

    def test_whitespace_only_rejected(self) -> None:
        """Whitespace-only input should be rejected."""
        result = execute_cell("reasoning", "   \t\n  ")
        assert result["status"] == "error"

    def test_script_injection(self) -> None:
        """Script tags in input should not crash the system."""
        result = execute_cell("reasoning", "<script>alert('xss')</script>")
        assert result["status"] == "ok"

    def test_sql_injection_like_input(self) -> None:
        """SQL-like input should not crash the system."""
        result = execute_cell("reasoning", "'; DROP TABLE users; --")
        assert result["status"] == "ok"


class TestNoHardcodedSecrets:
    """Verify no secrets are hardcoded in the codebase."""

    def test_no_api_keys_in_source(self) -> None:
        """Source code should not contain hardcoded API keys."""
        src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "mangomas_demo")
        src_dir = os.path.abspath(src_dir)

        secret_patterns = [
            r"(?:api[_-]?key|api[_-]?secret)\s*=\s*['\"][a-zA-Z0-9]{16,}['\"]",
            r"(?:password|passwd|pwd)\s*=\s*['\"][^'\"]{8,}['\"]",
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI-style keys
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub PATs
            r"hf_[a-zA-Z0-9]{34}",  # HuggingFace tokens
        ]

        violations: list[str] = []
        for root, _, files in os.walk(src_dir):
            for f in files:
                if not f.endswith(".py"):
                    continue
                fpath = os.path.join(root, f)
                with open(fpath, encoding="utf-8") as fh:
                    content = fh.read()
                    for pattern in secret_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        violations.extend(
                            f"{fpath}: {m}" for m in matches
                        )

        assert len(violations) == 0, f"Hardcoded secrets found: {violations}"
