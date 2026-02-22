"""
Sanity Tests — Documentation validation.

Verifies that all architecture documents exist, contain valid Mermaid
code blocks, and that the README contains no dead internal links.
"""

from __future__ import annotations

import os
import re

import pytest

# ---------------------------------------------------------------------------
# Paths derived from repo root (no hard-coded absolute paths)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DOCS_ARCH_DIR = os.path.join(_REPO_ROOT, "docs", "architecture")
_README_PATH = os.path.join(_REPO_ROOT, "README.md")

_EXPECTED_ARCH_DOCS = [
    "c4-context.md",
    "c4-container.md",
    "c4-component.md",
    "c4-code.md",
    "data-flow.md",
    "deployment.md",
]

_EXPECTED_ROOT_DOCS = [
    "CONTRIBUTING.md",
    "SECURITY.md",
    "LICENSE",
    "CHANGELOG.md",
]


class TestArchitectureDocsExist:
    """Verify all C4 architecture documents are present."""

    def test_docs_directory_exists(self) -> None:
        """docs/architecture/ directory should exist."""
        assert os.path.isdir(_DOCS_ARCH_DIR), (
            f"Missing directory: {_DOCS_ARCH_DIR}"
        )

    @pytest.mark.parametrize("filename", _EXPECTED_ARCH_DOCS)
    def test_arch_doc_exists(self, filename: str) -> None:
        """Each C4 architecture document should exist."""
        path = os.path.join(_DOCS_ARCH_DIR, filename)
        assert os.path.isfile(path), f"Missing architecture doc: {filename}"

    @pytest.mark.parametrize("filename", _EXPECTED_ARCH_DOCS)
    def test_arch_doc_not_empty(self, filename: str) -> None:
        """Architecture documents should not be empty."""
        path = os.path.join(_DOCS_ARCH_DIR, filename)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            assert size > 100, f"{filename} is too small ({size} bytes)"


class TestArchitectureDocsMermaid:
    """Verify architecture documents contain valid Mermaid code blocks."""

    @pytest.mark.parametrize("filename", _EXPECTED_ARCH_DOCS)
    def test_contains_mermaid_block(self, filename: str) -> None:
        """Each architecture doc should contain at least one ```mermaid block."""
        path = os.path.join(_DOCS_ARCH_DIR, filename)
        if not os.path.isfile(path):
            pytest.skip(f"{filename} does not exist yet")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        mermaid_blocks = re.findall(r"```mermaid", content)
        assert len(mermaid_blocks) >= 1, (
            f"{filename} has no ```mermaid code blocks"
        )

    @pytest.mark.parametrize("filename", _EXPECTED_ARCH_DOCS)
    def test_mermaid_blocks_are_closed(self, filename: str) -> None:
        """Every opened ```mermaid block should have a closing ```."""
        path = os.path.join(_DOCS_ARCH_DIR, filename)
        if not os.path.isfile(path):
            pytest.skip(f"{filename} does not exist yet")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # Count closing ``` that appear after mermaid blocks
        # Simple heuristic: total ``` count should be even
        all_fences = len(re.findall(r"```", content))
        assert all_fences % 2 == 0, (
            f"{filename} has unclosed code fence (total fences: {all_fences})"
        )


class TestRootDocsExist:
    """Verify supporting documentation files exist."""

    @pytest.mark.parametrize("filename", _EXPECTED_ROOT_DOCS)
    def test_root_doc_exists(self, filename: str) -> None:
        """Root-level documentation files should exist."""
        path = os.path.join(_REPO_ROOT, filename)
        assert os.path.isfile(path), f"Missing root doc: {filename}"


class TestReadmeIntegrity:
    """Verify README.md quality and link integrity."""

    def test_readme_exists(self) -> None:
        """README.md should exist at repo root."""
        assert os.path.isfile(_README_PATH)

    def test_readme_has_content(self) -> None:
        """README should have substantial content."""
        size = os.path.getsize(_README_PATH)
        assert size > 500, f"README is too small ({size} bytes)"

    def test_readme_has_mermaid_diagram(self) -> None:
        """README should contain at least one Mermaid diagram."""
        with open(_README_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "```mermaid" in content

    def test_readme_no_dead_internal_links(self) -> None:
        """Internal file links in README should point to existing files."""
        with open(_README_PATH, encoding="utf-8") as f:
            content = f.read()
        # Match markdown links: [text](path) excluding http(s) and # anchors
        link_pattern = r"\[.*?\]\((?!https?://)(?!#)([^)]+)\)"
        links = re.findall(link_pattern, content)
        missing: list[str] = []
        for link in links:
            # Strip anchor fragments
            file_path = link.split("#")[0]
            if not file_path:
                continue
            full_path = os.path.join(_REPO_ROOT, file_path)
            if not os.path.exists(full_path):
                missing.append(file_path)
        assert len(missing) == 0, f"Dead internal links in README: {missing}"

    def test_readme_has_table_of_contents(self) -> None:
        """README should contain a table of contents section."""
        with open(_README_PATH, encoding="utf-8") as f:
            content = f.read().lower()
        assert "table of contents" in content or "## contents" in content

    def test_notebook_directory_exists(self) -> None:
        """notebooks/ directory should exist."""
        nb_dir = os.path.join(_REPO_ROOT, "notebooks")
        assert os.path.isdir(nb_dir), "Missing notebooks/ directory"
