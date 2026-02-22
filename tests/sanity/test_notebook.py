"""
Sanity Tests — Jupyter Notebook validation.

Verifies that the demo notebook exists, is valid JSON,
and uses only public API imports.
"""

from __future__ import annotations

import json
import os
import re

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_NOTEBOOK_PATH = os.path.join(_REPO_ROOT, "notebooks", "demo.ipynb")

# Minimum expected cells (markdown + code)
_MIN_CELL_COUNT = 10

# Public API modules that are safe to import
_ALLOWED_IMPORT_SOURCES = {
    "mangomas_demo",
    "mangomas_demo.features",
    "mangomas_demo.cells",
    "mangomas_demo.cells.types",
    "mangomas_demo.cells.executor",
    "mangomas_demo.mcts",
    "mangomas_demo.mcts.engine",
    "mangomas_demo.mcts.benchmark",
    "mangomas_demo.mcts.viz",
    "mangomas_demo.routing",
    "mangomas_demo.routing.router",
    "mangomas_demo.routing.viz",
    "mangomas_demo.agents",
    "mangomas_demo.agents.orchestrator",
    "mangomas_demo.ui",
    "mangomas_demo.ui.builder",
    "mangomas_demo.ui.styles",
    "mangomas_demo.models",
    "mangomas_demo.logging_config",
}


class TestNotebookExists:
    """Verify the demo notebook is present."""

    def test_notebook_file_exists(self) -> None:
        """notebooks/demo.ipynb should exist."""
        assert os.path.isfile(_NOTEBOOK_PATH), (
            f"Missing notebook: {_NOTEBOOK_PATH}"
        )

    def test_notebook_not_empty(self) -> None:
        """Notebook should have non-trivial size."""
        if not os.path.isfile(_NOTEBOOK_PATH):
            pytest.skip("Notebook does not exist yet")
        size = os.path.getsize(_NOTEBOOK_PATH)
        assert size > 500, f"Notebook is too small ({size} bytes)"


class TestNotebookStructure:
    """Verify notebook JSON structure and content."""

    @pytest.fixture()
    def notebook(self) -> dict:
        """Load the notebook as a dict."""
        if not os.path.isfile(_NOTEBOOK_PATH):
            pytest.skip("Notebook does not exist yet")
        with open(_NOTEBOOK_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_valid_json(self) -> None:
        """Notebook should be valid JSON."""
        if not os.path.isfile(_NOTEBOOK_PATH):
            pytest.skip("Notebook does not exist yet")
        with open(_NOTEBOOK_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_has_nbformat(self, notebook: dict) -> None:
        """Notebook should declare nbformat version."""
        assert "nbformat" in notebook
        assert notebook["nbformat"] >= 4

    def test_has_cells(self, notebook: dict) -> None:
        """Notebook should contain cells."""
        assert "cells" in notebook
        assert isinstance(notebook["cells"], list)

    def test_minimum_cell_count(self, notebook: dict) -> None:
        """Notebook should have a minimum number of cells."""
        assert len(notebook["cells"]) >= _MIN_CELL_COUNT, (
            f"Expected >= {_MIN_CELL_COUNT} cells, got {len(notebook['cells'])}"
        )

    def test_has_markdown_cells(self, notebook: dict) -> None:
        """Notebook should contain explanatory markdown cells."""
        md_cells = [c for c in notebook["cells"] if c.get("cell_type") == "markdown"]
        assert len(md_cells) >= 5, "Expected at least 5 markdown cells"

    def test_has_code_cells(self, notebook: dict) -> None:
        """Notebook should contain executable code cells."""
        code_cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) >= 5, "Expected at least 5 code cells"

    def test_cells_have_source(self, notebook: dict) -> None:
        """Every cell should have a source field."""
        for i, cell in enumerate(notebook["cells"]):
            assert "source" in cell, f"Cell {i} missing 'source' field"

    def test_code_cells_use_public_api(self, notebook: dict) -> None:
        """Code cells should only import from public API modules."""
        code_cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
        violations: list[str] = []
        for i, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            # Match "from X import" and "import X"
            imports = re.findall(
                r"(?:from|import)\s+(mangomas_demo[\w.]*)", source
            )
            for imp in imports:
                if imp not in _ALLOWED_IMPORT_SOURCES:
                    violations.append(f"Cell {i}: {imp}")
        assert len(violations) == 0, (
            f"Non-public API imports found: {violations}"
        )

    def test_has_kernelspec(self, notebook: dict) -> None:
        """Notebook metadata should include a kernelspec."""
        metadata = notebook.get("metadata", {})
        assert "kernelspec" in metadata, "Missing kernelspec in metadata"


class TestNotebookContent:
    """Verify notebook covers key demo topics."""

    @pytest.fixture()
    def all_source(self) -> str:
        """Concatenate all cell sources into one string."""
        if not os.path.isfile(_NOTEBOOK_PATH):
            pytest.skip("Notebook does not exist yet")
        with open(_NOTEBOOK_PATH, encoding="utf-8") as f:
            nb = json.load(f)
        parts: list[str] = []
        for cell in nb.get("cells", []):
            parts.append("".join(cell.get("source", [])))
        return "\n".join(parts)

    def test_covers_feature_extraction(self, all_source: str) -> None:
        """Notebook should demonstrate feature extraction."""
        assert "featurize64" in all_source

    def test_covers_cognitive_cells(self, all_source: str) -> None:
        """Notebook should demonstrate cell execution."""
        assert "execute_cell" in all_source

    def test_covers_mcts(self, all_source: str) -> None:
        """Notebook should demonstrate MCTS planning."""
        assert "run_mcts" in all_source

    def test_covers_routing(self, all_source: str) -> None:
        """Notebook should demonstrate MoE routing."""
        assert "route_task" in all_source

    def test_covers_orchestration(self, all_source: str) -> None:
        """Notebook should demonstrate agent orchestration."""
        assert "orchestrate" in all_source
