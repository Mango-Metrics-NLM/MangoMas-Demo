"""
Sanity Tests — imports, config, parameter counts.
"""

from __future__ import annotations


class TestImports:
    """Verify all modules import correctly."""

    def test_import_package(self) -> None:
        """Top-level package should import."""
        import mangomas_demo
        assert hasattr(mangomas_demo, "__version__")
        assert mangomas_demo.__version__ == "1.0.0"

    def test_import_features(self) -> None:
        """Features module should import."""
        from mangomas_demo.features import featurize64
        assert callable(featurize64)

    def test_import_cells(self) -> None:
        """Cells module should import."""
        from mangomas_demo.cells import CELL_TYPES
        assert len(CELL_TYPES) == 10

    def test_import_mcts(self) -> None:
        """MCTS module should import."""
        from mangomas_demo.mcts import run_mcts
        assert callable(run_mcts)

    def test_import_routing(self) -> None:
        """Routing module should import."""
        from mangomas_demo.routing import EXPERT_NAMES
        assert len(EXPERT_NAMES) == 8

    def test_import_agents(self) -> None:
        """Agents module should import."""
        from mangomas_demo.agents import AGENTS
        assert len(AGENTS) == 8

    def test_import_models(self) -> None:
        """Models module should import."""
        from mangomas_demo.models import TORCH_AVAILABLE
        assert isinstance(TORCH_AVAILABLE, bool)


class TestConfig:
    """Verify configuration consistency."""

    def test_cell_types_count(self) -> None:
        """Should have exactly 10 cell types."""
        from mangomas_demo.cells.types import CELL_TYPES
        assert len(CELL_TYPES) == 10

    def test_expert_names_count(self) -> None:
        """Should have exactly 8 experts."""
        from mangomas_demo.routing.router import EXPERT_NAMES
        assert len(EXPERT_NAMES) == 8

    def test_agent_count(self) -> None:
        """Should have exactly 8 agents."""
        from mangomas_demo.agents.orchestrator import AGENTS
        assert len(AGENTS) == 8

    def test_task_categories(self) -> None:
        """Should have 5 task categories."""
        from mangomas_demo.mcts.engine import TASK_CATEGORIES
        assert len(TASK_CATEGORIES) == 5
        for cat, actions in TASK_CATEGORIES.items():
            assert len(actions) >= 3, f"Category {cat} has too few actions"
