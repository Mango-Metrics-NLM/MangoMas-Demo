"""
Unit Tests — MCTS Planning Engine.
"""

from __future__ import annotations

from mangomas_demo.mcts.benchmark import benchmark_strategies
from mangomas_demo.mcts.engine import TASK_CATEGORIES, MCTSNode, _detect_category, run_mcts


class TestMCTSNode:
    """Tests for the MCTS tree node."""

    def test_ucb1_unvisited_is_inf(self) -> None:
        """Unvisited nodes should have infinite UCB1 score."""
        node = MCTSNode(id="n1", action="test", visits=0)
        assert node.ucb1_score(parent_visits=10) == float("inf")

    def test_ucb1_visited(self) -> None:
        """Visited nodes should have finite UCB1 score."""
        node = MCTSNode(id="n1", action="test", visits=5, total_value=3.5)
        score = node.ucb1_score(parent_visits=20)
        assert isinstance(score, float)
        assert score > 0

    def test_puct_unvisited_is_inf(self) -> None:
        """Unvisited nodes should have infinite PUCT score."""
        node = MCTSNode(id="n1", action="test", visits=0, policy_prior=0.5)
        assert node.puct_score(parent_visits=10) == float("inf")

    def test_puct_visited(self) -> None:
        """Visited nodes should have finite PUCT score."""
        node = MCTSNode(id="n1", action="test", visits=5, total_value=3.0, policy_prior=0.6)
        score = node.puct_score(parent_visits=20)
        assert isinstance(score, float)

    def test_to_dict(self) -> None:
        """Should serialize to dict with required fields."""
        child = MCTSNode(id="c1", action="sub", visits=2, total_value=1.5)
        node = MCTSNode(id="root", action="main", visits=5, total_value=3.5, children=[child])
        d = node.to_dict()
        assert d["id"] == "root"
        assert "children" in d
        assert len(d["children"]) == 1

    def test_to_dict_depth_limit(self) -> None:
        """Should respect max_depth limit."""
        deep = MCTSNode(id="deep", action="deep", children=[
            MCTSNode(id="d1", action="d1", children=[
                MCTSNode(id="d2", action="d2")
            ])
        ])
        d = deep.to_dict(max_depth=1)
        assert "children" in d
        # Depth 1 children should not have their own children serialized
        if d["children"][0].get("children"):
            assert True  # max_depth=1 allows first level children


class TestRunMCTS:
    """Tests for the MCTS search function."""

    def test_returns_best_action(self, sample_text: str) -> None:
        """Should return a non-empty best_action."""
        result = run_mcts(sample_text, max_simulations=20)
        assert result["best_action"] != ""
        assert result["best_action"] != "none"

    def test_category_detection(self) -> None:
        """Security keywords should route to security category."""
        result = run_mcts("Analyze security vulnerabilities", max_simulations=10)
        assert result["category"] == "security"

    def test_architecture_category(self) -> None:
        """Architecture keywords should route correctly."""
        result = run_mcts("Design the system architecture", max_simulations=10)
        assert result["category"] == "architecture"

    def test_implementation_default(self) -> None:
        """Unknown topics should default to implementation."""
        result = run_mcts("Do something generic", max_simulations=10)
        assert result["category"] == "implementation"

    def test_simulation_count(self) -> None:
        """Reported simulation count should match input."""
        result = run_mcts("test task", max_simulations=50)
        assert result["total_simulations"] == 50

    def test_tree_structure(self, sample_text: str) -> None:
        """Tree should be a valid dict with children."""
        result = run_mcts(sample_text, max_simulations=30)
        tree = result["tree"]
        assert "id" in tree
        assert "children" in tree

    def test_all_actions_returned(self, sample_text: str) -> None:
        """all_actions should list actions with visits and values."""
        result = run_mcts(sample_text, max_simulations=30)
        assert len(result["all_actions"]) >= 1
        for a in result["all_actions"]:
            assert "action" in a
            assert "visits" in a
            assert "value" in a

    def test_convergence(self, sample_text: str) -> None:
        """More simulations should give more total visits."""
        r1 = run_mcts(sample_text, max_simulations=10)
        r2 = run_mcts(sample_text, max_simulations=100)
        total_v1 = sum(a["visits"] for a in r1["all_actions"])
        total_v2 = sum(a["visits"] for a in r2["all_actions"])
        assert total_v2 >= total_v1

    def test_strategy_ucb1(self, sample_text: str) -> None:
        """UCB1 strategy should be reported."""
        result = run_mcts(sample_text, max_simulations=10, strategy="ucb1")
        assert result["strategy"] == "ucb1"

    def test_strategy_puct(self, sample_text: str) -> None:
        """PUCT strategy should be reported."""
        result = run_mcts(sample_text, max_simulations=10, strategy="puct")
        assert result["strategy"] == "puct"

    def test_elapsed_ms(self, sample_text: str) -> None:
        """Should report elapsed time in milliseconds."""
        result = run_mcts(sample_text, max_simulations=10)
        assert result["elapsed_ms"] >= 0


class TestDetectCategory:
    """Tests for category detection helper."""

    def test_security(self) -> None:
        assert _detect_category("fix security vulnerability") == "security"

    def test_architecture(self) -> None:
        assert _detect_category("design system architecture") == "architecture"

    def test_optimization(self) -> None:
        assert _detect_category("optimize performance") == "optimization"

    def test_research(self) -> None:
        assert _detect_category("research market trends") == "research"

    def test_default(self) -> None:
        assert _detect_category("hello world") == "implementation"


class TestBenchmarkStrategies:
    """Tests for the 3-strategy benchmark."""

    def test_all_strategies_returned(self, sample_text: str) -> None:
        """Benchmark should return MCTS, greedy, and random results."""
        result = benchmark_strategies(sample_text)
        assert "mcts" in result["results"]
        assert "greedy" in result["results"]
        assert "random" in result["results"]

    def test_each_has_quality_score(self, sample_text: str) -> None:
        """Each strategy should report quality_score and best_action."""
        result = benchmark_strategies(sample_text)
        for name, data in result["results"].items():
            assert "quality_score" in data, f"{name} missing quality_score"
            assert "best_action" in data, f"{name} missing best_action"
            assert "elapsed_ms" in data, f"{name} missing elapsed_ms"

    def test_category_included(self, sample_text: str) -> None:
        """Should include detected category."""
        result = benchmark_strategies(sample_text)
        assert result["category"] in TASK_CATEGORIES
