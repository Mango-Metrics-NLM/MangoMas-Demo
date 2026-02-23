"""
Integration Tests - full pipeline and end-to-end flows.
"""

from __future__ import annotations

from mangomas_demo.agents.orchestrator import orchestrate
from mangomas_demo.cells.executor import compose_cells, execute_cell
from mangomas_demo.features import featurize64
from mangomas_demo.mcts.engine import run_mcts
from mangomas_demo.routing.router import route_task


class TestFullPipeline:
    """End-to-end pipeline tests."""

    def test_featurize_to_route_to_cell(self, sample_text: str) -> None:
        """Full flow: featurize → route → execute best cell."""
        features = featurize64(sample_text)
        assert len(features) == 64

        routing = route_task(sample_text, top_k=1)
        expert = routing["selected_experts"][0]["expert"]
        assert expert in [
            "Code Expert",
            "Test Expert",
            "Design Expert",
            "Research Expert",
            "Architecture Expert",
            "Security Expert",
            "Performance Expert",
            "Docs Expert",
        ]

        result = execute_cell("reasoning", sample_text)
        assert result["status"] == "ok"

    def test_orchestration_moe_strategy(self, sample_text: str) -> None:
        """Orchestration with MoE routing should produce real results."""
        result = orchestrate(sample_text, max_agents=3, strategy="moe_routing")
        assert result["agents_selected"] == 3
        for agent_result in result["results"]:
            assert "output" in agent_result
            assert agent_result["output"]["status"] == "ok"

    def test_orchestration_round_robin(self, sample_text: str) -> None:
        """Round-robin strategy should work."""
        result = orchestrate(sample_text, max_agents=2, strategy="round_robin")
        assert result["agents_selected"] == 2

    def test_orchestration_random(self, sample_text: str) -> None:
        """Random strategy should work."""
        result = orchestrate(sample_text, max_agents=2, strategy="random")
        assert result["agents_selected"] == 2

    def test_cell_composition_pipeline(self) -> None:
        """Multi-cell pipeline should chain correctly."""
        result = compose_cells(
            "reasoning, ethics, curiosity",
            "Analyze the security posture of our API at user@test.com",
        )
        assert result["total_cells"] == 3
        assert all(a["status"] == "ok" for a in result["activations"])

    def test_mcts_then_route(self, sample_text: str) -> None:
        """MCTS planning followed by routing should work."""
        plan = run_mcts(sample_text, max_simulations=20)
        assert plan["best_action"] != "none"

        routing = route_task(plan["best_action"], top_k=2)
        assert len(routing["selected_experts"]) == 2


class TestEndToEnd:
    """Smoke tests for the full application stack."""

    def test_all_cells_in_pipeline(self) -> None:
        """Running all 10 cells in sequence should not crash."""
        all_cells = (
            "reasoning,memory,causal,ethics,empathy,curiosity,figliteral,r2p,telemetry,aggregator"
        )
        result = compose_cells(all_cells, "Test all cells with this input text")
        assert result["total_cells"] == 10
        ok_count = sum(1 for a in result["activations"] if a["status"] == "ok")
        assert ok_count == 10

    def test_large_orchestration(self) -> None:
        """Orchestrating 8 agents should work."""
        result = orchestrate("Build a complete system", max_agents=8, strategy="round_robin")
        assert result["agents_selected"] == 8

    def test_mcts_high_simulation(self) -> None:
        """MCTS with many simulations should converge."""
        result = run_mcts("Complex planning task", max_simulations=200)
        total_visits = sum(a["visits"] for a in result["all_actions"])
        assert total_visits == 200
