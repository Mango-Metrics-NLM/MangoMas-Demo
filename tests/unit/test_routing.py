"""
Unit Tests — MoE Neural Router.
"""

from __future__ import annotations

import pytest

from mangomas_demo.routing.router import route_task, EXPERT_NAMES


class TestRouteTask:
    """Tests for MoE routing function."""

    def test_returns_selected_experts(self, sample_text: str) -> None:
        """Should return selected experts with weights."""
        result = route_task(sample_text, top_k=3)
        assert len(result["selected_experts"]) == 3

    def test_top_k_respected(self) -> None:
        """Selected count should match top_k."""
        for k in [1, 3, 5, 8]:
            result = route_task("test input", top_k=k)
            assert len(result["selected_experts"]) == k

    def test_all_weights_sum_to_one(self, sample_text: str) -> None:
        """All expert weights should approximately sum to 1."""
        result = route_task(sample_text)
        total = sum(result["all_weights"].values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"

    def test_deterministic_routing(self) -> None:
        """Same input should produce identical routing (fixed seed)."""
        r1 = route_task("analyze security threats", top_k=3)
        r2 = route_task("analyze security threats", top_k=3)
        assert r1["selected_experts"] == r2["selected_experts"]

    def test_semantic_boost_security(self) -> None:
        """Security keywords should boost Security Expert."""
        result = route_task("Fix the security vulnerability in auth", top_k=3)
        expert_names = [e["expert"] for e in result["selected_experts"]]
        assert "Security Expert" in expert_names

    def test_semantic_boost_code(self) -> None:
        """Code-related keywords should boost Code Expert."""
        result = route_task("Implement a new function in the module", top_k=3)
        expert_names = [e["expert"] for e in result["selected_experts"]]
        assert "Code Expert" in expert_names

    def test_semantic_boost_test(self) -> None:
        """Test keywords should boost Test Expert."""
        result = route_task("Write unit tests for the coverage module", top_k=3)
        expert_names = [e["expert"] for e in result["selected_experts"]]
        assert "Test Expert" in expert_names

    def test_features_64dim(self, sample_text: str) -> None:
        """Should include 64-dim feature vector."""
        result = route_task(sample_text)
        assert len(result["features"]) == 64

    def test_expert_names_complete(self) -> None:
        """All 8 experts should be in the all_weights."""
        result = route_task("test")
        assert len(result["all_weights"]) == len(EXPERT_NAMES)

    def test_elapsed_ms(self, sample_text: str) -> None:
        """Should report elapsed time."""
        result = route_task(sample_text)
        assert result["elapsed_ms"] >= 0

    def test_rank_ordering(self, sample_text: str) -> None:
        """Selected experts should be ordered by rank."""
        result = route_task(sample_text, top_k=5)
        ranks = [e["rank"] for e in result["selected_experts"]]
        assert ranks == sorted(ranks)

    def test_weights_positive(self, sample_text: str) -> None:
        """All weights should be non-negative."""
        result = route_task(sample_text)
        for w in result["all_weights"].values():
            assert w >= 0
