"""
Unit Tests - Cognitive Cells (all 10 cell types).
"""

from __future__ import annotations

import pytest

from mangomas_demo.cells.executor import compose_cells, execute_cell
from mangomas_demo.cells.types import CELL_TYPES


class TestCellRegistry:
    """Tests for cell type registry."""

    def test_ten_cell_types(self) -> None:
        """Registry should contain exactly 10 cell types."""
        assert len(CELL_TYPES) == 10

    def test_all_cells_have_name(self) -> None:
        """Every cell should have a name field."""
        for ct, info in CELL_TYPES.items():
            assert "name" in info, f"{ct} missing 'name'"

    def test_all_cells_have_heads(self) -> None:
        """Every cell should have at least one head."""
        for ct, info in CELL_TYPES.items():
            assert len(info["heads"]) >= 1, f"{ct} has no heads"


class TestExecuteCell:
    """Tests for execute_cell dispatcher."""

    @pytest.mark.parametrize("cell_type", list(CELL_TYPES.keys()))
    def test_all_cells_execute(self, cell_type: str, sample_text: str) -> None:
        """Every registered cell type should execute successfully."""
        result = execute_cell(cell_type, sample_text)
        assert result["status"] == "ok", f"{cell_type} failed: {result}"
        assert result["cell_type"] == cell_type
        assert "request_id" in result
        assert "elapsed_ms" in result

    @pytest.mark.parametrize("empty", ["", " ", "\t\n"])
    def test_empty_input_error(self, empty: str) -> None:
        """Empty/whitespace input should return error status."""
        result = execute_cell("reasoning", empty)
        assert result["status"] == "error"
        assert "required" in result.get("message", "").lower()

    def test_invalid_json_config(self, sample_text: str) -> None:
        """Invalid JSON config should return descriptive error."""
        result = execute_cell("reasoning", sample_text, config_json="{bad json}")
        assert result["status"] == "error"
        assert "Invalid JSON" in result.get("message", "")


class TestReasoningCell:
    """Tests for the reasoning cell."""

    def test_has_sections(self, sample_text: str) -> None:
        """Should produce sections with confidence scores."""
        result = execute_cell("reasoning", sample_text)
        assert "sections" in result
        assert len(result["sections"]) >= 1
        for s in result["sections"]:
            assert "confidence" in s
            assert 0 <= s["confidence"] <= 1.0

    def test_configurable_head(self, sample_text: str) -> None:
        """Head type config should be respected."""
        result = execute_cell("reasoning", sample_text, config_json='{"head_type": "nn"}')
        assert result["head_type"] == "nn"


class TestMemoryCell:
    """Tests for the memory cell."""

    def test_preference_extraction(self) -> None:
        """Should detect preference keywords."""
        result = execute_cell("memory", "I prefer Python over Java")
        assert "preferences" in result
        assert len(result["preferences"]) >= 1

    def test_opt_out_detection(self) -> None:
        """Should detect opt-out expressions."""
        result = execute_cell("memory", "Please don't remember this")
        assert result["opt_out"] is True

    def test_consent_status(self, sample_text: str) -> None:
        """Should always include consent status."""
        result = execute_cell("memory", sample_text)
        assert result["consent_status"] == "granted"


class TestEthicsCell:
    """Tests for the ethics/safety cell."""

    def test_pii_detection_email(self) -> None:
        """Should detect email addresses."""
        result = execute_cell("ethics", "Contact user@example.com for details")
        assert not result["is_safe"]
        assert len(result["pii_detected"]) >= 1
        assert result["pii_detected"][0]["type"] == "email"

    def test_pii_detection_phone(self) -> None:
        """Should detect phone numbers."""
        result = execute_cell("ethics", "Call 555-123-4567")
        assert not result["is_safe"]

    def test_pii_detection_ssn(self) -> None:
        """Should detect SSN patterns."""
        result = execute_cell("ethics", "SSN: 123-45-6789")
        assert not result["is_safe"]

    def test_clean_text_safe(self) -> None:
        """Text without PII should be marked safe."""
        result = execute_cell("ethics", "Design a secure API")
        assert result["is_safe"]

    def test_redaction(self) -> None:
        """PII should be redacted in output."""
        result = execute_cell("ethics", "Email me at john@test.com")
        assert "[REDACTED]" in result["redacted_text"]


class TestEmpathyCell:
    """Tests for the empathy cell."""

    def test_frustration_detection(self) -> None:
        """Should detect frustration keywords."""
        result = execute_cell("empathy", "I'm so frustrated with this broken system")
        assert result["detected_emotion"] == "frustration"
        assert (
            "frustrat" in result["empathetic_response"].lower()
            or "resolve" in result["empathetic_response"].lower()
        )

    def test_excitement_detection(self) -> None:
        """Should detect excitement keywords."""
        result = execute_cell("empathy", "This is amazing! I love it!")
        assert result["detected_emotion"] == "excitement"

    def test_neutral_fallback(self) -> None:
        """No emotional keywords should default to neutral."""
        result = execute_cell("empathy", "The system processes data")
        assert result["detected_emotion"] == "neutral"

    def test_confidence_bounded(self, sample_text: str) -> None:
        """Confidence should be in [0, 1]."""
        result = execute_cell("empathy", sample_text)
        assert 0 <= result["confidence"] <= 1.0


class TestCuriosityCell:
    """Tests for the curiosity cell."""

    def test_generates_questions(self, sample_text: str) -> None:
        """Should generate topic-aware questions."""
        result = execute_cell("curiosity", sample_text)
        assert "questions" in result
        assert len(result["questions"]) >= 1

    def test_questions_reference_topic(self, sample_text: str) -> None:
        """Questions should reference words from the input."""
        result = execute_cell("curiosity", "Security vulnerability analysis")
        questions_text = " ".join(result["questions"])
        assert any(w in questions_text.lower() for w in ["security", "vulnerability", "analysis"])

    def test_max_questions_config(self, sample_text: str) -> None:
        """max_questions config should limit output."""
        result = execute_cell("curiosity", sample_text, config_json='{"max_questions": 2}')
        assert len(result["questions"]) <= 2

    def test_novelty_score(self, sample_text: str) -> None:
        """Should return a novelty score."""
        result = execute_cell("curiosity", sample_text)
        assert 0 <= result["novelty_score"] <= 1.0


class TestFigLiteralCell:
    """Tests for the figurative/literal cell."""

    def test_figurative_detection(self, figurative_text: str) -> None:
        """Should classify figurative language correctly."""
        result = execute_cell("figliteral", figurative_text)
        assert result["classification"] == "figurative"

    def test_literal_detection(self) -> None:
        """Plain text should be classified as literal."""
        result = execute_cell("figliteral", "The server runs on port 8080")
        assert result["classification"] == "literal"

    def test_literal_interpretation(self) -> None:
        """Known idioms should have literal interpretations."""
        result = execute_cell("figliteral", "This task is a piece of cake")
        assert "literal_interpretation" in result
        assert len(result["literal_interpretation"]) > 5


class TestAggregatorCell:
    """Tests for the aggregator cell."""

    def test_aggregates_sub_cells(self, sample_text: str) -> None:
        """Should execute and aggregate sub-cells."""
        result = execute_cell("aggregator", sample_text)
        assert result["cells_aggregated"] >= 1
        assert len(result["sub_cell_results"]) >= 1

    def test_weighted_average_strategy(self, sample_text: str) -> None:
        """Default strategy should be weighted_average."""
        result = execute_cell("aggregator", sample_text)
        assert result["strategy"] == "weighted_average"

    def test_custom_strategy(self, sample_text: str) -> None:
        """max_confidence strategy should work."""
        result = execute_cell(
            "aggregator",
            sample_text,
            config_json='{"strategy": "max_confidence"}',
        )
        assert result["strategy"] == "max_confidence"

    def test_real_aggregated_output(self, sample_text: str) -> None:
        """Output should be real, not placeholder."""
        result = execute_cell("aggregator", sample_text)
        assert "Aggregated" in result["aggregated_output"]
        assert result["confidence"] > 0


class TestTelemetryCell:
    """Tests for the telemetry cell."""

    def test_event_recorded(self, sample_text: str) -> None:
        """Should record a telemetry event."""
        result = execute_cell("telemetry", sample_text)
        assert result["event_recorded"] is True
        assert "trace_id" in result

    def test_action_extraction(self) -> None:
        """Should extract action verbs from text."""
        result = execute_cell("telemetry", "User clicked the submit button on the login page")
        attrs = result.get("parsed_attributes", {})
        assert attrs.get("action") == "click"

    def test_duration_extraction(self) -> None:
        """Should extract numeric durations."""
        result = execute_cell("telemetry", "The operation took 5 seconds")
        attrs = result.get("parsed_attributes", {})
        assert attrs.get("duration_value") == 5

    def test_page_extraction(self) -> None:
        """Should extract page references."""
        result = execute_cell("telemetry", "User navigated on the dashboard page")
        attrs = result.get("parsed_attributes", {})
        assert attrs.get("page") == "dashboard"


class TestCausalCell:
    """Tests for the causal inference cell."""

    def test_causal_effect(self, sample_text: str) -> None:
        """Should produce a causal effect estimate."""
        result = execute_cell("causal", sample_text)
        assert "causal_effect" in result
        assert isinstance(result["causal_effect"], float)

    def test_confidence_interval(self, sample_text: str) -> None:
        """Should produce a confidence interval."""
        result = execute_cell("causal", sample_text)
        ci = result["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] < ci[1]


class TestR2PCell:
    """Tests for the requirements-to-plan cell."""

    def test_generates_plan(self, sample_text: str) -> None:
        """Should generate a structured plan."""
        result = execute_cell("r2p", sample_text)
        assert "plan" in result
        assert len(result["plan"]) >= 3

    def test_steps_have_effort(self, sample_text: str) -> None:
        """Each step should have estimated effort."""
        result = execute_cell("r2p", sample_text)
        for step in result["plan"]:
            assert "estimated_effort" in step


class TestComposeCells:
    """Tests for multi-cell pipeline composition."""

    def test_pipeline_execution(self, sample_text: str) -> None:
        """Should execute all cells in pipeline."""
        result = compose_cells("reasoning, ethics", sample_text)
        assert result["total_cells"] == 2
        assert len(result["activations"]) == 2

    def test_empty_pipeline(self, sample_text: str) -> None:
        """Empty pipeline should return error."""
        result = compose_cells("", sample_text)
        assert "error" in result

    def test_invalid_cell_in_pipeline(self, sample_text: str) -> None:
        """Unknown cell should be handled gracefully."""
        result = compose_cells("reasoning, unknown_cell, ethics", sample_text)
        assert result["total_cells"] == 3
        assert any(a["status"] == "error" for a in result["activations"])
