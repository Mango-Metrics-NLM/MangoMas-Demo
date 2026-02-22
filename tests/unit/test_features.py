"""
Unit Tests — Feature Extraction (featurize64).
"""

from __future__ import annotations

import math

from mangomas_demo.features import featurize64, plot_features


class TestFeaturize64:
    """Tests for the 64-dimensional feature extraction function."""

    def test_output_length(self, sample_text: str) -> None:
        """Should return exactly 64 features."""
        features = featurize64(sample_text)
        assert len(features) == 64

    def test_unit_normalized(self, sample_text: str) -> None:
        """Feature vector should be L2-normalized to unit length."""
        features = featurize64(sample_text)
        norm = math.sqrt(sum(f * f for f in features))
        assert abs(norm - 1.0) < 0.01, f"Norm was {norm}, expected ~1.0"

    def test_deterministic(self, sample_text: str) -> None:
        """Same input should produce identical features."""
        f1 = featurize64(sample_text)
        f2 = featurize64(sample_text)
        assert f1 == f2

    def test_different_inputs_different_features(self) -> None:
        """Different inputs should produce different features."""
        f1 = featurize64("hello world")
        f2 = featurize64("design a microservice")
        assert f1 != f2

    def test_domain_tags_detected(self) -> None:
        """Domain keywords should set tag features to non-zero."""
        features = featurize64("code security architecture test")
        # Tags are at indices 32-47 — code, security, architecture, test should activate
        assert features[32] > 0  # "code" tag
        assert features[36] > 0  # "security" tag

    def test_empty_text_handling(self) -> None:
        """Empty text should still produce 64-dim vector (no crash)."""
        features = featurize64("")
        assert len(features) == 64

    def test_all_floats(self, sample_text: str) -> None:
        """All features should be float values."""
        features = featurize64(sample_text)
        assert all(isinstance(f, float) for f in features)

    def test_values_bounded(self, sample_text: str) -> None:
        """Normalized features should be in [-1, 1] range."""
        features = featurize64(sample_text)
        assert all(-1.5 <= f <= 1.5 for f in features)

    def test_long_text(self) -> None:
        """Very long text should not crash or exceed 64 dims."""
        features = featurize64("word " * 10000)
        assert len(features) == 64

    def test_special_characters(self) -> None:
        """Text with special chars should be handled."""
        features = featurize64("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert len(features) == 64


class TestPlotFeatures:
    """Tests for the feature visualization function."""

    def test_returns_figure(self, sample_text: str) -> None:
        """Should return a Plotly Figure."""
        import plotly.graph_objects as go

        features = featurize64(sample_text)
        fig = plot_features(features)
        assert isinstance(fig, go.Figure)

    def test_custom_title(self) -> None:
        """Custom title should be reflected in the figure."""
        features = featurize64("test")
        fig = plot_features(features, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"
