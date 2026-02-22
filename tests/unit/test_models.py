"""
Unit Tests — Neural Network Models.
"""

from __future__ import annotations

import pytest

from mangomas_demo.models import TORCH_AVAILABLE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestModels:
    """Tests for neural network model shapes and param counts."""

    def test_expert_tower_output_shape(self) -> None:
        """ExpertTower should output (batch, 256)."""
        import torch
        from mangomas_demo.models import ExpertTower

        model = ExpertTower()
        x = torch.randn(1, 64)
        out = model(x)
        assert out.shape == (1, 256)

    def test_moe_7m_param_count(self) -> None:
        """MoE model should have ~7M parameters."""
        from mangomas_demo.models import MixtureOfExperts7M

        model = MixtureOfExperts7M()
        count = model.parameter_count
        # 7M ± 500K tolerance
        assert 6_000_000 < count < 8_000_000, f"Parameter count: {count}"

    def test_moe_7m_output_shape(self) -> None:
        """MoE model should output (logits, gate_weights)."""
        import torch
        from mangomas_demo.models import MixtureOfExperts7M

        model = MixtureOfExperts7M()
        x = torch.randn(1, 64)
        logits, gate_weights = model(x)
        assert logits.shape == (1, 10)
        assert gate_weights.shape == (1, 16)

    def test_router_net_softmax(self) -> None:
        """RouterNet forward should produce valid softmax output."""
        import torch
        from mangomas_demo.models import RouterNet

        model = RouterNet()
        x = torch.randn(1, 64)
        out = model(x)
        assert out.shape == (1, 8)
        assert abs(out.sum().item() - 1.0) < 0.01

    def test_policy_network_output(self) -> None:
        """PolicyNetwork should output action probabilities."""
        import torch
        from mangomas_demo.models import PolicyNetwork

        model = PolicyNetwork(d_in=128, n_actions=32)
        x = torch.randn(1, 128)
        out = model(x)
        assert out.shape == (1, 32)
        assert abs(out.sum().item() - 1.0) < 0.01

    def test_value_network_output(self) -> None:
        """ValueNetwork should output scalar in [-1, 1]."""
        import torch
        from mangomas_demo.models import ValueNetwork

        model = ValueNetwork()
        x = torch.randn(1, 192)
        out = model(x)
        assert out.shape == (1, 1)
        assert -1.0 <= out.item() <= 1.0
