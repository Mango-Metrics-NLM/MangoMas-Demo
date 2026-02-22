"""
Neural Network Models for MangoMAS.

Provides the five core NN components:
- ExpertTower: Single expert in the MoE ensemble
- MixtureOfExperts7M: ~7M-param gated MoE model
- RouterNet: Fast MLP routing gate
- PolicyNetwork: MCTS policy head
- ValueNetwork: MCTS value head

All models gracefully fall back to object stubs when PyTorch is unavailable.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Torch import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except ImportError:
    _TORCH = False

# Re-export for use by other modules
TORCH_AVAILABLE = _TORCH


class ExpertTower(nn.Module if _TORCH else object):  # type: ignore[misc]
    """Single expert tower: 64 → 512 → 512 → 256."""

    def __init__(self, d_in: int = 64, h1: int = 512, h2: int = 512, d_out: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, d_out)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the expert tower."""
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class MixtureOfExperts7M(nn.Module if _TORCH else object):  # type: ignore[misc]
    """
    ~7M parameter Mixture-of-Experts model.

    Architecture:
    - Gating network: 64 → 512 → N_experts (softmax)
    - Expert towers (×N): 64 → 512 → 512 → 256
    - Classifier head: 256 → N_classes
    """

    def __init__(self, num_classes: int = 10, num_experts: int = 16) -> None:
        super().__init__()
        self.num_experts = num_experts

        # Gating network
        self.gate_fc1 = nn.Linear(64, 512)
        self.gate_fc2 = nn.Linear(512, num_experts)

        # Expert towers
        self.experts = nn.ModuleList([ExpertTower() for _ in range(num_experts)])

        # Classifier head
        self.classifier = nn.Linear(256, num_classes)

    @property
    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x64: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
        """Forward pass: returns (logits, gate_weights)."""
        # Gating
        gate = F.relu(self.gate_fc1(x64))
        gate_weights = torch.softmax(self.gate_fc2(gate), dim=-1)

        # Expert outputs
        expert_outs = torch.stack([e(x64) for e in self.experts], dim=1)

        # Weighted aggregation
        agg = torch.sum(expert_outs * gate_weights.unsqueeze(-1), dim=1)

        # Classifier
        logits = self.classifier(agg)
        return logits, gate_weights


class RouterNet(nn.Module if _TORCH else object):  # type: ignore[misc]
    """
    Neural routing gate MLP: 64 → 128 → 64 → N_experts.

    Used for fast (~0.8ms) expert selection via softmax output.
    """

    EXPERTS = [
        "code_expert", "test_expert", "design_expert", "research_expert",
        "architecture_expert", "security_expert", "performance_expert",
        "documentation_expert",
    ]

    def __init__(self, d_in: int = 64, d_h: int = 128, n_out: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Linear(d_h // 2, n_out),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass returning softmax probabilities."""
        return torch.softmax(self.net(x), dim=-1)


class PolicyNetwork(nn.Module if _TORCH else object):  # type: ignore[misc]
    """MCTS policy network: 128 → 256 → 128 → N_actions."""

    def __init__(self, d_in: int = 128, n_actions: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions), nn.Softmax(dim=-1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass returning action probabilities."""
        return self.net(x)


class ValueNetwork(nn.Module if _TORCH else object):  # type: ignore[misc]
    """MCTS value network: 192 → 256 → 64 → 1 (tanh)."""

    def __init__(self, d_in: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass returning scalar value estimate in [-1, 1]."""
        return self.net(x)
