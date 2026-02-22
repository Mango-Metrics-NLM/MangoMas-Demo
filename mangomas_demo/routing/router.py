"""
MoE Neural Router — deterministic routing via singleton RouterNet.

Routes tasks to the top-K most relevant expert agents using a
64-dim feature vector and keyword-based semantic boosting.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mangomas_demo.features import featurize64
from mangomas_demo.models import TORCH_AVAILABLE, RouterNet

if TORCH_AVAILABLE:
    import torch

EXPERT_NAMES: list[str] = [
    "Code Expert", "Test Expert", "Design Expert", "Research Expert",
    "Architecture Expert", "Security Expert", "Performance Expert", "Docs Expert",
]

# Singleton router with fixed seed for deterministic routing
_ROUTER_SEED = 42
_router_net_singleton: RouterNet | None = None


def _get_router() -> RouterNet:
    """Get or create the singleton RouterNet with a fixed seed."""
    global _router_net_singleton
    if _router_net_singleton is None and TORCH_AVAILABLE:
        torch.manual_seed(_ROUTER_SEED)
        _router_net_singleton = RouterNet(d_in=64, n_out=len(EXPERT_NAMES))
        _router_net_singleton.eval()
    return _router_net_singleton  # type: ignore[return-value]


# Keyword → expert index mapping for semantic routing boost
_EXPERT_KEYWORDS: dict[int, list[str]] = {
    0: ["code", "implement", "function", "class", "program", "script", "module"],
    1: ["test", "unit test", "coverage", "qa", "assert", "mock", "fixture"],
    2: ["design", "ui", "ux", "layout", "wireframe", "mockup", "style"],
    3: ["research", "analyze", "study", "survey", "literature", "paper", "compare"],
    4: ["architect", "system", "microservice", "scale", "pattern", "infrastructure"],
    5: ["security", "auth", "encrypt", "threat", "vulnerab", "owasp", "pci", "compliance"],
    6: ["performance", "optimize", "latency", "throughput", "cache", "speed", "fast"],
    7: ["document", "readme", "docs", "comment", "explain", "write", "manual"],
}


def route_task(task: str, top_k: int = 3) -> dict[str, Any]:
    """Route a task through the neural MoE gate.

    Args:
        task: Task description string.
        top_k: Number of top experts to select (1–8).

    Returns:
        Dict with task, features, all_weights, selected_experts, top_k,
        nn_enabled, and elapsed_ms.
    """
    start = time.monotonic()

    features = featurize64(task)

    if TORCH_AVAILABLE:
        router = _get_router()
        feature_tensor = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            weights = router(feature_tensor)[0].numpy()
    else:
        # Fallback: deterministic routing from features
        weights = np.array([abs(f) for f in features[: len(EXPERT_NAMES)]])
        weights = weights / (weights.sum() + 1e-8)

    # Apply keyword-based semantic boost to expert routing
    lower_task = task.lower()
    boost = np.zeros(len(EXPERT_NAMES))
    for idx, kws in _EXPERT_KEYWORDS.items():
        for kw in kws:
            if kw in lower_task:
                boost[idx] += 0.15
    weights = weights + boost
    weights = weights / (weights.sum() + 1e-8)

    # Top-K selection
    top_indices = np.argsort(weights)[::-1][:top_k]
    selected = [
        {
            "expert": EXPERT_NAMES[i],
            "weight": round(float(weights[i]), 4),
            "rank": rank + 1,
        }
        for rank, i in enumerate(top_indices)
    ]

    elapsed = (time.monotonic() - start) * 1000

    return {
        "task": task,
        "features": features,
        "all_weights": {
            EXPERT_NAMES[i]: round(float(weights[i]), 4)
            for i in range(len(EXPERT_NAMES))
        },
        "selected_experts": selected,
        "top_k": top_k,
        "nn_enabled": TORCH_AVAILABLE,
        "elapsed_ms": round(elapsed, 2),
    }
