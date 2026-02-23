"""
MCTS Benchmark - compare MCTS vs Greedy vs Random strategies.
"""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from mangomas_demo.mcts.engine import TASK_CATEGORIES, _detect_category, run_mcts
from mangomas_demo.models import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from mangomas_demo.models import PolicyNetwork, ValueNetwork


def benchmark_strategies(task: str) -> dict[str, Any]:
    """Compare MCTS vs Greedy vs Random on the same task.

    Args:
        task: Task description string.

    Returns:
        Dict with task, category, and results for each strategy
        (quality_score, best_action, elapsed_ms).
    """
    results: dict[str, Any] = {}
    category = _detect_category(task)
    actions = TASK_CATEGORIES[category]

    # MCTS - full tree search
    start = time.monotonic()
    r = run_mcts(task, max_simulations=100)
    elapsed_mcts = (time.monotonic() - start) * 1000
    results["mcts"] = {
        "quality_score": r["best_value"],
        "best_action": r["best_action"],
        "elapsed_ms": round(elapsed_mcts, 2),
    }

    # Greedy - single-step: pick action with highest policy prior
    start = time.monotonic()
    if TORCH_AVAILABLE:
        policy_net = PolicyNetwork(d_in=128, n_actions=len(actions))
        policy_net.eval()
        torch.manual_seed(hash(task) % (2**31))
        embed = torch.randn(1, 128)
        with torch.no_grad():
            priors = policy_net(embed)[0].numpy()
        best_idx = int(np.argmax(priors))
        greedy_action = actions[best_idx]
        greedy_quality = float(priors[best_idx])
    else:
        greedy_quality = max(random.uniform(0.1, 0.3) for _ in actions)
        greedy_action = random.choice(actions)
    elapsed_greedy = (time.monotonic() - start) * 1000
    results["greedy"] = {
        "quality_score": round(greedy_quality, 3),
        "best_action": greedy_action,
        "elapsed_ms": round(elapsed_greedy, 2),
    }

    # Random - pick random action with random value
    start = time.monotonic()
    random_action = random.choice(actions)
    if TORCH_AVAILABLE:
        value_net = ValueNetwork(d_in=192)
        value_net.eval()
        torch.manual_seed(hash(task + random_action) % (2**31))
        state = torch.randn(1, 192)
        with torch.no_grad():
            random_quality = value_net(state).item()
    else:
        random_quality = random.uniform(-0.5, 0.5)
    elapsed_random = (time.monotonic() - start) * 1000
    results["random"] = {
        "quality_score": round(random_quality, 3),
        "best_action": random_action,
        "elapsed_ms": round(elapsed_random, 2),
    }

    return {"task": task, "category": category, "results": results}
