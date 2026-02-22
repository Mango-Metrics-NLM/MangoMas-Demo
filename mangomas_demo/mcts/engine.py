"""
MCTS Planning Engine — Monte Carlo Tree Search with NN priors.

Implements UCB1 and PUCT selection strategies with optional
policy/value neural network integration.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

from mangomas_demo.models import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from mangomas_demo.models import PolicyNetwork, ValueNetwork

TASK_CATEGORIES: dict[str, list[str]] = {
    "architecture": ["service_split", "api_gateway", "data_layer", "security_layer", "caching"],
    "implementation": ["requirements", "design", "code", "test", "deploy"],
    "optimization": ["profile", "identify_bottleneck", "optimize", "validate", "benchmark"],
    "security": ["asset_inventory", "threat_enumeration", "risk_scoring", "mitigations", "audit"],
    "research": ["literature_review", "comparison", "synthesis", "recommendations", "publish"],
}

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "architecture": ["architect", "design", "micro", "system"],
    "security": ["security", "threat", "vulnerability", "attack"],
    "optimization": ["optimize", "performance", "latency", "speed"],
    "research": ["research", "survey", "study", "analyze"],
}


def _detect_category(task: str) -> str:
    """Detect task category from keywords. Defaults to 'implementation'."""
    lower = task.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return cat
    return "implementation"


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""

    id: str
    action: str
    visits: int = 0
    total_value: float = 0.0
    policy_prior: float = 0.0
    children: list[MCTSNode] | None = None

    def ucb1_score(self, parent_visits: int, c: float = 1.414) -> float:
        """Upper Confidence Bound 1 score."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_value / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def puct_score(self, parent_visits: int, c: float = 1.0) -> float:
        """Predictor + UCB for Trees score (AlphaZero-style)."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_value / self.visits
        exploration = c * self.policy_prior * math.sqrt(parent_visits) / (1 + self.visits)
        return exploitation + exploration

    def to_dict(self, max_depth: int = 3) -> dict[str, Any]:
        """Serialize to dict for JSON output. Limits depth to prevent large payloads."""
        d: dict[str, Any] = {
            "id": self.id,
            "action": self.action,
            "visits": self.visits,
            "value": round(self.total_value / max(self.visits, 1), 3),
            "policy_prior": round(self.policy_prior, 3),
        }
        if self.children and max_depth > 0:
            d["children"] = [
                c.to_dict(max_depth - 1)
                for c in sorted(self.children, key=lambda n: -n.visits)[:5]
            ]
        return d


def run_mcts(
    task: str,
    max_simulations: int = 100,
    exploration_constant: float = 1.414,
    strategy: str = "ucb1",
) -> dict[str, Any]:
    """Run MCTS planning on a task and return the search tree.

    Args:
        task: Task description string.
        max_simulations: Number of MCTS rollouts (10–500).
        exploration_constant: UCB1/PUCT exploration constant.
        strategy: Selection strategy — 'ucb1' or 'puct'.

    Returns:
        Dict with task, category, best_action, best_value, tree, all_actions,
        elapsed_ms, and nn_enabled.
    """
    start = time.monotonic()
    category = _detect_category(task)
    actions = TASK_CATEGORIES[category]

    # Build tree
    root = MCTSNode(id="root", action=task[:50], children=[])

    # Use NN priors if torch available
    if TORCH_AVAILABLE:
        policy_net = PolicyNetwork(d_in=128, n_actions=len(actions))
        value_net = ValueNetwork(d_in=192)
        policy_net.eval()
        value_net.eval()

    for sim in range(max_simulations):
        node = root

        # EXPAND: add children if needed
        if not node.children:
            node.children = []
            for i, act in enumerate(actions):
                prior = random.uniform(0.1, 0.5)
                if TORCH_AVAILABLE:
                    embed = torch.randn(1, 128)
                    with torch.no_grad():
                        priors = policy_net(embed)[0]
                    prior = priors[i % len(priors)].item()
                node.children.append(
                    MCTSNode(
                        id=f"{act}-{sim}",
                        action=act,
                        policy_prior=prior,
                        children=[],
                    )
                )

        # Select best child
        score_fn = (
            (lambda n: n.ucb1_score(root.visits + 1, exploration_constant))
            if strategy == "ucb1"
            else (lambda n: n.puct_score(root.visits + 1, exploration_constant))
        )
        best_child = max(node.children, key=score_fn)

        # SIMULATE: get value estimate
        if TORCH_AVAILABLE:
            state = torch.randn(1, 192)
            with torch.no_grad():
                value = value_net(state).item()
        else:
            value = random.uniform(0.3, 0.9)

        # BACKPROPAGATE
        best_child.visits += 1
        best_child.total_value += value
        root.visits += 1

    elapsed = (time.monotonic() - start) * 1000

    # Best plan
    if root.children:
        best = max(root.children, key=lambda n: n.visits)
        best_action = best.action
        best_value = round(best.total_value / max(best.visits, 1), 3)
    else:
        best_action = "none"
        best_value = 0.0

    return {
        "task": task,
        "category": category,
        "strategy": strategy,
        "best_action": best_action,
        "best_value": best_value,
        "total_simulations": max_simulations,
        "exploration_constant": exploration_constant,
        "tree": root.to_dict(max_depth=2),
        "all_actions": [
            {
                "action": c.action,
                "visits": c.visits,
                "value": round(c.total_value / max(c.visits, 1), 3),
            }
            for c in sorted(root.children or [], key=lambda n: -n.visits)
        ],
        "elapsed_ms": round(elapsed, 2),
        "nn_enabled": TORCH_AVAILABLE,
    }
