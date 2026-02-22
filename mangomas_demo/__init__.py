"""
MangoMAS Demo — Multi-Agent Cognitive Architecture.

A production-grade interactive demo showcasing:
- 10 Cognitive Cells with NN heads
- MCTS Planning with policy/value networks
- 7M-param MoE Neural Router
- Multi-agent orchestration

Author: MangoMAS Engineering (Ian Cruickshank)
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Ian Cruickshank"

# ---------------------------------------------------------------------------
# Public re-exports for convenient top-level access
# ---------------------------------------------------------------------------
from mangomas_demo.features import featurize64, plot_features
from mangomas_demo.cells import execute_cell, compose_cells
from mangomas_demo.mcts.engine import run_mcts, MCTSNode
from mangomas_demo.mcts.benchmark import benchmark_strategies
from mangomas_demo.routing.router import route_task
from mangomas_demo.agents.orchestrator import orchestrate

__all__ = [
    "featurize64",
    "plot_features",
    "execute_cell",
    "compose_cells",
    "run_mcts",
    "MCTSNode",
    "benchmark_strategies",
    "route_task",
    "orchestrate",
]
