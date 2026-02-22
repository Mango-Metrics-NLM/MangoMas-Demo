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
# Logging — auto-configure on import
# ---------------------------------------------------------------------------
from mangomas_demo.logging_config import configure_logging, get_logger

configure_logging()

# ---------------------------------------------------------------------------
# Public re-exports for convenient top-level access
# ---------------------------------------------------------------------------
from mangomas_demo.agents.orchestrator import orchestrate  # noqa: E402
from mangomas_demo.cells import compose_cells, execute_cell  # noqa: E402
from mangomas_demo.features import featurize64, plot_features  # noqa: E402
from mangomas_demo.mcts.benchmark import benchmark_strategies  # noqa: E402
from mangomas_demo.mcts.engine import MCTSNode, run_mcts  # noqa: E402
from mangomas_demo.routing.router import route_task  # noqa: E402

__all__ = [
    "MCTSNode",
    "benchmark_strategies",
    "compose_cells",
    "configure_logging",
    "execute_cell",
    "featurize64",
    "get_logger",
    "orchestrate",
    "plot_features",
    "route_task",
    "run_mcts",
]
