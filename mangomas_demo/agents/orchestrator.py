"""
Agent Orchestration — Multi-agent task execution with learned routing.

Supports three routing strategies:
- moe_routing: Neural MoE gate selects top-K agents
- round_robin: Sequential agent selection
- random: Random agent selection
"""

from __future__ import annotations

import random as _rnd
import time
from typing import Any

from mangomas_demo.cells.executor import execute_cell
from mangomas_demo.routing.router import route_task

AGENTS: list[dict[str, str]] = [
    {"name": "SWE Agent", "specialization": "Code scaffold generation", "icon": "[SWE]"},
    {"name": "Architect Agent", "specialization": "System design and patterns", "icon": "[ARCH]"},
    {"name": "QA Agent", "specialization": "Test plan and case generation", "icon": "[QA]"},
    {"name": "Security Agent", "specialization": "Threat modeling (OWASP)", "icon": "[SEC]"},
    {"name": "DevOps Agent", "specialization": "Infrastructure planning", "icon": "[OPS]"},
    {"name": "Research Agent", "specialization": "Technical analysis", "icon": "[RES]"},
    {"name": "Performance Agent", "specialization": "Optimization analysis", "icon": "[PERF]"},
    {"name": "Documentation Agent", "specialization": "Technical writing", "icon": "[DOC]"},
]

# Agent-to-cell mapping for real processing
_AGENT_CELL_MAP: dict[str, str] = {
    "SWE Agent": "reasoning",
    "Architect Agent": "r2p",
    "QA Agent": "reasoning",
    "Security Agent": "ethics",
    "DevOps Agent": "telemetry",
    "Research Agent": "causal",
    "Performance Agent": "reasoning",
    "Documentation Agent": "reasoning",
}


def orchestrate(
    task: str, max_agents: int = 3, strategy: str = "moe_routing"
) -> dict[str, Any]:
    """Orchestrate multiple agents for a task using specified routing strategy.

    Args:
        task: Task description string.
        max_agents: Maximum number of agents to select (1–8).
        strategy: Routing strategy — 'moe_routing', 'round_robin', or 'random'.

    Returns:
        Dict with task, strategy, agents_selected, max_agents, results,
        and total_elapsed_ms.
    """
    start = time.monotonic()

    if strategy == "round_robin":
        agent_results = _orchestrate_round_robin(task, max_agents)
    elif strategy == "random":
        agent_results = _orchestrate_random(task, max_agents)
    else:  # moe_routing (default)
        agent_results = _orchestrate_moe(task, max_agents)

    elapsed = (time.monotonic() - start) * 1000

    return {
        "task": task,
        "strategy": strategy,
        "agents_selected": len(agent_results),
        "max_agents": max_agents,
        "results": agent_results,
        "total_elapsed_ms": round(elapsed, 2),
    }


def _orchestrate_round_robin(
    task: str, max_agents: int
) -> list[dict[str, Any]]:
    """Select agents in round-robin order."""
    selected_agents = AGENTS[:max_agents]
    results: list[dict[str, Any]] = []
    for agent in selected_agents:
        cell_type = _AGENT_CELL_MAP.get(agent["name"], "reasoning")
        cell_result = execute_cell(cell_type, task)
        results.append({
            "agent": agent["name"],
            "icon": agent["icon"],
            "specialization": agent["specialization"],
            "weight": round(1.0 / max_agents, 4),
            "cell_used": cell_type,
            "output": cell_result,
            "confidence": cell_result.get("confidence", round(0.8, 3)),
        })
    return results


def _orchestrate_random(
    task: str, max_agents: int
) -> list[dict[str, Any]]:
    """Randomly select agents."""
    shuffled = _rnd.sample(AGENTS, min(max_agents, len(AGENTS)))
    results: list[dict[str, Any]] = []
    for agent in shuffled:
        cell_type = _AGENT_CELL_MAP.get(agent["name"], "reasoning")
        cell_result = execute_cell(cell_type, task)
        results.append({
            "agent": agent["name"],
            "icon": agent["icon"],
            "specialization": agent["specialization"],
            "weight": round(1.0 / max_agents, 4),
            "cell_used": cell_type,
            "output": cell_result,
            "confidence": cell_result.get("confidence", round(0.8, 3)),
        })
    return results


def _orchestrate_moe(
    task: str, max_agents: int
) -> list[dict[str, Any]]:
    """Use MoE neural routing to select agents."""
    routing = route_task(task, top_k=max_agents)
    results: list[dict[str, Any]] = []
    for expert in routing["selected_experts"]:
        agent_name = expert["expert"].replace(" Expert", " Agent")
        agent = next((a for a in AGENTS if agent_name in a["name"]), AGENTS[0])
        cell_type = _AGENT_CELL_MAP.get(agent["name"], "reasoning")
        cell_result = execute_cell(cell_type, task)
        results.append({
            "agent": agent["name"],
            "icon": agent["icon"],
            "specialization": agent["specialization"],
            "weight": expert["weight"],
            "cell_used": cell_type,
            "output": cell_result,
            "confidence": cell_result.get("confidence", round(0.8, 3)),
        })
    return results
