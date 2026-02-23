"""
MCTS Visualization — Plotly sunburst chart of the MCTS search tree.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def plot_mcts_tree(tree_data: dict[str, Any]) -> go.Figure:
    """Create a sunburst visualization of the MCTS tree.

    Args:
        tree_data: Serialized MCTSNode dict from ``MCTSNode.to_dict()``.

    Returns:
        A Plotly Figure with a sunburst chart.
    """
    ids: list[str] = []
    labels: list[str] = []
    parents: list[str] = []
    values: list[int] = []
    colors: list[float] = []

    def _walk(node: dict[str, Any], parent_id: str = "") -> None:
        nid = node["id"]
        ids.append(nid)
        labels.append(f"{node['action']}\n(v={node.get('value', 0)}, n={node.get('visits', 0)})")
        parents.append(parent_id)
        values.append(max(node.get("visits", 1), 1))
        colors.append(node.get("value", 0))
        for child in node.get("children", []):
            _walk(child, nid)

    _walk(tree_data)

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors, colorscale="Viridis", showscale=True),
            branchvalues="total",
        )
    )
    fig.update_layout(
        title="MCTS Search Tree",
        height=500,
        template="plotly_dark",
        margin=dict(t=40, l=0, r=0, b=0),
    )
    return fig
