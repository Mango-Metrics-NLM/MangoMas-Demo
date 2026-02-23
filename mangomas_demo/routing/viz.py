"""
Routing Visualization — expert weight bar charts.
"""

from __future__ import annotations

import plotly.graph_objects as go


def plot_expert_weights(weights: dict[str, float]) -> go.Figure:
    """Create a bar chart of expert routing weights.

    Args:
        weights: Dict mapping expert name → routing weight.

    Returns:
        A Plotly Figure object.
    """
    names = list(weights.keys())
    vals = list(weights.values())
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#F0E68C",
        "#87CEEB",
    ]
    fig = go.Figure(
        data=[go.Bar(x=names, y=vals, marker_color=colors[: len(names)])],
        layout=go.Layout(
            title="Expert Routing Weights",
            yaxis=dict(title="Weight (softmax)", range=[0, max(vals) * 1.2]),
            height=350,
            template="plotly_dark",
            margin=dict(t=40),
        ),
    )
    return fig
