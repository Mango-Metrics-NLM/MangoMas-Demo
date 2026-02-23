"""
Feature Engineering - 64-dimensional vector extraction.

Combines hash-based sinusoidal, domain-tag, structural, sentiment,
and novelty/complexity signals into a unit-normalized feature vector.
"""

from __future__ import annotations

import hashlib
import math

import plotly.graph_objects as go

from mangomas_demo.logging_config import get_logger

_log = get_logger("features")


def featurize64(text: str) -> list[float]:
    """
    Extract a deterministic 64-dimensional feature vector from text.

    Combines:
    - 32 hash-based sinusoidal features (content fingerprint)
    - 16 domain-tag signals (code, security, architecture, data, etc.)
    - 8 structural signals (length, punctuation, questions, etc.)
    - 4 sentiment polarity estimates
    - 4 novelty/complexity scores

    Args:
        text: Input text string to featurize.

    Returns:
        A list of 64 floats, L2-normalized to unit length.
    """
    features: list[float] = []

    # 1. Hash-based sinusoidal features (32 dims)
    h = hashlib.sha256(text.encode()).hexdigest()
    for i in range(32):
        byte_val = int(h[i * 2 : i * 2 + 2], 16) / 255.0
        features.append(math.sin(byte_val * math.pi * (i + 1)))

    # 2. Domain tag signals (16 dims)
    lower = text.lower()
    domain_tags = [
        "code",
        "function",
        "class",
        "api",
        "security",
        "threat",
        "architecture",
        "design",
        "data",
        "database",
        "test",
        "deploy",
        "optimize",
        "performance",
        "research",
        "analyze",
    ]
    for tag in domain_tags:
        features.append(1.0 if tag in lower else 0.0)

    # 3. Structural signals (8 dims)
    features.append(min(len(text) / 500.0, 1.0))  # length
    features.append(text.count(".") / max(len(text), 1) * 10)  # period density
    features.append(text.count("?") / max(len(text), 1) * 10)  # question density
    features.append(text.count("!") / max(len(text), 1) * 10)  # exclamation density
    features.append(text.count(",") / max(len(text), 1) * 10)  # comma density
    features.append(len(text.split()) / 100.0)  # word count normalized
    features.append(1.0 if any(c.isupper() for c in text) else 0.0)  # has uppercase
    features.append(sum(1 for c in text if c.isdigit()) / max(len(text), 1))

    # 4. Sentiment polarity (4 dims)
    pos_words = ["good", "great", "excellent", "improve", "best", "optimize"]
    neg_words = ["bad", "fail", "error", "bug", "crash", "threat"]
    features.append(sum(1 for w in pos_words if w in lower) / len(pos_words))
    features.append(sum(1 for w in neg_words if w in lower) / len(neg_words))
    features.append(0.5)  # neutral baseline
    features.append(abs(features[-3] - features[-2]))  # polarity distance

    # 5. Novelty/complexity (4 dims)
    unique_words = len(set(text.lower().split()))
    total_words = max(len(text.split()), 1)
    features.append(unique_words / total_words)  # lexical diversity
    features.append(min(len(text.split("\n")) / 10.0, 1.0))  # line count
    features.append(text.count("(") / max(len(text), 1) * 20)  # nesting
    features.append(min(max(len(w) for w in text.split()) / 20.0, 1.0) if text.strip() else 0.0)

    # Normalize to unit vector
    norm = math.sqrt(sum(f * f for f in features)) + 1e-8
    normalized = [f / norm for f in features]
    _log.debug(
        "featurize64 produced %d-dim vector (norm=%.4f) for input length %d",
        len(normalized),
        norm,
        len(text),
    )
    return normalized


def plot_features(features: list[float], title: str = "64-D Feature Vector") -> go.Figure:
    """Create a Plotly bar chart of the 64-dim feature vector.

    Args:
        features: List of 64 floats (the feature vector).
        title: Chart title string.

    Returns:
        A Plotly Figure object.
    """
    labels = (
        [f"hash_{i}" for i in range(32)]
        + [
            f"tag_{t}"
            for t in [
                "code",
                "func",
                "class",
                "api",
                "sec",
                "threat",
                "arch",
                "design",
                "data",
                "db",
                "test",
                "deploy",
                "opt",
                "perf",
                "research",
                "analyze",
            ]
        ]
        + [f"struct_{i}" for i in range(8)]
        + [f"sent_{i}" for i in range(4)]
        + [f"novel_{i}" for i in range(4)]
    )
    colors = (
        ["#FF6B6B"] * 32 + ["#4ECDC4"] * 16 + ["#45B7D1"] * 8 + ["#96CEB4"] * 4 + ["#FFEAA7"] * 4
    )
    fig = go.Figure(
        data=[go.Bar(x=labels, y=features, marker_color=colors)],
        layout=go.Layout(
            title=title,
            xaxis=dict(title="Feature Dimension", tickangle=-45, tickfont=dict(size=7)),
            yaxis=dict(title="Value"),
            height=350,
            template="plotly_dark",
            margin=dict(b=120),
        ),
    )
    return fig
