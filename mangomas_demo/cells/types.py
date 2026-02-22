"""
Cognitive Cell Type Registry.

Defines the 10 biologically-inspired cell types and their available heads.
"""

from __future__ import annotations

from typing import Any

CELL_TYPES: dict[str, dict[str, Any]] = {
    "reasoning": {
        "name": "ReasoningCell",
        "description": "Structured reasoning with Rule or NN heads",
        "heads": ["rule", "nn"],
    },
    "memory": {
        "name": "MemoryCell",
        "description": "Privacy-preserving preference extraction",
        "heads": ["preference_extractor"],
    },
    "causal": {
        "name": "CausalCell",
        "description": "Pearl's do-calculus for causal inference",
        "heads": ["do_calculus"],
    },
    "ethics": {
        "name": "EthicsCell",
        "description": "Safety classification and PII detection",
        "heads": ["classifier", "pii_scanner"],
    },
    "empathy": {
        "name": "EmpathyCell",
        "description": "Emotional tone detection and empathetic responses",
        "heads": ["tone_detector"],
    },
    "curiosity": {
        "name": "CuriosityCell",
        "description": "Epistemic curiosity and hypothesis generation",
        "heads": ["hypothesis_generator"],
    },
    "figliteral": {
        "name": "FigLiteralCell",
        "description": "Figurative vs literal language classification",
        "heads": ["classifier"],
    },
    "r2p": {
        "name": "R2PCell",
        "description": "Requirements-to-Plan structured decomposition",
        "heads": ["planner"],
    },
    "telemetry": {
        "name": "TelemetryCell",
        "description": "Telemetry event capture and structuring",
        "heads": ["collector"],
    },
    "aggregator": {
        "name": "AggregatorCell",
        "description": "Multi-expert output aggregation",
        "heads": ["weighted_average", "max_confidence", "ensemble"],
    },
}
