"""
Cognitive Cell Executor — dispatch and execution logic for all 10 cell types.

Each cell follows the lifecycle: validate → infer → structure → return.
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from typing import Any

from mangomas_demo.cells.types import CELL_TYPES


def execute_cell(cell_type: str, text: str, config_json: str = "{}") -> dict[str, Any]:
    """Execute a cognitive cell and return structured results.

    Args:
        cell_type: One of the 10 registered cell types.
        text: Input text to process.
        config_json: Optional JSON string with cell-specific config.

    Returns:
        Dict with cell_type, request_id, status, elapsed_ms, and cell-specific fields.
    """
    start = time.monotonic()

    # Validate empty input
    if not text or not text.strip():
        return {
            "cell_type": cell_type,
            "status": "error",
            "message": "Input text is required. Please provide some text to process.",
            "elapsed_ms": 0.0,
        }

    # Parse config JSON
    try:
        config = json.loads(config_json) if config_json.strip() else {}
    except json.JSONDecodeError as e:
        return {
            "cell_type": cell_type,
            "status": "error",
            "message": f"Invalid JSON config: {e}",
            "elapsed_ms": 0.0,
        }

    request_id = f"req-{uuid.uuid4().hex[:12]}"

    result: dict[str, Any] = {
        "cell_type": cell_type,
        "request_id": request_id,
        "status": "ok",
    }

    # Dispatch to cell-specific logic
    if cell_type == "reasoning":
        _execute_reasoning(result, text, config)
    elif cell_type == "memory":
        _execute_memory(result, text)
    elif cell_type == "causal":
        _execute_causal(result, text, config)
    elif cell_type == "ethics":
        _execute_ethics(result, text)
    elif cell_type == "empathy":
        _execute_empathy(result, text)
    elif cell_type == "curiosity":
        _execute_curiosity(result, text, config)
    elif cell_type == "figliteral":
        _execute_figliteral(result, text)
    elif cell_type == "r2p":
        _execute_r2p(result)
    elif cell_type == "telemetry":
        _execute_telemetry(result, text)
    elif cell_type == "aggregator":
        _execute_aggregator(result, text, config)

    elapsed = (time.monotonic() - start) * 1000
    result["elapsed_ms"] = round(elapsed, 2)
    return result


def compose_cells(pipeline_str: str, text: str) -> dict[str, Any]:
    """Execute a pipeline of cells sequentially.

    Args:
        pipeline_str: Comma-separated list of cell type names.
        text: Input text to process through the pipeline.

    Returns:
        Dict with pipeline, activations, final_output, total_cells, and context_keys.
    """
    cell_types = [c.strip() for c in pipeline_str.split(",") if c.strip()]
    if not cell_types:
        return {"error": "No cell types specified"}

    activations: list[dict[str, Any]] = []
    context: dict[str, Any] = {}
    final_output: dict[str, Any] = {}

    for ct in cell_types:
        if ct not in CELL_TYPES:
            activations.append({
                "cell_type": ct,
                "status": "error",
                "message": f"Unknown cell type: {ct}",
            })
            continue
        result = execute_cell(ct, text)
        activations.append({
            "cell_type": ct,
            "status": result.get("status", "ok"),
            "elapsed_ms": result.get("elapsed_ms", 0),
        })
        context.update({
            k: v for k, v in result.items() if k not in ("request_id", "elapsed_ms")
        })
        final_output = result

    return {
        "pipeline": cell_types,
        "activations": activations,
        "final_output": final_output,
        "total_cells": len(cell_types),
        "context_keys": list(context.keys()),
    }


# ---------------------------------------------------------------------------
# Private cell implementations
# ---------------------------------------------------------------------------


def _execute_reasoning(
    result: dict[str, Any], text: str, config: dict[str, Any]
) -> None:
    """Structured reasoning with configurable head type."""
    head = config.get("head_type", "rule")
    words = text.split()
    sections = []
    chunk_size = max(len(words) // 3, 1)
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        sections.append({
            "text": chunk,
            "confidence": round(random.uniform(0.7, 0.99), 3),
            "boundary_type": random.choice(
                ["topic_shift", "elaboration", "conclusion"]
            ),
        })
    result["head_type"] = head
    result["sections"] = sections
    result["section_count"] = len(sections)


def _execute_memory(result: dict[str, Any], text: str) -> None:
    """Privacy-preserving preference extraction."""
    preferences = []
    lower = text.lower()
    if "prefer" in lower or "like" in lower:
        preferences.append({"type": "explicit", "value": text, "confidence": 0.95})
    if "always" in lower or "usually" in lower:
        preferences.append({"type": "implicit", "value": text, "confidence": 0.72})
    result["preferences"] = preferences
    result["opt_out"] = "don't remember" in lower
    result["consent_status"] = "granted"


def _execute_causal(
    result: dict[str, Any], text: str, config: dict[str, Any]
) -> None:
    """Simulated causal inference via do-calculus."""
    result["mode"] = config.get("mode", "do_calculus")
    result["variables"] = [w for w in text.split() if len(w) > 3][:5]
    result["causal_effect"] = round(random.uniform(-0.5, 0.8), 3)
    result["confidence_interval"] = [
        round(result["causal_effect"] - 0.15, 3),
        round(result["causal_effect"] + 0.15, 3),
    ]


def _execute_ethics(result: dict[str, Any], text: str) -> None:
    """Safety classification and PII detection."""
    pii_patterns = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }
    pii_found: list[dict[str, str]] = []
    redacted = text
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, text)
        for m in matches:
            pii_found.append({"type": pii_type, "value": "[REDACTED]"})
            redacted = redacted.replace(m, "[REDACTED]")

    result["is_safe"] = len(pii_found) == 0
    result["pii_detected"] = pii_found
    result["redacted_text"] = redacted
    result["risk_score"] = round(
        random.uniform(0.0, 0.3) if not pii_found else random.uniform(0.6, 0.9), 3
    )


def _execute_empathy(result: dict[str, Any], text: str) -> None:
    """Keyword-based emotion detection and empathetic responses."""
    lower = text.lower()
    emotion_keywords: dict[str, list[str]] = {
        "frustration": [
            "frustrat", "annoy", "angry", "upset", "fail", "broken", "stuck",
            "overwhelm",
        ],
        "anxiety": [
            "worry", "anxious", "nervous", "afraid", "fear", "concern", "stress",
            "uncertain",
        ],
        "excitement": [
            "excit", "amazing", "awesome", "great", "love", "fantastic", "thrilled",
            "happy",
        ],
        "satisfaction": [
            "satisfied", "pleased", "good", "well", "success", "accomplish", "done",
            "complete",
        ],
        "confusion": [
            "confus", "unclear", "don't understand", "what does", "how does", "lost",
            "puzzle",
        ],
    }
    emotion_scores: dict[str, int] = {}
    for emotion, keywords in emotion_keywords.items():
        emotion_scores[emotion] = sum(1 for kw in keywords if kw in lower)
    best_emotion = max(emotion_scores, key=lambda e: emotion_scores[e])
    detected = best_emotion if emotion_scores[best_emotion] > 0 else "neutral"
    confidence = min(0.95, 0.6 + emotion_scores.get(detected, 0) * 0.1)
    responses = {
        "neutral": "I understand your message. How can I help further?",
        "frustration": "I can see this is frustrating. Let me help resolve this.",
        "excitement": "That's great news! Let's build on that momentum.",
        "confusion": "Let me clarify that for you step by step.",
        "satisfaction": "Glad to hear things are going well!",
        "anxiety": "I understand your concern. Let's work through this together.",
    }
    result["detected_emotion"] = detected
    result["confidence"] = round(confidence, 3)
    result["empathetic_response"] = responses[detected]


def _execute_curiosity(
    result: dict[str, Any], text: str, config: dict[str, Any]
) -> None:
    """Topic-aware question generation from extracted keywords."""
    words = [w for w in text.split() if len(w) > 3]
    topics = list(dict.fromkeys(words[:5]))  # unique, order-preserved
    topic_str = ", ".join(topics[:3]) if topics else "this topic"
    questions = [
        f"What are the underlying assumptions behind {topic_str}?",
        f"How would the outcome differ if we changed the approach to "
        f"{topics[0] if topics else 'this'}?",
        "What related problems have been solved in adjacent domains?",
        f"What are the second-order effects of "
        f"{topics[1] if len(topics) > 1 else 'this decision'}?",
        f"What evidence would disprove our current hypothesis about {topic_str}?",
    ]
    max_q = config.get("max_questions", 3)
    result["questions"] = questions[:max_q]
    # Novelty based on lexical diversity
    unique_ratio = len(set(text.lower().split())) / max(len(text.split()), 1)
    result["novelty_score"] = round(min(unique_ratio + 0.3, 0.95), 3)


def _execute_figliteral(result: dict[str, Any], text: str) -> None:
    """Figurative vs literal language classification with idiom decomposition."""
    figurative_map: dict[str, str] = {
        "raining cats and dogs": "raining very heavily",
        "piece of cake": "something very easy to do",
        "break a leg": "good luck; perform well",
        "time flies": "time appears to pass quickly",
        "hit the nail on the head": "to be exactly right",
        "spill the beans": "to reveal a secret",
        "under the weather": "feeling sick or unwell",
        "bite the bullet": "to endure a painful situation with courage",
    }
    figurative_markers = ["like a", "as if"] + list(figurative_map.keys())
    is_figurative = any(m in text.lower() for m in figurative_markers)
    result["classification"] = "figurative" if is_figurative else "literal"
    result["confidence"] = round(0.9 if is_figurative else 0.85, 3)
    if is_figurative:
        literal_parts: list[str] = []
        for idiom, meaning in figurative_map.items():
            if idiom in text.lower():
                literal_parts.append(f"'{idiom}' = {meaning}")
        if not literal_parts and ("like a" in text.lower() or "as if" in text.lower()):
            literal_parts.append(
                "Contains simile/metaphor — direct comparison without figurative intent"
            )
        result["literal_interpretation"] = (
            "; ".join(literal_parts) if literal_parts
            else "No specific idiom decomposition available"
        )
        result["figurative_elements"] = literal_parts


def _execute_r2p(result: dict[str, Any]) -> None:
    """Requirements-to-Plan structured decomposition."""
    steps = [
        {"step": 1, "action": "Analyze requirements", "estimated_effort": "2h"},
        {"step": 2, "action": "Design solution architecture", "estimated_effort": "4h"},
        {"step": 3, "action": "Implement core logic", "estimated_effort": "8h"},
        {"step": 4, "action": "Write tests", "estimated_effort": "4h"},
        {"step": 5, "action": "Deploy and validate", "estimated_effort": "2h"},
    ]
    result["plan"] = steps
    result["total_effort"] = "20h"
    result["success_criteria"] = [
        "All tests pass", "Performance targets met", "Code reviewed",
    ]


def _execute_telemetry(result: dict[str, Any], text: str) -> None:
    """Telemetry event parsing — extract action, duration, page, element."""
    result["event_recorded"] = True
    result["trace_id"] = f"trace-{uuid.uuid4().hex[:8]}"
    result["timestamp"] = time.time()

    lower = text.lower()
    attrs: dict[str, Any] = {"source": "cognitive_cell", "cell_type": "telemetry"}

    # Extract action verbs
    action_map = {
        "click": "click", "submit": "submit", "scroll": "scroll",
        "navigate": "navigate", "hover": "hover", "type": "input",
        "select": "select", "drag": "drag", "drop": "drop", "open": "open",
    }
    for verb, action in action_map.items():
        if verb in lower:
            attrs["action"] = action
            break

    # Extract numeric durations
    dur_match = re.search(r"(\d+)\s*(?:second|sec|ms|minute|min)", lower)
    if dur_match:
        attrs["duration_value"] = int(dur_match.group(1))
        unit = dur_match.group(0).replace(dur_match.group(1), "").strip()
        attrs["duration_unit"] = unit

    # Extract page/element references
    page_match = re.search(r"(?:on|at|in)\s+(?:the\s+)?(\w+)\s+page", lower)
    if page_match:
        attrs["page"] = page_match.group(1)
    elem_match = re.search(
        r"(?:click|clicked|press|pressed|hit)\s+(?:the\s+)?(\w+)", lower
    )
    if elem_match:
        attrs["element"] = elem_match.group(1)

    result["metadata"] = attrs
    result["parsed_attributes"] = {
        k: v for k, v in attrs.items() if k not in ("source", "cell_type")
    }


def _execute_aggregator(
    result: dict[str, Any], text: str, config: dict[str, Any]
) -> None:
    """Multi-expert output aggregation with configurable strategy."""
    strategy = config.get("strategy", "weighted_average")
    sub_cells = config.get("sub_cells", ["reasoning", "ethics", "causal"])
    sub_results: list[dict[str, Any]] = []

    for sc in sub_cells:
        if sc in CELL_TYPES and sc != "aggregator":  # prevent recursion
            sr = execute_cell(sc, text)
            sub_results.append({
                "cell": sc,
                "status": sr.get("status", "ok"),
                "confidence": sr.get("confidence", sr.get("risk_score", 0.8)),
                "elapsed_ms": sr.get("elapsed_ms", 0),
            })

    # Compute aggregated confidence
    if sub_results:
        confidences = [
            r["confidence"] for r in sub_results
            if isinstance(r["confidence"], (int, float))
        ]
        if strategy == "max_confidence":
            agg_confidence = max(confidences) if confidences else 0.0
        elif strategy == "ensemble":
            agg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
        else:  # weighted_average
            weights = [1.0 / (i + 1) for i in range(len(confidences))]
            w_sum = sum(weights)
            agg_confidence = (
                sum(c * w for c, w in zip(confidences, weights)) / w_sum
                if w_sum else 0.0
            )
    else:
        agg_confidence = 0.0

    result["strategy"] = strategy
    result["sub_cell_results"] = sub_results
    result["cells_aggregated"] = len(sub_results)
    result["aggregated_output"] = f"Aggregated {len(sub_results)} cells via {strategy}"
    result["confidence"] = round(agg_confidence, 3)
