# Changelog

All notable changes to MangoMAS Demo will be documented in this file.

## [Unreleased]

### Fixed

- Fix incorrect GitHub URL in `notebooks/MangMas_demo.ipynb` summary cell (`mangomas/mangomas-demo` → `Mango-Metrics-NLM/MangoMas-Demo`)
- Remove duplicate `notebooks/demo.ipynb` (canonical file is `MangMas_demo.ipynb`)
- Update `test_notebook.py` to validate canonical notebook file and add link regression tests

## [1.0.0] — 2026-02-22

### 🎉 Initial Public Release

**Features:**

- 10 Cognitive Cells (Reasoning, Memory, Causal, Ethics, Empathy, Curiosity, FigLiteral, R2P, Telemetry, Aggregator) with real NN heads
- MCTS Planning Engine with UCB1 and PUCT selection strategies
- 7M-parameter Mixture-of-Experts Neural Router with semantic keyword boosting
- Multi-agent orchestration with 3 routing strategies (MoE, Round-Robin, Random)
- 6-tab Gradio interactive demo with dark-mode glassmorphism theme
- Comprehensive test suite (80%+ coverage) with unit, integration, security, and sanity tests

**Architecture:**

- Modular Python package (`mangomas_demo/`) with 6 sub-packages
- Clean separation: features → models → cells → mcts → routing → agents → ui
- Torch graceful fallback (CPU-only mode when PyTorch unavailable)

**Infrastructure:**

- GitHub Actions CI pipeline (lint → type-check → test → coverage → build)
- GitHub Actions CD pipeline (Docker build → deploy to HuggingFace Space)
- Docker support (Python 3.11-slim with health check)

**Bug Fixes Carried Forward (from development):**

- BUG-001: Aggregator cell real aggregation
- BUG-002: Invalid JSON config error handling
- BUG-003: Deterministic MoE routing with fixed seeds
- BUG-004: Semantic boost for expert routing
- BUG-005: Real values in MCTS benchmark
- BUG-007: Keyword-based emotion detection in Empathy cell
- BUG-009: Real output from agent orchestrations
- BUG-010: Functional strategy dropdown for orchestration
- BUG-011: Input validation for empty/whitespace strings
- BUG-012: Literal interpretation for figurative language
- BUG-013: Topic-aware curiosity questions
- BUG-014: Telemetry event attribute parsing
- BUG-015: Architecture diagram alignment
- BUG-016: Chart title truncation fix
- BUG-017: X-axis label clipping fix in plots
