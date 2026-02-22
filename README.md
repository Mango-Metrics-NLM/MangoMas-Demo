# 🥭 MangoMAS Demo

> **Multi-Agent Cognitive Architecture** — 10 Cognitive Cells · MCTS Planning · 7M MoE Router · 8-Agent Orchestration

[![CI](https://github.com/Mango-Metrics-NLM/MangoMas-Demo/actions/workflows/ci.yml/badge.svg)](https://github.com/Mango-Metrics-NLM/MangoMas-Demo/actions)
[![Coverage](https://img.shields.io/badge/coverage-%3E80%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-yellow)](https://huggingface.co/spaces/Mango-Metrics-NLM/MangoMAS)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Architecture

```mermaid
graph TB
    subgraph "MangoMAS-Demo"
        APP["app.py — Gradio Entrypoint"]
        FEAT["features.py — 64-D Vector Extraction"]
        MOD["models.py — 5 Neural Networks (~7M params)"]
        CELLS["cells/ — 10 Cognitive Cell Types"]
        MCTS["mcts/ — Monte Carlo Tree Search"]
        ROUTER["routing/ — MoE Neural Router"]
        AGENTS["agents/ — 8-Agent Orchestration"]
    end

    APP --> FEAT & CELLS & MCTS & ROUTER & AGENTS
    ROUTER --> FEAT & MOD
    MCTS --> MOD
    AGENTS --> ROUTER & CELLS
```

## Key Features

| Component | Description | Details |
|-----------|-------------|---------|
| **Cognitive Cells** | 10 biologically-inspired processing units | Reasoning, Memory, Causal, Ethics, Empathy, Curiosity, FigLiteral, R2P, Telemetry, Aggregator |
| **MCTS Engine** | Monte Carlo Tree Search with NN priors | UCB1 & PUCT strategies, policy/value networks |
| **MoE Router** | ~7M param neural routing gate | 16 expert towers, gated aggregation, semantic keyword boosting |
| **Agent Orchestration** | 8 specialized agents | SWE, Architect, QA, Security, DevOps, Research, Performance, Docs |

## Quick Start

### Local

```bash
# Clone and install
git clone https://github.com/Mango-Metrics-NLM/MangoMas-Demo.git
cd MangoMas-Demo
pip install -e ".[dev]"

# Run the demo
python app.py
# → Open http://localhost:7860
```

### Docker

```bash
docker build -t mangomas-demo .
docker run -p 7860:7860 mangomas-demo
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=mangomas_demo --cov-report=term-missing

# Lint & type check
ruff check mangomas_demo/ tests/
mypy mangomas_demo/
```

## Project Structure

```
MangoMas-Demo/
├── app.py                     # Gradio entrypoint (~15 lines)
├── mangomas_demo/
│   ├── __init__.py            # Package exports
│   ├── features.py            # 64-dim feature extraction
│   ├── models.py              # ExpertTower, MoE7M, RouterNet, Policy/ValueNet
│   ├── cells/                 # 10 cognitive cell implementations
│   ├── mcts/                  # MCTS engine, benchmark, visualization
│   ├── routing/               # MoE neural router
│   ├── agents/                # Multi-agent orchestrator
│   └── ui/                    # Gradio UI builder
├── tests/                     # 80%+ coverage test suite
├── .github/workflows/         # CI/CD pipelines
├── docs/                      # Blog posts & architecture
├── Dockerfile
└── pyproject.toml
```

## Technical Blog Posts

1. **[Building Biologically-Inspired Cognitive Cells](docs/blog/)** — Architecture of the 10-cell system
2. **[MCTS for Task Planning](docs/blog/)** — Neural-guided Monte Carlo Tree Search
3. **[7M-Param MoE Routing](docs/blog/)** — Mixture-of-Experts at inference time

## Citation

```bibtex
@software{mangomas_demo_2026,
  author = {Cruickshank, Ian},
  title = {{MangoMAS}: Multi-Agent Cognitive Architecture Demo},
  year = {2026},
  url = {https://github.com/Mango-Metrics-NLM/MangoMas-Demo}
}
```

## License

[MIT](LICENSE) © 2026 Ian Cruickshank / Mango-Metrics-NLM
