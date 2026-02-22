# C4 Level 2 -- Container Diagram

## Overview

The Container diagram zooms into the MangoMAS Demo system boundary and reveals the seven major containers (deployable/runnable units) that comprise the architecture. Each container represents a distinct area of responsibility within the `mangomas_demo` Python package, connected by in-process Python calls.

## Container Diagram

```mermaid
graph TB
    %% ---------------------------------------------------------------
    %% External actors
    %% ---------------------------------------------------------------
    User["<b>User / Researcher</b><br/><i>Person</i><br/>Interacts via browser"]

    %% ---------------------------------------------------------------
    %% System boundary
    %% ---------------------------------------------------------------
    subgraph MangoMAS["MangoMAS Demo System"]
        direction TB

        UI["<b>Gradio Web App</b><br/><i>Container: Python / Gradio</i><br/>6-tab interactive UI<br/>(app.py, ui/builder.py, ui/styles.py)<br/>Dark glassmorphism theme"]

        Features["<b>Feature Extraction</b><br/><i>Container: Python module</i><br/>64-dim vector extraction<br/>(features.py)<br/>Hash sinusoidal + domain tags<br/>+ structural + sentiment + novelty"]

        Models["<b>Neural Network Models</b><br/><i>Container: Python / PyTorch</i><br/>5 NN architectures<br/>(models.py)<br/>ExpertTower, MixtureOfExperts7M,<br/>RouterNet, PolicyNetwork, ValueNetwork"]

        Cells["<b>Cognitive Cell Engine</b><br/><i>Container: Python module</i><br/>10 biologically-inspired cell types<br/>(cells/executor.py, cells/types.py)<br/>Dispatch, compose, aggregate"]

        MCTS["<b>MCTS Planning Engine</b><br/><i>Container: Python module</i><br/>Monte Carlo Tree Search<br/>(mcts/engine.py, mcts/benchmark.py,<br/>mcts/viz.py)<br/>UCB1/PUCT strategies"]

        Router["<b>MoE Neural Router</b><br/><i>Container: Python module</i><br/>Singleton RouterNet gate<br/>(routing/router.py, routing/viz.py)<br/>Keyword boosting, top-K selection"]

        Orchestrator["<b>Agent Orchestrator</b><br/><i>Container: Python module</i><br/>8 specialized agents<br/>(agents/orchestrator.py)<br/>3 strategies: moe_routing,<br/>round_robin, random"]
    end

    %% ---------------------------------------------------------------
    %% External systems
    %% ---------------------------------------------------------------
    HF["<b>HuggingFace Spaces</b><br/><i>External System</i>"]
    PyTorch["<b>PyTorch Runtime</b><br/><i>External Library</i>"]

    %% ---------------------------------------------------------------
    %% Relationships -- User
    %% ---------------------------------------------------------------
    User -->|"HTTP/WebSocket<br/>via browser"| UI

    %% ---------------------------------------------------------------
    %% Internal relationships
    %% ---------------------------------------------------------------
    UI -->|"Calls featurize64(),<br/>plot_features()"| Features
    UI -->|"Calls execute_cell(),<br/>compose_cells()"| Cells
    UI -->|"Calls run_mcts(),<br/>benchmark_strategies(),<br/>plot_mcts_tree()"| MCTS
    UI -->|"Calls route_task(),<br/>plot_expert_weights()"| Router
    UI -->|"Calls orchestrate()"| Orchestrator

    Router -->|"Calls featurize64()<br/>for 64-dim vectors"| Features
    Router -->|"Uses singleton<br/>RouterNet for<br/>softmax gating"| Models

    MCTS -->|"Uses PolicyNetwork<br/>and ValueNetwork<br/>for NN priors"| Models

    Orchestrator -->|"Calls route_task()<br/>for MoE routing"| Router
    Orchestrator -->|"Calls execute_cell()<br/>via agent-cell mapping"| Cells

    Cells -->|"References CELL_TYPES<br/>registry for dispatch"| Cells

    %% ---------------------------------------------------------------
    %% External relationships
    %% ---------------------------------------------------------------
    Models -.->|"torch.nn.Module<br/>(optional import)"| PyTorch
    MCTS -.->|"torch tensors<br/>for NN inference"| PyTorch
    Router -.->|"torch tensors<br/>for forward pass"| PyTorch
    UI -.->|"Deployed to"| HF

    %% ---------------------------------------------------------------
    %% Styling
    %% ---------------------------------------------------------------
    style User fill:#08427B,stroke:#052E56,color:#fff
    style UI fill:#438DD5,stroke:#2E6295,color:#fff
    style Features fill:#438DD5,stroke:#2E6295,color:#fff
    style Models fill:#438DD5,stroke:#2E6295,color:#fff
    style Cells fill:#438DD5,stroke:#2E6295,color:#fff
    style MCTS fill:#438DD5,stroke:#2E6295,color:#fff
    style Router fill:#438DD5,stroke:#2E6295,color:#fff
    style Orchestrator fill:#438DD5,stroke:#2E6295,color:#fff
    style HF fill:#999999,stroke:#6B6B6B,color:#fff
    style PyTorch fill:#999999,stroke:#6B6B6B,color:#fff
    style MangoMAS fill:none,stroke:#1168BD,stroke-width:2px,stroke-dasharray:5
```

## Container Descriptions

| Container | Technology | Source Files | Purpose |
|-----------|-----------|--------------|---------|
| **Gradio Web App** | Python, Gradio, Plotly | `app.py`, `ui/builder.py`, `ui/styles.py` | Entrypoint and 6-tab interactive interface. Builds UI with `build_app()`, applies dark glassmorphism CSS theme, launches on port 7860 |
| **Feature Extraction** | Python, hashlib, math | `features.py` | Extracts deterministic 64-dimensional L2-normalized feature vectors from text. Five signal groups: 32 hash sinusoidal, 16 domain tags, 8 structural, 4 sentiment, 4 novelty |
| **Neural Network Models** | Python, PyTorch (optional) | `models.py` | Five `nn.Module` subclasses with graceful fallback to `object` when PyTorch is unavailable. Defines `ExpertTower` (64-512-512-256), `MixtureOfExperts7M` (~7M params, 16 experts), `RouterNet` (64-128-64-8), `PolicyNetwork` (128-256-128-N), `ValueNetwork` (192-256-64-1) |
| **Cognitive Cell Engine** | Python | `cells/types.py`, `cells/executor.py` | Registry of 10 cell types with dispatch via `execute_cell()`. Supports pipeline composition via `compose_cells()`. Lifecycle: validate, infer, structure, return |
| **MCTS Planning Engine** | Python, PyTorch (optional) | `mcts/engine.py`, `mcts/benchmark.py`, `mcts/viz.py` | Monte Carlo Tree Search with `MCTSNode` dataclass, UCB1 and PUCT selection strategies, optional PolicyNetwork/ValueNetwork priors, sunburst Plotly visualization, and strategy benchmarking (MCTS vs Greedy vs Random) |
| **MoE Neural Router** | Python, NumPy, PyTorch (optional) | `routing/router.py`, `routing/viz.py` | Singleton `RouterNet` with fixed seed (42) for deterministic routing. Applies semantic keyword boosting to expert weights, selects top-K experts, generates bar chart visualizations |
| **Agent Orchestrator** | Python | `agents/orchestrator.py` | Defines 8 specialized agents (SWE, Architect, QA, Security, DevOps, Research, Performance, Documentation) with agent-to-cell mapping. Three routing strategies: `moe_routing` (neural gate), `round_robin` (sequential), `random` |

## Inter-Container Communication

All communication between containers occurs as **in-process Python function calls** -- there are no network boundaries between containers. The system runs as a single Python process:

1. **UI to Feature Extraction**: `featurize64(text)` returns a `list[float]` of 64 values; `plot_features(features)` returns a Plotly `Figure`.
2. **UI to Cognitive Cells**: `execute_cell(cell_type, text, config_json)` returns a `dict`; `compose_cells(pipeline_str, text)` chains multiple cells sequentially.
3. **UI to MCTS**: `run_mcts(task, max_simulations, exploration_constant, strategy)` returns a dict with the search tree and best action.
4. **UI to Router**: `route_task(task, top_k)` returns routing weights and selected experts.
5. **UI to Orchestrator**: `orchestrate(task, max_agents, strategy)` returns a dict with per-agent results.
6. **Router to Features**: The router internally calls `featurize64(task)` to produce the 64-dim input vector.
7. **Router to Models**: The router uses the singleton `RouterNet` for softmax gating over 8 experts.
8. **MCTS to Models**: The engine uses `PolicyNetwork` for action priors and `ValueNetwork` for state evaluation.
9. **Orchestrator to Router**: When using `moe_routing` strategy, the orchestrator calls `route_task()` for expert selection.
10. **Orchestrator to Cells**: Each selected agent maps to a cell type via `_AGENT_CELL_MAP` and executes through `execute_cell()`.
