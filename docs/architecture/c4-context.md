# C4 Level 1 -- System Context Diagram

## Overview

The System Context diagram shows MangoMAS Demo as a single system and its relationships with the users and external systems it interacts with. MangoMAS is a Python-based Multi-Agent Cognitive Architecture that combines neural network routing, Monte Carlo Tree Search planning, and a pipeline of 10 cognitive cell types -- all exposed through an interactive Gradio web interface.

## System Context Diagram

```mermaid
graph TB
    %% ---------------------------------------------------------------
    %% People / Actors
    %% ---------------------------------------------------------------
    User["<b>User / Researcher</b><br/><i>Person</i><br/>Interacts with the 6-tab<br/>Gradio interface to explore<br/>cognitive cells, MCTS planning,<br/>MoE routing, and orchestration"]

    %% ---------------------------------------------------------------
    %% The MangoMAS System (boundary)
    %% ---------------------------------------------------------------
    MangoMAS["<b>MangoMAS Demo</b><br/><i>Software System</i><br/>Multi-Agent Cognitive Architecture<br/>with 10 cells, MCTS engine,<br/>7M-param MoE router,<br/>8-agent orchestrator"]

    %% ---------------------------------------------------------------
    %% External Systems
    %% ---------------------------------------------------------------
    HF["<b>HuggingFace Spaces</b><br/><i>External System</i><br/>Hosts the live Gradio app<br/>at Mango-Metrics-NLM/MangoMAS"]

    GHA["<b>GitHub Actions</b><br/><i>External System</i><br/>CI/CD pipeline: lint, type-check,<br/>test, coverage, build, deploy"]

    PyTorch["<b>PyTorch Runtime</b><br/><i>External Library</i><br/>Provides tensor ops for NN models.<br/>Optional -- system falls back<br/>gracefully when unavailable"]

    %% ---------------------------------------------------------------
    %% Relationships
    %% ---------------------------------------------------------------
    User -->|"Submits tasks and<br/>views results via browser"| MangoMAS
    MangoMAS -->|"Returns interactive plots,<br/>JSON outputs, and<br/>orchestration results"| User

    MangoMAS -->|"Deployed to<br/>(CD pipeline pushes<br/>via HfApi)"| HF
    HF -->|"Serves Gradio app<br/>to end users"| User

    GHA -->|"Runs CI on push/PR:<br/>ruff lint, mypy,<br/>pytest --cov, build"| MangoMAS
    GHA -->|"Deploys to HF Space<br/>on main branch push"| HF

    MangoMAS -.->|"Uses for NN inference<br/>(ExpertTower, RouterNet,<br/>PolicyNetwork, ValueNetwork)"| PyTorch

    %% ---------------------------------------------------------------
    %% Styling
    %% ---------------------------------------------------------------
    style User fill:#08427B,stroke:#052E56,color:#fff
    style MangoMAS fill:#1168BD,stroke:#0B4884,color:#fff
    style HF fill:#999999,stroke:#6B6B6B,color:#fff
    style GHA fill:#999999,stroke:#6B6B6B,color:#fff
    style PyTorch fill:#999999,stroke:#6B6B6B,color:#fff
```

## Element Descriptions

| Element | Type | Description |
|---------|------|-------------|
| **User / Researcher** | Person | Data scientists, ML engineers, or researchers who interact with the Gradio web UI to explore multi-agent cognitive architecture features |
| **MangoMAS Demo** | Software System | The core application -- a Python package (`mangomas_demo`) providing feature extraction, neural models, cognitive cells, MCTS planning, MoE routing, and multi-agent orchestration |
| **HuggingFace Spaces** | External System | Cloud hosting platform where the Gradio application is deployed and served publicly at `Mango-Metrics-NLM/MangoMAS` |
| **GitHub Actions** | External System | CI/CD automation running two workflows: `ci.yml` (lint, type-check, test with coverage, build) and `cd.yml` (deploy to HuggingFace Space) |
| **PyTorch Runtime** | External Library | Optional deep learning framework. When available, provides real neural network inference for the 5 NN models. When absent, the system uses deterministic fallback logic |

## Key Interactions

1. **User to MangoMAS**: The user submits natural language tasks through the Gradio 6-tab UI (Features, Cells, Composition, MCTS, Router, Orchestration). The system processes inputs through the feature extraction and cognitive pipeline, returning structured JSON results and interactive Plotly visualizations.

2. **MangoMAS to HuggingFace**: The CD pipeline (`cd.yml`) uses `huggingface_hub.HfApi.upload_folder()` to push the application code to the HuggingFace Space on every push to `main`.

3. **GitHub Actions to MangoMAS**: On every pull request and push to `main`, the CI pipeline runs Ruff linting, Mypy strict type checking, pytest with 80% minimum coverage, and a package build verification step.

4. **MangoMAS to PyTorch**: All five neural network classes (`ExpertTower`, `MixtureOfExperts7M`, `RouterNet`, `PolicyNetwork`, `ValueNetwork`) conditionally import `torch` and `torch.nn`. If PyTorch is unavailable, models inherit from `object` instead of `nn.Module`, and routing/MCTS engines use deterministic fallback calculations.
