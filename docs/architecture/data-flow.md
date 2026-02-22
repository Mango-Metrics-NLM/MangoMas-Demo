# Data Flow -- Sequence Diagrams

## Overview

This document contains three sequence diagrams that trace the primary data flows through MangoMAS Demo: (a) the full user input pipeline from Gradio to aggregated output, (b) the MCTS search cycle with expand/select/simulate/backpropagate phases, and (c) agent orchestration with MoE routing.

## (a) Full Pipeline Flow: User Input to UI Output

This diagram traces a complete request from the user entering text in the Gradio UI through feature extraction, neural routing, cell execution, and back to the rendered output.

```mermaid
sequenceDiagram
    actor User
    participant UI as Gradio Web App<br/>(ui/builder.py)
    participant Feat as Feature Extraction<br/>(features.py)
    participant Router as MoE Neural Router<br/>(routing/router.py)
    participant RNet as RouterNet<br/>(models.py)
    participant Cells as Cell Executor<br/>(cells/executor.py)
    participant Agg as AggregatorCell

    User->>UI: Submit task text via browser
    activate UI

    Note over UI: Tab 5 - MoE Router tab<br/>or Tab 6 - Orchestration tab

    UI->>Router: route_task(task, top_k=3)
    activate Router

    Router->>Feat: featurize64(task)
    activate Feat
    Note over Feat: 1. SHA-256 hash (32 sinusoidal dims)<br/>2. Domain tag scan (16 dims)<br/>3. Structural signals (8 dims)<br/>4. Sentiment polarity (4 dims)<br/>5. Novelty/complexity (4 dims)<br/>6. L2 normalize to unit vector
    Feat-->>Router: features: list[float] (64 dims)
    deactivate Feat

    Router->>RNet: forward(tensor([features]))
    activate RNet
    Note over RNet: Linear(64,128) -> ReLU -> Dropout(0.1)<br/>-> Linear(128,64) -> ReLU<br/>-> Linear(64,8) -> Softmax
    RNet-->>Router: weights: Tensor (8 probs)
    deactivate RNet

    Note over Router: Apply keyword semantic boost:<br/>+0.15 per matching keyword<br/>Re-normalize weights<br/>Sort descending, take top-K

    Router-->>UI: {selected_experts, all_weights, features, elapsed_ms}
    deactivate Router

    loop For each selected expert (top-K)
        UI->>Cells: execute_cell(cell_type, task)
        activate Cells
        Note over Cells: 1. Validate input (non-empty)<br/>2. Parse config JSON<br/>3. Generate request_id<br/>4. Dispatch to _execute_{type}()<br/>5. Record elapsed_ms
        Cells-->>UI: {cell_type, request_id, status, elapsed_ms, ...}
        deactivate Cells
    end

    Note over UI: Render Plotly bar chart<br/>(expert weights)<br/>Render feature vector chart<br/>Format JSON results

    UI-->>User: Interactive plots + JSON output
    deactivate UI
```

## (b) MCTS Search Cycle

This diagram shows the internal Monte Carlo Tree Search loop with the four phases: expand, select, simulate, and backpropagate. The cycle runs for the configured number of simulations (10-500).

```mermaid
sequenceDiagram
    participant Caller as UI / Benchmark
    participant Engine as run_mcts()<br/>(mcts/engine.py)
    participant Root as MCTSNode (root)
    participant Child as MCTSNode (child)
    participant PNet as PolicyNetwork<br/>(models.py)
    participant VNet as ValueNetwork<br/>(models.py)

    Caller->>Engine: run_mcts(task, max_simulations, exploration_constant, strategy)
    activate Engine

    Note over Engine: _detect_category(task)<br/>Lookup TASK_CATEGORIES[category]<br/>Get action list (5 actions)

    Engine->>Root: Create MCTSNode(id="root", action=task[:50])
    activate Root

    rect rgb(40, 60, 80)
        Note over Engine,VNet: Simulation Loop (repeat max_simulations times)

        Note over Engine: PHASE 1: EXPAND<br/>If root has no children,<br/>create child nodes for each action

        alt First simulation (no children)
            loop For each action in category
                Engine->>PNet: forward(torch.randn(1, 128))
                activate PNet
                PNet-->>Engine: priors: Tensor (N_actions)
                deactivate PNet
                Engine->>Root: Add child MCTSNode(action, policy_prior=prior)
            end
        end

        Note over Engine: PHASE 2: SELECT<br/>Choose best child by score function

        alt strategy == "ucb1"
            Engine->>Root: For each child: ucb1_score(parent_visits, c)
            Note over Root: UCB1 = Q/n + c * sqrt(ln(N)/n)<br/>Returns inf if visits == 0
        else strategy == "puct"
            Engine->>Root: For each child: puct_score(parent_visits, c)
            Note over Root: PUCT = Q/n + c * P * sqrt(N)/(1+n)<br/>Returns inf if visits == 0
        end

        Root-->>Engine: best_child (highest score)

        Note over Engine: PHASE 3: SIMULATE<br/>Get value estimate for selected node

        Engine->>VNet: forward(torch.randn(1, 192))
        activate VNet
        Note over VNet: Linear(192,256) -> ReLU<br/>-> Linear(256,64) -> ReLU<br/>-> Linear(64,1) -> Tanh<br/>Output in [-1, 1]
        VNet-->>Engine: value: float
        deactivate VNet

        Note over Engine: PHASE 4: BACKPROPAGATE<br/>Update visit counts and values

        Engine->>Child: visits += 1
        Engine->>Child: total_value += value
        Engine->>Root: visits += 1
    end

    Note over Engine: Find best child (max visits)<br/>Compute best_value = total_value / visits

    Engine->>Root: to_dict(max_depth=2)
    Root-->>Engine: Serialized tree dict

    deactivate Root

    Engine-->>Caller: {task, category, strategy, best_action, best_value,<br/>total_simulations, tree, all_actions, elapsed_ms, nn_enabled}
    deactivate Engine
```

## (c) Agent Orchestration with MoE Routing

This diagram shows the full agent orchestration flow when using the `moe_routing` strategy. The orchestrator calls the neural router to select agents, maps each agent to a cognitive cell, executes the cells, and collects results.

```mermaid
sequenceDiagram
    actor User
    participant UI as Gradio Orchestration Tab
    participant Orch as orchestrate()<br/>(agents/orchestrator.py)
    participant MoE as _orchestrate_moe()
    participant Router as route_task()<br/>(routing/router.py)
    participant Feat as featurize64()
    participant RNet as RouterNet (singleton)
    participant CellExec as execute_cell()<br/>(cells/executor.py)

    User->>UI: Enter task, select max_agents=3,<br/>strategy="moe_routing"
    UI->>Orch: orchestrate(task, 3, "moe_routing")
    activate Orch

    Note over Orch: Dispatch by strategy string

    Orch->>MoE: _orchestrate_moe(task, 3)
    activate MoE

    MoE->>Router: route_task(task, top_k=3)
    activate Router

    Router->>Feat: featurize64(task)
    activate Feat
    Feat-->>Router: 64-dim feature vector
    deactivate Feat

    Router->>RNet: forward(feature_tensor)
    activate RNet
    RNet-->>Router: 8-dim softmax weights
    deactivate RNet

    Note over Router: Apply keyword boost (+0.15)<br/>Re-normalize<br/>argsort descending[:3]

    Router-->>MoE: {selected_experts: [{expert, weight, rank}, ...]}
    deactivate Router

    Note over MoE: Map expert names to agents:<br/>"Code Expert" -> "SWE Agent"<br/>"Security Expert" -> "Security Agent"<br/>"Architecture Expert" -> "Architect Agent"

    loop For each selected expert (top 3)
        Note over MoE: Lookup _AGENT_CELL_MAP:<br/>SWE Agent -> "reasoning"<br/>Security Agent -> "ethics"<br/>Architect Agent -> "r2p"

        MoE->>CellExec: execute_cell(cell_type, task)
        activate CellExec
        Note over CellExec: Dispatch to _execute_{type}()<br/>Returns structured result dict
        CellExec-->>MoE: {cell_type, request_id, status, elapsed_ms, ...}
        deactivate CellExec

        Note over MoE: Build agent result:<br/>{agent, icon, specialization,<br/>weight, cell_used, output, confidence}
    end

    MoE-->>Orch: agent_results: list[dict]
    deactivate MoE

    Orch-->>UI: {task, strategy, agents_selected: 3,<br/>results: [...], total_elapsed_ms}
    deactivate Orch

    UI-->>User: Rendered JSON with per-agent<br/>results and confidence scores
```

## Agent-to-Cell Mapping Reference

The following table shows the fixed mapping used by the orchestrator to route each agent to its cognitive cell type:

```mermaid
graph LR
    subgraph Agents["Agents"]
        SWE["SWE Agent"]
        Arch["Architect Agent"]
        QA["QA Agent"]
        Sec["Security Agent"]
        Ops["DevOps Agent"]
        Res["Research Agent"]
        Perf["Performance Agent"]
        Doc["Documentation Agent"]
    end

    subgraph Cells["Cognitive Cells"]
        reasoning["reasoning"]
        r2p["r2p"]
        ethics["ethics"]
        telemetry["telemetry"]
        causal["causal"]
    end

    SWE -->|"_AGENT_CELL_MAP"| reasoning
    Arch -->|"_AGENT_CELL_MAP"| r2p
    QA -->|"_AGENT_CELL_MAP"| reasoning
    Sec -->|"_AGENT_CELL_MAP"| ethics
    Ops -->|"_AGENT_CELL_MAP"| telemetry
    Res -->|"_AGENT_CELL_MAP"| causal
    Perf -->|"_AGENT_CELL_MAP"| reasoning
    Doc -->|"_AGENT_CELL_MAP"| reasoning

    style SWE fill:#264653,stroke:#1A3040,color:#fff
    style Arch fill:#264653,stroke:#1A3040,color:#fff
    style QA fill:#264653,stroke:#1A3040,color:#fff
    style Sec fill:#264653,stroke:#1A3040,color:#fff
    style Ops fill:#264653,stroke:#1A3040,color:#fff
    style Res fill:#264653,stroke:#1A3040,color:#fff
    style Perf fill:#264653,stroke:#1A3040,color:#fff
    style Doc fill:#264653,stroke:#1A3040,color:#fff
    style reasoning fill:#CA6702,stroke:#9A4E01,color:#fff
    style r2p fill:#CA6702,stroke:#9A4E01,color:#fff
    style ethics fill:#CA6702,stroke:#9A4E01,color:#fff
    style telemetry fill:#CA6702,stroke:#9A4E01,color:#fff
    style causal fill:#CA6702,stroke:#9A4E01,color:#fff
```

## Benchmark Strategy Comparison Flow

The benchmark module compares three strategies on the same task:

```mermaid
sequenceDiagram
    participant UI as MCTS Tab
    participant Bench as benchmark_strategies()<br/>(mcts/benchmark.py)
    participant MCTS as run_mcts()
    participant PNet as PolicyNetwork
    participant VNet as ValueNetwork

    UI->>Bench: benchmark_strategies(task)
    activate Bench

    Note over Bench: Strategy 1: MCTS (full tree search)
    Bench->>MCTS: run_mcts(task, max_simulations=100)
    activate MCTS
    MCTS-->>Bench: {best_value, best_action, elapsed_ms}
    deactivate MCTS

    Note over Bench: Strategy 2: Greedy (single-step policy)
    Bench->>PNet: forward(torch.randn(1, 128))
    activate PNet
    PNet-->>Bench: priors (action probabilities)
    deactivate PNet
    Note over Bench: argmax(priors) -> greedy_action

    Note over Bench: Strategy 3: Random (random action + value)
    Bench->>VNet: forward(torch.randn(1, 192))
    activate VNet
    VNet-->>Bench: random value estimate
    deactivate VNet
    Note over Bench: random.choice(actions) -> random_action

    Bench-->>UI: {task, category, results: {mcts, greedy, random}}
    deactivate Bench
```
