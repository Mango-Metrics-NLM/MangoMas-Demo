# C4 Level 4 -- Code-Level Diagrams

## Overview

The Code-level diagrams show the actual class structures, method signatures, data types, and relationships as implemented in the MangoMAS codebase. This level is derived directly from the source code and represents the finest grain of architectural detail.

## Neural Network Model Hierarchy

All five neural network classes conditionally inherit from `torch.nn.Module` (or `object` when PyTorch is unavailable). The `TORCH_AVAILABLE` boolean flag controls this at import time.

```mermaid
classDiagram
    class nnModule {
        <<PyTorch Base>>
        +parameters() Iterator
        +forward(x) Tensor
        +eval() Self
        +train() Self
    }

    class ExpertTower {
        -fc1: nn.Linear(64, 512)
        -fc2: nn.Linear(512, 512)
        -fc3: nn.Linear(512, 256)
        +__init__(d_in=64, h1=512, h2=512, d_out=256)
        +forward(x: Tensor) Tensor
    }

    class MixtureOfExperts7M {
        -num_experts: int = 16
        -gate_fc1: nn.Linear(64, 512)
        -gate_fc2: nn.Linear(512, num_experts)
        -experts: nn.ModuleList~ExpertTower~
        -classifier: nn.Linear(256, num_classes)
        +__init__(num_classes=10, num_experts=16)
        +parameter_count: int
        +forward(x64: Tensor) tuple~Tensor, Tensor~
    }

    class RouterNet {
        -net: nn.Sequential
        +EXPERTS: list~str~ = 8 expert names
        +__init__(d_in=64, d_h=128, n_out=8)
        +forward(x: Tensor) Tensor
    }

    class PolicyNetwork {
        -net: nn.Sequential
        +__init__(d_in=128, n_actions=32)
        +forward(x: Tensor) Tensor
    }

    class ValueNetwork {
        -net: nn.Sequential
        +__init__(d_in=192)
        +forward(x: Tensor) Tensor
    }

    nnModule <|-- ExpertTower : inherits (when torch available)
    nnModule <|-- MixtureOfExperts7M : inherits (when torch available)
    nnModule <|-- RouterNet : inherits (when torch available)
    nnModule <|-- PolicyNetwork : inherits (when torch available)
    nnModule <|-- ValueNetwork : inherits (when torch available)

    MixtureOfExperts7M "1" *-- "16" ExpertTower : experts (ModuleList)
```

### RouterNet Internal Architecture

The `RouterNet` uses an `nn.Sequential` with three linear layers, ReLU activations, and dropout:

```mermaid
graph LR
    Input["Input<br/>64-dim"] --> L1["nn.Linear<br/>64 &rarr; 128"]
    L1 --> R1["ReLU"]
    R1 --> D1["Dropout<br/>p=0.1"]
    D1 --> L2["nn.Linear<br/>128 &rarr; 64"]
    L2 --> R2["ReLU"]
    R2 --> L3["nn.Linear<br/>64 &rarr; 8"]
    L3 --> SM["Softmax<br/>dim=-1"]
    SM --> Output["Output<br/>8-dim probs"]

    style Input fill:#438DD5,stroke:#2E6295,color:#fff
    style Output fill:#2D6A4F,stroke:#1B4332,color:#fff
    style L1 fill:#E9C46A,stroke:#C9A24A,color:#000
    style L2 fill:#E9C46A,stroke:#C9A24A,color:#000
    style L3 fill:#E9C46A,stroke:#C9A24A,color:#000
    style R1 fill:#F4A261,stroke:#D48241,color:#000
    style R2 fill:#F4A261,stroke:#D48241,color:#000
    style D1 fill:#E76F51,stroke:#C5533A,color:#fff
    style SM fill:#2A9D8F,stroke:#1E7A6E,color:#fff
```

### MixtureOfExperts7M Forward Pass

```mermaid
graph TB
    Input64["Input x64<br/>(batch, 64)"] --> GateFC1["gate_fc1<br/>Linear(64, 512)"]
    GateFC1 --> GateReLU["ReLU"]
    GateReLU --> GateFC2["gate_fc2<br/>Linear(512, 16)"]
    GateFC2 --> Softmax["Softmax(dim=-1)"]
    Softmax --> GateWeights["gate_weights<br/>(batch, 16)"]

    Input64 --> Expert0["ExpertTower 0<br/>(64&rarr;256)"]
    Input64 --> Expert1["ExpertTower 1<br/>(64&rarr;256)"]
    Input64 --> ExpertDots["... x16 ..."]
    Input64 --> Expert15["ExpertTower 15<br/>(64&rarr;256)"]

    Expert0 --> Stack["torch.stack<br/>(batch, 16, 256)"]
    Expert1 --> Stack
    ExpertDots --> Stack
    Expert15 --> Stack

    GateWeights --> WeightedSum["Weighted Sum<br/>gate_weights * expert_outs"]
    Stack --> WeightedSum
    WeightedSum --> Agg["Aggregated<br/>(batch, 256)"]
    Agg --> Classifier["classifier<br/>Linear(256, N_classes)"]
    Classifier --> Logits["logits<br/>(batch, N_classes)"]

    style Input64 fill:#438DD5,stroke:#2E6295,color:#fff
    style GateWeights fill:#E9C46A,stroke:#C9A24A,color:#000
    style Logits fill:#2D6A4F,stroke:#1B4332,color:#fff
    style Stack fill:#7B2CBF,stroke:#5A189A,color:#fff
    style WeightedSum fill:#E76F51,stroke:#C5533A,color:#fff
```

## MCTSNode Dataclass

The `MCTSNode` is a `@dataclass` that forms the tree structure for Monte Carlo Tree Search. Each node tracks visit counts, accumulated value, and an optional neural policy prior.

```mermaid
classDiagram
    class MCTSNode {
        +id: str
        +action: str
        +visits: int = 0
        +total_value: float = 0.0
        +policy_prior: float = 0.0
        +children: list~MCTSNode~ | None = None
        +ucb1_score(parent_visits: int, c: float = 1.414) float
        +puct_score(parent_visits: int, c: float = 1.0) float
        +to_dict(max_depth: int = 3) dict
    }

    MCTSNode "1" o-- "0..*" MCTSNode : children (self-referential tree)
```

### UCB1 and PUCT Formulas

The two selection strategies implemented in `MCTSNode`:

```mermaid
graph TB
    subgraph UCB1["UCB1 Strategy"]
        direction LR
        UCB1_E["Exploitation<br/>Q(s,a) = total_value / visits"]
        UCB1_X["Exploration<br/>c * sqrt(ln(N_parent) / N_child)"]
        UCB1_S["Score = Q + c * sqrt(ln(N) / n)"]
    end

    subgraph PUCT["PUCT Strategy (AlphaZero-style)"]
        direction LR
        PUCT_E["Exploitation<br/>Q(s,a) = total_value / visits"]
        PUCT_X["Exploration<br/>c * P(a) * sqrt(N_parent) / (1 + N_child)"]
        PUCT_S["Score = Q + c * P * sqrt(N) / (1+n)"]
    end

    style UCB1 fill:#0077B6,stroke:#005A8A,color:#fff
    style PUCT fill:#023E8A,stroke:#012A5E,color:#fff
```

## Cell Type Registry Schema

The `CELL_TYPES` dictionary in `cells/types.py` defines the 10 cognitive cell types. Each entry maps a string key to a configuration dict.

```mermaid
classDiagram
    class CELL_TYPES {
        <<Registry: dict~str, dict~>>
    }

    class CellTypeEntry {
        +name: str
        +description: str
        +heads: list~str~
    }

    class ReasoningCell {
        +name = "ReasoningCell"
        +heads = ["rule", "nn"]
        +description = "Structured reasoning with Rule or NN heads"
    }

    class MemoryCell {
        +name = "MemoryCell"
        +heads = ["preference_extractor"]
        +description = "Privacy-preserving preference extraction"
    }

    class CausalCell {
        +name = "CausalCell"
        +heads = ["do_calculus"]
        +description = "Pearl's do-calculus for causal inference"
    }

    class EthicsCell {
        +name = "EthicsCell"
        +heads = ["classifier", "pii_scanner"]
        +description = "Safety classification and PII detection"
    }

    class EmpathyCell {
        +name = "EmpathyCell"
        +heads = ["tone_detector"]
        +description = "Emotional tone detection and empathetic responses"
    }

    class CuriosityCell {
        +name = "CuriosityCell"
        +heads = ["hypothesis_generator"]
        +description = "Epistemic curiosity and hypothesis generation"
    }

    class FigLiteralCell {
        +name = "FigLiteralCell"
        +heads = ["classifier"]
        +description = "Figurative vs literal language classification"
    }

    class R2PCell {
        +name = "R2PCell"
        +heads = ["planner"]
        +description = "Requirements-to-Plan structured decomposition"
    }

    class TelemetryCell {
        +name = "TelemetryCell"
        +heads = ["collector"]
        +description = "Telemetry event capture and structuring"
    }

    class AggregatorCell {
        +name = "AggregatorCell"
        +heads = ["weighted_average", "max_confidence", "ensemble"]
        +description = "Multi-expert output aggregation"
    }

    CELL_TYPES --> CellTypeEntry : schema
    CellTypeEntry <|-- ReasoningCell
    CellTypeEntry <|-- MemoryCell
    CellTypeEntry <|-- CausalCell
    CellTypeEntry <|-- EthicsCell
    CellTypeEntry <|-- EmpathyCell
    CellTypeEntry <|-- CuriosityCell
    CellTypeEntry <|-- FigLiteralCell
    CellTypeEntry <|-- R2PCell
    CellTypeEntry <|-- TelemetryCell
    CellTypeEntry <|-- AggregatorCell
```

## Executor Function Signatures

The `cells/executor.py` module exposes two public functions and 10 private handler functions:

```mermaid
classDiagram
    class CellExecutor {
        <<Module: cells/executor.py>>
        +execute_cell(cell_type: str, text: str, config_json: str = "{}") dict~str, Any~
        +compose_cells(pipeline_str: str, text: str) dict~str, Any~
        -_execute_reasoning(result: dict, text: str, config: dict) None
        -_execute_memory(result: dict, text: str) None
        -_execute_causal(result: dict, text: str, config: dict) None
        -_execute_ethics(result: dict, text: str) None
        -_execute_empathy(result: dict, text: str) None
        -_execute_curiosity(result: dict, text: str, config: dict) None
        -_execute_figliteral(result: dict, text: str) None
        -_execute_r2p(result: dict) None
        -_execute_telemetry(result: dict, text: str) None
        -_execute_aggregator(result: dict, text: str, config: dict) None
    }

    class CellResult {
        <<Return Type>>
        +cell_type: str
        +request_id: str
        +status: str
        +elapsed_ms: float
        +... cell-specific fields
    }

    class PipelineResult {
        <<Return Type>>
        +pipeline: list~str~
        +activations: list~dict~
        +final_output: dict
        +total_cells: int
        +context_keys: list~str~
    }

    CellExecutor ..> CellResult : returns from execute_cell
    CellExecutor ..> PipelineResult : returns from compose_cells
```

## Routing Module Internals

```mermaid
classDiagram
    class RouterModule {
        <<Module: routing/router.py>>
        +EXPERT_NAMES: list~str~ = 8 names
        -_ROUTER_SEED: int = 42
        -_router_net_singleton: RouterNet | None
        -_EXPERT_KEYWORDS: dict~int, list~str~~
        +route_task(task: str, top_k: int = 3) dict~str, Any~
        -_get_router() RouterNet
    }

    class RouteResult {
        <<Return Type>>
        +task: str
        +features: list~float~
        +all_weights: dict~str, float~
        +selected_experts: list~dict~
        +top_k: int
        +nn_enabled: bool
        +elapsed_ms: float
    }

    class SelectedExpert {
        <<Nested Dict>>
        +expert: str
        +weight: float
        +rank: int
    }

    RouterModule ..> RouteResult : returns from route_task
    RouteResult *-- SelectedExpert : selected_experts list
```

## Logging Configuration Classes

```mermaid
classDiagram
    class LoggingModule {
        <<Module: logging_config.py>>
        -_DEFAULT_LOG_LEVEL: str = "INFO"
        -_DEFAULT_LOG_FORMAT: str = "text"
        -_ENV_LOG_LEVEL: str = "MANGOMAS_LOG_LEVEL"
        -_ENV_LOG_FORMAT: str = "MANGOMAS_LOG_FORMAT"
        -_LOGGER_NAME: str = "mangomas_demo"
        +get_logger(name: str | None) Logger
        +configure_logging() Logger
    }

    class _JSONFormatter {
        <<logging.Formatter subclass>>
        +format(record: LogRecord) str
    }

    class JSONLogEntry {
        <<Output Schema>>
        +timestamp: str
        +level: str
        +logger: str
        +message: str
        +exception: str | None
        +component: str | None
        +elapsed_ms: float | None
        +cell_type: str | None
        +strategy: str | None
    }

    LoggingModule --> _JSONFormatter : uses when format=json
    _JSONFormatter ..> JSONLogEntry : produces
```

## Agent Orchestration Types

```mermaid
classDiagram
    class AgentDefinition {
        <<Dict Schema in AGENTS list>>
        +name: str
        +specialization: str
        +icon: str
    }

    class OrchestrationResult {
        <<Return Type>>
        +task: str
        +strategy: str
        +agents_selected: int
        +max_agents: int
        +results: list~AgentResult~
        +total_elapsed_ms: float
    }

    class AgentResult {
        <<Nested Dict>>
        +agent: str
        +icon: str
        +specialization: str
        +weight: float
        +cell_used: str
        +output: dict
        +confidence: float
    }

    OrchestrationResult *-- AgentResult : results list
    AgentResult ..> AgentDefinition : derived from
```
