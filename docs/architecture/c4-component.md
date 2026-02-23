# C4 Level 3 -- Component Diagram

## Overview

The Component diagram drills into each container to reveal the internal components -- the individual classes, functions, registries, and visualizations that make up the MangoMAS system. This level shows all 10 cognitive cell types, all 5 neural network models, all 8 agents, and all MCTS/routing components with their dependencies.

## Neural Network Models Container

The `models.py` module defines five `nn.Module` subclasses (with graceful `object` fallback) that power the MoE routing, MCTS planning, and expert classification subsystems.

```mermaid
graph TB
    subgraph ModelsContainer["Neural Network Models (models.py)"]
        direction TB

        ET["<b>ExpertTower</b><br/><i>Component: nn.Module</i><br/>Single expert MLP<br/>64 &rarr; 512 &rarr; 512 &rarr; 256<br/>ReLU activations"]

        MoE["<b>MixtureOfExperts7M</b><br/><i>Component: nn.Module</i><br/>~7M params, 16 expert towers<br/>Gating: 64 &rarr; 512 &rarr; 16 softmax<br/>Classifier: 256 &rarr; N_classes"]

        RN["<b>RouterNet</b><br/><i>Component: nn.Module</i><br/>Routing gate MLP<br/>64 &rarr; 128 &rarr; 64 &rarr; 8 softmax<br/>Dropout 0.1, ~0.8ms latency"]

        PN["<b>PolicyNetwork</b><br/><i>Component: nn.Module</i><br/>MCTS policy head<br/>128 &rarr; 256 &rarr; 128 &rarr; N_actions<br/>Softmax output"]

        VN["<b>ValueNetwork</b><br/><i>Component: nn.Module</i><br/>MCTS value head<br/>192 &rarr; 256 &rarr; 64 &rarr; 1<br/>Tanh output in [-1, 1]"]

        TA["<b>TORCH_AVAILABLE</b><br/><i>Component: bool flag</i><br/>Runtime check for<br/>PyTorch availability"]
    end

    MoE -->|"Contains 16x"| ET
    MoE -->|"Gating selects<br/>expert weights"| ET

    style ET fill:#2D6A4F,stroke:#1B4332,color:#fff
    style MoE fill:#2D6A4F,stroke:#1B4332,color:#fff
    style RN fill:#2D6A4F,stroke:#1B4332,color:#fff
    style PN fill:#2D6A4F,stroke:#1B4332,color:#fff
    style VN fill:#2D6A4F,stroke:#1B4332,color:#fff
    style TA fill:#40916C,stroke:#2D6A4F,color:#fff
    style ModelsContainer fill:none,stroke:#2D6A4F,stroke-width:2px,stroke-dasharray:5
```

## Cognitive Cell Engine Container

The cell engine consists of a type registry (`types.py`) and an executor (`executor.py`) with 10 private handler functions, plus a composition pipeline.

```mermaid
graph TB
    subgraph CellsContainer["Cognitive Cell Engine (cells/)"]
        direction TB

        Registry["<b>CELL_TYPES Registry</b><br/><i>Component: dict</i><br/>10 cell type definitions<br/>with names, descriptions,<br/>and available heads"]

        Executor["<b>execute_cell()</b><br/><i>Component: function</i><br/>Dispatches to cell-specific<br/>handler by cell_type string.<br/>Lifecycle: validate &rarr; infer<br/>&rarr; structure &rarr; return"]

        Composer["<b>compose_cells()</b><br/><i>Component: function</i><br/>Sequential multi-cell pipeline.<br/>Comma-separated cell types,<br/>accumulates context dict"]

        R1["<b>ReasoningCell</b><br/><i>heads: rule, nn</i><br/>Structured reasoning<br/>with boundary detection"]

        R2["<b>MemoryCell</b><br/><i>heads: preference_extractor</i><br/>Privacy-preserving<br/>preference extraction"]

        R3["<b>CausalCell</b><br/><i>heads: do_calculus</i><br/>Pearl's do-calculus for<br/>causal inference"]

        R4["<b>EthicsCell</b><br/><i>heads: classifier, pii_scanner</i><br/>Safety classification<br/>and PII detection/redaction"]

        R5["<b>EmpathyCell</b><br/><i>heads: tone_detector</i><br/>Emotional tone detection<br/>with empathetic responses"]

        R6["<b>CuriosityCell</b><br/><i>heads: hypothesis_generator</i><br/>Epistemic curiosity and<br/>question generation"]

        R7["<b>FigLiteralCell</b><br/><i>heads: classifier</i><br/>Figurative vs literal<br/>language classification"]

        R8["<b>R2PCell</b><br/><i>heads: planner</i><br/>Requirements-to-Plan<br/>structured decomposition"]

        R9["<b>TelemetryCell</b><br/><i>heads: collector</i><br/>Telemetry event capture,<br/>action/duration/page parsing"]

        R10["<b>AggregatorCell</b><br/><i>heads: weighted_average,<br/>max_confidence, ensemble</i><br/>Multi-expert output aggregation"]
    end

    Executor -->|"Dispatches to"| R1
    Executor -->|"Dispatches to"| R2
    Executor -->|"Dispatches to"| R3
    Executor -->|"Dispatches to"| R4
    Executor -->|"Dispatches to"| R5
    Executor -->|"Dispatches to"| R6
    Executor -->|"Dispatches to"| R7
    Executor -->|"Dispatches to"| R8
    Executor -->|"Dispatches to"| R9
    Executor -->|"Dispatches to"| R10

    Composer -->|"Calls sequentially"| Executor
    Executor -->|"Validates against"| Registry
    R10 -->|"Aggregates output from<br/>other cells (no recursion)"| Executor

    style Registry fill:#9B2226,stroke:#6B1518,color:#fff
    style Executor fill:#AE2012,stroke:#7A160D,color:#fff
    style Composer fill:#AE2012,stroke:#7A160D,color:#fff
    style R1 fill:#CA6702,stroke:#9A4E01,color:#fff
    style R2 fill:#CA6702,stroke:#9A4E01,color:#fff
    style R3 fill:#CA6702,stroke:#9A4E01,color:#fff
    style R4 fill:#CA6702,stroke:#9A4E01,color:#fff
    style R5 fill:#CA6702,stroke:#9A4E01,color:#fff
    style R6 fill:#CA6702,stroke:#9A4E01,color:#fff
    style R7 fill:#CA6702,stroke:#9A4E01,color:#fff
    style R8 fill:#CA6702,stroke:#9A4E01,color:#fff
    style R9 fill:#CA6702,stroke:#9A4E01,color:#fff
    style R10 fill:#CA6702,stroke:#9A4E01,color:#fff
    style CellsContainer fill:none,stroke:#AE2012,stroke-width:2px,stroke-dasharray:5
```

## MCTS Planning Engine Container

The MCTS engine implements Monte Carlo Tree Search with two selection strategies and integrates optional neural network priors for policy and value estimation.

```mermaid
graph TB
    subgraph MCTSContainer["MCTS Planning Engine (mcts/)"]
        direction TB

        MCTSNode["<b>MCTSNode</b><br/><i>Component: dataclass</i><br/>Fields: id, action, visits,<br/>total_value, policy_prior, children<br/>Methods: ucb1_score(), puct_score(),<br/>to_dict()"]

        RunMCTS["<b>run_mcts()</b><br/><i>Component: function</i><br/>Main search loop:<br/>expand &rarr; select &rarr; simulate<br/>&rarr; backpropagate<br/>10-500 simulations"]

        Categories["<b>TASK_CATEGORIES</b><br/><i>Component: dict</i><br/>5 categories: architecture,<br/>implementation, optimization,<br/>security, research<br/>Each with 5 action steps"]

        DetectCat["<b>_detect_category()</b><br/><i>Component: function</i><br/>Keyword-based task<br/>category classification"]

        Benchmark["<b>benchmark_strategies()</b><br/><i>Component: function</i><br/>Compares MCTS vs Greedy<br/>vs Random on same task.<br/>Returns quality scores and timing"]

        SunburstViz["<b>plot_mcts_tree()</b><br/><i>Component: function</i><br/>Plotly sunburst chart<br/>of search tree with<br/>Viridis color scale"]
    end

    RunMCTS -->|"Creates and traverses"| MCTSNode
    RunMCTS -->|"Detects category via"| DetectCat
    DetectCat -->|"Looks up actions in"| Categories
    Benchmark -->|"Calls run_mcts() for<br/>MCTS strategy"| RunMCTS
    SunburstViz -->|"Walks MCTSNode.to_dict()<br/>tree structure"| MCTSNode

    PN_ext["PolicyNetwork"]
    VN_ext["ValueNetwork"]

    RunMCTS -.->|"Action priors<br/>(when torch available)"| PN_ext
    RunMCTS -.->|"Value estimates<br/>(when torch available)"| VN_ext
    Benchmark -.->|"Uses PolicyNetwork<br/>for greedy strategy"| PN_ext
    Benchmark -.->|"Uses ValueNetwork<br/>for random strategy"| VN_ext

    style MCTSNode fill:#023E8A,stroke:#012A5E,color:#fff
    style RunMCTS fill:#0077B6,stroke:#005A8A,color:#fff
    style Categories fill:#0096C7,stroke:#006F94,color:#fff
    style DetectCat fill:#0096C7,stroke:#006F94,color:#fff
    style Benchmark fill:#0077B6,stroke:#005A8A,color:#fff
    style SunburstViz fill:#48CAE4,stroke:#2BA3C0,color:#000
    style PN_ext fill:#2D6A4F,stroke:#1B4332,color:#fff
    style VN_ext fill:#2D6A4F,stroke:#1B4332,color:#fff
    style MCTSContainer fill:none,stroke:#0077B6,stroke-width:2px,stroke-dasharray:5
```

## MoE Neural Router Container

The routing subsystem combines neural network gating with keyword-based semantic boosting to select top-K experts for a given task.

```mermaid
graph TB
    subgraph RouterContainer["MoE Neural Router (routing/)"]
        direction TB

        RouteTask["<b>route_task()</b><br/><i>Component: function</i><br/>Main routing entry point.<br/>Featurize &rarr; NN gate &rarr;<br/>keyword boost &rarr; top-K select"]

        GetRouter["<b>_get_router()</b><br/><i>Component: function</i><br/>Singleton factory for RouterNet<br/>with fixed seed 42 for<br/>deterministic routing"]

        ExpertNames["<b>EXPERT_NAMES</b><br/><i>Component: list</i><br/>8 experts: Code, Test, Design,<br/>Research, Architecture, Security,<br/>Performance, Docs"]

        Keywords["<b>_EXPERT_KEYWORDS</b><br/><i>Component: dict</i><br/>Per-expert keyword lists<br/>for semantic routing boost<br/>(+0.15 per keyword match)"]

        WeightsViz["<b>plot_expert_weights()</b><br/><i>Component: function</i><br/>Plotly bar chart of<br/>expert routing weights<br/>with color-coded bars"]
    end

    Feat_ext["featurize64()"]
    RN_ext["RouterNet (singleton)"]

    RouteTask -->|"Gets singleton"| GetRouter
    GetRouter -->|"Creates/returns"| RN_ext
    RouteTask -->|"Calls for 64-dim vector"| Feat_ext
    RouteTask -->|"Applies semantic boost"| Keywords
    RouteTask -->|"Selects from"| ExpertNames
    WeightsViz -->|"Visualizes weights<br/>from route_task()"| RouteTask

    style RouteTask fill:#7B2CBF,stroke:#5A189A,color:#fff
    style GetRouter fill:#9D4EDD,stroke:#7B2CBF,color:#fff
    style ExpertNames fill:#C77DFF,stroke:#9D4EDD,color:#000
    style Keywords fill:#C77DFF,stroke:#9D4EDD,color:#000
    style WeightsViz fill:#E0AAFF,stroke:#C77DFF,color:#000
    style Feat_ext fill:#438DD5,stroke:#2E6295,color:#fff
    style RN_ext fill:#2D6A4F,stroke:#1B4332,color:#fff
    style RouterContainer fill:none,stroke:#7B2CBF,stroke-width:2px,stroke-dasharray:5
```

## Agent Orchestrator Container

The orchestrator coordinates 8 specialized agents, mapping each to a cognitive cell type and supporting three routing strategies.

```mermaid
graph TB
    subgraph OrchestratorContainer["Agent Orchestrator (agents/)"]
        direction TB

        Orchestrate["<b>orchestrate()</b><br/><i>Component: function</i><br/>Entry point: selects agents<br/>by strategy, executes cells,<br/>collects results"]

        MoEStrat["<b>_orchestrate_moe()</b><br/><i>Component: function</i><br/>Uses route_task() for<br/>neural expert selection"]

        RRStrat["<b>_orchestrate_round_robin()</b><br/><i>Component: function</i><br/>Sequential selection<br/>AGENTS[:max_agents]"]

        RandStrat["<b>_orchestrate_random()</b><br/><i>Component: function</i><br/>Random sampling from<br/>agent pool"]

        CellMap["<b>_AGENT_CELL_MAP</b><br/><i>Component: dict</i><br/>Agent &rarr; Cell mapping:<br/>SWE &rarr; reasoning<br/>Architect &rarr; r2p<br/>QA &rarr; reasoning<br/>Security &rarr; ethics<br/>DevOps &rarr; telemetry<br/>Research &rarr; causal<br/>Performance &rarr; reasoning<br/>Documentation &rarr; reasoning"]

        A1["<b>SWE Agent</b><br/>Code scaffold generation"]
        A2["<b>Architect Agent</b><br/>System design and patterns"]
        A3["<b>QA Agent</b><br/>Test plan and case generation"]
        A4["<b>Security Agent</b><br/>Threat modeling - OWASP"]
        A5["<b>DevOps Agent</b><br/>Infrastructure planning"]
        A6["<b>Research Agent</b><br/>Technical analysis"]
        A7["<b>Performance Agent</b><br/>Optimization analysis"]
        A8["<b>Documentation Agent</b><br/>Technical writing"]
    end

    Orchestrate -->|"strategy: moe_routing"| MoEStrat
    Orchestrate -->|"strategy: round_robin"| RRStrat
    Orchestrate -->|"strategy: random"| RandStrat

    MoEStrat -->|"Maps expert to agent"| A1
    MoEStrat -->|"Maps expert to agent"| A2
    MoEStrat -->|"Maps expert to agent"| A3
    MoEStrat -->|"Maps expert to agent"| A4
    MoEStrat -->|"Maps expert to agent"| A5
    MoEStrat -->|"Maps expert to agent"| A6
    MoEStrat -->|"Maps expert to agent"| A7
    MoEStrat -->|"Maps expert to agent"| A8

    CellMap -->|"Resolves cell type<br/>for each agent"| Orchestrate

    RT_ext["route_task()"]
    EC_ext["execute_cell()"]

    MoEStrat -->|"Calls for expert selection"| RT_ext
    MoEStrat -->|"Executes mapped cell"| EC_ext
    RRStrat -->|"Executes mapped cell"| EC_ext
    RandStrat -->|"Executes mapped cell"| EC_ext

    style Orchestrate fill:#D4A373,stroke:#B08451,color:#000
    style MoEStrat fill:#E9C46A,stroke:#C9A24A,color:#000
    style RRStrat fill:#E9C46A,stroke:#C9A24A,color:#000
    style RandStrat fill:#E9C46A,stroke:#C9A24A,color:#000
    style CellMap fill:#F4A261,stroke:#D48241,color:#000
    style A1 fill:#264653,stroke:#1A3040,color:#fff
    style A2 fill:#264653,stroke:#1A3040,color:#fff
    style A3 fill:#264653,stroke:#1A3040,color:#fff
    style A4 fill:#264653,stroke:#1A3040,color:#fff
    style A5 fill:#264653,stroke:#1A3040,color:#fff
    style A6 fill:#264653,stroke:#1A3040,color:#fff
    style A7 fill:#264653,stroke:#1A3040,color:#fff
    style A8 fill:#264653,stroke:#1A3040,color:#fff
    style RT_ext fill:#7B2CBF,stroke:#5A189A,color:#fff
    style EC_ext fill:#AE2012,stroke:#7A160D,color:#fff
    style OrchestratorContainer fill:none,stroke:#D4A373,stroke-width:2px,stroke-dasharray:5
```

## Gradio Web App Container

The UI container assembles six interactive tabs, each wired to the corresponding backend containers.

```mermaid
graph TB
    subgraph UIContainer["Gradio Web App (ui/)"]
        direction TB

        BuildApp["<b>build_app()</b><br/><i>Component: function</i><br/>Assembles gr.Blocks with<br/>6 tabs, applies THEME_CSS"]

        ThemeCSS["<b>THEME_CSS</b><br/><i>Component: string constant</i><br/>Dark glassmorphism theme<br/>with gradient backgrounds<br/>and glass-effect panels"]

        Tab1["<b>Features Tab</b><br/>featurize64 + bar chart<br/>Input: text<br/>Output: plot + JSON"]

        Tab2["<b>Cognitive Cells Tab</b><br/>execute_cell with dropdown<br/>Input: cell_type, text, config<br/>Output: JSON"]

        Tab3["<b>Composition Tab</b><br/>compose_cells pipeline<br/>Input: pipeline_str, text<br/>Output: JSON"]

        Tab4["<b>MCTS Planning Tab</b><br/>run_mcts + sunburst chart<br/>+ benchmark_strategies<br/>Input: task, sims, strategy<br/>Output: plot + JSON"]

        Tab5["<b>MoE Router Tab</b><br/>route_task + weight chart<br/>+ feature chart<br/>Input: task, top_k<br/>Output: 2 plots + JSON"]

        Tab6["<b>Orchestration Tab</b><br/>orchestrate with strategy<br/>Input: task, max_agents, strategy<br/>Output: JSON"]
    end

    BuildApp --> Tab1
    BuildApp --> Tab2
    BuildApp --> Tab3
    BuildApp --> Tab4
    BuildApp --> Tab5
    BuildApp --> Tab6
    BuildApp -->|"Applies CSS"| ThemeCSS

    style BuildApp fill:#E76F51,stroke:#C5533A,color:#fff
    style ThemeCSS fill:#F4A261,stroke:#D48241,color:#000
    style Tab1 fill:#2A9D8F,stroke:#1E7A6E,color:#fff
    style Tab2 fill:#2A9D8F,stroke:#1E7A6E,color:#fff
    style Tab3 fill:#2A9D8F,stroke:#1E7A6E,color:#fff
    style Tab4 fill:#2A9D8F,stroke:#1E7A6E,color:#fff
    style Tab5 fill:#2A9D8F,stroke:#1E7A6E,color:#fff
    style Tab6 fill:#2A9D8F,stroke:#1E7A6E,color:#fff
    style UIContainer fill:none,stroke:#E76F51,stroke-width:2px,stroke-dasharray:5
```

## Feature Extraction Container

The feature extraction module produces the 64-dimensional vectors that serve as the universal input representation for the neural models and router.

```mermaid
graph TB
    subgraph FeaturesContainer["Feature Extraction (features.py)"]
        direction TB

        Featurize["<b>featurize64()</b><br/><i>Component: function</i><br/>Text &rarr; 64-dim L2-normalized<br/>feature vector"]

        HashSin["<b>Hash Sinusoidal</b><br/><i>Signal Group: 32 dims</i><br/>SHA-256 &rarr; byte values<br/>&rarr; sin(byte * pi * i)"]

        DomainTags["<b>Domain Tags</b><br/><i>Signal Group: 16 dims</i><br/>Binary presence of: code,<br/>function, class, api, security,<br/>threat, architecture, design,<br/>data, database, test, deploy,<br/>optimize, performance,<br/>research, analyze"]

        Structural["<b>Structural Signals</b><br/><i>Signal Group: 8 dims</i><br/>Length, period/question/<br/>exclamation/comma density,<br/>word count, uppercase,<br/>digit ratio"]

        Sentiment["<b>Sentiment Polarity</b><br/><i>Signal Group: 4 dims</i><br/>Positive score, negative score,<br/>neutral baseline, polarity distance"]

        Novelty["<b>Novelty/Complexity</b><br/><i>Signal Group: 4 dims</i><br/>Lexical diversity, line count,<br/>nesting depth, max word length"]

        PlotFeats["<b>plot_features()</b><br/><i>Component: function</i><br/>Plotly bar chart with<br/>color-coded signal groups"]
    end

    Featurize --> HashSin
    Featurize --> DomainTags
    Featurize --> Structural
    Featurize --> Sentiment
    Featurize --> Novelty
    PlotFeats -->|"Visualizes output of"| Featurize

    style Featurize fill:#438DD5,stroke:#2E6295,color:#fff
    style HashSin fill:#6BAED6,stroke:#4A8AB5,color:#000
    style DomainTags fill:#6BAED6,stroke:#4A8AB5,color:#000
    style Structural fill:#6BAED6,stroke:#4A8AB5,color:#000
    style Sentiment fill:#6BAED6,stroke:#4A8AB5,color:#000
    style Novelty fill:#6BAED6,stroke:#4A8AB5,color:#000
    style PlotFeats fill:#9ECAE1,stroke:#6BAED6,color:#000
    style FeaturesContainer fill:none,stroke:#438DD5,stroke-width:2px,stroke-dasharray:5
```

## Full System Component Interaction

This diagram shows how the major components across all containers connect at the function-call level.

```mermaid
graph LR
    subgraph UI["Gradio Web App"]
        BuildApp["build_app()"]
    end

    subgraph Features["Feature Extraction"]
        F64["featurize64()"]
        PF["plot_features()"]
    end

    subgraph Models["Neural Network Models"]
        RN["RouterNet"]
        PN["PolicyNetwork"]
        VN["ValueNetwork"]
        ET["ExpertTower x16"]
        MoE7M["MixtureOfExperts7M"]
    end

    subgraph Cells["Cognitive Cell Engine"]
        EC["execute_cell()"]
        CC["compose_cells()"]
        REG["CELL_TYPES registry"]
    end

    subgraph MCTSEng["MCTS Planning Engine"]
        RM["run_mcts()"]
        BS["benchmark_strategies()"]
        PMT["plot_mcts_tree()"]
    end

    subgraph Routing["MoE Neural Router"]
        RT["route_task()"]
        PEW["plot_expert_weights()"]
    end

    subgraph Agents["Agent Orchestrator"]
        ORCH["orchestrate()"]
        ACM["_AGENT_CELL_MAP"]
    end

    BuildApp --> F64
    BuildApp --> PF
    BuildApp --> EC
    BuildApp --> CC
    BuildApp --> RM
    BuildApp --> BS
    BuildApp --> PMT
    BuildApp --> RT
    BuildApp --> PEW
    BuildApp --> ORCH

    RT --> F64
    RT --> RN
    RM --> PN
    RM --> VN
    BS --> RM
    BS --> PN
    BS --> VN
    ORCH --> RT
    ORCH --> EC
    ORCH --> ACM
    EC --> REG
    CC --> EC
    MoE7M --> ET

    style UI fill:#E76F51,stroke:#C5533A,color:#fff
    style Features fill:#438DD5,stroke:#2E6295,color:#fff
    style Models fill:#2D6A4F,stroke:#1B4332,color:#fff
    style Cells fill:#AE2012,stroke:#7A160D,color:#fff
    style MCTSEng fill:#0077B6,stroke:#005A8A,color:#fff
    style Routing fill:#7B2CBF,stroke:#5A189A,color:#fff
    style Agents fill:#D4A373,stroke:#B08451,color:#000
```
