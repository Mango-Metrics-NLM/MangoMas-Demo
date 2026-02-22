"""
Gradio UI Builder — assembles the 6-tab interactive demo.

Tabs:
1. Feature Extraction — featurize64 + bar chart
2. Cognitive Cells — execute any of 10 cells
3. Cell Composition — multi-cell pipeline
4. MCTS Planning — tree search with sunburst chart
5. MoE Router — expert routing with visualization
6. Agent Orchestration — multi-agent task execution
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from mangomas_demo.agents.orchestrator import orchestrate
from mangomas_demo.cells.executor import compose_cells, execute_cell
from mangomas_demo.cells.types import CELL_TYPES
from mangomas_demo.features import featurize64, plot_features
from mangomas_demo.mcts.benchmark import benchmark_strategies
from mangomas_demo.mcts.engine import run_mcts
from mangomas_demo.mcts.viz import plot_mcts_tree
from mangomas_demo.routing.router import route_task
from mangomas_demo.routing.viz import plot_expert_weights
from mangomas_demo.ui.styles import THEME_CSS


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application.

    Returns:
        A configured ``gr.Blocks`` instance ready for ``.launch()``.
    """
    with gr.Blocks(css=THEME_CSS, title="MangoMAS Demo") as demo:
        gr.Markdown(
            "# 🥭 MangoMAS — Multi-Agent Cognitive Architecture\n"
            "**10 Cognitive Cells | MCTS Planning | 7M MoE Router | 8-Agent Orchestration**"
        )

        with gr.Tabs():
            _build_features_tab()
            _build_cells_tab()
            _build_composition_tab()
            _build_mcts_tab()
            _build_routing_tab()
            _build_orchestration_tab()

    return demo


# ---------------------------------------------------------------------------
# Tab builders (private)
# ---------------------------------------------------------------------------

def _build_features_tab() -> None:
    """Tab 1: Feature Extraction."""
    with gr.Tab("🔬 Features"):
        gr.Markdown("### Feature Extraction (64-D Vector)")
        text_in = gr.Textbox(
            label="Input Text", value="Design a secure API gateway with rate limiting",
            lines=2,
        )
        btn = gr.Button("Extract Features", variant="primary")
        plot_out = gr.Plot(label="Feature Vector")
        json_out = gr.JSON(label="Raw Vector")

        def _run_features(text: str) -> tuple[Any, list[float]]:
            feats = featurize64(text)
            fig = plot_features(feats, title=f"Features for: {text[:40]}...")
            return fig, feats

        btn.click(_run_features, inputs=[text_in], outputs=[plot_out, json_out])


def _build_cells_tab() -> None:
    """Tab 2: Individual Cell Execution."""
    with gr.Tab("🧬 Cognitive Cells"):
        gr.Markdown("### Execute Cognitive Cells")
        with gr.Row():
            cell_drop = gr.Dropdown(
                choices=list(CELL_TYPES.keys()),
                value="reasoning",
                label="Cell Type",
            )
            config_box = gr.Textbox(label="Config JSON", value="{}", lines=1)
        text_in = gr.Textbox(
            label="Input Text", value="Analyze the performance bottleneck in our API",
            lines=2,
        )
        btn = gr.Button("Execute Cell", variant="primary")
        json_out = gr.JSON(label="Cell Output")

        def _run_cell(ct: str, text: str, config: str) -> dict[str, Any]:
            return execute_cell(ct, text, config)

        btn.click(_run_cell, inputs=[cell_drop, text_in, config_box], outputs=[json_out])


def _build_composition_tab() -> None:
    """Tab 3: Multi-Cell Pipeline."""
    with gr.Tab("🔗 Composition"):
        gr.Markdown("### Cell Composition Pipeline")
        pipe_in = gr.Textbox(
            label="Pipeline (comma-separated)", value="reasoning, ethics, causal",
            lines=1,
        )
        text_in = gr.Textbox(
            label="Input Text", value="User john@example.com reported a security issue",
            lines=2,
        )
        btn = gr.Button("Run Pipeline", variant="primary")
        json_out = gr.JSON(label="Pipeline Output")

        btn.click(compose_cells, inputs=[pipe_in, text_in], outputs=[json_out])


def _build_mcts_tab() -> None:
    """Tab 4: MCTS Planning."""
    with gr.Tab("🌲 MCTS Planning"):
        gr.Markdown("### Monte Carlo Tree Search — Task Decomposition")
        with gr.Row():
            task_in = gr.Textbox(
                label="Task", value="Design a microservices architecture for e-commerce",
                lines=2,
            )
            with gr.Column():
                sims_in = gr.Slider(10, 500, value=100, step=10, label="Simulations")
                strategy_in = gr.Dropdown(
                    choices=["ucb1", "puct"], value="ucb1", label="Strategy"
                )
        btn = gr.Button("Run MCTS", variant="primary")
        with gr.Row():
            tree_out = gr.Plot(label="Search Tree")
            json_out = gr.JSON(label="Results")
        bench_btn = gr.Button("Benchmark Strategies", variant="secondary")
        bench_out = gr.JSON(label="Benchmark Results")

        def _run_mcts(task: str, sims: int, strat: str) -> tuple[Any, dict[str, Any]]:
            result = run_mcts(task, int(sims), strategy=strat)
            fig = plot_mcts_tree(result["tree"])
            return fig, result

        def _run_bench(task: str) -> dict[str, Any]:
            return benchmark_strategies(task)

        btn.click(_run_mcts, inputs=[task_in, sims_in, strategy_in], outputs=[tree_out, json_out])
        bench_btn.click(_run_bench, inputs=[task_in], outputs=[bench_out])


def _build_routing_tab() -> None:
    """Tab 5: MoE Neural Router."""
    with gr.Tab("🧠 MoE Router"):
        gr.Markdown("### Neural Mixture-of-Experts Routing (~7M params)")
        with gr.Row():
            task_in = gr.Textbox(
                label="Task", value="Implement authentication with JWT tokens",
                lines=2,
            )
            topk_in = gr.Slider(1, 8, value=3, step=1, label="Top-K Experts")
        btn = gr.Button("Route Task", variant="primary")
        with gr.Row():
            weights_out = gr.Plot(label="Expert Weights")
            feature_out = gr.Plot(label="Input Features")
        json_out = gr.JSON(label="Full Routing Data")

        def _run_route(task: str, top_k: int) -> tuple[Any, Any, dict[str, Any]]:
            result = route_task(task, int(top_k))
            w_fig = plot_expert_weights(result["all_weights"])
            f_fig = plot_features(result["features"], title="Input Feature Vector")
            return w_fig, f_fig, result

        btn.click(_run_route, inputs=[task_in, topk_in], outputs=[weights_out, feature_out, json_out])


def _build_orchestration_tab() -> None:
    """Tab 6: Agent Orchestration."""
    with gr.Tab("🤖 Orchestration"):
        gr.Markdown("### Multi-Agent Task Orchestration")
        with gr.Row():
            task_in = gr.Textbox(
                label="Task", value="Build a REST API with authentication and tests",
                lines=2,
            )
            with gr.Column():
                agent_count = gr.Slider(1, 8, value=3, step=1, label="Max Agents")
                strategy_in = gr.Dropdown(
                    choices=["moe_routing", "round_robin", "random"],
                    value="moe_routing",
                    label="Strategy",
                )
        btn = gr.Button("Orchestrate", variant="primary")
        json_out = gr.JSON(label="Orchestration Results")

        btn.click(
            orchestrate,
            inputs=[task_in, agent_count, strategy_in],
            outputs=[json_out],
        )
