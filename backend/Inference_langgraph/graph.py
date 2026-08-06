"""
backend/langgraph_pipeline/graph.py
=====================================
Assembles the AIOps LangGraph StateGraph.

Node order:
  collect → feature_eng → prelim_severity → classify → tumbling_window
  → forecasting → severity_update → reliability → human_gate → db_writer → END

Routing:
  - After collect: if error == "no_data"  → END (outer loop retries)
                   otherwise              → feature_eng
  - After any other node: if error is set → db_writer (write partial state) → END
  - All other nodes: linear chain

Checkpointing:
  - SqliteSaver is used to persist the full state after every node.
  - If the pipeline crashes mid-episode, it resumes from the last completed node.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SAVER = True
except ImportError:
    _HAS_SAVER = False

from Inference_langgraph.state import AIOpsLangState
from Inference_langgraph.nodes import (
    n01_collect,
    n02_feature_engineering,
    n03_prelim_severity,
    n04_classify,
    n05_tumbling_window,
    n06_forecasting,
    n07_severity_update,
    n08_reliability,
    n09_human_gate,
    n10_db_writer,
)
from Simulator.app_data_generator_for_offline.config import LANGGRAPH_CHECKPOINT_DB


# =============================================================================
# Routing helpers
# =============================================================================

def _route_after_collect(state: AIOpsLangState) -> str:
    """
    After collect:
      - no_data   → END (outer loop will retry after POLL_INTERVAL_MS)
      - any error → END (log already printed by the node)
      - success   → feature_eng
    """
    err = state.get("error")
    if err == "no_data" or (err and "error" in err.lower()):
        return END
    return "feature_eng"


def _route_on_error(state: AIOpsLangState) -> str:
    """
    After any middle node: if error is set → skip remaining compute, write what
    we have to DB, then END.
    """
    if state.get("error"):
        return "db_writer"
    return "ok"


# =============================================================================
# Graph factory
# =============================================================================

def build_graph(checkpoint_db: str | None = None):
    """
    Build and compile the AIOps StateGraph.

    Args:
        checkpoint_db: Path to the SQLite file for checkpointing.
                       If None, uses LANGGRAPH_CHECKPOINT_DB from config.
                       Set to "" to disable checkpointing.

    Returns:
        Compiled LangGraph graph (supports .invoke() and .stream()).
    """
    g = StateGraph(AIOpsLangState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("collect",           n01_collect.run)
    g.add_node("feature_eng",       n02_feature_engineering.run)
    g.add_node("prelim_severity",   n03_prelim_severity.run)
    g.add_node("classify",          n04_classify.run)
    g.add_node("tumbling_window",   n05_tumbling_window.run)
    g.add_node("forecasting",       n06_forecasting.run)
    g.add_node("severity_update",   n07_severity_update.run)
    g.add_node("reliability",       n08_reliability.run)
    g.add_node("human_gate",        n09_human_gate.run)
    g.add_node("db_writer",         n10_db_writer.run)

    # ── Entry point ───────────────────────────────────────────────────────────
    g.set_entry_point("collect")

    # ── Routing after collect ─────────────────────────────────────────────────
    g.add_conditional_edges(
        "collect",
        _route_after_collect,
        {
            END:          END,
            "feature_eng": "feature_eng",
        },
    )

    # ── Linear chain with error short-circuit ─────────────────────────────────
    # Each node can set state["error"]. If set, we skip remaining compute nodes
    # and jump to db_writer to at least persist the partial state.
    for src, dst in [
        ("feature_eng",     "prelim_severity"),
        ("prelim_severity", "classify"),
        ("classify",        "tumbling_window"),
        ("tumbling_window", "forecasting"),
        ("forecasting",     "severity_update"),
        ("severity_update", "reliability"),
        ("reliability",     "human_gate"),
        ("human_gate",      "db_writer"),
    ]:
        g.add_conditional_edges(
            src,
            _route_on_error,
            {
                "ok":       dst,
                "db_writer": "db_writer",
            },
        )

    # db_writer always terminates
    g.add_edge("db_writer", END)

    # ── Compile with optional checkpointing ───────────────────────────────────
    db_path = checkpoint_db if checkpoint_db is not None else LANGGRAPH_CHECKPOINT_DB

    if db_path and _HAS_SAVER:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        checkpointer = SqliteSaver.from_conn_string(db_path)
        return g.compile(checkpointer=checkpointer)
    else:
        return g.compile()


def get_graph_ascii() -> str:
    """Return Mermaid diagram string for the compiled graph (for debugging)."""
    try:
        graph = build_graph(checkpoint_db="")  # no checkpointing needed for viz
        return graph.get_graph().draw_mermaid()
    except Exception as exc:
        return f"Graph visualisation unavailable: {exc}"
