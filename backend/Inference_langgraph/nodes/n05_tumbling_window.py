"""
backend/langgraph_pipeline/nodes/n05_tumbling_window.py
=========================================================
LangGraph Node 5 — Tumbling Window.

Wraps TumblingWindow.update() — maintains a majority-vote window over the
last N classification labels to produce a stable dominant_state.

The TumblingWindow instance is held as a module-level singleton so the
window accumulates predictions across cycles (as in run_pipeline.py).

Returns:
    dominant_state    — the majority-vote failure mode
    vote_distribution — dict of {mode: count}
    window_margin     — fraction difference between top two modes
    window_full       — True when window has N entries
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from nodes.tumbling_window.tumbling_window import TumblingWindow
from Simulator.app_data_generator_for_offline.state import PipelineState
from Inference_langgraph.state import AIOpsLangState

# ── TumblingWindow singleton ──────────────────────────────────────────────────
_lock   = threading.Lock()
_window: TumblingWindow | None = None


def init() -> None:
    """Create TumblingWindow singleton. Called once at pipeline startup."""
    global _window
    with _lock:
        if _window is None:
            print("[n05_tumbling_window] Initialising TumblingWindow...")
            _window = TumblingWindow()


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """Tumbling window node — majority-vote smoother over classification labels."""
    global _window
    if _window is None:
        init()

    cycle = state.get("cycle", 0)

    ps = PipelineState(
        raw_metric            = state.get("raw_metric", {}),
        raw_log               = {},
        raw_traces            = [],
        episode_id            = state.get("episode_id", ""),
        failure_mode          = state.get("failure_mode", "NONE"),
        timestamp             = state.get("timestamp", 0.0),
        elapsed_s             = state.get("elapsed_s", 0.0),
        step                  = cycle,
        service               = state.get("service", ""),
        classifier_input      = state.get("classifier_input", {}),
        evidence              = state.get("evidence", {}),
        predicted_failure     = state.get("predicted_failure", "NONE"),
        prediction_probability = state.get("prediction_probability", 0.0),
        preliminary_severity  = state.get("preliminary_severity", "P4"),
    )

    try:
        updated_ps = _window.update(ps, cycle)
    except Exception as exc:
        print(f"[n05_tumbling_window] Error: {exc}")
        return {
            "dominant_state":    state.get("predicted_failure", "NONE"),
            "vote_distribution": {},
            "window_margin":     0.0,
            "window_full":       False,
            "error":             f"TumblingWindow error: {exc}",
        }

    return {
        "dominant_state":    str(updated_ps.summarized_failure   or "NONE"),
        "vote_distribution": updated_ps.vote_distribution        or {},
        "window_margin":     float(updated_ps.window_margin      or 0.0),
        "window_full":       bool(updated_ps.window_full),
    }
