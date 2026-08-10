"""
backend/langgraph_pipeline/nodes/n04_classify.py
=================================================
LangGraph Node 4 — Classification.

Wraps load_classifier() + classify() from nodes/classification/classifier.py.
Model is loaded ONCE at startup and reused for every subsequent cycle.

Auto-training behaviour:
  - If lgbm_model.pkl is missing, returns {"predicted_failure": "NONE", ...}
    with a note in the console. The model must be trained manually first via:
        python backend/nodes/classification/train_classifier.py --tune --gpu
  - Once trained, the .pkl is saved and reloaded here automatically.

Returns:
    predicted_failure       — failure mode string
    prediction_probability  — confidence score 0.0–1.0
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from Inference_langgraph.nodes.classification.classifier import load_classifier, classify
from Simulator.app_data_generator_for_offline.state import PipelineState
from Inference_langgraph.state import AIOpsLangState

# ── Singleton model load state ────────────────────────────────────────────────
_lock         = threading.Lock()
_model_loaded = False
_load_tried   = False     # avoid hammering the file check every cycle


def init() -> bool:
    """
    Attempt to load the LightGBM model PKL.
    Returns True if successfully loaded.
    Called once at pipeline startup from run_langgraph.py.
    """
    global _model_loaded, _load_tried
    with _lock:
        if not _model_loaded:
            print("[n04_classify] Loading LightGBM classifier...")
            _model_loaded = load_classifier()
            _load_tried   = True
            if not _model_loaded:
                print("[n04_classify] WARNING: model PKL not found. Run:")
                print("  python backend/nodes/classification/train_classifier.py --tune --gpu")
    return _model_loaded


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """Classification node — predicts failure mode from engineered features."""
    global _model_loaded, _load_tried

    # Try lazy load if model wasn't ready at startup
    if not _model_loaded:
        _model_loaded = load_classifier()
        if not _model_loaded:
            return {
                "predicted_failure":     "NONE",
                "prediction_probability": 0.0,
            }

    # Build minimal PipelineState for classify()
    ps = PipelineState(
        raw_metric       = state.get("raw_metric", {}),
        raw_log          = {},
        raw_traces       = [],
        episode_id       = state.get("episode_id", ""),
        failure_mode     = state.get("failure_mode", "NONE"),
        timestamp        = state.get("timestamp", 0.0),
        elapsed_s        = state.get("elapsed_s", 0.0),
        step             = state.get("cycle", 0),
        service          = state.get("service", ""),
        classifier_input = state.get("classifier_input", {}),
        evidence         = state.get("evidence", {}),
    )

    try:
        classified_ps = classify(ps)
    except Exception as exc:
        print(f"[n04_classify] Classify error: {exc}")
        return {
            "predicted_failure":      "NONE",
            "prediction_probability": 0.0,
            "error":                  f"Classify error: {exc}",
        }

    return {
        "predicted_failure":     str(classified_ps.predicted_failure or "NONE"),
        "prediction_probability": float(classified_ps.prediction_probability or 0.0),
    }
