"""
app_simulator/pipeline/classifier.py
======================================
Runtime LightGBM classification node.

Loads the frozen model once at startup.
Per cycle: builds 1-row DataFrame, runs inference, stores result in PipelineState.

Auto-train:
  If the model .pkl is missing when the pipeline starts, this module checks
  engineered_features.csv row count. If enough rows exist (or MIN_TRAIN_ROWS is None),
  it launches the offline training script as a subprocess, waits for it to complete,
  then loads the freshly trained model.

  MIN_TRAIN_ROWS = None → train on whatever data is available.
  MIN_TRAIN_ROWS = N    → wait until N rows are collected before training.

Graceful degradation:
  If model is not found AND not enough data yet, returns "PENDING" with 0.0 probability
  and prints a clear message once per session.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from app_data_generator.state import PipelineState
from app_data_generator.config import (
    ENGINEERED_FEAT_CSV,
    FEATURE_NAMES_JSON,
    LABEL_ENCODER_PKL,
    LGBM_MODEL_PKL,
    MIN_TRAIN_ROWS,
)

# Circuit breaker encoding (must match metrics_features.py)
CB_ENCODE = {"closed": 0, "half-open": 1, "open": 2}

_MODEL        = None
_LABEL_ENC    = None
_FEATURE_NAMES: list[str] = []
_WARN_SHOWN   = False


# =============================================================================
# Internal helpers
# =============================================================================

def _count_engineered_rows() -> int:
    """Count rows in engineered_features.csv (header not counted). Returns 0 if missing."""
    csv_path = ENGINEERED_FEAT_CSV
    if not csv_path.exists():
        return 0
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            # subtract 1 for header; use sum for efficiency on large files
            return max(0, sum(1 for _ in fh) - 1)
    except Exception:
        return 0


def _run_training() -> bool:
    """
    Launch offline/train_classifier.py as a subprocess.
    Blocks until training completes.

    Returns True if training succeeded, False otherwise.
    """
    train_script = Path(__file__).resolve().parent.parent / "offline" / "train_classifier.py"
    if not train_script.exists():
        print(f"[Classifier] ERROR: training script not found at {train_script}")
        return False

    print(f"[Classifier] Starting auto-training ({_count_engineered_rows():,} rows available)...")
    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            check=False,
            capture_output=False,   # let stdout/stderr pass through
        )
        if result.returncode == 0:
            print("[Classifier] Auto-training completed successfully.")
            return True
        else:
            print(f"[Classifier] Auto-training exited with code {result.returncode}.")
            return False
    except Exception as exc:
        print(f"[Classifier] Auto-training subprocess error: {exc}")
        return False


def _load_from_disk() -> bool:
    """
    Load model, label encoder and feature names from disk.
    Returns True on success.
    """
    global _MODEL, _LABEL_ENC, _FEATURE_NAMES
    try:
        import joblib
        _MODEL     = joblib.load(LGBM_MODEL_PKL)
        _LABEL_ENC = joblib.load(LABEL_ENCODER_PKL)
        with open(FEATURE_NAMES_JSON, "r", encoding="utf-8") as f:
            _FEATURE_NAMES = json.load(f)
        print(f"[Classifier] Loaded LightGBM model ({len(_FEATURE_NAMES)} features)")
        print(f"[Classifier] Classes: {list(_LABEL_ENC.classes_)}")
        return True
    except Exception as exc:
        print(f"[Classifier] ERROR loading model: {exc}")
        return False


# =============================================================================
# Public API
# =============================================================================

def load_classifier() -> bool:
    """
    Load the LightGBM model, label encoder, and feature names.

    Auto-train logic:
      1. If model .pkl already exists → load it and return True.
      2. If model missing:
         a. Count rows in engineered_features.csv.
         b. If MIN_TRAIN_ROWS is None OR row_count >= MIN_TRAIN_ROWS:
            → trigger offline training, then load.
         c. Otherwise → print PENDING warning, return False.

    Returns:
        True if model loaded successfully, False otherwise.
    """
    global _MODEL

    # Case 1: model already trained
    if LGBM_MODEL_PKL.exists():
        return _load_from_disk()

    # Case 2: model missing → check data availability
    row_count = _count_engineered_rows()
    threshold = MIN_TRAIN_ROWS  # None = no minimum

    if threshold is not None and row_count < threshold:
        print(
            f"[Classifier] PENDING: model not trained yet. "
            f"Need {threshold:,} rows, have {row_count:,}.\n"
            f"[Classifier]          Classification will output PENDING until data is ready."
        )
        return False

    if row_count == 0:
        print(
            "[Classifier] PENDING: no engineered_features.csv found yet.\n"
            "[Classifier]          Run the simulator first to generate data."
        )
        return False

    # Auto-train
    print(f"[Classifier] Model not found. Auto-training on {row_count:,} rows ...")
    success = _run_training()
    if success:
        return _load_from_disk()

    print("[Classifier] Auto-training failed. Classification disabled this session.")
    return False


def try_auto_train() -> bool:
    """
    Re-attempt auto-training mid-session (called each cycle when _MODEL is None).
    Only triggers once enough rows exist; thereafter, trains once and loads.

    Returns True if model is now loaded, False if still waiting.
    """
    global _WARN_SHOWN

    if _MODEL is not None:
        return True

    if not LGBM_MODEL_PKL.exists():
        row_count = _count_engineered_rows()
        threshold = MIN_TRAIN_ROWS

        # Still not enough data — just warn once and skip
        if threshold is not None and row_count < threshold:
            if not _WARN_SHOWN:
                print(
                    f"[Classifier] PENDING: {row_count:,}/{threshold:,} rows. "
                    f"Waiting for more data before training."
                )
                _WARN_SHOWN = True
            return False

        if row_count > 0:
            _WARN_SHOWN = False
            print(f"[Classifier] Enough data detected ({row_count:,} rows). Auto-training now...")
            success = _run_training()
            if success:
                return _load_from_disk()
        return False

    # Model file appeared mid-session (e.g. trained externally)
    return _load_from_disk()


def classify(state: PipelineState) -> PipelineState:
    """
    Run LightGBM inference on state.classifier_input.
    Fills state.predicted_failure and state.prediction_probability.

    If the model is not yet loaded, attempts auto-train once.
    Falls back to predicted_failure = "PENDING" if still not ready.

    Args:
        state: PipelineState with classifier_input filled.

    Returns:
        Same state with predicted_failure and prediction_probability set.
    """
    global _WARN_SHOWN

    if _MODEL is None:
        # Try auto-training mid-session
        loaded = try_auto_train()
        if not loaded:
            state.predicted_failure      = "PENDING"
            state.prediction_probability = 0.0
            return state

    feats = state.classifier_input

    # Build feature vector in the exact trained order
    row = []
    for col in _FEATURE_NAMES:
        val = feats.get(col, 0)
        if col == "circuit_breaker_state":
            val = CB_ENCODE.get(str(val).lower(), 0) if isinstance(val, str) else int(val)
        try:
            row.append(float(val))
        except (TypeError, ValueError):
            row.append(0.0)

    X = np.array([row], dtype=float)

    try:
        pred_enc  = _MODEL.predict(X)[0]
        proba     = _MODEL.predict_proba(X)[0]

        # Decode int → string label
        predicted_mode = _LABEL_ENC.inverse_transform([int(pred_enc)])[0]
        probability    = float(np.max(proba))

        state.predicted_failure      = predicted_mode
        state.prediction_probability = round(probability, 4)

    except Exception as exc:
        print(f"[Classifier] ERROR during inference: {exc}")
        state.predicted_failure      = "NONE"
        state.prediction_probability = 0.0

    return state
