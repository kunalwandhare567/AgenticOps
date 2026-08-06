"""
backend/langgraph_pipeline/nodes/n06_forecasting.py
======================================================
LangGraph Node 6 — Forecasting.

Wraps route_forecast() from nodes/forecasting/router.py.

Design:
  - Only runs when window_full == True (tumbling window has settled).
  - Only runs every FORECAST_EVERY_N_CYCLES cycles to avoid running expensive
    ARIMA fits on every 2-second telemetry tick (controlled by config).
  - If the dominant_state is "NONE", returns empty forecast fields immediately.
  - Writes to forecasting_output.csv.

Returns:
    forecast_result, forecast_algorithm, time_to_failure,
    forecast_confidence, threshold_crossed, earliest_ttf_feature
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from Simulator.app_data_generator_for_offline.config import FORECASTING_OUTPUT_CSV, FORECAST_EVERY_N_CYCLES
from nodes.forecasting.router import route_forecast
from Inference_langgraph.state import AIOpsLangState

# ── CSV ───────────────────────────────────────────────────────────────────────
_csv_fh     = None
_csv_writer = None
_csv_lock   = threading.Lock()

# ── Cycle throttle ────────────────────────────────────────────────────────────
_last_forecast_cycle: int = -1
_throttle_lock = threading.Lock()


def _write_csv(row: dict) -> None:
    global _csv_writer, _csv_fh
    with _csv_lock:
        if _csv_fh is None:
            FORECASTING_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            _csv_fh = FORECASTING_OUTPUT_CSV.open("a", newline="", encoding="utf-8")
        if _csv_writer is None:
            write_header = _csv_fh.tell() == 0
            _csv_writer  = csv.DictWriter(
                _csv_fh, fieldnames=list(row.keys()), extrasaction="ignore"
            )
            if write_header:
                _csv_writer.writeheader()
        _csv_writer.writerow(row)
        _csv_fh.flush()


# ── Empty forecast result ─────────────────────────────────────────────────────
_EMPTY = {
    "forecast_result":      {},
    "forecast_algorithm":   None,
    "time_to_failure":      None,
    "forecast_confidence":  None,
    "threshold_crossed":    None,
    "earliest_ttf_feature": None,
}


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """
    Forecasting node — predicts time-to-failure for the dominant failure mode.

    Skips computation if:
    - window_full is False (not enough labels for a stable mode yet)
    - dominant_state is "NONE" (no active failure to forecast)
    - fewer than FORECAST_EVERY_N_CYCLES since last run
    """
    global _last_forecast_cycle

    cycle        = state.get("cycle", 0)
    window_full  = state.get("window_full", False)
    dominant     = state.get("dominant_state", "NONE")
    episode_id   = state.get("episode_id", "")

    # Guard: window must be full and mode must be active
    if not window_full or not dominant or dominant == "NONE":
        return _EMPTY

    # Throttle: only run every FORECAST_EVERY_N_CYCLES cycles
    with _throttle_lock:
        cycles_since = cycle - _last_forecast_cycle
        if cycles_since < FORECAST_EVERY_N_CYCLES and _last_forecast_cycle >= 0:
            # Return last known forecast values (already in state from previous run)
            return {}   # empty dict = no state update = keep previous values
        _last_forecast_cycle = cycle

    features = state.get("classifier_input", {})

    try:
        raw_result = route_forecast(
            failure_mode     = dominant,
            episode_id       = episode_id,
            current_features = features,
        )
    except Exception as exc:
        print(f"[n06_forecasting] Error: {exc}")
        return {**_EMPTY, "error": f"Forecasting error: {exc}"}

    if not raw_result:
        return _EMPTY

    # Extract top-level convenience fields from the formatted result
    ttf        = raw_result.get("time_to_failure")
    conf       = raw_result.get("forecast_confidence", 0.0)
    crossed    = raw_result.get("threshold_crossed", False)
    fc_payload = raw_result.get("forecast", {})
    algorithm  = (
        fc_payload.get("algorithm_used")
        or fc_payload.get("algorithm")
        or raw_result.get("confidence_reason", "unknown")
    )
    earliest_feat = fc_payload.get("earliest_ttf_feature") or fc_payload.get("primary_metric")

    # Write to CSV
    try:
        _write_csv({
            "cycle":             cycle,
            "episode_id":        episode_id,
            "failure_mode":      dominant,
            "timestamp":         state.get("timestamp", 0.0),
            "elapsed_s":         state.get("elapsed_s", 0.0),
            "algorithm":         algorithm,
            "time_to_failure":   ttf,
            "forecast_confidence": conf,
            "threshold_crossed": int(crossed) if crossed is not None else 0,
            "earliest_ttf_feature": earliest_feat,
        })
    except Exception as exc:
        print(f"[n06_forecasting] CSV write error: {exc}")

    return {
        "forecast_result":      raw_result,
        "forecast_algorithm":   algorithm,
        "time_to_failure":      ttf,
        "forecast_confidence":  float(conf) if conf is not None else None,
        "threshold_crossed":    bool(crossed) if crossed is not None else None,
        "earliest_ttf_feature": earliest_feat,
    }
