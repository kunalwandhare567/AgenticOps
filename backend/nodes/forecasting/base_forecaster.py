
"""
d:/Before_done/forecasting_node/base_forecaster.py
===================================================
Core forecasting engine — called by all 13 mode-specific forecasting functions.

Flow per call:
  1. Look up this failure mode's config (primary metric, algorithm, threshold).
  2. Pull the raw feature trajectory from buffer.py (episode history).
  3. Run the selected algorithm → predictions list.
  4. Compute TTF (minutes until threshold breach).
  5. Compute confidence score.
  6. Check secondary metric for compound breach confirmation.
  7. Return structured result dict.

Public API:
  run_forecast(episode_id, failure_mode, current_features) → dict
"""
from __future__ import annotations

from typing import Any, Optional

from .algorithms import (
    arima_forecast,
    change_point_forecast,
    compute_ttf,
    exponential_forecast,
    linear_forecast,
)
from .buffer import append_feature_row, get_metric_series, get_episode_length
from .thresholds import (
    ALGO_ARIMA, ALGO_CHANGE, ALGO_EXP, ALGO_LINEAR,
    ARIMA_ORDER, FORECAST_STEPS, STEP_INTERVAL_MIN,
    get_algorithm, get_config, get_critical_threshold,
    get_direction, get_primary_metric, get_secondary,
)


def run_forecast(
    episode_id:       str,
    failure_mode:     str,
    current_features: dict[str, Any],
) -> dict:
    """
    Unified forecasting entry-point for all 13 failure modes.

    Args:
        episode_id:        Current episode identifier (e.g. "ep_MEMORY_LEAK_20260723_...").
        failure_mode:      Failure mode string (e.g. "MEMORY_LEAK", "CPU_SATURATION").
        current_features:  The latest feature row dict produced by Stage 1 FE.
                           This is appended to the buffer FIRST, then the full
                           episode history is used to forecast.

    Returns:
        {
          "failure_mode":          str,
          "primary_metric":        str,
          "algorithm":             str,
          "critical_threshold":    float,
          "history_steps":         int,
          "forecast_steps":        int,
          "current_value":         float,
          "predictions":           list[float],
          "timestamps_min":        list[float],
          "time_to_failure":       float | None,   ← minutes
          "forecast_confidence":   float,          ← 0.0–1.0
          "threshold_crossed":     bool,
          "secondary_check":       dict | None,
        }
        Returns empty dict for NONE mode (no forecast needed).
    """
    # ── NONE mode → no forecast ───────────────────────────────────────────────
    if failure_mode == "NONE" or failure_mode is None:
        return {}

    # ── 1. Append latest feature row to buffer ────────────────────────────────
    append_feature_row(episode_id, current_features)

    # ── 2. Load config ────────────────────────────────────────────────────────
    primary_metric     = get_primary_metric(failure_mode)
    algorithm          = get_algorithm(failure_mode)
    critical_threshold = get_critical_threshold(failure_mode)
    direction          = get_direction(failure_mode)
    sec_metric, sec_threshold = get_secondary(failure_mode)

    if primary_metric is None or algorithm is None:
        return {}

    # ── 3. Pull series from buffer ────────────────────────────────────────────
    series = get_metric_series(episode_id, primary_metric)
    history_steps = get_episode_length(episode_id)
    current_value = float(current_features.get(primary_metric, series[-1] if series else 0.0))

    # ── 4. Run algorithm ──────────────────────────────────────────────────────
    if algorithm == ALGO_LINEAR:
        predictions, confidence = linear_forecast(series, FORECAST_STEPS)

    elif algorithm == ALGO_EXP:
        predictions, confidence = exponential_forecast(series, FORECAST_STEPS)

    elif algorithm == ALGO_ARIMA:
        predictions, confidence = arima_forecast(series, FORECAST_STEPS, ARIMA_ORDER)

    elif algorithm == ALGO_CHANGE:
        predictions, confidence = change_point_forecast(series, FORECAST_STEPS)

    else:
        predictions, confidence = linear_forecast(series, FORECAST_STEPS)

    # ── 5. Compute TTF ────────────────────────────────────────────────────────
    ttf = compute_ttf(predictions, critical_threshold, direction, STEP_INTERVAL_MIN)

    # ── 6. Secondary metric check ─────────────────────────────────────────────
    secondary_result = None
    if sec_metric and sec_threshold is not None:
        sec_series  = get_metric_series(episode_id, sec_metric)
        sec_current = float(current_features.get(sec_metric, sec_series[-1] if sec_series else 0.0))

        # Simple current-value check (secondary confirms the mode, no forecast needed)
        sec_breached = sec_current >= sec_threshold  # all secondaries are higher_worse

        secondary_result = {
            "metric":       sec_metric,
            "current":      round(sec_current, 4),
            "threshold":    sec_threshold,
            "breached":     sec_breached,
        }

    # ── 7. Build output ───────────────────────────────────────────────────────
    timestamps_min = [round((i + 1) * STEP_INTERVAL_MIN, 4) for i in range(FORECAST_STEPS)]

    return {
        "failure_mode":        failure_mode,
        "primary_metric":      primary_metric,
        "algorithm":           algorithm,
        "critical_threshold":  critical_threshold,
        "direction":           direction,
        "history_steps":       history_steps,
        "forecast_steps":      FORECAST_STEPS,
        "current_value":       round(current_value, 4),
        "predictions":         [round(p, 4) for p in predictions],
        "timestamps_min":      timestamps_min,
        "time_to_failure":     ttf,                     # minutes or None
        "forecast_confidence": confidence,
        "threshold_crossed":   ttf is not None,
        "secondary_check":     secondary_result,
    }
