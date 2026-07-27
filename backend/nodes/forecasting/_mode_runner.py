"""
d:/Before_done/forecasting_node/_mode_runner.py
================================================
Shared forecasting engine used by all 12 mode files.

Each mode file calls run_auto_arima_forecast() with its own critical feature
table and tier-2 algorithm preference. This module handles:
  1. Appending the current feature row to the episode buffer.
  2. Resolving feature series from the buffer (with alias fallback).
  3. Running the appropriate algorithm tier per feature.
  4. Computing per-feature TTF via slope extrapolation.
  5. Aggregating to a single time_to_failure (minimum valid TTF).
  6. Computing the final confidence score with full reason string.
  7. Returning a complete convergence schema dict.

NOT part of the public API — imported only by modes/*.py.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .algorithms import (
    apply_auto_arima,
    apply_exponential_forecast,
    compute_confidence,
    compute_signal_ttf,
)
from .buffer import append_feature_row, get_episode_length
from .feature_lookup import get_feature, get_feature_series
from .thresholds import STEP_INTERVAL_S

# Number of future steps to predict (10 steps × 2s = 20s forecast window)
_FORECAST_STEPS = 10
# Cycles needed for Signal B to reach full 0.30 weight
_MIN_CYCLES_FULL_CONF = 20


def run_auto_arima_forecast(
    episode_id: str,
    failure_mode: str,
    current_features: dict[str, Any],
    critical_features: dict[str, float],
    extra_features: list[str],
    tier2_algorithm: str = "linear",
    sampling_interval: float = STEP_INTERVAL_S,
    forecast_steps: int = _FORECAST_STEPS,
    is_upper_threshold: bool = True,
) -> dict:
    """
    Run the unified Auto-ARIMA multi-feature forecast for one failure mode.

    Algorithm tier selection:
      Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear inside apply_auto_arima)
      Tier 2: Only invoked when Tier-1 degrades to 'linear' or 'constant'.
              - tier2_algorithm='exponential' → apply_exponential_forecast()
                (used for RETRY_STORM and CASCADING_FAILURE)
              - tier2_algorithm='linear' → keep the linear result from Tier-1
                (all other modes)

    Args:
        episode_id:          Active episode identifier.
        failure_mode:        Mode string (e.g. 'MEMORY_LEAK').
        current_features:    Latest feature row dict from the pipeline.
        critical_features:   {feature_name: critical_threshold} — features
                             used for TTF computation.
        extra_features:      Additional feature names to forecast and include
                             in predictions dict (but NOT used for TTF).
        tier2_algorithm:     'exponential' | 'linear'
        sampling_interval:   Seconds between steps (default 2.0).
        forecast_steps:      How many future steps to forecast (default 10).
        is_upper_threshold:  True if crossing ABOVE threshold = failure.

    Returns:
        Convergence schema dict (all keys always present):
        {
            failure_mode, algorithm_used, history_steps, forecast_horizon_s,
            timestamps_s, predictions, current_values, critical_thresholds,
            feature_ttfs, feature_slopes, threshold_crossing_status,
            time_to_failure, earliest_ttf_feature,
            forecast_confidence, confidence_reason, threshold_crossed
        }
    """
    # ── 1. Append current cycle to episode buffer ─────────────────────────────
    append_feature_row(episode_id, current_features)
    n_history = get_episode_length(episode_id)

    # ── 2. Build full feature list ────────────────────────────────────────────
    all_features: list[str] = list(critical_features.keys()) + [
        f for f in extra_features if f not in critical_features
    ]
    timestamps_s = [i * sampling_interval for i in range(1, forecast_steps + 1)]

    # ── 3. Forecast each feature ──────────────────────────────────────────────
    feature_histories: dict[str, list[float]] = {}
    current_values: dict[str, float] = {}
    forecasts: dict[str, list[float]] = {}
    model_types: dict[str, str] = {}
    fitted_models: dict[str, object | None] = {}

    for feat in all_features:
        history = get_feature_series(episode_id, feat)
        current_val = get_feature(current_features, feat, default=0.0)

        feature_histories[feat] = history
        current_values[feat] = round(current_val, 6)

        if not history or len(history) < 2:
            forecasts[feat] = [round(current_val, 4)] * forecast_steps
            model_types[feat] = "constant"
            fitted_models[feat] = None
            continue

        # Tier 1: Auto-ARIMA (internally falls back to linear if needed)
        fc_vals, model_type, fitted_model = apply_auto_arima(history, forecast_steps)

        # Tier 2: If Tier-1 fell all the way to linear/constant AND
        #          this mode requests exponential as tier-2, try it.
        if model_type in ("linear", "constant") and tier2_algorithm == "exponential":
            exp_vals, exp_type, exp_coeffs = apply_exponential_forecast(history, forecast_steps)
            if exp_type == "exponential":
                fc_vals, model_type, fitted_model = exp_vals, exp_type, exp_coeffs

        forecasts[feat] = [round(v, 4) for v in fc_vals]
        model_types[feat] = model_type
        fitted_models[feat] = fitted_model

    # ── 4. Compute per-feature TTF ────────────────────────────────────────────
    feature_ttfs: dict[str, float | None] = {}
    feature_slopes: dict[str, float] = {}
    threshold_crossing_status: dict[str, bool] = {}

    for feat, threshold in critical_features.items():
        current_val = current_values.get(feat, 0.0)
        fc_vals = forecasts.get(feat, [current_val] * forecast_steps)

        ttf, slope = compute_signal_ttf(
            current_value=current_val,
            threshold=threshold,
            forecast_values=fc_vals,
            sampling_interval=sampling_interval,
            is_upper_threshold=is_upper_threshold,
        )
        feature_ttfs[feat] = ttf
        feature_slopes[feat] = round(slope, 6)
        threshold_crossing_status[feat] = (
            ttf is not None and not (isinstance(ttf, float) and math.isnan(ttf))
        )

    # ── 5. Final TTF = minimum valid positive TTF across all critical features ─
    valid_pairs = [
        (ttf, feat)
        for feat, ttf in feature_ttfs.items()
        if ttf is not None and not math.isnan(ttf) and ttf >= 0.0
    ]
    if valid_pairs:
        final_ttf, earliest_feature = min(valid_pairs, key=lambda p: p[0])
        final_ttf = float(round(final_ttf, 2))
    else:
        final_ttf = None
        earliest_feature = None

    # ── 6. Confidence — computed from PRIMARY critical feature's model ─────────
    primary_feat = next(iter(critical_features))
    primary_history = feature_histories.get(primary_feat, [])
    primary_model_type = model_types.get(primary_feat, "constant")
    primary_model = fitted_models.get(primary_feat)

    confidence, confidence_reason = compute_confidence(
        data=primary_history,
        model_type=primary_model_type,
        fitted_model=primary_model,
        n_history=n_history,
        min_cycles_for_full_conf=_MIN_CYCLES_FULL_CONF,
    )

    # ── 7. Overall algorithm label (most common across all features) ──────────
    algo_counts = Counter(model_types.values())
    algorithm_used = algo_counts.most_common(1)[0][0] if algo_counts else "unknown"

    # ── 8. Return convergence schema ──────────────────────────────────────────
    return {
        "failure_mode":              failure_mode,
        "algorithm_used":            algorithm_used,
        "history_steps":             n_history,
        "forecast_horizon_s":        forecast_steps * sampling_interval,
        "timestamps_s":              timestamps_s,
        "predictions":               forecasts,
        "current_values":            current_values,
        "critical_thresholds":       critical_features,
        "feature_ttfs":              feature_ttfs,
        "feature_slopes":            feature_slopes,
        "threshold_crossing_status": threshold_crossing_status,
        "time_to_failure":           final_ttf,
        "earliest_ttf_feature":      earliest_feature,
        "forecast_confidence":       confidence,
        "confidence_reason":         confidence_reason,
        "threshold_crossed":         final_ttf is not None,
    }
