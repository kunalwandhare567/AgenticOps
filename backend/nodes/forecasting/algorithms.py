"""
d:/Before_done/forecasting_node/algorithms.py
==============================================
Pure statistical forecasting algorithms — no state, no imports from pipeline.

NEW (v2) — Three unified functions for the redesigned forecasting nodes:
  apply_auto_arima       — 3-tier Auto-ARIMA (pmdarima → statsmodels → linear)
  apply_exponential_forecast — exponential regression with linear fallback
  compute_confidence     — fully-computed confidence: Signal A (model fit) + Signal B (data sufficiency)
  compute_signal_ttf     — per-feature TTF via slope extrapolation

LEGACY (v1, kept for backward compatibility):
  linear_forecast, exponential_forecast (old), arima_forecast,
  change_point_forecast, compute_ttf

All new functions return structured dicts or typed tuples — no hardcoded constants.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

from .thresholds import (
    ARIMA_MIN_HISTORY, ARIMA_ORDER, CHANGE_POINT_CONFIG,
    FORECAST_STEPS, LINEAR_MIN_HISTORY,
    SHORT_HISTORY_PENALTY_FACTOR, SHORT_HISTORY_PENALTY_THRESHOLD,
    STEP_INTERVAL_MIN, STEP_INTERVAL_S,
)

# ── Optional heavy dependencies (graceful degradation) ───────────────────────
_pmdarima_auto_arima = None
try:
    from pmdarima import auto_arima as _pmdarima_auto_arima
except ImportError:
    pass

_statsmodels_ARIMA = None
try:
    from statsmodels.tsa.arima.model import ARIMA as _statsmodels_ARIMA
except ImportError:
    pass


# =============================================================================
# NEW v2 — CORE FORECASTING ENGINE
# =============================================================================

def apply_auto_arima(
    data: list[float],
    steps: int = 10,
) -> tuple[list[float], str, object | None]:
    """
    3-tier Auto-ARIMA forecaster.

    Tier 1: pmdarima auto_arima (optimal order selection via AIC search)
    Tier 2: statsmodels ARIMA with AIC-ranked candidate orders
    Tier 3: linear regression fallback (numpy polyfit)

    Args:
        data:  Historical series values (oldest → newest).
        steps: Number of future steps to forecast.

    Returns:
        (forecast_values, model_type_str, fitted_model_or_None)
        model_type_str values:
          'pmdarima'
          'statsmodels_arima_p_d_q'   (e.g. 'statsmodels_arima_1_1_0')
          'linear'
          'constant'
    """
    if not data:
        return [0.0] * steps, "constant", None

    n = len(data)
    arr = np.array(data, dtype=float)

    # Constant / flat signal — no trend to model
    if n < 2 or np.allclose(arr, arr[0], atol=1e-6):
        return [float(arr[-1])] * steps, "constant", None

    # Tier 1: pmdarima
    if _pmdarima_auto_arima is not None and n >= 5:
        try:
            model = _pmdarima_auto_arima(
                data,
                seasonal=False,
                max_p=2, max_q=2, max_d=1,
                suppress_warnings=True,
                error_action="ignore",
                stepwise=True,
            )
            forecast = [float(v) for v in model.predict(n_periods=steps)]
            if any(math.isnan(v) or math.isinf(v) for v in forecast):
                raise ValueError("pmdarima forecast contains NaN/Inf")
            return forecast, "pmdarima", model
        except Exception:
            pass

    # Tier 2: statsmodels ARIMA (AIC-ranked candidates)
    if _statsmodels_ARIMA is not None and n >= 8:
        candidate_orders = [(1, 1, 0), (0, 1, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
        best_aic = float("inf")
        best_result = None
        best_order = None
        for order in candidate_orders:
            min_req = max(order[0], order[2]) + order[1] + 4
            if n < min_req:
                continue
            try:
                res = _statsmodels_ARIMA(data, order=order).fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_result = res
                    best_order = order
            except Exception:
                continue
        if best_result is not None:
            try:
                fc = list(best_result.forecast(steps=steps))
                if not any(math.isnan(v) or math.isinf(v) for v in fc):
                    tag = f"statsmodels_arima_{best_order[0]}_{best_order[1]}_{best_order[2]}"
                    return [float(v) for v in fc], tag, best_result
            except Exception:
                pass

    # Tier 3: linear fallback
    return _linear_fallback(data, steps)


def apply_exponential_forecast(
    data: list[float],
    steps: int = 10,
) -> tuple[list[float], str, np.ndarray | None]:
    """
    Exponential regression: fit log(y) = a·x + b in log-space.

    Best for: RETRY_STORM (feedback amplification), CASCADING_FAILURE (multi-stage escalation).

    Falls back to linear if:
      - Any data value is non-positive (cannot take log).
      - Fewer than 3 data points.

    Returns:
        (forecast_values, 'exponential' | 'linear' | 'constant', coeffs_or_None)
    """
    if not data or len(data) < 3:
        val = float(data[-1]) if data else 0.0
        return [val] * steps, "constant", None

    # Require all positive for log transform
    if any(v <= 0 for v in data):
        vals, mtype, _ = _linear_fallback(data, steps)
        return vals, mtype, None

    try:
        n = len(data)
        x = np.arange(n, dtype=float)
        log_y = np.log(np.array(data, dtype=float))
        coeffs = np.polyfit(x, log_y, 1)
        future_x = np.arange(n, n + steps, dtype=float)
        forecast = [float(np.exp(np.polyval(coeffs, xi))) for xi in future_x]
        # Sanity: clip extreme values
        max_val = float(max(data)) * 100.0
        forecast = [min(v, max_val) for v in forecast]
        return forecast, "exponential", coeffs
    except Exception:
        vals, mtype, _ = _linear_fallback(data, steps)
        return vals, mtype, None


def _linear_fallback(
    data: list[float],
    steps: int,
) -> tuple[list[float], str, None]:
    """Simple numpy polyfit linear regression fallback. Always succeeds."""
    if not data:
        return [0.0] * steps, "constant", None
    if len(data) < 2:
        return [float(data[-1])] * steps, "constant", None
    try:
        n = len(data)
        x = np.arange(n, dtype=float)
        y = np.array(data, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(n, n + steps, dtype=float)
        forecast = [float(slope * xi + intercept) for xi in future_x]
        return forecast, "linear", None
    except Exception:
        return [float(data[-1])] * steps, "constant", None


# =============================================================================
# NEW v2 — CONFIDENCE COMPUTATION (fully mathematical, zero hardcoding)
# =============================================================================

def compute_confidence(
    data: list[float],
    model_type: str,
    fitted_model: object | None,
    n_history: int,
    min_cycles_for_full_conf: int = 20,
) -> tuple[float, str]:
    """
    Compute confidence score [0.0 – 1.0] with full explanation string.

    Two independent signals are summed:

    Signal A — Model Fit Quality (0.0 → 0.70):
        Measures how well the chosen algorithm captures the historical trend.
        • pmdarima / statsmodels  → residual noise ratio:
              conf_A = max(0, min(0.70, 1 – 2 × (std(residuals) / mean(|y|))))
        • exponential             → log-space R²:
              conf_A = max(0, min(0.70, R²_log))
        • linear                  → R²:
              conf_A = max(0, min(0.70, R²))
        • constant / unknown      → 0.0

    Signal B — Data Sufficiency (0.0 → 0.30):
        Measures how many cycles of history are available.
              conf_B = min(0.30, 0.30 × (n_history / min_cycles_for_full_conf))

    Total: confidence = conf_A + conf_B  (range: 0.0 – 1.0)

    Args:
        data:                     Historical feature series.
        model_type:               String returned by apply_auto_arima/apply_exponential_forecast.
        fitted_model:             Fitted model object (or None for linear/constant).
        n_history:                Total number of cycles in the episode buffer.
        min_cycles_for_full_conf: Cycles needed for full 0.30 Signal B.

    Returns:
        (confidence_score, reason_string)
    """
    if not data:
        conf_B = 0.0
        data_reason = f"0/{min_cycles_for_full_conf} history cycles -> data_quality={conf_B:.4f}"
        return 0.0, f"no data available; {data_reason}; total confidence=0.0000"

    arr = np.array(data, dtype=float)
    n = len(arr)
    mean_signal = float(np.mean(np.abs(arr))) or 1e-9

    # ── Signal B: data sufficiency ────────────────────────────────────────────
    conf_B = min(0.30, 0.30 * (n_history / max(min_cycles_for_full_conf, 1)))
    data_reason = f"{n_history}/{min_cycles_for_full_conf} history cycles -> data_quality={conf_B:.4f}"

    # ── Signal A: model fit quality ───────────────────────────────────────────
    conf_A: float
    model_reason: str

    if model_type == "constant":
        conf_A = 0.0
        model_reason = "constant signal: no active degradation trend -> fit_quality=0.0000"

    elif model_type == "pmdarima" and fitted_model is not None:
        try:
            insample = np.array(fitted_model.predict_in_sample(), dtype=float)
            residuals = arr[-len(insample):] - insample
            noise_ratio = float(np.std(residuals)) / mean_signal
            conf_A = max(0.0, min(0.70, 1.0 - 2.0 * noise_ratio))
            model_reason = (
                f"pmdarima auto_arima: noise_ratio={noise_ratio:.4f} "
                f"-> fit_quality={conf_A:.4f}"
            )
        except Exception:
            # pmdarima model exists but residuals unavailable — mid-range estimate
            conf_A = 0.35
            model_reason = f"pmdarima auto_arima: in-sample unavailable -> fit_quality={conf_A:.4f}"

    elif model_type.startswith("statsmodels_arima") and fitted_model is not None:
        try:
            residuals = np.array(list(fitted_model.resid), dtype=float)
            noise_ratio = float(np.std(residuals)) / mean_signal
            conf_A = max(0.0, min(0.70, 1.0 - 2.0 * noise_ratio))
            aic = getattr(fitted_model, "aic", float("nan"))
            model_reason = (
                f"{model_type}: AIC={aic:.2f}, noise_ratio={noise_ratio:.4f} "
                f"-> fit_quality={conf_A:.4f}"
            )
        except Exception:
            conf_A = 0.30
            model_reason = f"{model_type}: residuals unavailable -> fit_quality={conf_A:.4f}"

    elif model_type == "exponential":
        try:
            x = np.arange(n, dtype=float)
            log_y = np.log(np.maximum(arr, 1e-9))
            coeffs = np.polyfit(x, log_y, 1)
            log_y_pred = np.polyval(coeffs, x)
            ss_res = float(np.sum((log_y - log_y_pred) ** 2))
            ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
            r2 = max(0.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-9)))
            conf_A = max(0.0, min(0.70, r2))
            model_reason = f"exponential regression: log-space R2={r2:.4f} -> fit_quality={conf_A:.4f}"
        except Exception:
            conf_A = 0.25
            model_reason = f"exponential regression: R2 unavailable -> fit_quality={conf_A:.4f}"

    elif model_type == "linear":
        try:
            x = np.arange(n, dtype=float)
            coeffs = np.polyfit(x, arr, 1)
            y_pred = np.polyval(coeffs, x)
            ss_res = float(np.sum((arr - y_pred) ** 2))
            ss_tot = float(np.sum((arr - np.mean(arr)) ** 2))
            r2 = max(0.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-9)))
            conf_A = max(0.0, min(0.70, r2))
            model_reason = f"linear regression: R2={r2:.4f} -> fit_quality={conf_A:.4f}"
        except Exception:
            conf_A = 0.20
            model_reason = f"linear regression: R2 unavailable -> fit_quality={conf_A:.4f}"

    else:
        conf_A = 0.20
        model_reason = f"unknown model '{model_type}' -> fit_quality={conf_A:.4f}"

    confidence = round(min(1.0, conf_A + conf_B), 4)
    reason = f"{model_reason}; {data_reason}; total confidence={confidence:.4f}"
    return confidence, reason


# =============================================================================
# NEW v2 — SIGNAL-LEVEL TTF COMPUTATION
# =============================================================================

def compute_signal_ttf(
    current_value: float,
    threshold: float,
    forecast_values: list[float],
    sampling_interval: float = STEP_INTERVAL_S,
    is_upper_threshold: bool = True,
) -> tuple[float | None, float]:
    """
    Compute per-feature TTF (seconds) and trend slope using the forecast trajectory.

    Algorithm:
        1. If already breached → TTF = 0.0.
        2. Compute slope (units/second) by fitting a line through the forecast points.
        3. Extrapolate: TTF = (threshold – current_value) / slope.
        4. If slope moves away from threshold → TTF = None.

    Args:
        current_value:      Current observed metric value.
        threshold:          Critical threshold value.
        forecast_values:    List of predicted future values (Tier-1/2/3 output).
        sampling_interval:  Seconds between each forecast step (default 2.0s).
        is_upper_threshold: True if crossing ABOVE threshold is failure.
                            False if crossing BELOW threshold is failure.

    Returns:
        (ttf_seconds_or_None, slope_per_second)
        ttf = 0.0  → already breached
        ttf = float → predicted seconds to breach
        ttf = None → moving away or flat (no breach predicted)
    """
    if current_value is None:
        return None, 0.0

    try:
        current_value = float(current_value)
    except (TypeError, ValueError):
        return None, 0.0

    if math.isnan(current_value) or math.isinf(current_value):
        return None, 0.0

    # Already breached
    if is_upper_threshold and current_value >= threshold:
        return 0.0, 0.0
    if not is_upper_threshold and current_value <= threshold:
        return 0.0, 0.0

    if not forecast_values:
        return None, 0.0

    steps = len(forecast_values)
    time_pts = np.arange(1, steps + 1, dtype=float) * sampling_interval
    vals = np.array(forecast_values, dtype=float)

    # Compute slope via linear fit through forecast points
    try:
        slope_coeff, _ = np.polyfit(time_pts, vals, 1)
        slope_per_sec = float(slope_coeff)
    except Exception:
        slope_per_sec = float((vals[-1] - current_value) / (steps * sampling_interval + 1e-9))

    # Extrapolate to threshold crossing
    if is_upper_threshold:
        if slope_per_sec > 1e-9:
            distance = threshold - current_value
            ttf = distance / slope_per_sec
            return (round(float(ttf), 2), slope_per_sec) if ttf >= 0 else (None, slope_per_sec)
        else:
            return None, slope_per_sec   # moving away or flat
    else:
        if slope_per_sec < -1e-9:
            distance = threshold - current_value  # negative distance
            ttf = distance / slope_per_sec        # distance/slope → positive
            return (round(float(ttf), 2), slope_per_sec) if ttf >= 0 else (None, slope_per_sec)
        else:
            return None, slope_per_sec   # moving away or flat


# =============================================================================
# LEGACY v1 — kept intact for backward compatibility with base_forecaster.py
# =============================================================================

def linear_forecast(
    series: list[float],
    steps: int = FORECAST_STEPS,
) -> tuple[list[float], float]:
    """LEGACY: fit y = a·x + b, return (predictions, confidence)."""
    n = len(series)
    if n < LINEAR_MIN_HISTORY:
        val = series[-1] if series else 0.0
        return [val] * steps, _low_confidence(n)
    x = np.arange(n, dtype=float)
    try:
        coeffs = np.polyfit(x, series, 1)
        x_future = np.arange(n, n + steps, dtype=float)
        predictions = list(np.polyval(coeffs, x_future))
        confidence  = _confidence_from_r2(x, series, coeffs, n)
        return predictions, confidence
    except Exception:
        val = series[-1]
        return [val] * steps, 0.20


def exponential_forecast(
    series: list[float],
    steps: int = FORECAST_STEPS,
) -> tuple[list[float], float]:
    """LEGACY: fit log(y) = a·x + b, return (predictions, confidence)."""
    n = len(series)
    if n < LINEAR_MIN_HISTORY:
        val = series[-1] if series else 0.0
        return [val] * steps, _low_confidence(n)
    s_pos = [max(v, 1e-3) for v in series]
    try:
        x = np.arange(n, dtype=float)
        log_s = np.log(s_pos)
        coeffs = np.polyfit(x, log_s, 1)
        x_future = np.arange(n, n + steps, dtype=float)
        predictions = [float(math.exp(np.polyval(coeffs, xi))) for xi in x_future]
        confidence = _confidence_from_r2(x, log_s, coeffs, n)
        return predictions, confidence
    except Exception:
        return linear_forecast(series, steps)


def arima_forecast(
    series: list[float],
    steps: int = FORECAST_STEPS,
    order: tuple = ARIMA_ORDER,
) -> tuple[list[float], float]:
    """LEGACY: ARIMA(p,d,q) via statsmodels, falls back to linear."""
    n = len(series)
    if n < ARIMA_MIN_HISTORY:
        return linear_forecast(series, steps)
    try:
        if _statsmodels_ARIMA is None:
            return linear_forecast(series, steps)
        min_required = max(order[0], order[2]) + order[1] + 5
        if n < min_required:
            return linear_forecast(series, steps)
        model  = _statsmodels_ARIMA(series, order=order)
        result = model.fit()
        forecast = list(result.forecast(steps=steps))
        confidence = _arima_confidence(series, result, n)
        return forecast, confidence
    except Exception:
        return linear_forecast(series, steps)


def change_point_forecast(
    series: list[float],
    steps: int = FORECAST_STEPS,
) -> tuple[list[float], float]:
    """LEGACY: step-function change-point detection for BAD_DEPLOYMENT."""
    n = len(series)
    if n < 2:
        val = series[-1] if series else 0.0
        return [val] * steps, 0.20
    window    = CHANGE_POINT_CONFIG["history_window"]
    slope_thr = CHANGE_POINT_CONFIG["slope_threshold"]
    recent    = series[-window:] if n >= window else series
    rx = np.arange(len(recent), dtype=float)
    try:
        coeffs = np.polyfit(rx, recent, 1)
        slope  = float(coeffs[0])
    except Exception:
        slope = 0.0
    current_val = series[-1]
    if abs(slope) > slope_thr:
        x_future   = np.arange(len(recent), len(recent) + steps, dtype=float)
        predictions = [float(np.polyval(coeffs, xi)) for xi in x_future]
        confidence  = min(1.0, abs(slope) / slope_thr * 0.70)
    else:
        predictions = [current_val] * steps
        confidence  = 0.30
    return predictions, confidence


def compute_ttf(
    predictions: list[float],
    critical_threshold: float,
    direction: str = "higher_worse",
    step_interval_min: float = STEP_INTERVAL_MIN,
) -> Optional[float]:
    """LEGACY: scan predictions for threshold crossing, return TTF in minutes."""
    for i, val in enumerate(predictions):
        crossed = (
            direction == "higher_worse" and val >= critical_threshold
        ) or (
            direction == "lower_worse"  and val <= critical_threshold
        )
        if crossed:
            return round((i + 1) * step_interval_min, 3)
    return None


# ── Legacy confidence helpers (unchanged) ────────────────────────────────────

def _low_confidence(n_samples: int) -> float:
    return round(0.20 + (n_samples / SHORT_HISTORY_PENALTY_THRESHOLD) * 0.10, 3)


def _confidence_from_r2(
    x: np.ndarray,
    y,
    coeffs: np.ndarray,
    n: int,
) -> float:
    try:
        y_pred = np.polyval(coeffs, x)
        y_arr  = np.array(y, dtype=float)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r2     = 1.0 - ss_res / max(ss_tot, 1e-9)
        r2     = max(0.0, min(1.0, r2))
    except Exception:
        r2 = 0.40
    if n < SHORT_HISTORY_PENALTY_THRESHOLD:
        r2 *= SHORT_HISTORY_PENALTY_FACTOR
    return round(r2, 4)


def _arima_confidence(series: list[float], result, n: int) -> float:
    try:
        residuals   = list(result.resid)
        std_resid   = float(np.std(residuals))
        mean_signal = float(np.mean(np.abs(series))) or 1e-9
        noise_ratio = std_resid / mean_signal
        confidence  = max(0.0, min(1.0, 1.0 - noise_ratio * 2.0))
    except Exception:
        confidence = 0.50
    if n < SHORT_HISTORY_PENALTY_THRESHOLD:
        confidence *= SHORT_HISTORY_PENALTY_FACTOR
    return round(confidence, 4)
