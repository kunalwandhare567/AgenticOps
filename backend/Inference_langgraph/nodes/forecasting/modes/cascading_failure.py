"""
d:/Before_done/forecasting_node/modes/cascading_failure.py
===========================================================
CASCADING_FAILURE Forecasting Node.

Failure Physics:
    A multi-stage compound failure that spreads exponentially through the system:
        Stage 1 (0–25s):  DB slowdown begins.
        Stage 2 (26–50s): Cache degrades as DB can't serve cache-miss requests.
        Stage 3 (51–70s): Error storm erupts (30–70% error rate).
        Stage 4 (71–120s): CPU exhaustion — full infrastructure collapse.

    Each stage amplifies the next stage — creating an EXPONENTIAL escalation.
    P99 latency is the universal symptom that escalates through all 4 stages.
    Auto-ARIMA captures the multi-stage non-linear acceleration.

    From design image: Exponential-growth rate extrapolation is the identified
    algorithm for this mode (PROPAGATION branch: "Exponential-growth rate
    extrapolation + blast-radius tracking from trace spans").

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: EXPONENTIAL (multi-stage amplification creates exponential escalation)
    Tier 3: linear (final safety net inside apply_exponential_forecast)

Critical Features for TTF:
    p99_latency    → threshold 1500.0 (ms) — Stage 4 full collapse
    error_rate     → threshold 0.50   — Stage 3 error storm confirmed
    cpu_utilization → threshold 90.0  (%) — Stage 4 CPU exhaustion

Extra features forecasted (context only):
    db_p99, active_connections, cache_miss_rate
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "CASCADING_FAILURE"

_CRITICAL = {
    "p99_latency":     1500.0,   # ms — full infrastructure collapse
    "error_rate":      0.50,     # fraction — error storm confirmed
    "cpu_utilization": 90.0,     # % — CPU exhaustion (Stage 4)
}
_EXTRA = ["db_p99", "active_connections", "cache_miss_rate"]


def forecast_cascading_failure(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for CASCADING_FAILURE failure mode.

    Uses Auto-ARIMA as tier-1.
    Tier-2 is EXPONENTIAL — each cascade stage amplifies the next,
    creating exponential growth that a straight line underestimates.

    Args:
        episode_id:       Current episode identifier.
        current_features: Latest feature row dict from the pipeline.

    Returns:
        Convergence schema dict (see _mode_runner.run_auto_arima_forecast).
    """
    return run_auto_arima_forecast(
        episode_id        = episode_id,
        failure_mode      = FAILURE_MODE,
        current_features  = current_features,
        critical_features = _CRITICAL,
        extra_features    = _EXTRA,
        tier2_algorithm   = "exponential",   # Multi-stage cascade = exponential growth
    )
