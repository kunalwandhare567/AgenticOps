"""
d:/Before_done/forecasting_node/modes/cpu_saturation.py
========================================================
CPU_SATURATION Forecasting Node.

Failure Physics:
    CPU climbs stochastically with a rising trend under sustained load.
    OS scheduling degrades sharply at 90% — threads begin to starve.
    Auto-ARIMA captures oscillating saturation better than a straight line.

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear fallback

Critical Features for TTF:
    cpu_utilization  → threshold 90.0 (%)
    cpu_saturation   → threshold 0.90 (fraction)

Extra features forecasted (context only):
    thread_pool_queue, rps, p99_latency
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "CPU_SATURATION"

_CRITICAL = {
    "cpu_utilization": 90.0,   # % — OS scheduling degrades
    "cpu_saturation":  0.90,   # fraction — fully saturated
}
_EXTRA = ["thread_pool_queue", "rps", "p99_latency"]


def forecast_cpu_saturation(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for CPU_SATURATION failure mode.

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
        tier2_algorithm   = "linear",
    )
