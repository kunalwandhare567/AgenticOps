"""
d:/Before_done/forecasting_node/modes/queue_backup.py
======================================================
QUEUE_BACKUP Forecasting Node.

Failure Physics:
    Work queue depth grows as consumers cannot keep pace with producers.
    Thread pool saturates → requests pile up → latency climbs stochastically.
    Auto-ARIMA captures the stochastic oscillation around the rising trend.

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear fallback

Critical Features for TTF:
    queue_lag         → threshold 500.0  — severe backlog (queue depth units)
    thread_pool_queue → threshold 100.0  — thread pool saturated

Extra features forecasted (context only):
    p99_latency, rps, error_rate
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "QUEUE_BACKUP"

_CRITICAL = {
    "queue_lag":         500.0,   # queue depth — severe backlog
    "thread_pool_queue": 100.0,   # thread pool saturated
}
_EXTRA = ["p99_latency", "rps", "error_rate"]


def forecast_queue_backup(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for QUEUE_BACKUP failure mode.

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
