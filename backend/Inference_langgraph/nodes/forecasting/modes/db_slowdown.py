"""
d:/Before_done/forecasting_node/modes/db_slowdown.py
=====================================================
DB_SLOWDOWN Forecasting Node.

Failure Physics:
    DB latency grows non-linearly as each slow query adds lock contention,
    which slows the next query (multiplicative degradation).
    Auto-ARIMA captures the non-linear escalation pattern.
    At 1500ms, connection pool exhaustion begins (pool size ~100).

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear fallback

Critical Features for TTF:
    db_p99             → threshold 1500.0 (ms) — pool exhaustion begins
    active_connections → threshold 800.0  — connection pool pressure
    db_connection_wait → threshold 50.0   (ms) — lock wait confirms slowdown

Extra features forecasted (context only):
    db_connection_pool, cache_hit_rate, p99_latency
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "DB_SLOWDOWN"

_CRITICAL = {
    "db_p99":             1500.0,   # ms — pool exhaustion begins
    "active_connections": 800.0,    # connections — pool pressure
    "db_connection_wait": 50.0,     # ms — lock wait time
}
_EXTRA = ["db_connection_pool", "cache_hit_rate", "p99_latency"]


def forecast_db_slowdown(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for DB_SLOWDOWN failure mode.

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
