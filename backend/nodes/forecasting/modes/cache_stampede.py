"""
d:/Before_done/forecasting_node/modes/cache_stampede.py
========================================================
CACHE_STAMPEDE Forecasting Node.

Failure Physics:
    Cache miss rate climbs as large batches of keys expire simultaneously.
    At 90% miss rate, virtually all requests bypass cache and hit the DB directly,
    causing a secondary DB latency surge (feedback loop).

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear fallback

Critical Features for TTF:
    cache_miss_rate → threshold 0.90 (90% miss = DB fully overwhelmed)
    db_p99          → threshold 500.0 (ms) — secondary DB surge

Extra features forecasted (context only):
    cache_hit_rate, active_connections, p99_latency
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "CACHE_STAMPEDE"

_CRITICAL = {
    "cache_miss_rate": 0.90,    # fraction — DB overwhelmed
    "db_p99":          500.0,   # ms — secondary DB surge
}
_EXTRA = ["cache_hit_rate", "active_connections", "p99_latency"]


def forecast_cache_stampede(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for CACHE_STAMPEDE failure mode.

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
