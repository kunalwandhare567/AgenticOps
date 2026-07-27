"""
d:/Before_done/forecasting_node/modes/error_storm.py
=====================================================
ERROR_STORM Forecasting Node.

Failure Physics:
    Error rate climbs stochastically as cascading application faults spread
    across service instances. The pattern is non-linear and oscillating before
    crossing the 50% threshold — Auto-ARIMA is well-suited to this.

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: linear fallback

Critical Features for TTF:
    error_rate    → threshold 0.50  (50% error rate = storm confirmed)
    http_5xx_rate → threshold 0.50  (50% 5xx confirms server-side storm)

Extra features forecasted (context only):
    network_errors, rps, p99_latency
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "ERROR_STORM"

_CRITICAL = {
    "error_rate":    0.50,   # fraction — storm confirmed
    "http_5xx_rate": 0.50,   # fraction — server-side storm
}
_EXTRA = ["network_errors", "rps", "p99_latency"]


def forecast_error_storm(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for ERROR_STORM failure mode.

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
