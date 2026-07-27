"""
d:/Before_done/forecasting_node/modes/retry_storm.py
=====================================================
RETRY_STORM Forecasting Node.

Failure Physics:
    Client-side retries create a FEEDBACK LOOP:
        more errors → more retries → higher RPS → more errors → ...
    This feedback amplification causes EXPONENTIAL growth in retry count
    and RPS. A small initial degradation (~5% error rate) triggers a
    runaway escalation to 25%+ retry rate very quickly.

    From design image: Exponential Regression is the identified model for
    this mode because "retry storms amplify quickly" and "risk grows slowly
    at first, then very fast" — a curved upward exponential pattern.

Algorithm Tiers:
    Tier 1: Auto-ARIMA (pmdarima → statsmodels → linear)
    Tier 2: EXPONENTIAL (explicitly chosen — feedback loops grow exponentially)
    Tier 3: linear (final safety net inside apply_exponential_forecast)

Critical Features for TTF:
    retry_count_per_request → threshold 2.0   (5× healthy baseline of ~0.4)
    error_rate              → threshold 0.25  (25% — ≈5× healthy 5%)
    rps                     → threshold 500.0 (3× normal ~175 rps)

Extra features forecasted (context only):
    active_connections, http_5xx_rate, p99_latency
"""
from .._mode_runner import run_auto_arima_forecast

FAILURE_MODE = "RETRY_STORM"

_CRITICAL = {
    "retry_count_per_request": 2.0,    # 5× healthy baseline
    "error_rate":              0.25,   # 25% — ≈5× healthy 5%
    "rps":                     500.0,  # 3× normal ~175 rps
}
_EXTRA = ["active_connections", "http_5xx_rate", "p99_latency"]


def forecast_retry_storm(episode_id: str, current_features: dict) -> dict:
    """
    Run TTF forecast for RETRY_STORM failure mode.

    Uses Auto-ARIMA as tier-1.
    Tier-2 is EXPONENTIAL — retry feedback loops create accelerating growth
    that a straight line underestimates.

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
        tier2_algorithm   = "exponential",   # Feedback loop = exponential growth
    )
