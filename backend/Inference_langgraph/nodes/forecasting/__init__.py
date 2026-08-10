"""
d:/Before_done/forecasting_node/__init__.py
============================================
Public API for the forecasting_node package.

Recommended import (from outside the package):

    from nodes.forecasting import route_forecast, route_forecast_batch
    from nodes.forecasting import open_episode, close_episode, expire_stale_episodes
    from nodes.forecasting import active_episode_summary, list_supported_modes

Or directly from the router:
    from nodes.forecasting.router import route_forecast

Buffer operations (low-level):
    from nodes.forecasting.buffer import append_feature_row, get_metric_series
"""
from .router import (
    route_forecast,
    route_forecast_batch,
    open_episode,
    close_episode,
    expire_stale_episodes,
    active_episode_summary,
    list_supported_modes,
    is_mode_supported,
)
from .buffer import (
    append_feature_row,
    get_metric_series,
    get_episode_length,
    get_all_metrics,
    clear_episode,
    reset_all,
    active_episodes,
)
from .thresholds import (
    get_config,
    get_primary_metric,
    get_algorithm,
    get_critical_threshold,
    get_direction,
    get_secondary,
    MODE_CONFIG,
)

__all__ = [
    # Router
    "route_forecast",
    "route_forecast_batch",
    "open_episode",
    "close_episode",
    "expire_stale_episodes",
    "active_episode_summary",
    "list_supported_modes",
    "is_mode_supported",
    # Buffer
    "append_feature_row",
    "get_metric_series",
    "get_episode_length",
    "get_all_metrics",
    "clear_episode",
    "reset_all",
    "active_episodes",
    # Thresholds
    "get_config",
    "get_primary_metric",
    "get_algorithm",
    "get_critical_threshold",
    "get_direction",
    "get_secondary",
    "MODE_CONFIG",
]
