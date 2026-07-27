"""
app_simulator/feature_engineering/orchestrator.py
===================================================
Simplified feature engineering orchestrator.

Changes from previous version:
  - Traces FE REMOVED (trace_features.py deleted per spec).
  - Metrics: raw passthrough only (no rolling window aggregation).
  - Logs: exactly 5 Drain3 classifier features.
  - Output: merged feature row appended to pipeline/output/engineered_features.csv.

Public API:
    load_drain_artifacts()       -> (miner, known_ids)   — call once at startup
    run_feature_engineering()    -> async coroutine
    run_feature_engineering_sync() -> sync wrapper for testing
"""
from __future__ import annotations

import asyncio
import csv
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app_data_generator.config import (
    ALL_CLASSIFIER_COLS, DRAIN_INI, DRAIN_STATE,
    ENGINEERED_FEAT_CSV, KNOWN_TEMPLATES_JSON,
)
from .metrics_features import compute_metrics_features
from .log_features import (
    compute_log_features,
    load_template_miner,
    load_known_template_ids,
)

# Thread pool — 2 workers: 1 per feature type (metrics + logs)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fe-worker")


# =============================================================================
# Startup loaders (call once)
# =============================================================================

def load_drain_artifacts() -> tuple[Any, set]:
    """
    Load Drain3 state binary and known template ID set.

    Returns:
        (template_miner, known_template_ids)
        template_miner may be None if drain_state.bin not found.
    """
    miner     = load_template_miner(str(DRAIN_STATE), str(DRAIN_INI))
    known_ids = load_known_template_ids(str(KNOWN_TEMPLATES_JSON))
    return miner, known_ids


# =============================================================================
# CSV append helper
# =============================================================================

def _append_feature_row(row: dict) -> None:
    """Append one classifier feature row to pipeline/output/engineered_features.csv."""
    out_path = ENGINEERED_FEAT_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    # Only write classifier columns (no underscore evidence fields)
    clean_row = {k: v for k, v in row.items() if not k.startswith("_")}

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_row.keys()), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(clean_row)


# =============================================================================
# Main async orchestrator
# =============================================================================

async def run_feature_engineering(
    conn: sqlite3.Connection,
    episode_id: str,
    timestamp: float,
    failure_mode: str,
    elapsed_s: float,
    template_miner: Any,
    known_template_ids: set,
    window: int = 30,                  # kept for API compat, unused
) -> dict | None:
    """
    Run metrics passthrough and log feature extraction concurrently.
    Merge results → append to engineered_features.csv.

    Args:
        conn:               SQLite connection.
        episode_id:         Current episode ID.
        timestamp:          Current tick Unix timestamp.
        failure_mode:       Current failure mode label.
        elapsed_s:          Seconds into episode.
        template_miner:     Drain3 TemplateMiner (or None).
        known_template_ids: Set of known cluster IDs.
        window:             Unused (kept for compatibility).

    Returns:
        {
          "classifier_input": {episode_id, failure_mode, timestamp, elapsed_s,
                               ...27 raw metric features, ...5 log features},
          "evidence":         {_log_template_ids_seen, _log_raw_lines}
        }
        Returns None on error.
    """
    loop = asyncio.get_event_loop()

    try:
        metric_feats, log_feats = await asyncio.gather(
            loop.run_in_executor(
                _executor,
                compute_metrics_features,
                conn, episode_id, timestamp, window,
            ),
            loop.run_in_executor(
                _executor,
                compute_log_features,
                conn, episode_id, timestamp, template_miner, known_template_ids,
            ),
        )
    except Exception as exc:
        print(f"[FE] ERROR in gather: {exc}")
        return None

    # Build classifier input: metadata + 27 raw metrics + 5 log features
    classifier_row: dict = {
        "episode_id":   episode_id,
        "failure_mode": failure_mode,
        "timestamp":    timestamp,
        "elapsed_s":    elapsed_s,
    }
    classifier_row.update(metric_feats)
    classifier_row.update({k: v for k, v in log_feats.items() if not k.startswith("_")})

    # Evidence: underscore fields only (never in CSV)
    evidence: dict = {k: v for k, v in log_feats.items() if k.startswith("_")}

    # Persist to CSV
    try:
        _append_feature_row(classifier_row)
    except Exception as exc:
        print(f"[FE] WARN: could not append to engineered_features.csv: {exc}")

    return {"classifier_input": classifier_row, "evidence": evidence}


# =============================================================================
# Sync wrapper for testing / CLI use
# =============================================================================

def run_feature_engineering_sync(
    conn: sqlite3.Connection,
    episode_id: str,
    timestamp: float,
    failure_mode: str,
    elapsed_s: float,
    template_miner: Any,
    known_template_ids: set,
) -> dict | None:
    """Synchronous wrapper around run_feature_engineering."""
    return asyncio.run(run_feature_engineering(
        conn, episode_id, timestamp, failure_mode, elapsed_s,
        template_miner, known_template_ids,
    ))


# =============================================================================
# Direct feature engineering from raw dicts (used by pipeline/feature_engineering.py)
# =============================================================================

def run_feature_engineering_from_raw(
    metric: dict,
    log: dict,
    episode_id: str,
    failure_mode: str,
    timestamp: float,
    elapsed_s: float,
    template_miner: Any,
    known_template_ids: set,
) -> dict:
    """
    Run FE directly from raw generator output dicts (no SQLite needed).
    Used by the pipeline runner which gets items from TelemetryQueue.

    Args:
        metric:  Raw metric dict from MetricsGenerator.generate()
        log:     Raw log dict from LogGenerator.generate()
        episode_id, failure_mode, timestamp, elapsed_s: Episode metadata.
        template_miner, known_template_ids: Drain3 artifacts.

    Returns:
        {"classifier_input": {...}, "evidence": {...}}
    """
    from .metrics_features import passthrough_metrics
    from .log_features import engineer_log_features

    # 27 raw metrics (passthrough)
    metric_feats = passthrough_metrics(metric)

    # 5 log features (Drain3)
    log_feats = engineer_log_features(
        [log] if isinstance(log, dict) else log,
        template_miner,
        known_template_ids,
    )

    # Merge
    classifier_row: dict = {
        "episode_id":   episode_id,
        "failure_mode": failure_mode,
        "timestamp":    timestamp,
        "elapsed_s":    elapsed_s,
    }
    classifier_row.update(metric_feats)
    classifier_row.update({k: v for k, v in log_feats.items() if not k.startswith("_")})

    evidence = {k: v for k, v in log_feats.items() if k.startswith("_")}

    # Persist
    try:
        _append_feature_row(classifier_row)
    except Exception as exc:
        print(f"[FE] WARN: could not append to CSV: {exc}")

    return {"classifier_input": classifier_row, "evidence": evidence}
