"""
backend/Inference_langgraph/nodes/n01_collect.py
================================================
LangGraph Node 1 — Collect.

Reads the next unprocessed telemetry row from simulator_db.sqlite.
Maintains `last_processed_id` as a module-level counter so each graph
invocation picks up exactly where the previous one left off.

Returns:
    State keys updated:
        raw_metric, raw_log, raw_traces, episode_id, failure_mode,
        timestamp, elapsed_s, service, last_processed_id
    OR:
        error = "no_data"   ← graph router sends to END; outer loop retries
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from Simulator.app_data_generator_for_offline.config import DB_PATH
from Inference_langgraph.state import AIOpsLangState, validate_state

# ── Module-level state (survives across graph invocations in the same process) ─
_lock               = threading.Lock()
_db_conn:    sqlite3.Connection | None = None
_last_processed_id: int = 0

# ── Live mode flag & target DB path ──────────────────────────────────────────
# Set to True by run_langgraph.py --live before the graph starts.
_LIVE_MODE: bool = False
_target_db_path: Path = DB_PATH


def set_live_mode(enabled: bool, db_path: str | Path | None = None) -> None:
    """Enable or disable live feed mode. Called once at startup by run_langgraph.py."""
    global _LIVE_MODE, _target_db_path, _db_conn
    _LIVE_MODE = enabled
    if db_path is not None:
        _target_db_path = Path(db_path)
    else:
        from Simulator.app_data_generator_for_offline.config import LIVE_FEED_DB_PATH
        _target_db_path = LIVE_FEED_DB_PATH
    _db_conn = None  # force connection re-open with new DB path
    print(f"[n01_collect] Live mode: {'ENABLED — reading from ' + str(_target_db_path) if enabled else 'DISABLED — reading from simulator_db.sqlite'}")


def _get_connection() -> sqlite3.Connection | None:
    """Return the shared SQLite connection, opening it lazily."""
    global _db_conn
    with _lock:
        if _db_conn is None and _target_db_path.exists():
            _db_conn = sqlite3.connect(str(_target_db_path), check_same_thread=False)
            _db_conn.row_factory = sqlite3.Row
            _db_conn.execute("PRAGMA journal_mode=WAL")
        return _db_conn


def _fetch_next_row(
    conn: sqlite3.Connection,
    last_id: int,
) -> tuple[int, dict, list[dict], list[dict]] | None:
    """
    Query the next metrics row after `last_id`.

    Returns (row_id, metric_dict, logs_list, spans_list) or None if no new data.
    """
    try:
        # Check table exists
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
        )
        if not cur.fetchone():
            return None

        cur = conn.execute(
            "SELECT * FROM metrics WHERE id > ? ORDER BY id ASC LIMIT 1", (last_id,)
        )
        row = cur.fetchone()
        if not row:
            return None

        metric  = dict(row)
        row_id  = metric["id"]
        ep_id   = metric["episode_id"]
        ts      = metric["timestamp"]

        # Fetch associated log rows for this (episode_id, timestamp) pair
        cur = conn.execute(
            "SELECT log_level, exception_type, log_message "
            "FROM logs WHERE episode_id = ? AND timestamp = ?",
            (ep_id, ts),
        )
        logs = [
            {"log_level": r[0], "exception_type": r[1], "log_message": r[2]}
            for r in cur.fetchall()
        ]

        # Fetch associated trace spans
        cur    = conn.execute(
            "SELECT * FROM traces WHERE episode_id = ? AND timestamp = ?",
            (ep_id, ts),
        )
        s_cols = [d[0] for d in cur.description]
        spans  = [dict(zip(s_cols, r)) for r in cur.fetchall()]

        return row_id, metric, logs, spans

    except Exception as exc:
        print(f"[n01_collect] DB error: {exc}")
        return None


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run(state: AIOpsLangState) -> dict[str, Any]:
    """
    Collect node — reads one telemetry tick.

    Live mode (--live):
      1. Checks LiveTelemetryQueue (in-process deque, for single-process runs).
      2. If queue is empty, reads next row from live_feed_db.sqlite (for multi-process runs).

    Historical mode (default):
      Reads next row from simulator_db.sqlite.

    If no new data is available returns {"error": "no_data"}.
    """
    global _last_processed_id

    # ── 1. Check in-process LiveTelemetryQueue first (fast in-memory path) ────
    if _LIVE_MODE:
        try:
            from Simulator.live_feed_simulator.live_queue import LiveTelemetryQueue
            item = LiveTelemetryQueue.pop()
            if item is not None:
                metric, logs, spans = item
                if isinstance(logs, dict):
                    logs = [logs]
                return {
                    "last_processed_id": 0,
                    "raw_metric":        metric,
                    "raw_log":           logs if isinstance(logs, list) else [],
                    "raw_traces":        spans if isinstance(spans, list) else [],
                    "episode_id":        str(metric.get("episode_id", "")),
                    "failure_mode":      str(metric.get("failure_mode", "NONE")),
                    "timestamp":         float(metric.get("timestamp", 0.0)),
                    "elapsed_s":         float(metric.get("elapsed_s",  0.0)),
                    "service":           str(metric.get("service", "")),
                    "error":             None,
                }
        except ImportError:
            pass

    # ── 2. Query target SQLite database (live_feed_db or simulator_db) ────────
    last_id = state.get("last_processed_id", 0)
    if last_id == 0:
        last_id = _last_processed_id

    conn = _get_connection()
    if conn is None:
        return {"error": "no_data"}

    result = _fetch_next_row(conn, last_id)
    if result is None:
        return {"error": "no_data"}

    row_id, metric, logs, spans = result
    _last_processed_id = row_id

    update: dict[str, Any] = {
        "last_processed_id": row_id,
        "raw_metric":        metric,
        "raw_log":           logs,
        "raw_traces":        spans,
        "episode_id":        str(metric.get("episode_id", "")),
        "failure_mode":      str(metric.get("failure_mode", "NONE")),
        "timestamp":         float(metric.get("timestamp", 0.0)),
        "elapsed_s":         float(metric.get("elapsed_s",  0.0)),
        "service":           str(metric.get("service", "")),
        "error":             None,
    }

    return update
