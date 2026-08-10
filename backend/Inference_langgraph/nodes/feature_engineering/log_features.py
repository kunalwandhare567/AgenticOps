"""
app_simulator/feature_engineering/log_features.py
===================================================
Online log feature extraction using Drain3.

Produces exactly 5 classifier features per cycle:
    log_count              int >= 0
    log_max_severity       int 0-4  (INFO=1, WARN=2, ERROR=3, CRITICAL=4)
    log_critical_count     int >= 0
    log_has_exception      0 or 1
    log_has_novel_template 0 or 1

Evidence fields (never in classifier, stored in PipelineState.evidence):
    _log_template_ids_seen   list[int]
    _log_raw_lines           list[str]

Severity encoding (matches DEVOPS log_feature_engineering.py exactly):
    ""      → 0
    INFO    → 1
    WARNING → 2  (also WARN)
    ERROR   → 3
    CRITICAL/FATAL → 4
"""
from __future__ import annotations

import json
import os
import pickle
import sqlite3
from typing import Any

# ── Severity map ──────────────────────────────────────────────────────────────
SEVERITY_MAP: dict[str, int] = {
    "":         0,
    "INFO":     1,
    "WARNING":  2,
    "WARN":     2,
    "ERROR":    3,
    "CRITICAL": 4,
    "FATAL":    4,
}

# ── The 5 classifier feature names ───────────────────────────────────────────
CLASSIFIER_FIELDS = [
    "log_count",
    "log_max_severity",
    "log_critical_count",
    "log_has_exception",
    "log_has_novel_template",
]

# ── Evidence-only fields (underscore prefix, never in classifier CSV) ─────────
EVIDENCE_FIELDS = [
    "_log_template_ids_seen",
    "_log_raw_lines",
]

# ── Empty result (returned when no logs) ─────────────────────────────────────
_EMPTY = {
    "log_count":              0,
    "log_max_severity":       0,
    "log_critical_count":     0,
    "log_has_exception":      0,
    "log_has_novel_template": 0,
    "_log_template_ids_seen": [],
    "_log_raw_lines":         [],
}


# =============================================================================
# Core function — called once per inference cycle
# =============================================================================

def engineer_log_features(
    log_lines_this_cycle: list[dict],
    template_miner: Any,            # drain3.TemplateMiner, loaded once at startup
    known_template_ids: set[int],   # loaded once at startup
) -> dict:
    """
    Convert a list of raw log dicts for one cycle into exactly 5 classifier features.

    Args:
        log_lines_this_cycle: List of raw log dicts with keys:
                              log_level, exception_type, log_message
        template_miner:       Drain3 TemplateMiner (read-only at runtime).
                              Pass None to disable novelty detection.
        known_template_ids:   Set of cluster IDs seen during offline training.

    Returns:
        dict with 5 classifier features + 2 evidence fields.
    """
    if not log_lines_this_cycle:
        return dict(_EMPTY)

    severities:    list[int] = []
    template_ids:  list[int] = []
    raw_lines:     list[str] = []
    has_exception: bool      = False
    has_novel:     bool      = False

    for line in log_lines_this_cycle:
        log_level   = str(line.get("log_level",    "")).strip().upper()
        exc_type    = str(line.get("exception_type", "")).strip()
        log_message = str(line.get("log_message",   "")).strip()

        # Severity
        severities.append(SEVERITY_MAP.get(log_level, 0))

        # Exception presence
        if exc_type:
            has_exception = True

        raw_lines.append(log_message)

        # Drain3 template matching (read-only)
        if template_miner is not None and log_message:
            cluster = template_miner.match(log_message)
            if cluster is None:
                has_novel = True
                template_ids.append(-1)          # -1 sentinel = genuinely novel
            else:
                tid = cluster.cluster_id
                template_ids.append(tid)
                if tid not in known_template_ids:
                    has_novel = True

    log_count        = len(log_lines_this_cycle)
    log_max_sev      = max(severities) if severities else 0
    log_critical_cnt = sum(1 for s in severities if s >= 4)

    return {
        # ── 5 classifier features ─────────────────────────────────────────
        "log_count":              log_count,
        "log_max_severity":       log_max_sev,
        "log_critical_count":     log_critical_cnt,
        "log_has_exception":      int(has_exception),
        "log_has_novel_template": int(has_novel),
        # ── Evidence (never in classifier) ────────────────────────────────
        "_log_template_ids_seen": template_ids,
        "_log_raw_lines":         raw_lines,
    }


# =============================================================================
# SQLite wrapper — used by orchestrator.py to get logs for a cycle
# =============================================================================

def compute_log_features(
    conn: sqlite3.Connection,
    episode_id: str,
    timestamp: float,
    template_miner: Any,
    known_template_ids: set[int],
) -> dict:
    """
    Query raw logs for (episode_id, timestamp) from SQLite and compute features.
    Used by the async orchestrator when running from the DB.
    """
    try:
        cur = conn.execute(
            """
            SELECT log_level, exception_type, log_message
            FROM   logs
            WHERE  episode_id = ?
              AND  timestamp  = ?
            """,
            (episode_id, timestamp),
        )
        rows = [{"log_level": r[0], "exception_type": r[1], "log_message": r[2]}
                for r in cur.fetchall()]
    except Exception as exc:
        print(f"[LogFE] ERROR querying logs: {exc}")
        rows = []

    return engineer_log_features(rows, template_miner, known_template_ids)


# =============================================================================
# Artifact loaders — called once at startup
# =============================================================================

def load_template_miner(state_bin: str, drain_ini: str) -> Any:
    """Load frozen Drain3 TemplateMiner from disk. Returns None on failure."""
    try:
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig
        from drain3.file_persistence import FilePersistence
    except ImportError:
        print("[LogFE] WARN: drain3 not installed. log_has_novel_template will be 0.")
        return None

    if not os.path.exists(state_bin):
        print(f"[LogFE] WARN: {state_bin} not found. Run offline/train_drain.py first.")
        return None

    config = TemplateMinerConfig()
    if os.path.exists(drain_ini):
        config.load(drain_ini)

    # Try FilePersistence loader first
    try:
        persistence = FilePersistence(state_bin)
        miner = TemplateMiner(persistence_handler=persistence, config=config)
        n = len(list(miner.drain.clusters))
        if n > 0:
            print(f"[LogFE] Loaded Drain3 state: {n} templates from {state_bin}")
            return miner
    except Exception:
        pass

    # Fallback: pickle
    try:
        with open(state_bin, "rb") as fh:
            miner = pickle.load(fh)
        n = len(list(miner.drain.clusters))
        print(f"[LogFE] Loaded Drain3 state (pickle): {n} templates")
        return miner
    except Exception as exc:
        print(f"[LogFE] ERROR: Could not load Drain3 state: {exc}")
        return None


def load_known_template_ids(templates_json: str) -> set[int]:
    """Load set of known cluster IDs from frozen JSON artifact."""
    if not os.path.exists(templates_json):
        print(f"[LogFE] WARN: {templates_json} not found. Novelty detection disabled.")
        return set()
    with open(templates_json, "r", encoding="utf-8") as f:
        ids = json.load(f)
    print(f"[LogFE] Loaded {len(ids)} known template IDs.")
    return set(ids)
