"""
backend/nodes/human_gate/audit_logger.py
=========================================
Module 6 — Audit Logger.

Responsibility
--------------
Persist every Human Gate decision to a permanent audit record.
This record is the "future learning data" your mentor described —
it captures what the AI recommended, what the human decided, and
the resulting final severity, so the model can be improved offline.

Two storage layers:
1. SQLite table `human_gate_review` — queryable, used by the dashboard.
2. human_gate_output.csv — append-only flat file for offline ML ingestion.

Design notes:
- The AuditLogger is intentionally decoupled from InterruptManager.
  InterruptManager stores pending/live state; AuditLogger stores settled history.
- The SQLite table is created in HUMAN_GATE_AUDIT_DB (same file as pending_reviews),
  keeping all Human Gate data in one database.
- CSV output mirrors the same schema for easy pandas ingestion.

Usage
-----
    logger = AuditLogger()
    logger.record(approval_result, review_request)
    rows = logger.get_history(limit=50)
"""
from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulator.app_data_generator_for_offline.config import HUMAN_GATE_AUDIT_DB, HUMAN_GATE_OUTPUT_CSV
from nodes.human_gate.approval_engine import ApprovalResult
from nodes.human_gate.review_builder  import HumanReviewRequest

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Simulator.app_data_generator_for_offline.storage.db_writer import DbWriter


# ---------------------------------------------------------------------------
# DDL — human_gate_review table
# ---------------------------------------------------------------------------

_CREATE_REVIEW_TABLE = """
CREATE TABLE IF NOT EXISTS human_gate_review (
    review_id       TEXT PRIMARY KEY,
    incident_id     TEXT NOT NULL,
    episode_id      TEXT NOT NULL,
    failure_mode    TEXT NOT NULL,
    failure_label   TEXT NOT NULL,
    old_severity    TEXT NOT NULL,
    new_severity    TEXT NOT NULL,
    final_severity  TEXT NOT NULL,
    decision        TEXT NOT NULL,
    operator        TEXT NOT NULL,
    reason          TEXT,
    confidence      REAL,
    ttf_seconds     REAL,
    impact_band     TEXT,
    urgency_band    TEXT,
    is_large_jump   INTEGER DEFAULT 0,
    escalation_summary TEXT,
    response_ms     INTEGER,
    timeout_seconds INTEGER,
    created_at      TEXT,
    decided_at      TEXT,
    recorded_at     TEXT NOT NULL
);
"""

# CSV column order — matches table schema above
_CSV_COLUMNS = [
    "review_id", "incident_id", "episode_id", "failure_mode", "failure_label",
    "old_severity", "new_severity", "final_severity", "decision",
    "operator", "reason", "confidence", "ttf_seconds",
    "impact_band", "urgency_band", "is_large_jump", "escalation_summary",
    "response_ms", "timeout_seconds", "created_at", "decided_at", "recorded_at",
]


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Writes every settled Human Gate decision to SQLite + CSV.

    Both the pipeline runner and the API can instantiate this logger —
    they share the same SQLite file (WAL mode handles concurrent access).
    """

    def __init__(
        self,
        db_path:     Path | None = None,
        csv_path:    Path | None = None,
        pipeline_db: "DbWriter | None" = None,
    ) -> None:
        self._db_path    = db_path  or HUMAN_GATE_AUDIT_DB
        self._csv_path   = csv_path or HUMAN_GATE_OUTPUT_CSV
        self._pipeline_db = pipeline_db   # optional: mirrors to simulator_db.sqlite
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Primary write method
    # ------------------------------------------------------------------

    def record(
        self,
        result:  ApprovalResult,
        request: HumanReviewRequest,
    ) -> None:
        """
        Persist one settled Human Gate decision.

        Writes to:
         - human_gate_review table in SQLite
         - human_gate_output.csv (append)

        Args:
            result:  ApprovalResult from ApprovalEngine.compute_result().
            request: Original HumanReviewRequest (carries context fields).
        """
        recorded_at = datetime.now(timezone.utc).isoformat()

        row: dict[str, Any] = {
            "review_id":         result.review_id,
            "incident_id":       result.incident_id,
            "episode_id":        result.episode_id,
            "failure_mode":      request.failure_mode,
            "failure_label":     request.failure_label,
            "old_severity":      result.old_severity,
            "new_severity":      result.new_severity,
            "final_severity":    result.final_severity,
            "decision":          result.decision,
            "operator":          result.operator,
            "reason":            result.reason,
            "confidence":        request.confidence,
            "ttf_seconds":       request.ttf_seconds,
            "impact_band":       request.impact_band,
            "urgency_band":      request.urgency_band,
            "is_large_jump":     int(request.is_large_jump),
            "escalation_summary": request.escalation_summary,
            "response_ms":       result.response_ms,
            "timeout_seconds":   request.timeout_seconds,
            "created_at":        request.created_at,
            "decided_at":        result.decided_at,
            "recorded_at":       recorded_at,
        }

        self._write_sqlite(row)
        self._write_csv(row)
        # Mirror to unified simulator_db.sqlite (node_human_gate table)
        if self._pipeline_db is not None:
            try:
                self._pipeline_db.write_human_gate(row)
            except Exception as _e:
                print(f"[AuditLogger] WARN: pipeline DB write failed: {_e}")

    # ------------------------------------------------------------------
    # Read methods (for dashboard / metrics endpoints)
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent audit records, newest first."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM human_gate_review ORDER BY recorded_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_metrics(self) -> dict[str, Any]:
        """
        Compute Human Gate KPI metrics from the full audit history.

        Returns dict with keys:
            total_reviews, approved_count, rejected_count, auto_approved_count,
            human_approval_pct, human_rejection_pct, auto_approval_pct,
            avg_response_ms, false_escalation_count
        """
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM human_gate_review")
            total = cur.fetchone()[0] or 0
            if total == 0:
                return self._empty_metrics()

            cur = conn.execute(
                "SELECT decision, COUNT(*) as cnt FROM human_gate_review GROUP BY decision"
            )
            counts: dict[str, int] = {r[0]: r[1] for r in cur.fetchall()}

            approved      = counts.get("APPROVED",      0)
            rejected      = counts.get("REJECTED",      0)
            auto_approved = counts.get("AUTO_APPROVED",  0)

            cur = conn.execute(
                "SELECT AVG(response_ms) FROM human_gate_review WHERE response_ms > 0"
            )
            avg_ms_row = cur.fetchone()
            avg_ms = round(float(avg_ms_row[0]), 1) if avg_ms_row and avg_ms_row[0] else 0.0

            return {
                "total_reviews":        total,
                "approved_count":       approved,
                "rejected_count":       rejected,
                "auto_approved_count":  auto_approved,
                "human_approval_pct":   round((approved / total) * 100, 1),
                "human_rejection_pct":  round((rejected / total) * 100, 1),
                "auto_approval_pct":    round((auto_approved / total) * 100, 1),
                "avg_response_ms":      avg_ms,
                "false_escalation_count": rejected,  # rejected = AI was wrong
            }

    def get_episode_decisions(self, episode_id: str) -> list[dict[str, Any]]:
        """Return all Human Gate decisions for a specific episode."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM human_gate_review WHERE episode_id = ? ORDER BY recorded_at DESC",
                (episode_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_REVIEW_TABLE)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _write_sqlite(self, row: dict[str, Any]) -> None:
        """Insert one audit row into human_gate_review table."""
        cols        = list(row.keys())
        placeholders = ", ".join(f":{c}" for c in cols)
        col_str      = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO human_gate_review ({col_str}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, row)
            conn.commit()

    def _write_csv(self, row: dict[str, Any]) -> None:
        """Append one audit row to the CSV output file."""
        write_header = not self._csv_path.exists()
        with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        return {
            "total_reviews":        0,
            "approved_count":       0,
            "rejected_count":       0,
            "auto_approved_count":  0,
            "human_approval_pct":   0.0,
            "human_rejection_pct":  0.0,
            "auto_approval_pct":    0.0,
            "avg_response_ms":      0.0,
            "false_escalation_count": 0,
        }
