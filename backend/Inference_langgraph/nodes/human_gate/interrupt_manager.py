"""
backend/nodes/human_gate/interrupt_manager.py
==============================================
Module 3 — Interrupt Manager (LangGraph-style interrupt, SQLite-backed).

Responsibility
--------------
Acts as the shared message bus between the pipeline runner process and
the FastAPI server process.  Both processes see the same SQLite database,
so no message queue or Redis is required.

How the "interrupt" works
--------------------------
1. Pipeline runner calls  post_review(request)
   → Writes a WAITING row to pending_reviews table in SQLite.
   → Calls poll_for_decision(review_id, timeout) which spins every 0.1 s
     reading the row's status.  When status changes to APPROVED / REJECTED /
     AUTO_APPROVED the poll returns.

2. FastAPI server calls   submit_decision(review_id, decision, operator, reason)
   → Updates the pending_reviews row in SQLite.
   → The polling loop in the pipeline runner process picks up the change.

This is the "LangGraph interrupt" pattern without requiring LangGraph —
identical semantics (pause → decision → resume) via SQLite polling.

Key methods
-----------
    post_review(request)             → insert WAITING row
    poll_for_decision(id, timeout)   → block until decision or timeout
    submit_decision(id, ...)         → resolve from API side
    get_pending()                    → return all WAITING reviews (for API)
    get_review(id)                   → return one review dict (for API)
    get_all_reviews()                → full history (for audit endpoint)
"""
from __future__ import annotations

import sqlite3
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulator.app_data_generator_for_offline.config import HUMAN_GATE_AUDIT_DB, HUMAN_GATE_TIMEOUT_SECONDS
from nodes.human_gate.review_builder import HumanReviewRequest


# ---------------------------------------------------------------------------
# DDL — pending_reviews table schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pending_reviews (
    review_id       TEXT PRIMARY KEY,
    incident_id     TEXT NOT NULL,
    episode_id      TEXT NOT NULL,
    failure_mode    TEXT NOT NULL,
    failure_label   TEXT NOT NULL,
    old_severity    TEXT NOT NULL,
    new_severity    TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.0,
    ttf_seconds     REAL NOT NULL DEFAULT -1.0,
    impact_band     TEXT NOT NULL DEFAULT 'None',
    urgency_band    TEXT NOT NULL DEFAULT 'Distant',
    root_cause      TEXT NOT NULL DEFAULT '',
    escalation_summary TEXT NOT NULL DEFAULT '',
    is_large_jump   INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    timeout_seconds INTEGER NOT NULL DEFAULT 2,
    status          TEXT NOT NULL DEFAULT 'WAITING',
    decision        TEXT,
    operator        TEXT,
    reason          TEXT,
    decided_at      TEXT,
    response_ms     INTEGER
);
"""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class InterruptManager:
    """
    SQLite-backed pending review queue.

    Thread-safety: SQLite WAL mode is used so the API process (reader/writer)
    and pipeline process (writer then reader) can coexist safely.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or HUMAN_GATE_AUDIT_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    # ------------------------------------------------------------------
    # Pipeline-side: post a review and block until decision
    # ------------------------------------------------------------------

    def post_review(self, request: HumanReviewRequest) -> None:
        """
        Insert a new WAITING review into the pending_reviews table.
        Called by run_human_gate.py immediately before poll_for_decision().
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_reviews (
                    review_id, incident_id, episode_id, failure_mode, failure_label,
                    old_severity, new_severity, confidence, ttf_seconds,
                    impact_band, urgency_band, root_cause, escalation_summary,
                    is_large_jump, created_at, expires_at, timeout_seconds, status
                ) VALUES (
                    :review_id, :incident_id, :episode_id, :failure_mode, :failure_label,
                    :old_severity, :new_severity, :confidence, :ttf_seconds,
                    :impact_band, :urgency_band, :root_cause, :escalation_summary,
                    :is_large_jump, :created_at, :expires_at, :timeout_seconds, 'WAITING'
                )
                """,
                {
                    "review_id":          request.review_id,
                    "incident_id":        request.incident_id,
                    "episode_id":         request.episode_id,
                    "failure_mode":       request.failure_mode,
                    "failure_label":      request.failure_label,
                    "old_severity":       request.old_severity,
                    "new_severity":       request.new_severity,
                    "confidence":         request.confidence,
                    "ttf_seconds":        request.ttf_seconds,
                    "impact_band":        request.impact_band,
                    "urgency_band":       request.urgency_band,
                    "root_cause":         request.root_cause,
                    "escalation_summary": request.escalation_summary,
                    "is_large_jump":      int(request.is_large_jump),
                    "created_at":         request.created_at,
                    "expires_at":         request.expires_at,
                    "timeout_seconds":    request.timeout_seconds,
                },
            )
            conn.commit()

    def poll_for_decision(
        self,
        review_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Block (poll every 0.1 s) until the review row is no longer WAITING,
        or until timeout seconds elapse — whichever comes first.

        On timeout the review is auto-approved directly in the DB so the
        API side also sees the final state.

        Args:
            review_id: UUID string of the pending review.
            timeout:   Max seconds to wait. Defaults to HUMAN_GATE_TIMEOUT_SECONDS.

        Returns:
            Dict with at least: status, decision, operator, reason, decided_at.
        """
        deadline = time.monotonic() + (timeout or HUMAN_GATE_TIMEOUT_SECONDS)
        poll_interval = 0.1   # seconds

        while time.monotonic() < deadline:
            row = self._fetch_row(review_id)
            if row and row["status"] != "WAITING":
                return dict(row)
            time.sleep(poll_interval)

        # Timeout — auto-approve
        now_str = datetime.now(timezone.utc).isoformat()
        elapsed_ms = int((timeout or HUMAN_GATE_TIMEOUT_SECONDS) * 1000)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE pending_reviews
                SET    status     = 'AUTO_APPROVED',
                       decision   = 'AUTO_APPROVED',
                       operator   = 'system',
                       reason     = 'No human response within timeout window.',
                       decided_at = ?,
                       response_ms = ?
                WHERE  review_id = ? AND status = 'WAITING'
                """,
                (now_str, elapsed_ms, review_id),
            )
            conn.commit()

        row = self._fetch_row(review_id)
        return dict(row) if row else {
            "status":     "AUTO_APPROVED",
            "decision":   "AUTO_APPROVED",
            "operator":   "system",
            "reason":     "Timeout — row not found after auto-approve.",
            "decided_at": now_str,
        }

    # ------------------------------------------------------------------
    # API-side: submit a human decision
    # ------------------------------------------------------------------

    def submit_decision(
        self,
        review_id:  str,
        decision:   str,      # "APPROVED" | "REJECTED"
        operator:   str,
        reason:     str = "",
    ) -> dict[str, Any]:
        """
        Record the operator's decision.  Called by the FastAPI endpoint.

        The pipeline's poll_for_decision() will pick up the status change
        within the next 0.1-second poll cycle.

        Args:
            review_id: UUID of the pending review.
            decision:  "APPROVED" or "REJECTED".
            operator:  Operator username / name.
            reason:    Optional rejection reason.

        Returns:
            Updated row dict, or error dict if review_id not found.
        """
        decision = decision.upper().strip()
        if decision not in ("APPROVED", "REJECTED"):
            return {"error": f"Invalid decision '{decision}'. Use APPROVED or REJECTED."}

        # Check if review exists and is still WAITING / REVIEWING
        row = self._fetch_row(review_id)
        if not row:
            return {"error": f"Review {review_id} not found."}
        if row["status"] not in ("WAITING", "REVIEWING"):
            return {"error": f"Review {review_id} already settled: {row['status']}."}

        now_str = datetime.now(timezone.utc).isoformat()

        # Calculate response time in milliseconds
        try:
            created = datetime.fromisoformat(row["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            now_dt = datetime.now(timezone.utc)
            response_ms = int((now_dt - created).total_seconds() * 1000)
        except Exception:
            response_ms = 0

        # Final status = decision for approved; REJECTED for rejected
        final_status = decision  # "APPROVED" or "REJECTED"

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE pending_reviews
                SET    status      = ?,
                       decision    = ?,
                       operator    = ?,
                       reason      = ?,
                       decided_at  = ?,
                       response_ms = ?
                WHERE  review_id = ?
                """,
                (final_status, decision, operator, reason, now_str, response_ms, review_id),
            )
            conn.commit()

        updated = self._fetch_row(review_id)
        return dict(updated) if updated else {"status": final_status, "review_id": review_id}

    def mark_reviewing(self, review_id: str) -> None:
        """
        Transition a WAITING review to REVIEWING when the operator opens it.
        Called by the GET /api/human-gate/review/{id} endpoint.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE pending_reviews SET status = 'REVIEWING' WHERE review_id = ? AND status = 'WAITING'",
                (review_id,),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Query methods (used by FastAPI service layer)
    # ------------------------------------------------------------------

    def get_pending(self) -> list[dict[str, Any]]:
        """Return all reviews with status WAITING or REVIEWING."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM pending_reviews WHERE status IN ('WAITING', 'REVIEWING') ORDER BY created_at DESC"
            )
            return [dict(r) for r in cur.fetchall()]

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        """Return a single review by review_id, or None if not found."""
        row = self._fetch_row(review_id)
        return dict(row) if row else None

    def get_all_reviews(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return most recent reviews across all statuses (for audit/history)."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM pending_reviews ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fetch_row(self, review_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM pending_reviews WHERE review_id = ?",
                (review_id,),
            )
            return cur.fetchone()


# ---------------------------------------------------------------------------
# Module-level singleton (shared within same process)
# ---------------------------------------------------------------------------

_GLOBAL_INTERRUPT_MANAGER: InterruptManager | None = None


def get_interrupt_manager() -> InterruptManager:
    """
    Return the module-level singleton InterruptManager.
    Creating it lazily ensures the DB path from config is available.
    """
    global _GLOBAL_INTERRUPT_MANAGER
    if _GLOBAL_INTERRUPT_MANAGER is None:
        _GLOBAL_INTERRUPT_MANAGER = InterruptManager()
    return _GLOBAL_INTERRUPT_MANAGER
