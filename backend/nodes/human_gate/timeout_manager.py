"""
backend/nodes/human_gate/timeout_manager.py
============================================
Module 4 — Timeout Manager.

Responsibility
--------------
Checks whether a review has exceeded its deadline and triggers
auto-approval.  In this version (v1 — no threading), the timeout
is enforced inside InterruptManager.poll_for_decision() itself.

This module provides:
1. TimeoutManager class — a simple utility to check / compute deadline state.
2. compute_auto_approve_reason() — standardised reason string for audit logs.

For the future threading version, this module would run as a background
thread calling check_and_expire() every 0.5 s across all pending reviews.
That is left as a future enhancement (see TODO below).

Usage
-----
    manager = TimeoutManager()
    remaining = manager.seconds_remaining(request)   # float ≥ 0
    is_done   = manager.is_expired(request)           # bool
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_data_generator.config import HUMAN_GATE_TIMEOUT_SECONDS


class TimeoutManager:
    """
    Stateless utility for computing timeout state of a pending review.

    All methods accept either a HumanReviewRequest object or a plain dict
    (as returned by InterruptManager.get_review()).
    """

    def __init__(self, timeout_seconds: int | None = None) -> None:
        """
        Args:
            timeout_seconds: Override for HUMAN_GATE_TIMEOUT_SECONDS.
                             Useful in tests.
        """
        self.timeout_seconds = timeout_seconds or HUMAN_GATE_TIMEOUT_SECONDS

    # ------------------------------------------------------------------
    # Core state checks
    # ------------------------------------------------------------------

    def is_expired(self, review: Any) -> bool:
        """
        Return True if the review's expires_at timestamp is in the past.

        Args:
            review: HumanReviewRequest | dict — must have 'expires_at' key.
        """
        expires_at = self._get_expires(review)
        if expires_at is None:
            return True   # Unknown expiry → treat as expired to be safe
        return datetime.now(timezone.utc) >= expires_at

    def seconds_remaining(self, review: Any) -> float:
        """
        Return seconds remaining before auto-approve fires.
        Returns 0.0 if already expired.
        """
        expires_at = self._get_expires(review)
        if expires_at is None:
            return 0.0
        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, remaining)

    def percentage_elapsed(self, review: Any) -> float:
        """
        Return 0.0–100.0 representing how much of the timeout window has elapsed.
        Useful for rendering a progress bar in the dashboard.
        """
        remaining = self.seconds_remaining(review)
        total     = float(self.timeout_seconds)
        elapsed   = total - remaining
        return round(min(100.0, max(0.0, (elapsed / total) * 100.0)), 1)

    # ------------------------------------------------------------------
    # Reason string factory
    # ------------------------------------------------------------------

    @staticmethod
    def compute_auto_approve_reason(timeout_seconds: int) -> str:
        """
        Standard reason string written to the audit log when auto-approval
        fires due to timeout.
        """
        return (
            f"Auto-approved: no human response within {timeout_seconds}s timeout window. "
            "AI escalation recommendation accepted automatically."
        )

    # ------------------------------------------------------------------
    # Future: batch expiry check (threading version)
    # ------------------------------------------------------------------
    # TODO (v2 — threading enhancement):
    #   def check_and_expire_all(self, interrupt_manager: InterruptManager) -> int:
    #       """
    #       Check all WAITING reviews. Auto-approve any that are expired.
    #       Returns number of reviews auto-approved.
    #       Run this in a background thread every 0.5 s.
    #       """
    #       count = 0
    #       for review in interrupt_manager.get_pending():
    #           if self.is_expired(review):
    #               interrupt_manager.submit_decision(
    #                   review_id = review["review_id"],
    #                   decision  = "AUTO_APPROVED",
    #                   operator  = "system",
    #                   reason    = self.compute_auto_approve_reason(self.timeout_seconds),
    #               )
    #               count += 1
    #       return count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_expires(review: Any) -> datetime | None:
        """Extract expires_at datetime from a review object or dict."""
        try:
            raw = review["expires_at"] if isinstance(review, dict) else review.expires_at
            dt  = datetime.fromisoformat(str(raw))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
