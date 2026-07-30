"""
backend/nodes/human_gate/approval_engine.py
============================================
Module 5 — Approval Engine & State Machine.

Responsibility
--------------
Implements the formal state machine that governs each review's lifecycle,
and computes the final ApprovalResult — the object the rest of the pipeline
uses to determine the actual severity that goes to notify_node.

State Machine
-------------

    ┌─────────┐
    │ WAITING │  ← Initial state when review is posted
    └────┬────┘
         │
    ┌────▼──────────────────────┐
    │ REVIEWING                 │  ← Operator has opened the review panel
    └────┬───────────┬──────────┘
         │           │
    ┌────▼───┐  ┌────▼────┐  ┌───────────────┐
    │APPROVED│  │REJECTED │  │ AUTO_APPROVED  │ ← timeout fires
    └────┬───┘  └────┬────┘  └───────┬───────┘
         │           │               │
    └────▼───────────▼───────────────▼──┐
                  COMPLETED              │
    └───────────────────────────────────┘

ApprovalResult
--------------
    decision        : APPROVED | REJECTED | AUTO_APPROVED
    final_severity  : new_severity if approved/auto, old_severity if rejected
    operator        : reviewer username
    reason          : operator comment or auto reason
    response_ms     : ms from created_at to decision

Usage
-----
    engine = ApprovalEngine()
    result = engine.compute_result(review_dict)
    print(result.final_severity)   # "P1" or "P4" etc.
    print(result.decision)         # "APPROVED"
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# State machine states
# ---------------------------------------------------------------------------

class ReviewState(str, Enum):
    """
    Formal states of a Human Gate review.
    Using str-enum so values are directly JSON-serialisable.
    """
    WAITING       = "WAITING"        # Posted, awaiting operator
    REVIEWING     = "REVIEWING"      # Operator has opened the review panel
    APPROVED      = "APPROVED"       # Operator approved escalation
    REJECTED      = "REJECTED"       # Operator rejected escalation
    AUTO_APPROVED = "AUTO_APPROVED"  # Timeout fired — auto accepted
    COMPLETED     = "COMPLETED"      # Terminal state (archived)


# Valid state transitions (from → set of allowed to-states)
_VALID_TRANSITIONS: dict[ReviewState, set[ReviewState]] = {
    ReviewState.WAITING:       {ReviewState.REVIEWING, ReviewState.AUTO_APPROVED},
    ReviewState.REVIEWING:     {ReviewState.APPROVED, ReviewState.REJECTED, ReviewState.AUTO_APPROVED},
    ReviewState.APPROVED:      {ReviewState.COMPLETED},
    ReviewState.REJECTED:      {ReviewState.COMPLETED},
    ReviewState.AUTO_APPROVED: {ReviewState.COMPLETED},
    ReviewState.COMPLETED:     set(),   # Terminal — no further transitions
}

# States considered "terminal" (no longer pending)
TERMINAL_STATES: frozenset[ReviewState] = frozenset({
    ReviewState.APPROVED,
    ReviewState.REJECTED,
    ReviewState.AUTO_APPROVED,
    ReviewState.COMPLETED,
})


# ---------------------------------------------------------------------------
# ApprovalResult
# ---------------------------------------------------------------------------

@dataclass
class ApprovalResult:
    """
    Outcome of a completed Human Gate review.

    The pipeline uses final_severity to determine which severity value
    proceeds to notify_node. Everything else is recorded in the audit log.
    """

    # Core decision
    decision:       str    # "APPROVED" | "REJECTED" | "AUTO_APPROVED"
    final_severity: str    # Actual severity to apply: old or new

    # Provenance
    old_severity:   str    # What it was before
    new_severity:   str    # What the AI recommended
    operator:       str    # Who decided (or "system" for auto)
    reason:         str    # Operator comment / auto-reason

    # Timing
    response_ms:    int    # Milliseconds from created_at to decision
    decided_at:     str    # ISO-8601 UTC timestamp of decision

    # Identity (forwarded from HumanReviewRequest)
    review_id:      str
    incident_id:    str
    episode_id:     str

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def was_escalation_accepted(self) -> bool:
        """True if the AI's escalation was accepted (approved or auto-approved)."""
        return self.decision in ("APPROVED", "AUTO_APPROVED")

    @property
    def was_rejected_by_human(self) -> bool:
        """True only for explicit human rejection."""
        return self.decision == "REJECTED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id":             self.review_id,
            "incident_id":           self.incident_id,
            "episode_id":            self.episode_id,
            "decision":              self.decision,
            "final_severity":        self.final_severity,
            "old_severity":          self.old_severity,
            "new_severity":          self.new_severity,
            "operator":              self.operator,
            "reason":                self.reason,
            "response_ms":           self.response_ms,
            "decided_at":            self.decided_at,
            "was_escalation_accepted": self.was_escalation_accepted,
            "was_rejected_by_human": self.was_rejected_by_human,
        }


# ---------------------------------------------------------------------------
# Approval Engine
# ---------------------------------------------------------------------------

class ApprovalEngine:
    """
    Converts a settled review row (from InterruptManager / SQLite) into
    an ApprovalResult, applying state-machine rules for final_severity.

    This class is stateless — instantiate once and call compute_result()
    for each settled review.
    """

    # ------------------------------------------------------------------
    # State machine transition validator
    # ------------------------------------------------------------------

    @staticmethod
    def is_valid_transition(from_state: str, to_state: str) -> bool:
        """
        Return True if transitioning from_state → to_state is allowed.
        Accepts string values (raw from SQLite) and coerces to ReviewState.
        """
        try:
            f = ReviewState(from_state)
            t = ReviewState(to_state)
            return t in _VALID_TRANSITIONS.get(f, set())
        except ValueError:
            return False

    @staticmethod
    def is_terminal(status: str) -> bool:
        """Return True if the review status is a terminal state."""
        try:
            return ReviewState(status) in TERMINAL_STATES
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def compute_result(self, review: dict[str, Any]) -> ApprovalResult:
        """
        Given a settled review row dict (from InterruptManager), compute
        the final ApprovalResult that determines which severity the pipeline
        should use.

        Decision → final_severity mapping:
            APPROVED      → new_severity  (AI escalation accepted)
            AUTO_APPROVED → new_severity  (timeout: accept AI recommendation)
            REJECTED      → old_severity  (human says no — keep current)

        Args:
            review: Dict from InterruptManager.get_review() or poll_for_decision().
                    Must contain: decision, old_severity, new_severity, operator,
                                  reason, response_ms, decided_at, review_id,
                                  incident_id, episode_id.

        Returns:
            ApprovalResult — fully populated.
        """
        decision   = str(review.get("decision",   "AUTO_APPROVED")).upper()
        old_sev    = str(review.get("old_severity", "P4"))
        new_sev    = str(review.get("new_severity", "P4"))
        operator   = str(review.get("operator",   "system"))
        reason     = str(review.get("reason",     ""))
        decided_at = str(review.get("decided_at", datetime.now(timezone.utc).isoformat()))

        # Determine final severity based on decision
        if decision in ("APPROVED", "AUTO_APPROVED"):
            final_sev = new_sev   # Accept AI recommendation
        elif decision == "REJECTED":
            final_sev = old_sev   # Keep current severity
        else:
            # Unexpected decision value — default to auto-approve for safety
            decision  = "AUTO_APPROVED"
            final_sev = new_sev
            reason    = f"Unknown decision '{decision}' — defaulted to auto-approve."

        # Compute response_ms (from DB if available, else calculate)
        try:
            response_ms = int(review.get("response_ms", 0))
        except (TypeError, ValueError):
            response_ms = 0

        return ApprovalResult(
            decision        = decision,
            final_severity  = final_sev,
            old_severity    = old_sev,
            new_severity    = new_sev,
            operator        = operator,
            reason          = reason,
            response_ms     = response_ms,
            decided_at      = decided_at,
            review_id       = str(review.get("review_id",   "")),
            incident_id     = str(review.get("incident_id", "")),
            episode_id      = str(review.get("episode_id",  "")),
        )
