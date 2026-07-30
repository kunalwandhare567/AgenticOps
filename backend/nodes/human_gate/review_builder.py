"""
backend/nodes/human_gate/review_builder.py
==========================================
Module 2 — Review Request Builder.

Responsibility
--------------
Given a severity_update output row and forecast context, construct a
fully-enriched HumanReviewRequest dataclass. This is the object the
dashboard renders and the operator reviews before approving / rejecting
an escalation.

Design
------
- All fields are typed and defaulted — no field can be None unexpectedly.
- `expires_at` = created_at + HUMAN_GATE_TIMEOUT_SECONDS  (from config).
- `review_id` is a UUID4 string — unique across all runs.
- `incident_id` follows the dashboard convention: INC-{hash(episode_id) % 10000:04d}.

Usage
-----
    builder = ReviewRequestBuilder()
    request = builder.from_severity_update_row(row_dict)
    print(request.to_dict())
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_data_generator.config import HUMAN_GATE_TIMEOUT_SECONDS
from nodes.human_gate.escalation_detector import EscalationDetector, SEVERITY_LABEL


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HumanReviewRequest:
    """
    Fully-enriched review request forwarded to the Human Gate dashboard.

    Every field the operator needs to make a decision is embedded here.
    The object is serialisable to dict / JSON via .to_dict().
    """

    # Identity
    review_id:      str       # UUID4 — primary key for this review session
    incident_id:    str       # INC-XXXX display label
    episode_id:     str       # Raw episode identifier

    # Classification context
    failure_mode:   str       # e.g. MEMORY_LEAK
    failure_label:  str       # Pretty label for UI: "Memory Leak"

    # Severity change being proposed
    old_severity:   str       # Current committed severity before escalation
    new_severity:   str       # AI-recommended severity after severity_update

    # AI supporting evidence
    confidence:     float     # Forecast confidence (0.0–1.0)
    ttf_seconds:    float     # Time to failure in seconds (−1 if unknown)
    impact_band:    str       # "High" | "Moderate" | "None"
    urgency_band:   str       # "Imminent" | "Near" | "Distant"
    root_cause:     str       # Reason string from SeverityUpdater
    escalation_summary: str   # Human-readable "P4 → P1 — 3-level jump"
    is_large_jump:  bool      # True if jump ≥ 2 levels

    # Timing
    created_at:     str       # ISO-8601 UTC timestamp
    expires_at:     str       # ISO-8601 UTC timestamp (auto-approve deadline)
    timeout_seconds: int      # Configured timeout for this review

    # State tracking (updated by InterruptManager)
    status:         str = "WAITING"  # WAITING | REVIEWING | APPROVED | REJECTED | AUTO_APPROVED

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict (JSON-serialisable) copy of this request."""
        return asdict(self)

    @property
    def is_expired(self) -> bool:
        """True if the review deadline has passed."""
        now = datetime.now(timezone.utc)
        exp = datetime.fromisoformat(self.expires_at)
        # Make tz-aware if needed
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return now >= exp

    @property
    def seconds_remaining(self) -> float:
        """Seconds left before auto-approve. Returns 0.0 if already expired."""
        now = datetime.now(timezone.utc)
        exp = datetime.fromisoformat(self.expires_at)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        delta = (exp - now).total_seconds()
        return max(0.0, delta)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class ReviewRequestBuilder:
    """
    Constructs HumanReviewRequest objects from severity_update output rows.

    The builder is stateless and re-usable. Instantiate once and call
    from_severity_update_row() for every escalation.
    """

    def __init__(self) -> None:
        self._detector = EscalationDetector()

    # ------------------------------------------------------------------
    # Primary factory method
    # ------------------------------------------------------------------

    def from_severity_update_row(
        self,
        row: dict[str, Any],
        preliminary_severity: str | None = None,
    ) -> HumanReviewRequest:
        """
        Build a HumanReviewRequest from a severity_update_output.csv row.

        Args:
            row: Dict with keys matching severity_update output schema:
                 episode_id, failure_mode, preliminary_severity,
                 revised_severity, forecast_confidence, time_to_failure,
                 impact_band, urgency_band, reason, …
            preliminary_severity: Override for old severity if not in row.

        Returns:
            HumanReviewRequest — ready to post to InterruptManager.
        """
        episode_id   = str(row.get("episode_id", "unknown"))
        failure_mode = str(row.get("failure_mode", "NONE"))

        # Severity mapping —————————————————————————————————————————
        old_sev = str(
            preliminary_severity
            or row.get("preliminary_severity", "P4")
        ).strip().upper()
        new_sev = str(row.get("revised_severity", "P4")).strip().upper()

        # Normalise legacy labels
        old_sev = self._detector._normalise(old_sev)
        new_sev = self._detector._normalise(new_sev)

        # Forecast values ——————————————————————————————————————————
        try:
            confidence = float(row.get("forecast_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        try:
            ttf_raw = row.get("time_to_failure", -1)
            ttf = float(ttf_raw) if ttf_raw not in (None, "", "nan") else -1.0
        except (TypeError, ValueError):
            ttf = -1.0

        # Timing ————————————————————————————————————————————————————
        now        = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=HUMAN_GATE_TIMEOUT_SECONDS)

        return HumanReviewRequest(
            review_id           = str(uuid.uuid4()),
            incident_id         = f"INC-{abs(hash(episode_id)) % 10000:04d}",
            episode_id          = episode_id,
            failure_mode        = failure_mode,
            failure_label       = failure_mode.replace("_", " ").title(),
            old_severity        = old_sev,
            new_severity        = new_sev,
            confidence          = round(confidence, 4),
            ttf_seconds         = round(ttf, 2),
            impact_band         = str(row.get("impact_band",  "None")),
            urgency_band        = str(row.get("urgency_band", "Distant")),
            root_cause          = str(row.get("reason", "No detailed reason provided.")),
            escalation_summary  = self._detector.escalation_summary(old_sev, new_sev),
            is_large_jump       = self._detector.is_large_jump(old_sev, new_sev),
            created_at          = now.isoformat(),
            expires_at          = expires_at.isoformat(),
            timeout_seconds     = HUMAN_GATE_TIMEOUT_SECONDS,
            status              = "WAITING",
        )
