"""
backend/nodes/human_gate/escalation_detector.py
================================================
Module 1 — Escalation Detector.

Responsibility
--------------
Determine whether a severity change from old_severity → new_severity
constitutes an *escalation* that requires human validation before
it can affect production operations.

Design
------
Escalation is defined as: the new severity is a HIGHER priority than
the old severity. In P-notation, lower number = higher priority, so:

    P4 → P1  : escalation  (1 < 4)
    P3 → P2  : escalation  (2 < 3)
    P2 → P1  : escalation  (1 < 2)
    P3 → P3  : no change   — skip Human Gate
    P1 → P3  : de-escalation — skip Human Gate (hysteresis handles this)

This module is intentionally self-contained — it does NOT import from
severity_update/matrix.py to keep the Human Gate dependency-free.

Usage
-----
    detector = EscalationDetector()
    needs_review = detector.needs_review("P4", "P1")  # True
    needs_review = detector.needs_review("P3", "P3")  # False
    needs_review = detector.needs_review("P2", "P3")  # False (de-escalation)
"""
from __future__ import annotations


# Numeric rank: lower number = higher priority / worse severity
SEVERITY_RANK: dict[str, int] = {
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "P4": 4,
}

# Severity display labels for UI / audit
SEVERITY_LABEL: dict[str, str] = {
    "P1": "Critical",
    "P2": "High",
    "P3": "Moderate",
    "P4": "Low",
}

# Escalation jump size threshold — escalations ≥ this many levels get
# flagged as "large jump" in the review request for operator attention.
LARGE_JUMP_THRESHOLD: int = 2   # e.g. P4 → P2 (jump of 2 levels)


class EscalationDetector:
    """
    Determines whether a severity change requires human validation.

    The detector is stateless — it can be shared as a singleton or
    instantiated fresh per run.
    """

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def needs_review(self, old_severity: str, new_severity: str) -> bool:
        """
        Return True only when new_severity is a HIGHER priority than
        old_severity (i.e., the AI wants to escalate the incident).

        Args:
            old_severity: Current committed severity (P1–P4 or normalised string).
            new_severity: AI-recommended severity from severity_update node.

        Returns:
            bool — True if human review is required before accepting change.
        """
        old_norm = self._normalise(old_severity)
        new_norm = self._normalise(new_severity)

        old_rank = SEVERITY_RANK.get(old_norm, 4)
        new_rank = SEVERITY_RANK.get(new_norm, 4)

        # Escalation: new rank numerically LOWER = higher priority
        return new_rank < old_rank

    # ------------------------------------------------------------------
    # Auxiliary helpers (used by ReviewRequestBuilder)
    # ------------------------------------------------------------------

    def jump_size(self, old_severity: str, new_severity: str) -> int:
        """
        Return the number of severity levels being jumped.

        Example:
            jump_size("P4", "P1") → 3
            jump_size("P3", "P2") → 1
        """
        old_norm = self._normalise(old_severity)
        new_norm = self._normalise(new_severity)
        old_rank = SEVERITY_RANK.get(old_norm, 4)
        new_rank = SEVERITY_RANK.get(new_norm, 4)
        return max(0, old_rank - new_rank)

    def is_large_jump(self, old_severity: str, new_severity: str) -> bool:
        """Return True if the escalation jumps ≥ LARGE_JUMP_THRESHOLD levels."""
        return self.jump_size(old_severity, new_severity) >= LARGE_JUMP_THRESHOLD

    def escalation_summary(self, old_severity: str, new_severity: str) -> str:
        """
        Human-readable one-line summary of the escalation.

        Example:
            "P4 (Low) → P1 (Critical) — 3-level jump"
        """
        old_norm = self._normalise(old_severity)
        new_norm = self._normalise(new_severity)
        jump = self.jump_size(old_norm, new_norm)
        old_label = SEVERITY_LABEL.get(old_norm, old_norm)
        new_label = SEVERITY_LABEL.get(new_norm, new_norm)
        flag = " ⚠ LARGE JUMP" if jump >= LARGE_JUMP_THRESHOLD else ""
        return f"{old_norm} ({old_label}) → {new_norm} ({new_label}) — {jump}-level jump{flag}"

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(sev: str | None) -> str:
        """
        Normalise severity string to P1/P2/P3/P4.
        Handles legacy labels like CRITICAL, HIGH, WARNING, OK.
        """
        if not sev:
            return "P4"
        mapping = {
            "CRITICAL": "P1",
            "HIGH":     "P2",
            "WARNING":  "P3",
            "MODERATE": "P3",
            "LOW":      "P4",
            "OK":       "P4",
            "NONE":     "P4",
        }
        s = str(sev).upper().strip()
        return mapping.get(s, s if s in SEVERITY_RANK else "P4")
