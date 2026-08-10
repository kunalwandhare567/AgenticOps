"""
d:/Before_done/severity_update/hysteresis.py
==============================================
Step 4 — Hysteresis against recent severity history.

Rules:
  - Escalation (candidate worse than current updated_severity) -> applied immediately (0-cycle delay).
  - De-escalation (candidate better than current) -> applied only after candidate persists
    for K consecutive cycles (dwell time).
"""
from __future__ import annotations

from .matrix import is_escalation, is_deescalation

class HysteresisTracker:
    """
    Stateful hysteresis tracker per episode_id.
    Maintains current committed severity, pending candidate, and dwell counter.
    """
    def __init__(self, default_k: int = 5):
        """
        Args:
            default_k: Dwell time in cycles for de-escalation (default 5 cycles = 10 seconds).
        """
        self.default_k = default_k
        # { episode_id: {"current_severity": str, "candidate_severity": str, "dwell_count": int} }
        self._episodes: dict[str, dict] = {}

    def reconcile(
        self,
        episode_id: str,
        candidate_severity: str,
        k: int | None = None,
    ) -> tuple[str, bool, bool, int]:
        """
        Reconciles candidate_severity against the episode's history state.

        Args:
            episode_id:          Episode identifier.
            candidate_severity:  Output from Step 3 matrix lookup ("P1".."P4").
            k:                   Optional override for dwell time K cycles.

        Returns:
            (committed_severity, is_escalated, is_deescalated, dwell_count)
        """
        dwell_limit = k if k is not None else self.default_k

        if episode_id not in self._episodes:
            # First cycle for this episode -> initialize directly
            self._episodes[episode_id] = {
                "current_severity": candidate_severity,
                "candidate_severity": candidate_severity,
                "dwell_count": 0,
            }
            return candidate_severity, False, False, 0

        state = self._episodes[episode_id]
        current = state["current_severity"]

        # Case 1: Candidate is identical to current committed severity
        if candidate_severity == current:
            state["candidate_severity"] = candidate_severity
            state["dwell_count"] = 0
            return current, False, False, 0

        # Case 2: Escalation (Candidate is worse/higher priority than current)
        if is_escalation(candidate_severity, current):
            state["current_severity"] = candidate_severity
            state["candidate_severity"] = candidate_severity
            state["dwell_count"] = 0
            return candidate_severity, True, False, 0

        # Case 3: De-escalation (Candidate is better/lower priority than current)
        if is_deescalation(candidate_severity, current):
            if state["candidate_severity"] == candidate_severity:
                state["dwell_count"] += 1
            else:
                state["candidate_severity"] = candidate_severity
                state["dwell_count"] = 1

            # Check if dwell threshold is met
            if state["dwell_count"] >= dwell_limit:
                state["current_severity"] = candidate_severity
                state["dwell_count"] = 0
                return candidate_severity, False, True, dwell_limit
            else:
                # Hold current severity until dwell count reaches K
                return current, False, False, state["dwell_count"]

        # Fallback
        return current, False, False, 0

    def clear_episode(self, episode_id: str) -> None:
        """Clear state for a completed episode."""
        self._episodes.pop(episode_id, None)

    def reset_all(self) -> None:
        """Clear all episode states."""
        self._episodes.clear()
