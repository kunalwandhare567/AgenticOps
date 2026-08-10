"""
d:/Before_done/severity_update/updater.py
==========================================
Main orchestrator class tying together Steps 1–4.

Entry point:
    updater = SeverityUpdater(dwell_k=5, min_confidence=0.75)
    result = updater.process_cycle(episode_id, preliminary_severity, forecast_result)
"""
from __future__ import annotations

from typing import Any
from .bands import get_impact_band, get_urgency_band
from .matrix import get_candidate_severity
from .hysteresis import HysteresisTracker

class SeverityUpdater:
    """
    Main orchestrator for severity_update (Stage 7).
    """
    def __init__(self, dwell_k: int = 5, min_confidence: float = 0.75):
        """
        Args:
            dwell_k:        Dwell time K cycles for de-escalation (default 5 cycles = 10s).
            min_confidence: Minimum confidence threshold for urgency gate (default 0.75).
        """
        self.tracker = HysteresisTracker(default_k=dwell_k)
        self.min_confidence = min_confidence

    def process_cycle(
        self,
        episode_id: str,
        preliminary_severity: str,
        forecast_result: dict[str, Any],
        k: int | None = None,
    ) -> dict[str, Any]:
        """
        Process one cycle and compute revised severity.

        Args:
            episode_id:           Active episode identifier.
            preliminary_severity: Impact input string ("P1", "P2", "P3", "P4", "CRITICAL", "WARNING", "OK").
            forecast_result:      Output dict from forecasting node / convergence schema.
            k:                    Optional override for dwell time K cycles.

        Returns:
            Dict containing:
                "revised_severity":    str ("P1" | "P2" | "P3" | "P4"),
                "candidate_severity":  str ("P1" | "P2" | "P3" | "P4"),
                "impact_band":         str ("High" | "Moderate" | "None"),
                "urgency_band":        str ("Imminent" | "Near" | "Distant"),
                "gate_passed":         bool,
                "is_escalated":        bool,
                "is_deescalated":      bool,
                "dwell_count":         int,
                "reason":              str,
        """
        # Extract parameters from forecast_result (handles both nested and flat schemas)
        fc = forecast_result.get("forecast", forecast_result) if forecast_result else {}
        ttf = forecast_result.get("time_to_failure", fc.get("time_to_failure"))
        confidence = forecast_result.get("forecast_confidence", fc.get("forecast_confidence", 0.0))
        ttf_source = fc.get("earliest_ttf_feature", fc.get("algorithm_used", "auto_arima"))

        # Step 1 — Impact Band
        impact_band = get_impact_band(preliminary_severity)

        # Step 2 — Urgency Band (gated)
        urgency_band, gate_passed = get_urgency_band(
            ttf=ttf,
            ttf_source=ttf_source,
            confidence=confidence,
            min_confidence=self.min_confidence,
        )

        # Step 3 — Matrix Lookup
        candidate_sev = get_candidate_severity(impact_band, urgency_band)

        # Step 4 — Hysteresis Reconciliation
        revised_sev, is_escalated, is_deescalated, dwell_count = self.tracker.reconcile(
            episode_id=episode_id,
            candidate_severity=candidate_sev,
            k=k,
        )

        # Reason string construction for observability / dashboard UI
        k_used = k if k is not None else self.tracker.default_k
        reason = (
            f"Impact={impact_band} ({preliminary_severity}), "
            f"Urgency={urgency_band} (ttf={ttf}s, conf={confidence:.2f}, gate={'PASS' if gate_passed else 'FAIL'}) "
            f"-> candidate={candidate_sev}, revised={revised_sev}"
        )
        if is_escalated:
            reason += " [IMMEDIATE ESCALATION]"
        elif is_deescalated:
            reason += f" [DE-ESCALATED (Dwell Met {k_used}/{k_used})]"
        elif candidate_sev != revised_sev:
            reason += f" [HOLDING (Dwell {dwell_count}/{k_used})]"

        return {
            "revised_severity":   revised_sev,
            "candidate_severity": candidate_sev,
            "impact_band":        impact_band,
            "urgency_band":       urgency_band,
            "gate_passed":        gate_passed,
            "is_escalated":       is_escalated,
            "is_deescalated":     is_deescalated,
            "dwell_count":        dwell_count,
            "reason":             reason,
        }

    def clear_episode(self, episode_id: str) -> None:
        self.tracker.clear_episode(episode_id)

    def reset_all(self) -> None:
        self.tracker.reset_all()


# Global singleton instance & convenience function
_GLOBAL_UPDATER = SeverityUpdater()

def update_severity(
    episode_id: str,
    preliminary_severity: str,
    forecast_result: dict[str, Any],
    k: int | None = None,
) -> dict[str, Any]:
    """Convenience helper using global SeverityUpdater instance."""
    return _GLOBAL_UPDATER.process_cycle(episode_id, preliminary_severity, forecast_result, k=k)
