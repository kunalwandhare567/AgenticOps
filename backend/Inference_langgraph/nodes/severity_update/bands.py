"""
d:/Before_done/severity_update/bands.py
=========================================
Step 1: Impact Band (from preliminary_severity)
Step 2: Urgency Band (from ttf, gated by confidence & ttf_source)
"""
from __future__ import annotations

def get_impact_band(preliminary_severity: str) -> str:
    """
    Step 1 — Impact band (from preliminary_severity)

    Mapping:
        P1 / CRITICAL  -> High
        P2             -> High
        P3 / WARNING   -> Moderate
        P4 / OK / NONE -> None

    Args:
        preliminary_severity: String severity level ("P1", "P2", "P3", "P4", "CRITICAL", "WARNING", "OK")

    Returns:
        "High" | "Moderate" | "None"
    """
    if not preliminary_severity:
        return "None"
    sev = str(preliminary_severity).upper().strip()
    if sev in ("P1", "P2", "CRITICAL", "HIGH"):
        return "High"
    elif sev in ("P3", "WARNING", "MODERATE", "MEDIUM"):
        return "Moderate"
    elif sev in ("P4", "OK", "NONE", "LOW"):
        return "None"
    return "None"


def get_urgency_band(
    ttf: float | int | None,
    ttf_source: str | None,
    confidence: float,
    min_confidence: float = 0.75,
    invalid_sources: set[str] | None = None,
) -> tuple[str, bool]:
    """
    Step 2 — Urgency band (from ttf, gated)

    Gate:
        confidence >= 0.75 AND ttf_source NOT IN {not_applicable, rollback_decision}

    If gate fails -> urgency band = Distant
    If gate passes:
        ttf < 30s                 -> Imminent
        30s <= ttf <= 120s        -> Near
        ttf > 120s or ttf is None -> Distant

    Args:
        ttf:             Predicted Time to Failure in seconds (or None)
        ttf_source:      Source feature or decision string (e.g. "heap_mb", "not_applicable")
        confidence:      Forecast confidence score (0.0 to 1.0)
        min_confidence:  Minimum confidence threshold for gate (default 0.75)
        invalid_sources: Set of excluded ttf_source strings

    Returns:
        (urgency_band, gate_passed_boolean)
        urgency_band: "Imminent" | "Near" | "Distant"
    """
    if invalid_sources is None:
        invalid_sources = {"not_applicable", "rollback_decision"}

    source_clean = str(ttf_source).lower().strip() if ttf_source else ""
    
    # Evaluate Gate
    gate_passed = (confidence >= min_confidence) and (source_clean not in invalid_sources)

    if not gate_passed:
        return "Distant", False

    if ttf is None:
        return "Distant", True

    try:
        ttf_val = float(ttf)
    except (TypeError, ValueError):
        return "Distant", True

    if ttf_val < 0:
        # Already breached or negative extrapolation
        return "Imminent", True

    if ttf_val < 30.0:
        return "Imminent", True
    elif 30.0 <= ttf_val <= 120.0:
        return "Near", True
    else:
        return "Distant", True
