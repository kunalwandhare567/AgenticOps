"""
d:/Before_done/severity_update/matrix.py
==========================================
Step 3 — Combination matrix (Impact × Urgency → candidate severity)
Fixed 12-cell table (4 Impact values × 3 Urgency values).
"""
from __future__ import annotations

# Matrix lookup: (Impact, Urgency) -> Candidate Severity
# Priority order: P1 > P2 > P3 > P4
COMBINATION_MATRIX: dict[tuple[str, str], str] = {
    ("High", "Imminent"): "P1",
    ("High", "Near"):     "P1",
    ("High", "Distant"):  "P2",

    ("Moderate", "Imminent"): "P1",
    ("Moderate", "Near"):     "P2",
    ("Moderate", "Distant"):  "P3",

    ("None", "Imminent"): "P2",
    ("None", "Near"):     "P3",
    ("None", "Distant"):  "P4",
}

# Numeric rank for priority comparison (lower number = higher severity / worse)
SEVERITY_RANK: dict[str, int] = {
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "P4": 4,
}


def get_candidate_severity(impact_band: str, urgency_band: str) -> str:
    """
    Look up candidate severity in the 4×3 combination matrix.

    Args:
        impact_band:  "High" | "Moderate" | "None"
        urgency_band: "Imminent" | "Near" | "Distant"

    Returns:
        Candidate severity: "P1" | "P2" | "P3" | "P4"
    """
    key = (impact_band, urgency_band)
    return COMBINATION_MATRIX.get(key, "P4")


def is_escalation(candidate: str, current: str) -> bool:
    """
    Escalation check: Candidate is worse/higher priority than Current.
    (Rank is a smaller number, e.g. P1 < P4).
    """
    c_rank = SEVERITY_RANK.get(candidate, 4)
    curr_rank = SEVERITY_RANK.get(current, 4)
    return c_rank < curr_rank


def is_deescalation(candidate: str, current: str) -> bool:
    """
    De-escalation check: Candidate is better/lower priority than Current.
    (Rank is a larger number, e.g. P4 > P1).
    """
    c_rank = SEVERITY_RANK.get(candidate, 4)
    curr_rank = SEVERITY_RANK.get(current, 4)
    return c_rank > curr_rank
