"""
backend/Inference_langgraph/state.py
=====================================
AIOps LangGraph state schema — the single TypedDict that flows through
every node in the StateGraph.

Pydantic v2 validators are applied at each node boundary via the
`validate_state()` helper to catch bad data early without polluting node code.

Design rules:
  - TypedDict is required by LangGraph's StateGraph — it cannot accept Pydantic models.
  - Pydantic BaseModel `_AIOpsStateModel` mirrors every field for validation only.
  - `validate_state(state)` coerces and validates; raises ValidationError on failure.
  - All fields have sensible defaults so nodes only need to return the keys they changed.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# 1. LangGraph State — TypedDict  (the graph operates on this)
# =============================================================================

class AIOpsLangState(TypedDict, total=False):
    """
    Full pipeline state carried through every LangGraph node.
    All fields are Optional (total=False) so nodes return only the keys they update.
    LangGraph merges the returned dict into the existing state.
    """
    # ── Cycle identity ────────────────────────────────────────────────────────
    cycle:              int
    last_processed_id:  int         # last DB row id consumed by n01_collect

    # ── Raw telemetry (from simulator DB) ────────────────────────────────────
    raw_metric:         dict
    raw_log:            list        # list[dict]
    raw_traces:         list        # list[dict]
    episode_id:         str
    failure_mode:       str
    timestamp:          float
    elapsed_s:          float
    service:            str

    # ── Feature Engineering ───────────────────────────────────────────────────
    classifier_input:   dict        # 32-feature vector for LightGBM
    evidence:           dict        # diagnostic metadata

    # ── Preliminary Severity ─────────────────────────────────────────────────
    preliminary_severity:   str     # P1 / P2 / P3 / P4
    severity_result:        dict    # full SeverityResult fields as plain dict

    # ── Classification ────────────────────────────────────────────────────────
    predicted_failure:          str
    prediction_probability:     float

    # ── Tumbling Window ───────────────────────────────────────────────────────
    dominant_state:     str
    vote_distribution:  dict        # {mode_name: count}
    window_margin:      float
    window_full:        bool

    # ── Forecasting ───────────────────────────────────────────────────────────
    forecast_result:        dict    # full raw result from route_forecast()
    forecast_algorithm:     Optional[str]
    time_to_failure:        Optional[float]   # seconds until threshold breach
    forecast_confidence:    Optional[float]   # 0.0 – 1.0
    threshold_crossed:      Optional[bool]
    earliest_ttf_feature:   Optional[str]

    # ── Severity Update ───────────────────────────────────────────────────────
    revised_severity:   Optional[str]
    candidate_severity: Optional[str]
    impact_band:        Optional[str]
    urgency_band:       Optional[str]
    gate_passed:        Optional[bool]
    is_escalated:       Optional[bool]
    is_deescalated:     Optional[bool]
    su_reason:          Optional[str]
    dwell_count:        Optional[int]

    # ── Reliability ───────────────────────────────────────────────────────────
    active_failure_group:   Optional[str]
    survival_probability:   Optional[float]
    weibull_beta:           Optional[float]
    weibull_eta:            Optional[float]

    # ── Human Gate ────────────────────────────────────────────────────────────
    # Committed severity BEFORE this cycle's human gate (used for escalation detection)
    committed_severity:     Optional[str]

    hg_needed:          Optional[bool]      # True if human review was triggered
    hg_review_id:       Optional[str]
    hg_decision:        Optional[str]       # APPROVED | REJECTED | AUTO_APPROVED
    hg_final_severity:  Optional[str]
    hg_operator:        Optional[str]
    hg_response_ms:     Optional[int]
    hg_escalation_summary: Optional[str]
    hg_is_large_jump:   Optional[bool]

    # ── Internal routing ──────────────────────────────────────────────────────
    error:              Optional[str]       # set by any node on exception → routes to END


# =============================================================================
# 2. Pydantic validation model  (mirrors TypedDict for field-level validation)
# =============================================================================

class _AIOpsStateModel(BaseModel):
    """
    Pydantic v2 model used ONLY for validation at node boundaries.
    Not used by LangGraph directly — call validate_state() after each node.

    All fields are Optional with defaults so partial state dicts validate fine.
    """
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    # Identity
    cycle:              int   = Field(default=0, ge=0)
    last_processed_id:  int   = Field(default=0, ge=0)

    # Telemetry
    raw_metric:         dict  = Field(default_factory=dict)
    raw_log:            list  = Field(default_factory=list)
    raw_traces:         list  = Field(default_factory=list)
    episode_id:         str   = Field(default="")
    failure_mode:       str   = Field(default="NONE")
    timestamp:          float = Field(default=0.0, ge=0.0)
    elapsed_s:          float = Field(default=0.0, ge=0.0)
    service:            str   = Field(default="")

    # Feature Engineering
    classifier_input:   dict  = Field(default_factory=dict)
    evidence:           dict  = Field(default_factory=dict)

    # Preliminary Severity
    preliminary_severity: str = Field(default="P4")
    severity_result:      dict = Field(default_factory=dict)

    @field_validator("preliminary_severity", mode="before")
    @classmethod
    def validate_severity_label(cls, v: Any) -> str:
        """Normalise raw severity strings to P1–P4."""
        if v is None:
            return "P4"
        mapping = {"CRITICAL": "P1", "HIGH": "P2", "WARNING": "P3",
                   "MODERATE": "P3", "LOW": "P4", "OK": "P4", "NONE": "P4"}
        s = str(v).upper().strip()
        return mapping.get(s, s if s in {"P1", "P2", "P3", "P4"} else "P4")

    # Classification
    predicted_failure:       str   = Field(default="NONE")
    prediction_probability:  float = Field(default=0.0, ge=0.0, le=1.0)

    # Tumbling Window
    dominant_state:    str   = Field(default="NONE")
    vote_distribution: dict  = Field(default_factory=dict)
    window_margin:     float = Field(default=0.0, ge=0.0)
    window_full:       bool  = Field(default=False)

    # Forecasting
    forecast_result:       dict            = Field(default_factory=dict)
    forecast_algorithm:    Optional[str]   = None
    time_to_failure:       Optional[float] = None
    forecast_confidence:   Optional[float] = Field(default=None, ge=0.0, le=1.0)
    threshold_crossed:     Optional[bool]  = None
    earliest_ttf_feature:  Optional[str]  = None

    # Severity Update
    revised_severity:   Optional[str]   = None
    candidate_severity: Optional[str]   = None
    impact_band:        Optional[str]   = None
    urgency_band:       Optional[str]   = None
    gate_passed:        Optional[bool]  = None
    is_escalated:       Optional[bool]  = None
    is_deescalated:     Optional[bool]  = None
    su_reason:          Optional[str]   = None
    dwell_count:        Optional[int]   = None

    # Reliability
    active_failure_group:  Optional[str]   = None
    survival_probability:  Optional[float] = Field(default=None, ge=0.0, le=100.0)
    weibull_beta:          Optional[float] = Field(default=None, gt=0.0)
    weibull_eta:           Optional[float] = Field(default=None, gt=0.0)

    # Human Gate
    committed_severity:     Optional[str]  = None
    hg_needed:              Optional[bool] = None
    hg_review_id:           Optional[str]  = None
    hg_decision:            Optional[str]  = None
    hg_final_severity:      Optional[str]  = None
    hg_operator:            Optional[str]  = None
    hg_response_ms:         Optional[int]  = Field(default=None, ge=0)
    hg_escalation_summary:  Optional[str]  = None
    hg_is_large_jump:       Optional[bool] = None

    # Routing
    error:  Optional[str] = None


# =============================================================================
# 3. Public helper — validate & coerce a raw state dict
# =============================================================================

def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and coerce a state dict using Pydantic.

    Returns the validated state as a plain dict (not a Pydantic model) so
    LangGraph can merge it normally.

    Raises:
        pydantic.ValidationError: if a field value is out of range or wrong type.

    Usage (inside any node function):
        from Inference_langgraph.state import validate_state
        state = validate_state(state)
    """
    model = _AIOpsStateModel.model_validate(state)
    # model_dump includes only fields that were actually set (exclude_unset=True
    # would omit defaults we want to keep, so we use exclude_none=False here).
    return model.model_dump(exclude_none=False)


def make_empty_state(cycle: int = 0) -> AIOpsLangState:
    """Return a blank initial state dict for cycle `cycle`."""
    return _AIOpsStateModel(cycle=cycle).model_dump()
