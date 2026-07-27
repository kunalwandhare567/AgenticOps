"""
app_simulator/pipeline/state.py
================================
PipelineState dataclass — one instance per telemetry cycle.

Flows through the pipeline in order:
  Data Generation → Feature Engineering → Severity → Classification → Tumbling Window
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """Per-cycle shared state passed between all pipeline nodes."""

    # ── Raw telemetry (from generator / queue) ────────────────────────────────
    raw_metric:   dict = field(default_factory=dict)
    raw_log:      dict = field(default_factory=dict)
    raw_traces:   list = field(default_factory=list)
    episode_id:   str  = ""
    failure_mode: str  = ""
    timestamp:    float = 0.0
    elapsed_s:    float = 0.0
    step:         int   = 0
    service:      str   = ""

    # ── Feature Engineering output ────────────────────────────────────────────
    # 27 raw metrics + 5 log features + metadata
    classifier_input: dict = field(default_factory=dict)
    # Evidence fields (never in classifier CSV)
    evidence: dict = field(default_factory=dict)

    # ── Preliminary Severity output ───────────────────────────────────────────
    preliminary_severity: str = "P4"          # "P1" | "P2" | "P3" | "P4"
    severity_reasons:     list[str] = field(default_factory=list)
    # Full SeverityResult from DEVOPS engine (None until severity_node runs)
    severity_result:      object = None       # SeverityResult | None


    # ── Classification output ─────────────────────────────────────────────────
    predicted_failure:      str   = "NONE"
    prediction_probability: float = 0.0

    # ── Tumbling Window output ────────────────────────────────────────────────
    window_predictions: list[str] = field(default_factory=list)
    summarized_failure:  str = "NONE"
    vote_distribution:   dict[str, int] = field(default_factory=dict)
    window_margin:       int = 0
    window_full:         bool = False    # True once 10 predictions buffered

    def summary_line(self, cycle: int) -> str:
        """One-line console summary for this cycle."""
        prob_str = f"{self.prediction_probability:.2f}" if self.prediction_probability else "0.00"
        window_str = (
            f"{self.summarized_failure}({self.window_predictions.count(self.summarized_failure)}/{len(self.window_predictions)})"
            if self.window_predictions else "cold-start"
        )
        return (
            f"[cycle {cycle:>5}] "
            f"ep={self.failure_mode:<22} "
            f"sev={self.preliminary_severity}  "
            f"pred={self.predicted_failure:<22}({prob_str})  "
            f"window={window_str}"
        )
