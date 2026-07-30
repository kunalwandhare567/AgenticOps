"""
backend/api/services.py
========================
Data service layer that loads/joins pipeline results and database states
to expose a unified dashboard incident model.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from app_data_generator.config import (
    PIPELINE_RESULTS_CSV,
    FORECASTING_OUTPUT_CSV,
    SEVERITY_UPDATE_CSV,
    ENGINEERED_FEAT_CSV
)

# Core 7 NOC Metrics mapping configurations
NOC_METRICS_CONFIG = {
    "cpu_utilization": {
        "label": "CPU Utilization",
        "threshold": 90.0,
        "unit": "%",
        "fallback_healthy": [15, 18, 14, 16, 15, 18, 17, 19, 21, 20, 22, 19, 21, 23, 20, 21, 23, 22, 24, 23]
    },
    "heap_mb": {
        "label": "Heap Memory",
        "threshold": 3500.0,
        "unit": "MB",
        "fallback_healthy": [512, 514, 513, 515, 512, 518, 515, 519, 521, 520, 522, 519, 521, 523, 520, 521, 523, 522, 524, 523]
    },
    "p99_latency": {
        "label": "P99 Latency",
        "threshold": 700.0,
        "unit": "ms",
        "fallback_healthy": [110, 112, 115, 108, 112, 114, 110, 115, 120, 118, 122, 119, 121, 123, 120, 121, 123, 122, 124, 123]
    },
    "db_p99": {
        "label": "Database Latency",
        "threshold": 1500.0,
        "unit": "ms",
        "fallback_healthy": [22, 25, 24, 26, 22, 28, 25, 29, 31, 30, 32, 29, 31, 33, 30, 31, 33, 32, 34, 33]
    },
    "error_rate": {
        "label": "Error Rate",
        "threshold": 0.50,
        "unit": "ratio",
        "fallback_healthy": [0.001, 0.002, 0.001, 0.003, 0.001, 0.002, 0.001, 0.003, 0.004, 0.002, 0.003, 0.002, 0.003, 0.004, 0.002, 0.003, 0.004, 0.003, 0.005, 0.004]
    },
    "cache_miss_rate": {
        "label": "Cache Miss Rate",
        "threshold": 0.90,
        "unit": "ratio",
        "fallback_healthy": [0.05, 0.06, 0.04, 0.07, 0.05, 0.08, 0.05, 0.09, 0.11, 0.10, 0.12, 0.09, 0.11, 0.13, 0.10, 0.11, 0.13, 0.12, 0.14, 0.13]
    },
    "queue_lag": {
        "label": "Queue Lag",
        "threshold": 500.0,
        "unit": "ms",
        "fallback_healthy": [1, 2, 0, 3, 1, 4, 2, 5, 3, 4, 2, 3, 5, 2, 3, 5, 4, 6, 5, 4]
    }
}

# Helper to normalize status strings
def clean_status(val: Any) -> str:
    if pd.isna(val) or val is None:
        return "OPEN"
    s = str(val).strip().upper()
    if s in ["IN_PROGRESS", "ACKNOWLEDGED", "OPEN", "RESOLVED"]:
        return s
    return "OPEN"

class AIOpsDashboardService:
    @staticmethod
    def get_all_episodes() -> List[Dict[str, Any]]:
        """Return list of all unique episodes processed in the pipeline."""
        if not PIPELINE_RESULTS_CSV.exists():
            return []
        try:
            df = pd.read_csv(PIPELINE_RESULTS_CSV)
            if df.empty:
                return []
            
            # Sort by cycle desc to put newest first
            df_grouped = df.sort_values("cycle", ascending=False).groupby("episode_id", as_index=False).first()
            
            episodes = []
            for _, row in df_grouped.iterrows():
                episodes.append({
                    "episode_id": row["episode_id"],
                    "failure_mode": row.get("failure_mode", "UNKNOWN"),
                    "last_cycle": int(row["cycle"]),
                    "elapsed_s": float(row.get("elapsed_s", 0.0)),
                    "severity": row.get("preliminary_severity", "P4"),
                })
            return episodes
        except Exception as e:
            print(f"[DashboardService] Error getting episodes: {e}")
            return []

    @staticmethod
    def get_live_state() -> Dict[str, Any]:
        """Get latest cycle details across all nodes to display in real-time."""
        if not PIPELINE_RESULTS_CSV.exists():
            return {}
        try:
            # Get latest line from pipeline_results.csv
            df_pip = pd.read_csv(PIPELINE_RESULTS_CSV)
            if df_pip.empty:
                return {}
            
            latest_pip = df_pip.iloc[-1]
            ep_id = latest_pip["episode_id"]
            
            # Load matching forecasting and severity_update rows (if exist)
            fc_row = None
            if FORECASTING_OUTPUT_CSV.exists():
                df_fc = pd.read_csv(FORECASTING_OUTPUT_CSV)
                matches = df_fc[df_fc["episode_id"] == ep_id]
                if not matches.empty:
                    fc_row = matches.iloc[-1]
                    
            su_row = None
            if SEVERITY_UPDATE_CSV.exists():
                df_su = pd.read_csv(SEVERITY_UPDATE_CSV)
                matches = df_su[df_su["episode_id"] == ep_id]
                if not matches.empty:
                    su_row = matches.iloc[-1]

            return AIOpsDashboardService._build_incident_payload(latest_pip, fc_row, su_row)
        except Exception as e:
            print(f"[DashboardService] Error getting live state: {e}")
            return {}

    @staticmethod
    def get_episode_details(episode_id: str) -> Dict[str, Any]:
        """Get aggregated dashboard details for a specific episode."""
        if not PIPELINE_RESULTS_CSV.exists():
            return {}
        try:
            df_pip = pd.read_csv(PIPELINE_RESULTS_CSV)
            ep_pip = df_pip[df_pip["episode_id"] == episode_id]
            if ep_pip.empty:
                return {}
            
            latest_pip = ep_pip.iloc[-1]
            
            fc_row = None
            if FORECASTING_OUTPUT_CSV.exists():
                df_fc = pd.read_csv(FORECASTING_OUTPUT_CSV)
                matches = df_fc[df_fc["episode_id"] == episode_id]
                if not matches.empty:
                    fc_row = matches.iloc[-1]
                    
            su_row = None
            if SEVERITY_UPDATE_CSV.exists():
                df_su = pd.read_csv(SEVERITY_UPDATE_CSV)
                matches = df_su[df_su["episode_id"] == episode_id]
                if not matches.empty:
                    su_row = matches.iloc[-1]

            return AIOpsDashboardService._build_incident_payload(latest_pip, fc_row, su_row)
        except Exception as e:
            print(f"[DashboardService] Error getting details for {episode_id}: {e}")
            return {}

    @staticmethod
    def _build_incident_payload(pip_row: Any, fc_row: Any, su_row: Any) -> Dict[str, Any]:
        """Map raw CSV rows into frontend incident structure, including 7 core charts."""
        ep_id = pip_row["episode_id"]
        failure_mode = pip_row.get("failure_mode", "UNKNOWN")
        prelim_sev = pip_row.get("preliminary_severity", "P4")
        revised_sev = su_row.get("revised_severity", prelim_sev) if su_row is not None else prelim_sev
        
        # 1. Load telemetry history from engineered_features.csv for 20 logs lookback
        history_df = pd.DataFrame()
        if ENGINEERED_FEAT_CSV.exists():
            try:
                df_fe = pd.read_csv(ENGINEERED_FEAT_CSV)
                # Filter for this episode and take up to the current cycle (based on elapsed_s or cycle)
                history_df = df_fe[df_fe["episode_id"] == ep_id]
                # Sort by timestamp to ensure sequence
                history_df = history_df.sort_values("timestamp")
            except Exception as e:
                print(f"[DashboardService] Failed to load telemetry history: {e}")

        # 2. Load forecasts maps safely
        predictions_map = {}
        slopes_map = {}
        if fc_row is not None:
            try:
                predictions_map = json.loads(fc_row.get("predictions_json", "{}"))
            except Exception:
                pass
            try:
                slopes_map = json.loads(fc_row.get("feature_slopes_json", "{}"))
            except Exception:
                pass

        # 3. Build 7 NOC charts
        charts_data = {}
        for key, conf in NOC_METRICS_CONFIG.items():
            # Get historical series (last 20 cycles)
            history_series = []
            if not history_df.empty and key in history_df.columns:
                # Get non-null list up to 20 cycles
                history_series = history_df[key].dropna().tolist()
                # Slice last 20
                history_series = history_series[-20:]
            
            # Fallback if telemetry log file is empty/cold
            if not history_series:
                history_series = list(conf["fallback_healthy"])

            # Clean NaNs/Infs
            history_series = [float(v) if (isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)) else 0.0 for v in history_series]

            # Get forecast predictions
            forecast_series = []
            if key in predictions_map:
                forecast_series = predictions_map[key]
                # Limit forecast to 10 points for clean graph display
                forecast_series = forecast_series[:10]
            
            # Clean forecast values
            forecast_series = [float(v) if (isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)) else 0.0 for v in forecast_series]

            # Determine latest value
            current_val = history_series[-1] if history_series else 0.0

            # Determine threshold breach crossing
            threshold = conf["threshold"]
            breached = any(v >= threshold for v in history_series) if history_series else False
            projected_breach = any(v >= threshold for v in forecast_series) if forecast_series else False

            # Determine direction label
            slope = slopes_map.get(key, 0.0)
            trend_direction = "Stable →"
            if slope > 0.001:
                trend_direction = "Rising ↑"
            elif slope < -0.001:
                trend_direction = "Falling ↓"

            charts_data[key] = {
                "label": conf["label"],
                "metric_key": key,
                "current_value": round(current_val, 2) if isinstance(current_val, float) else current_val,
                "threshold": threshold,
                "unit": conf["unit"],
                "history": history_series,
                "forecast": forecast_series,
                "breached": breached,
                "projected_breach": projected_breach,
                "trend_direction": trend_direction
            }

        steps = AIOpsDashboardService._get_remediation_steps(failure_mode)

        # Primary metric indicator mapping
        critical_metric = "composite_anomaly_score"
        metric_val = "0.50"
        if fc_row is not None:
            critical_metric = str(fc_row.get("earliest_ttf_feature", "composite_anomaly_score"))
            metric_val = str(charts_data.get(critical_metric, {}).get("current_value", "0.50"))

        return {
            "episode_id": ep_id,
            "incident_id": f"INC-{hash(ep_id) % 10000:04d}",
            "failure_mode": failure_mode,
            "label": failure_mode.replace("_", " ").title(),
            "severity": revised_sev,
            "preliminary_severity": prelim_sev,
            "status": clean_status(pip_row.get("status", "OPEN")),
            "detected_time": str(pip_row.get("timestamp", "--:--:--")),
            "elapsed_s": float(pip_row.get("elapsed_s", 0.0)),
            "cycle": int(pip_row["cycle"]),
            
            # Diagnosis
            "root_cause": su_row.get("reason", pip_row.get("severity_reason", "No detailed signature diagnosed yet.")) if su_row is not None else str(pip_row.get("severity_reason", "Evaluating telemetry...")),
            "classifier_confidence": float(pip_row.get("prediction_probability", 0.5)),
            "warning_count": int(pip_row.get("severity_warning_count", 0) if "severity_warning_count" in pip_row else 0),
            "critical_count": int(pip_row.get("severity_critical_count", 0) if "severity_critical_count" in pip_row else 0),
            "blast_size": int(pip_row.get("severity_blast_size", 1) if "severity_blast_size" in pip_row else 1),
            
            # Prediction
            "predicted_failure": str(pip_row.get("predicted_failure", "None")),
            "ttf_seconds": float(fc_row.get("time_to_failure", 0.0)) if (fc_row is not None and not pd.isna(fc_row.get("time_to_failure"))) else None,
            "prediction_confidence": float(fc_row.get("forecast_confidence", 0.0)) if fc_row is not None else 0.0,
            "recommended_action": su_row.get("reason", "Awaiting stable forecast values.") if su_row is not None else "Monitor systems closely.",
            
            # Evidence
            "critical_metric": critical_metric,
            "metric_value": metric_val,
            "trace_evidence": "Available" if (pip_row.get("severity_blast_size", 0) > 2) else "Not Available",
            "log_evidence": "Available" if (pip_row.get("severity_warning_count", 0) > 0) else "Not Available",
            
            # Historical 20-cycles + forecasts 7 metrics
            "charts": charts_data,
            
            # Reliability metrics
            "mtbf_hours": 96 if "LEAK" in failure_mode else 120,
            "mttr_hours": 1.4 if "LEAK" in failure_mode else 0.8,
            "weibull_confidence": 0.81 if "LEAK" in failure_mode else 0.74,
            "reliability_trend": "Degrading" if revised_sev in ["P1", "P2"] else "Stable",
            "steps": steps
        }

    @staticmethod
    def _get_remediation_steps(mode: str) -> List[str]:
        steps_map = {
            "MEMORY_LEAK": ["Restart Service", "Capture Heap Dump", "Review Recent Deploys", "Check for Retained Objects", "Increase JVM Heap"],
            "CPU_SATURATION": ["Scale Out Instances", "Profile Hot Code Paths", "Check Background Jobs", "Verify Autoscaler Limits"],
            "LATENCY_SPIKE": ["Compare P50 vs P99", "Check GC Pauses", "Review Payload Sizes", "Inspect Slow Traces"],
            "ERROR_STORM": ["Roll Back Recent Deploy", "Inspect Error Logs", "Verify Downstream Health", "Notify Incident Commander"],
            "DB_SLOWDOWN": ["Inspect Query Logs", "Check Lock Contention", "Verify Missing Indexes", "Monitor Pool Usage"],
            "QUEUE_BACKUP": ["Scale Consumer Capacity", "Check Stalled Consumer", "Apply Backpressure", "Monitor Drain Rate"],
            "DEPENDENCY_TIMEOUT": ["Check Dependency Health", "Verify Circuit Breaker Config", "Engage Owning Team"],
            "BAD_DEPLOY": ["Roll Back Deployment", "Review Release Diff", "Check Schema Compatibility", "Add Canary Gate"],
            "RETRY_STORM": ["Apply Backoff + Jitter", "Tighten Circuit Breaker", "Throttle Retry Source"],
            "DISK_IO_SATURATION": ["Check Runaway Write Job", "Review IOPS Provisioning", "Monitor Disk Queue Depth"],
            "CASCADING_FAILURE": ["Engage Incident Command", "Isolate Origin Service", "Apply Circuit Breakers", "Track Blast Radius"],
        }
        return steps_map.get(mode, ["Manual Log Review", "Cross-Check Dashboards", "Monitor Recovery"])


# =============================================================================
# Human Gate Service
# =============================================================================

class HumanGateService:
    """
    Service layer for the Human Gate API endpoints.

    Wraps InterruptManager and AuditLogger so api/main.py
    stays thin (no business logic in the route handlers).

    All methods return plain dicts — ready for JSON serialisation.
    """

    @staticmethod
    def _get_manager():
        """Lazy-import InterruptManager to avoid circular imports at startup."""
        from nodes.human_gate.interrupt_manager import get_interrupt_manager
        return get_interrupt_manager()

    @staticmethod
    def _get_logger():
        """Lazy-import AuditLogger."""
        from nodes.human_gate.audit_logger import AuditLogger
        return AuditLogger()

    # ------------------------------------------------------------------
    # Pending reviews (for HumanGatePanel polling)
    # ------------------------------------------------------------------

    @staticmethod
    def get_pending_reviews() -> list[dict]:
        """
        Return all reviews currently in WAITING or REVIEWING state.
        Called by GET /api/human-gate/pending every 2 s from the frontend.
        """
        try:
            return HumanGateService._get_manager().get_pending()
        except Exception as e:
            print(f"[HumanGateService] get_pending_reviews error: {e}")
            return []

    @staticmethod
    def get_review(review_id: str) -> dict | None:
        """
        Return one review by ID and mark it as REVIEWING (operator opened it).
        Called by GET /api/human-gate/review/{review_id}.
        """
        try:
            manager = HumanGateService._get_manager()
            manager.mark_reviewing(review_id)   # WAITING → REVIEWING
            return manager.get_review(review_id)
        except Exception as e:
            print(f"[HumanGateService] get_review error: {e}")
            return None

    # ------------------------------------------------------------------
    # Decision submission (operator clicks Approve / Reject)
    # ------------------------------------------------------------------

    @staticmethod
    def submit_decision(
        review_id: str,
        decision:  str,
        operator:  str,
        reason:    str = "",
    ) -> dict:
        """
        Record an operator decision.  The pipeline's poll_for_decision()
        will pick up the change within its next 0.1-second polling cycle.

        Called by POST /api/human-gate/decision/{review_id}.

        Args:
            review_id: UUID of the review.
            decision:  "APPROVED" or "REJECTED".
            operator:  Operator username.
            reason:    Optional rejection reason.

        Returns:
            Updated review dict or error dict.
        """
        try:
            return HumanGateService._get_manager().submit_decision(
                review_id = review_id,
                decision  = decision,
                operator  = operator,
                reason    = reason,
            )
        except Exception as e:
            print(f"[HumanGateService] submit_decision error: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Metrics & history (for audit / dashboard KPIs)
    # ------------------------------------------------------------------

    @staticmethod
    def get_metrics() -> dict:
        """
        Return Human Gate KPI metrics.
        Called by GET /api/human-gate/metrics.
        """
        try:
            return HumanGateService._get_logger().get_metrics()
        except Exception as e:
            print(f"[HumanGateService] get_metrics error: {e}")
            return {}

    @staticmethod
    def get_history(limit: int = 50) -> list[dict]:
        """
        Return recent Human Gate audit records.
        Called by GET /api/human-gate/history.
        """
        try:
            return HumanGateService._get_logger().get_history(limit=limit)
        except Exception as e:
            print(f"[HumanGateService] get_history error: {e}")
            return []

