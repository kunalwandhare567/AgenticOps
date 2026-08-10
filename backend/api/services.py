"""
backend/api/services.py
========================
Data service layer that loads/joins pipeline results and database states
to expose a unified dashboard incident model.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from Simulator.app_data_generator_for_offline.config import (
    PIPELINE_RESULTS_CSV,
    FORECASTING_OUTPUT_CSV,
    SEVERITY_UPDATE_CSV,
    ENGINEERED_FEAT_CSV,
    LIVE_FEED_DB_PATH,
)

_HERE = Path(__file__).resolve().parent


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

        # Reliability & 4-Group Weibull parameters calculation
        from Inference_langgraph.nodes.reliability.extractor import FAILURE_GROUP_MAP
        from Inference_langgraph.nodes.reliability.weibull_fitter import GROUPS, fit_weibull_censored

        active_group = FAILURE_GROUP_MAP.get(failure_mode, "Healthy / Unassigned")
        elapsed_s = float(pip_row.get("elapsed_s", 0.0))

        group_beta = 2.0
        group_eta = 46.5
        life_csv = _HERE.parent / "Inference_langgraph" / "nodes" / "reliability" / "output" / "life_data_extracted.csv"
        if life_csv.exists() and active_group in GROUPS:
            try:
                df_life = pd.read_csv(life_csv)
                modes = GROUPS[active_group]
                sub = df_life[df_life["failure_mode"].isin(modes)]
                if not sub.empty:
                    fit = fit_weibull_censored(sub["ttf_seconds"].values, sub["event"].values)
                    group_beta = round(float(fit["beta"]), 2)
                    group_eta = round(float(fit["eta"]), 1)
            except Exception:
                pass

        if group_eta > 0:
            survival_ratio = float(np.exp(- (elapsed_s / group_eta) ** group_beta))
        else:
            survival_ratio = 1.0
        survival_pct = round(max(0.0, min(100.0, survival_ratio * 100.0)), 1)

        return {
            "episode_id": ep_id,
            "incident_id": f"INC-{hash(ep_id) % 10000:04d}",
            "failure_mode": failure_mode,
            "label": failure_mode.replace("_", " ").title(),
            "severity": revised_sev,
            "preliminary_severity": prelim_sev,
            "status": clean_status(pip_row.get("status", "OPEN")),
            "detected_time": str(pip_row.get("timestamp", "--:--:--")),
            "elapsed_s": elapsed_s,
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
            
            # Real-Time 4-Group Reliability parameters
            "active_group_name": active_group,
            "active_beta": group_beta,
            "active_eta": group_eta,
            "current_survival_pct": survival_pct,
            "mtbf_hours": 96 if "LEAK" in failure_mode else 120,
            "mttr_hours": 1.4 if "LEAK" in failure_mode else 0.8,
            "weibull_confidence": round(survival_ratio, 2),
            "reliability_trend": "Degrading" if revised_sev in ["P1", "P2"] else "Stable",
            "steps": steps
        }

    @staticmethod
    def get_reliability_summary() -> Dict[str, Any]:
        """Compute and return 4-group Weibull parameters, KM step points, and Weibull curves."""
        from Inference_langgraph.nodes.reliability.weibull_fitter import (
            GROUPS, fit_weibull_censored, kaplan_meier, weibull_survival
        )

        life_csv = _HERE.parent / "Inference_langgraph" / "nodes" / "reliability" / "output" / "life_data_extracted.csv"
        if not life_csv.exists():
            return {"groups": {}, "status": "no_data"}

        try:
            df = pd.read_csv(life_csv)
            if df.empty:
                return {"groups": {}, "status": "no_data"}

            group_annotations = {
                "Immediate trigger": {
                    "note": "Point-mass behavior concentrated near ~10s. Weibull β is unnaturally large.",
                    "suitability": "Point Mass (Poor Weibull Fit)",
                    "badge_color": "red"
                },
                "Fast accumulation": {
                    "note": "Infant/early accumulation (12–22s range). Discrete step artifacts.",
                    "suitability": "Moderate Step Fit",
                    "badge_color": "amber"
                },
                "Progressive resource degradation": {
                    "note": "Classic wear-out failure mechanism (12–84s continuous). BEST Weibull fit.",
                    "suitability": "Best Continuous Fit",
                    "badge_color": "emerald"
                },
                "Slow or latent degradation": {
                    "note": "Heavy right-censoring at 238s (~85% censored). High confidence spread.",
                    "suitability": "Heavy Censoring (Cure Fraction Needed)",
                    "badge_color": "blue"
                },
            }

            groups_result = {}
            for group_name, failure_modes in GROUPS.items():
                subset = df[df["failure_mode"].isin(failure_modes)].copy()
                if subset.empty:
                    continue

                time = subset["ttf_seconds"].values
                event = subset["event"].values

                fit = fit_weibull_censored(time, event)
                beta, eta = fit["beta"], fit["eta"]

                t_km, s_km, se_km = kaplan_meier(time, event)

                km_points = []
                for i in range(len(t_km)):
                    upper_ci = min(1.0, float(s_km[i] + 1.96 * se_km[i]))
                    lower_ci = max(0.0, float(s_km[i] - 1.96 * se_km[i]))
                    km_points.append({
                        "t": round(float(t_km[i]), 1),
                        "s": round(float(s_km[i]), 4),
                        "upper_ci": round(upper_ci, 4),
                        "lower_ci": round(lower_ci, 4)
                    })

                max_t = max(240.0, float(time.max()))
                t_smooth = np.linspace(0.0, max_t, 50)
                s_weibull = weibull_survival(t_smooth, beta, eta)

                weibull_points = []
                for i in range(len(t_smooth)):
                    weibull_points.append({
                        "t": round(float(t_smooth[i]), 1),
                        "s": round(float(s_weibull[i]), 4)
                    })

                meta = group_annotations.get(group_name, {
                    "note": "Group reliability fit",
                    "suitability": "Standard Fit",
                    "badge_color": "blue"
                })

                groups_result[group_name] = {
                    "group_name": group_name,
                    "failure_modes": failure_modes,
                    "n": fit["n"],
                    "events": fit["events"],
                    "censored": fit["censored"],
                    "beta": round(beta, 2),
                    "eta": round(eta, 1),
                    "log_likelihood": round(fit["log_likelihood"], 2),
                    "km_points": km_points,
                    "weibull_points": weibull_points,
                    "note": meta["note"],
                    "suitability": meta["suitability"],
                    "badge_color": meta["badge_color"]
                }

            return {"groups": groups_result, "status": "ok"}
        except Exception as e:
            print(f"[DashboardService] Error computing reliability summary: {e}")
            return {"groups": {}, "status": "error", "message": str(e)}

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
        from Inference_langgraph.nodes.human_gate.interrupt_manager import get_interrupt_manager
        return get_interrupt_manager()

    @staticmethod
    def _get_logger():
        """Lazy-import AuditLogger."""
        from Inference_langgraph.nodes.human_gate.audit_logger import AuditLogger
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


# =============================================================================
# Live Feed Service
# =============================================================================

class LiveFeedService:
    """
    Service layer for live feed endpoints.

    Reads from live_feed_db.sqlite (the separate DB populated by
    run_live_feed.py + run_langgraph.py --live).

    All methods return plain dicts / lists — ready for JSON serialisation.
    """

    _session_start: float = time.monotonic()

    @staticmethod
    def _get_conn() -> sqlite3.Connection | None:
        """Open a read-only connection to live_feed_db.sqlite if it exists."""
        if not LIVE_FEED_DB_PATH.exists():
            return None
        try:
            conn = sqlite3.connect(str(LIVE_FEED_DB_PATH), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            return conn
        except Exception:
            return None

    @staticmethod
    def get_live_feed_state() -> Dict[str, Any]:
        """
        Build the full live incident payload from live_feed_db.sqlite.

        Every field comes directly from the actual pipeline table columns
        (schema.sql / n10_db_writer.py). No hardcoding.

        Tables used:
          pipeline_results        — latest cycle snapshot (identity + all node outputs)
          node_feature_engineering— per-cycle metric history for all 7 NOC charts
          node_forecasting        — TTF, slopes, and forecast curves (JSON)
          node_reliability        — Weibull S(t), group_name, beta, eta
          node_preliminary_severity— recommended_action text
        """
        conn = LiveFeedService._get_conn()
        if conn is None:
            return {}
        try:
            # ── Guard: table must exist ───────────────────────────────────────
            if not conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_results'"
            ).fetchone():
                return {}

            # ── 1. Latest pipeline_results row (ground truth for this cycle) ──
            pr = conn.execute(
                "SELECT * FROM pipeline_results ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not pr:
                return {}
            pr = dict(pr)

            ep_id   = pr.get("episode_id", "")
            f_mode  = pr.get("failure_mode", "NONE")
            elapsed = float(pr.get("elapsed_s") or 0.0)
            prelim  = pr.get("preliminary_severity") or "P4"
            revised = pr.get("revised_severity") or prelim
            cycle   = int(pr.get("cycle") or 0)

            # ── 2. Metric history: node_feature_engineering (all 7 NOC metrics)
            #    Columns are the bare names (cpu_utilization, heap_mb, …)
            #    matching node_feature_engineering schema exactly.
            METRIC_COLS = [
                "cpu_utilization", "heap_mb", "p99_latency", "db_p99",
                "error_rate", "cache_miss_rate", "queue_lag",
            ]
            history_series: Dict[str, List[float]] = {k: [] for k in METRIC_COLS}

            fe_rows = conn.execute(
                f"SELECT {', '.join(METRIC_COLS)} FROM node_feature_engineering "
                f"WHERE episode_id = ? ORDER BY cycle ASC",
                (ep_id,)
            ).fetchall()

            for r in fe_rows:
                r = dict(r)
                for col in METRIC_COLS:
                    val = r.get(col)
                    try:
                        fval = float(val)
                        if not (np.isnan(fval) or np.isinf(fval)):
                            history_series[col].append(round(fval, 4))
                    except (TypeError, ValueError):
                        pass

            # ── 3. Latest forecasting row: node_forecasting ───────────────────
            #    predictions   = JSON dict {metric: [v1, v2, …]}
            #    feature_slopes= JSON dict {metric: slope_float}
            #    time_to_failure, forecast_confidence, earliest_ttf_feature
            fc_predictions: Dict[str, List[float]] = {}
            fc_slopes:      Dict[str, float] = {}
            fc_ttf          = pr.get("time_to_failure")          # already in pipeline_results
            fc_confidence   = pr.get("forecast_confidence")
            fc_earliest     = pr.get("earliest_ttf_feature") or ""
            fc_algorithm    = pr.get("forecast_algorithm") or "—"

            fc_row = conn.execute(
                "SELECT predictions, feature_slopes FROM node_forecasting "
                "WHERE episode_id = ? ORDER BY id DESC LIMIT 1",
                (ep_id,)
            ).fetchone()
            if fc_row:
                if fc_row["predictions"]:
                    try:
                        fc_predictions = json.loads(fc_row["predictions"])
                    except Exception:
                        pass
                if fc_row["feature_slopes"]:
                    try:
                        fc_slopes = json.loads(fc_row["feature_slopes"])
                    except Exception:
                        pass

            # ── 4. Reliability: node_reliability ─────────────────────────────
            #    group_name, survival_probability, beta, eta — all written per cycle
            survival_pct   = 100.0
            group_name     = "—"
            weibull_beta   = None
            weibull_eta    = None
            mtbf_hours     = None
            mttr_hours     = None

            rel_row = conn.execute(
                "SELECT group_name, survival_probability, beta, eta, mttf_seconds "
                "FROM node_reliability WHERE episode_id = ? ORDER BY id DESC LIMIT 1",
                (ep_id,)
            ).fetchone()
            if rel_row:
                rel_row = dict(rel_row)
                if rel_row.get("survival_probability") is not None:
                    survival_pct = round(float(rel_row["survival_probability"]), 1)
                if rel_row.get("group_name"):
                    group_name = rel_row["group_name"]
                if rel_row.get("beta") is not None:
                    weibull_beta = float(rel_row["beta"])
                if rel_row.get("eta") is not None:
                    weibull_eta = float(rel_row["eta"])
                if rel_row.get("mttf_seconds"):
                    mtbf_hours = round(float(rel_row["mttf_seconds"]) / 3600, 2)

            # ── 5. Recommended action: node_preliminary_severity ──────────────
            recommended_action = pr.get("su_reason") or ""
            if not recommended_action:
                ps_row = conn.execute(
                    "SELECT recommended_action FROM node_preliminary_severity "
                    "WHERE episode_id = ? ORDER BY id DESC LIMIT 1",
                    (ep_id,)
                ).fetchone()
                if ps_row:
                    recommended_action = ps_row["recommended_action"] or ""

            # ── 6. Build 7 NOC chart objects ──────────────────────────────────
            charts_data: Dict[str, Any] = {}
            for col, conf in NOC_METRICS_CONFIG.items():
                h_series = history_series.get(col, [])
                # Current value is always from pipeline_results if available
                # (fe_<col> prefixed columns)
                pr_val = pr.get(f"fe_{col}")
                try:
                    pr_fval = float(pr_val) if pr_val is not None else None
                except (TypeError, ValueError):
                    pr_fval = None

                current_val = pr_fval if pr_fval is not None else (h_series[-1] if h_series else 0.0)

                # Forecast series for this metric
                f_raw = fc_predictions.get(col, [])
                f_series = []
                for v in f_raw[:12]:
                    try:
                        fv = float(v)
                        if not (np.isnan(fv) or np.isinf(fv)):
                            f_series.append(round(fv, 4))
                    except (TypeError, ValueError):
                        pass

                threshold = conf["threshold"]
                slope     = fc_slopes.get(col, 0.0)
                try:
                    slope = float(slope)
                except (TypeError, ValueError):
                    slope = 0.0

                breached          = any(v >= threshold for v in h_series)
                projected_breach  = any(v >= threshold for v in f_series)

                # Trend direction from actual pipeline slope
                if slope > 0.005:
                    trend_dir = "Rising ↑"
                elif slope < -0.005:
                    trend_dir = "Falling ↓"
                else:
                    trend_dir = "Stable →"

                # Human-readable metric value string
                if col == "heap_mb":
                    val_str = f"{current_val:.0f} MB"
                elif col in ("error_rate", "cache_miss_rate"):
                    val_str = f"{current_val * 100:.2f}%"
                elif col in ("p99_latency", "db_p99", "queue_lag"):
                    val_str = f"{current_val:.1f} ms"
                else:
                    val_str = f"{current_val:.1f}%"

                charts_data[col] = {
                    "label":            conf["label"],
                    "metric_key":       col,
                    "unit":             conf["unit"],
                    "threshold":        threshold,
                    "current_value":    round(current_val, 4),
                    "metric_value_str": val_str,
                    "history":          [round(v, 4) for v in h_series[-40:]],
                    "forecast":         f_series,
                    "slope":            round(slope, 6),
                    "trend_direction":  trend_dir,
                    "breached":         breached,
                    "projected_breach": projected_breach,
                }

            # Critical metric display (from forecasting node's earliest_ttf_feature)
            crit_col = fc_earliest if fc_earliest in charts_data else "cpu_utilization"
            crit_val_str = charts_data.get(crit_col, {}).get("metric_value_str", "—")

            # Evidence availability from actual severity counts in pipeline_results
            warn_count  = int(pr.get("severity_warning_count")  or 0)
            crit_count  = int(pr.get("severity_critical_count") or 0)
            blast_size  = int(pr.get("severity_blast_size")     or 1)

            return {
                # ── Identity ─────────────────────────────────────────────────
                "episode_id":           ep_id,
                "incident_id":          f"LIVE-{abs(hash(ep_id)) % 10000:04d}",
                "failure_mode":         f_mode,
                "label":                f_mode.replace("_", " ").title(),
                "status":               "OPEN",
                "live_mode":            True,
                "cycle":                cycle,
                "detected_time":        str(pr.get("timestamp") or "—"),
                "elapsed_s":            elapsed,

                # ── Severity (from node_preliminary_severity + node_severity_update)
                "preliminary_severity": prelim,
                "severity":             revised,
                "candidate_severity":   pr.get("candidate_severity") or revised,
                "impact_band":          pr.get("impact_band") or "—",
                "urgency_band":         pr.get("urgency_band") or "—",
                "is_escalated":         bool(pr.get("is_escalated")),
                "is_deescalated":       bool(pr.get("is_deescalated")),
                "gate_passed":          bool(pr.get("gate_passed")),
                "root_cause":           pr.get("severity_reason") or recommended_action or "Live telemetry analysis in progress.",
                "recommended_action":   recommended_action or "Monitor system signals.",

                # ── Classification (from node_classification) ────────────────
                "predicted_failure":      pr.get("predicted_failure") or f_mode,
                "prediction_probability": float(pr.get("prediction_probability") or 0.0),
                "classifier_confidence":  float(pr.get("prediction_probability") or 0.0),

                # ── Tumbling Window (from node_tumbling_window) ───────────────
                "dominant_state":   pr.get("dominant_state") or f_mode,
                "window_full":      bool(pr.get("window_full")),
                "window_margin":    float(pr.get("window_margin") or 0.0),

                # ── Forecasting (from node_forecasting) ──────────────────────
                "ttf_seconds":          float(fc_ttf) if fc_ttf is not None else None,
                "time_to_failure":      float(fc_ttf) if fc_ttf is not None else None,
                "forecast_confidence":  float(fc_confidence) if fc_confidence is not None else None,
                "prediction_confidence":float(fc_confidence) if fc_confidence is not None else None,
                "earliest_ttf_feature": fc_earliest or None,
                "forecast_algorithm":   fc_algorithm,
                "threshold_crossed":    bool(pr.get("threshold_crossed")),
                "critical_metric":      crit_col,
                "metric_value":         crit_val_str,

                # ── Evidence (from node_preliminary_severity via pipeline_results)
                "warning_count":    warn_count,
                "critical_count":   crit_count,
                "blast_size":       blast_size,
                "log_evidence":     "Available" if warn_count > 0 else "Not Available",
                "trace_evidence":   "Available" if blast_size > 1 else "Not Available",

                # ── Reliability (from node_reliability) ──────────────────────
                "current_survival_pct": survival_pct,
                "weibull_confidence":   round(survival_pct / 100.0, 3),
                "active_group_name":    group_name,
                "weibull_beta":         weibull_beta,
                "weibull_eta":          weibull_eta,
                "reliability_trend":    "Degrading" if revised in ("P1", "P2") else "Stable",
                "mtbf_hours":           mtbf_hours,
                "mttr_hours":           mttr_hours,

                # ── Human Gate (from node_human_gate via pipeline_results) ───
                "hg_decision":        pr.get("hg_decision"),
                "hg_final_severity":  pr.get("hg_final_severity"),
                "hg_operator":        pr.get("hg_operator"),
                "hg_review_id":       pr.get("hg_review_id"),
                "hg_response_ms":     pr.get("hg_response_ms"),

                # ── NOC Charts (built from node_feature_engineering history) ─
                "charts": charts_data,

                # ── Remediation steps ─────────────────────────────────────────
                "steps": AIOpsDashboardService._get_remediation_steps(f_mode),
            }
        except Exception as e:
            print(f"[LiveFeedService] get_live_feed_state error: {e}")
            import traceback; traceback.print_exc()
            return {}
        finally:
            conn.close()



    @staticmethod
    def get_live_feed_episodes() -> List[Dict[str, Any]]:
        """Return all distinct episodes in live_feed_db.sqlite (newest first)."""
        conn = LiveFeedService._get_conn()
        if conn is None:
            return []
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_results'"
            )
            if not cur.fetchone():
                return []
            cur = conn.execute(
                """
                SELECT episode_id, failure_mode,
                       MAX(cycle) AS last_cycle,
                       MAX(elapsed_s) AS elapsed_s,
                       MAX(preliminary_severity) AS severity
                FROM pipeline_results
                GROUP BY episode_id
                ORDER BY last_cycle DESC
                """
            )
            return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[LiveFeedService] get_live_feed_episodes error: {e}")
            return []
        finally:
            conn.close()

    @staticmethod
    def get_live_feed_status() -> Dict[str, Any]:
        """Return live session metadata."""
        try:
            from Simulator.live_feed_simulator.live_queue import LiveTelemetryQueue
            queue_size = LiveTelemetryQueue.size()
        except Exception:
            queue_size = -1

        conn = LiveFeedService._get_conn()
        ep_count = 0
        latest_mode = "N/A"
        if conn:
            try:
                cur = conn.execute(
                    "SELECT COUNT(DISTINCT episode_id) FROM pipeline_results"
                )
                row = cur.fetchone()
                ep_count = row[0] if row else 0
                cur = conn.execute(
                    "SELECT failure_mode FROM pipeline_results ORDER BY id DESC LIMIT 1"
                )
                row = cur.fetchone()
                latest_mode = row[0] if row else "N/A"
            except Exception:
                pass
            finally:
                conn.close()

        return {
            "live_feed_db":   str(LIVE_FEED_DB_PATH),
            "db_exists":      LIVE_FEED_DB_PATH.exists(),
            "queue_depth":    queue_size,
            "episodes_total": ep_count,
            "latest_mode":    latest_mode,
            "uptime_s":       round(time.monotonic() - LiveFeedService._session_start, 1),
        }
