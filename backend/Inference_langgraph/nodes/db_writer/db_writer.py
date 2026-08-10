"""
backend/Inference_langgraph/nodes/db_writer/db_writer.py
=========================================================
Atomic 3-table SQLite writer for raw telemetry + all pipeline node outputs
and combined pipeline_results table.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_SCHEMA_PATH = _HERE.parent.parent.parent / "Simulator" / "app_data_generator_for_offline" / "storage" / "schema.sql"


class DbWriter:
    """Thread-safe SQLite writer. Call setup() once, then write methods per tick."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def setup(self, schema_path: Path | None = None) -> None:
        """Create/open DB, apply WAL mode, execute schema DDL."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")    # concurrent reads OK
        self._conn.execute("PRAGMA synchronous=NORMAL")  # faster, still crash-safe
        self._conn.execute("PRAGMA foreign_keys=ON")
        
        target_schema = schema_path if schema_path is not None else DEFAULT_SCHEMA_PATH
        if not target_schema.exists():
            # Fallback relative lookup
            target_schema = Path(__file__).resolve().parent.parent.parent.parent / "Simulator" / "app_data_generator_for_offline" / "storage" / "schema.sql"
            
        self._conn.executescript(target_schema.read_text(encoding="utf-8"))
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        """Expose connection for FE feature queries (read-only use)."""
        if self._conn is None:
            raise RuntimeError("DbWriter.setup() has not been called")
        return self._conn

    # =========================================================================
    # RAW TELEMETRY WRITES (written by run_simulator.py)
    # =========================================================================

    def write_tick(self, metric: dict, log: dict, spans: list[dict]) -> None:
        """
        Atomic 3-table INSERT.
        metrics uses INSERT OR IGNORE because of UNIQUE(episode_id, timestamp).
        logs and traces use plain INSERT.
        """
        conn = self._conn
        with conn:
            # metrics
            m_cols = list(metric.keys())
            conn.execute(
                f"INSERT OR IGNORE INTO metrics ({','.join(m_cols)}) "
                f"VALUES ({','.join('?' * len(m_cols))})",
                list(metric.values()),
            )
            # log
            l_cols = list(log.keys())
            conn.execute(
                f"INSERT INTO logs ({','.join(l_cols)}) "
                f"VALUES ({','.join('?' * len(l_cols))})",
                list(log.values()),
            )
            # spans
            for span in spans:
                s_cols = list(span.keys())
                conn.execute(
                    f"INSERT INTO traces ({','.join(s_cols)}) "
                    f"VALUES ({','.join('?' * len(s_cols))})",
                    list(span.values()),
                )

    def write_severity(self, result: dict) -> None:
        """Insert one row into the severity table from a SeverityResult dict."""
        _alias = {
            "severity":            "Severity",
            "raw_severity":        "RawSeverity",
            "weighted_score":      "WeightedScore",
            "critical_count":      "CriticalCount",
            "warning_count":       "WarningCount",
            "blast_size":          "BlastSize",
            "high_risk_mode":      "HighRiskMode",
            "blast_radius_growing":"BlastRadiusGrowing",
            "reason":              "Reason",
            "recommended_action":  "RecommendedAction",
        }
        row = {}
        for k, v in result.items():
            normalised = _alias.get(k, k)
            row[normalised] = v

        _SEVERITY_COLS = {
            "episode_id", "timestamp", "elapsed_s", "failure_mode",
            "Severity", "RawSeverity", "WeightedScore",
            "CriticalCount", "WarningCount", "BlastSize",
            "HighRiskMode", "BlastRadiusGrowing", "Reason", "RecommendedAction",
        }
        row = {k: v for k, v in row.items() if k in _SEVERITY_COLS}

        cols = list(row.keys())
        vals = list(row.values())
        with self._conn:
            self._conn.execute(
                f"INSERT OR IGNORE INTO severity ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                vals,
            )

    # =========================================================================
    # NODE OUTPUT WRITES — Option 2 (dedicated per-node tables)
    # =========================================================================

    def _insert(self, table: str, row: dict) -> None:
        """Generic helper: INSERT OR IGNORE into any table."""
        cols = list(row.keys())
        vals = list(row.values())
        with self._conn:
            self._conn.execute(
                f"INSERT INTO {table} ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                vals,
            )

    def write_feature_engineering(self, row: dict) -> None:
        """Persist one cycle's Feature Engineering output to node_feature_engineering."""
        _COLS = {
            "cycle", "episode_id", "failure_mode", "timestamp", "elapsed_s",
            "cpu_utilization", "memory_utilization", "heap_mb", "db_p99",
            "disk_read_latency", "disk_write_latency", "error_rate", "gc_pause_p99",
            "cache_hit_rate", "cache_miss_rate", "active_connections", "network_errors",
            "p50_latency", "p95_latency", "p99_latency", "queue_lag",
            "retry_count_per_request", "rps", "upstream_timeout_rate",
            "circuit_breaker_state", "http_4xx_rate", "http_5xx_rate",
            "iops_utilization", "thread_pool_queue", "cpu_saturation",
            "db_connection_pool", "db_connection_wait",
            "log_count", "log_max_severity", "log_critical_count",
            "log_has_exception", "log_has_novel_template",
        }
        filtered = {k: v for k, v in row.items() if k in _COLS}
        self._insert("node_feature_engineering", filtered)

    def write_preliminary_severity(self, row: dict) -> None:
        """Persist one cycle's Preliminary Severity output to node_preliminary_severity."""
        _alias = {
            "Severity":           "preliminary_severity",
            "RawSeverity":        "severity_raw",
            "WeightedScore":      "weighted_score",
            "CriticalCount":      "critical_count",
            "WarningCount":       "warning_count",
            "BlastSize":          "blast_size",
            "HighRiskMode":       "high_risk_mode",
            "BlastRadiusGrowing": "blast_radius_growing",
            "Reason":             "reason",
            "RecommendedAction":  "recommended_action",
        }
        normalised_row = {}
        for k, v in row.items():
            norm_key = _alias.get(k, k)
            normalised_row[norm_key] = v

        _COLS = {
            "cycle", "episode_id", "failure_mode", "timestamp", "elapsed_s",
            "preliminary_severity", "severity_raw", "weighted_score",
            "critical_count", "warning_count", "blast_size",
            "high_risk_mode", "blast_radius_growing", "reason", "recommended_action",
        }
        filtered = {k: v for k, v in normalised_row.items() if k in _COLS}
        self._insert("node_preliminary_severity", filtered)

    def write_classification(self, row: dict) -> None:
        """Persist one cycle's Classification output to node_classification."""
        _COLS = {
            "cycle", "episode_id", "failure_mode", "timestamp", "elapsed_s",
            "predicted_failure", "prediction_probability",
        }
        filtered = {k: v for k, v in row.items() if k in _COLS}
        self._insert("node_classification", filtered)

    def write_tumbling_window(self, row: dict) -> None:
        """Persist one cycle's Tumbling Window output to node_tumbling_window."""
        _COLS = {
            "cycle", "episode_id", "failure_mode", "timestamp", "elapsed_s",
            "dominant_state", "vote_distribution", "window_margin",
            "window_full", "window_size",
        }
        filtered = {k: v for k, v in row.items() if k in _COLS}
        if isinstance(filtered.get("vote_distribution"), dict):
            filtered["vote_distribution"] = json.dumps(filtered["vote_distribution"])
        if isinstance(filtered.get("window_full"), bool):
            filtered["window_full"] = int(filtered["window_full"])
        self._insert("node_tumbling_window", filtered)

    def write_forecasting(self, row: dict) -> None:
        """Persist one cycle's Forecasting + Convergence output to node_forecasting."""
        _COLS = {
            "cycle", "episode_id", "failure_mode", "timestamp", "elapsed_s",
            "algorithm_used", "history_steps", "forecast_horizon_s",
            "time_to_failure", "earliest_ttf_feature", "forecast_confidence",
            "confidence_reason", "threshold_crossed",
            "feature_ttfs", "feature_slopes", "predictions", "current_values",
        }
        filtered = {k: v for k, v in row.items() if k in _COLS}
        for key in ("feature_ttfs", "feature_slopes", "predictions", "current_values"):
            if key in filtered and isinstance(filtered[key], (dict, list)):
                filtered[key] = json.dumps(filtered[key])
        if isinstance(filtered.get("threshold_crossed"), bool):
            filtered["threshold_crossed"] = int(filtered["threshold_crossed"])
        self._insert("node_forecasting", filtered)

    def write_severity_update(self, row: dict) -> None:
        """Persist one cycle's Severity Update output to node_severity_update."""
        _COLS = {
            "cycle", "episode_id", "failure_mode", "timestamp", "elapsed_s",
            "preliminary_severity", "forecast_confidence", "time_to_failure",
            "earliest_ttf_feature", "impact_band", "urgency_band", "gate_passed",
            "candidate_severity", "revised_severity", "is_escalated",
            "is_deescalated", "dwell_count", "reason",
        }
        filtered = {k: v for k, v in row.items() if k in _COLS}
        for key in ("gate_passed", "is_escalated", "is_deescalated"):
            if key in filtered and isinstance(filtered[key], bool):
                filtered[key] = int(filtered[key])
        self._insert("node_severity_update", filtered)

    def write_reliability(self, row: dict) -> None:
        """Persist one episode's Reliability & Weibull output to node_reliability."""
        _COLS = {
            "cycle", "episode_id", "failure_mode", "group_name", "ttf_seconds",
            "mttf_seconds", "survival_probability", "hazard_rate", "event",
            "data_source", "beta", "eta", "recorded_at",
        }
        filtered = {k: v for k, v in row.items() if k in _COLS}
        if isinstance(filtered.get("event"), bool):
            filtered["event"] = int(filtered["event"])
        self._insert("node_reliability", filtered)

    def write_human_gate(self, row: dict) -> None:
        """Persist one Human Gate decision to node_human_gate."""
        _COLS = {
            "review_id", "incident_id", "episode_id", "failure_mode",
            "failure_label", "old_severity", "new_severity", "final_severity",
            "decision", "operator", "reason", "confidence", "ttf_seconds",
            "impact_band", "urgency_band", "is_large_jump", "escalation_summary",
            "response_ms", "timeout_seconds", "created_at", "decided_at", "recorded_at",
        }
        filtered = {k: v for k, v in row.items() if k in _COLS}
        if isinstance(filtered.get("is_large_jump"), bool):
            filtered["is_large_jump"] = int(filtered["is_large_jump"])
        cols = list(filtered.keys())
        vals = list(filtered.values())
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO node_human_gate ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                vals,
            )

    # =========================================================================
    # COMBINED PIPELINE RESULTS WRITE
    # =========================================================================

    def write_pipeline_results(self, row: dict) -> None:
        """Persist a full combined pipeline cycle snapshot to pipeline_results."""
        _COLS = {
            "cycle", "episode_id", "failure_mode", "timestamp", "elapsed_s",
            "fe_cpu_utilization", "fe_memory_utilization", "fe_heap_mb",
            "fe_error_rate", "fe_p99_latency", "fe_p95_latency", "fe_db_p99",
            "fe_queue_lag", "fe_log_count", "fe_log_critical_count",
            "fe_log_has_exception", "fe_log_has_novel_template",
            "preliminary_severity", "severity_weighted_score",
            "severity_critical_count", "severity_warning_count",
            "severity_blast_size", "severity_reason",
            "predicted_failure", "prediction_probability",
            "dominant_state", "vote_distribution", "window_margin",
            "window_full", "window_size",
            "forecast_algorithm", "time_to_failure", "forecast_confidence",
            "threshold_crossed", "earliest_ttf_feature",
            "revised_severity", "candidate_severity", "impact_band",
            "urgency_band", "gate_passed", "is_escalated", "is_deescalated",
            "su_reason",
            "hg_review_id", "hg_decision", "hg_final_severity",
            "hg_operator", "hg_response_ms",
        }
        filtered = {k: v for k, v in row.items() if k in _COLS}
        if isinstance(filtered.get("vote_distribution"), dict):
            filtered["vote_distribution"] = json.dumps(filtered["vote_distribution"])
        for key in ("window_full", "threshold_crossed", "gate_passed",
                    "is_escalated", "is_deescalated",
                    "fe_log_has_exception", "fe_log_has_novel_template"):
            if key in filtered and isinstance(filtered[key], bool):
                filtered[key] = int(filtered[key])
        self._insert("pipeline_results", filtered)

    def close(self) -> None:
        """Flush WAL and close connection."""
        if self._conn:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
            self._conn = None
