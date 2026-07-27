"""
app_simulator/storage/db_writer.py
====================================
Atomic 3-table SQLite writer.

write_tick() inserts metrics + log + spans in a single transaction.
All three rows succeed or all three roll back — no partial ticks.

WAL mode allows concurrent readers (FE queries) without blocking the writer.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path


class DbWriter:
    """Thread-safe SQLite writer. Call setup() once, then write_tick() per tick."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def setup(self) -> None:
        """Create/open DB, apply WAL mode, execute schema DDL."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")    # concurrent reads OK
        self._conn.execute("PRAGMA synchronous=NORMAL")  # faster, still crash-safe
        self._conn.execute("PRAGMA foreign_keys=ON")
        schema_path = Path(__file__).parent / "schema.sql"
        self._conn.executescript(schema_path.read_text(encoding="utf-8"))
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        """Expose connection for FE feature queries (read-only use)."""
        if self._conn is None:
            raise RuntimeError("DbWriter.setup() has not been called")
        return self._conn

    def write_tick(self, metric: dict, log: dict, spans: list[dict]) -> None:
        """
        Atomic 3-table INSERT.

        metrics uses INSERT OR IGNORE because of UNIQUE(episode_id, timestamp).
        logs and traces use plain INSERT.
        All three run inside one transaction — context manager auto-commits or rolls back.
        """
        conn = self._conn
        with conn:                               # BEGIN / COMMIT / ROLLBACK
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
        """
        Insert one row into the severity table from a SeverityResult dict.

        Expected keys (subset of SeverityResult.to_dict()):
            episode_id, timestamp, elapsed_s, failure_mode,
            Severity (or severity), RawSeverity (or raw_severity),
            WeightedScore (or weighted_score), CriticalCount, WarningCount,
            BlastSize, HighRiskMode, BlastRadiusGrowing, Reason, RecommendedAction
        """
        # Normalise: accept both snake_case and PascalCase keys from SeverityResult
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

        # Keep only columns that exist in the severity table
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

    def close(self) -> None:
        """Flush WAL and close connection."""
        if self._conn:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
            self._conn = None

