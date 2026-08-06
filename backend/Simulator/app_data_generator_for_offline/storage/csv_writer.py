"""
app_simulator/storage/csv_writer.py
=====================================
Versioned append-mode CSV writer.

Each time the simulator starts, it scans the output directory for existing
metrics_N.csv files and creates the NEXT numbered version:
  Run 1 → metrics_1.csv, logs_1.csv, traces_1.csv
  Run 2 → metrics_2.csv, logs_2.csv, traces_2.csv
  Run 3 → metrics_3.csv, logs_3.csv, traces_3.csv

Headers written once per file. Subsequent calls always append.
Safe for long-running processes.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from ..config import METRIC_FIELDS, LOG_FIELDS, TRACE_FIELDS


class CsvWriter:
    """Versioned writer. Call setup() once, then write_tick() per tick."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._paths: dict[str, Path] = {}

    def setup(self) -> None:
        """
        Create output directory, find next version number, open versioned files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        version = self._next_version()
        self._paths = {
            "metrics": self.output_dir / f"metrics_{version}.csv",
            "logs":    self.output_dir / f"logs_{version}.csv",
            "traces":  self.output_dir / f"traces_{version}.csv",
        }
        # Write headers (files don't exist yet for this version)
        self._write_header("metrics", METRIC_FIELDS)
        self._write_header("logs",    LOG_FIELDS)
        self._write_header("traces",  TRACE_FIELDS)

        print(f"  [CSV] Writing to:")
        for name, path in self._paths.items():
            print(f"         {path.name}")

    def _next_version(self) -> int:
        """
        Scan output_dir for metrics_N.csv files and return max(N) + 1.
        Returns 1 if no versioned files exist.
        """
        pattern = re.compile(r"^metrics_(\d+)\.csv$")
        max_n = 0
        if self.output_dir.exists():
            for f in self.output_dir.iterdir():
                m = pattern.match(f.name)
                if m:
                    max_n = max(max_n, int(m.group(1)))
        return max_n + 1

    def _write_header(self, name: str, fields: list[str]) -> None:
        with self._paths[name].open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fields)

    def write_tick(self, metric: dict, log: dict, spans: list[dict]) -> None:
        """Append metric row, log row, and all span rows to their versioned CSVs."""
        self._append("metrics", [metric],  METRIC_FIELDS)
        self._append("logs",    [log],     LOG_FIELDS)
        self._append("traces",  spans,     TRACE_FIELDS)

    def _append(self, name: str, rows: list[dict], fields: list[str]) -> None:
        with self._paths[name].open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writerows(rows)

    @property
    def paths(self) -> dict[str, Path]:
        """Return current versioned paths dict."""
        return self._paths
