"""
app_simulator/pipeline/tumbling_window.py
==========================================
Tumbling Window Node — 10-cycle / 20-second label smoother.

WHAT it smooths
---------------
Only the label sequence (Y) — the 10 predicted failure_mode strings produced
by the classifier across 10 cycles. Nothing else.

WHAT it does NOT touch
----------------------
The 27 raw metric features and 5 log features are left completely untouched.
Re-averaging them would corrupt correctly computed per-cycle values:
  - A P99 averaged across 10 cycles is NOT a meaningful P99.
  - Diluting a genuine CPU spike at cycle 8 with the 7 calm cycles before it
    hides the real transition.

The window's sole job is classification stability — fighting label flipping,
not feature reduction.

Worked example (from design document):
  Buffer: [CPU, CPU, LATENCY, CPU, CPU, CPU, CPU, LATENCY, CPU, CPU]
  dominant_state    = CPU_SATURATION
  vote_distribution = {CPU_SATURATION: 8, LATENCY_SPIKE: 2}
  margin            = 6

Episode boundary behaviour
--------------------------
The buffer is NOT reset when the episode_id changes.
Labels carry forward across episode boundaries for stable voting continuity.
(If episode isolation is needed in the future, call window.reset() explicitly.)

Output file
-----------
tumbling_window_output.csv — one row per cycle, written by update().
"""
from __future__ import annotations

import csv
from collections import Counter, deque

from app_data_generator.state import PipelineState
from app_data_generator.config import PIPELINE_OUTPUT_DIR, TUMBLING_WINDOW_CSV, WINDOW_SIZE

# ── CSV columns ────────────────────────────────────────────────────────────────
_WINDOW_CSV_COLS = [
    "cycle",
    "episode_id",
    "timestamp",
    "elapsed_s",
    "failure_mode",          # simulator ground-truth label
    "predicted_failure",     # single-cycle classifier label
    "dominant_state",        # majority vote winner across buffer
    "vote_distribution",     # JSON-like dict e.g. {"CPU_SATURATION": 8, "LATENCY_SPIKE": 2}
    "margin",                # votes_for_winner − votes_for_runner_up
    "window_full",           # True once 10 predictions buffered
    "window_size",           # current buffer fill (0–10)
]


class TumblingWindow:
    """
    10-cycle label buffer with majority vote summarization.

    Stateful — one instance per pipeline session.
    Buffer carries forward across episode boundaries (no reset on episode change).
    Writes one row per cycle to tumbling_window_output.csv.
    """

    def __init__(self, size: int = WINDOW_SIZE) -> None:
        self._size   = size
        self._buffer: deque[str] = deque(maxlen=size)

        # Open CSV for append
        PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_header = not TUMBLING_WINDOW_CSV.exists()
        self._csv_fh = TUMBLING_WINDOW_CSV.open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_fh, fieldnames=_WINDOW_CSV_COLS, extrasaction="ignore"
        )
        if write_header:
            self._csv_writer.writeheader()

    def update(self, state: PipelineState, cycle: int) -> PipelineState:
        """
        Append state.predicted_failure to buffer and compute majority vote.

        Smooths ONLY the label sequence — raw features in state are unchanged.

        Args:
            state: PipelineState after classification. Raw features untouched.
            cycle: Global pipeline cycle counter (for CSV row numbering).

        Returns:
            Same state with window fields filled:
              state.window_predictions  – list of labels in current buffer
              state.summarized_failure  – dominant state (majority vote winner)
              state.vote_distribution   – {label: count} dict
              state.window_margin       – winner_votes − runner_up_votes
              state.window_full         – True once buffer has >= size labels
        """
        label = state.predicted_failure or "NONE"
        self._buffer.append(label)

        labels    = list(self._buffer)
        vote_dist = dict(Counter(labels))

        # Majority vote: label with highest count
        dominant = max(vote_dist, key=vote_dist.get)

        # Margin: votes for winner minus votes for runner-up
        sorted_counts = sorted(vote_dist.values(), reverse=True)
        margin = sorted_counts[0] - (sorted_counts[1] if len(sorted_counts) > 1 else 0)

        state.window_predictions = labels
        state.summarized_failure  = dominant
        state.vote_distribution   = vote_dist
        state.window_margin       = margin
        state.window_full         = len(self._buffer) >= self._size

        # ── Write to tumbling_window_output.csv ───────────────────────────────
        row = {
            "cycle":             cycle,
            "episode_id":        state.episode_id,
            "timestamp":         state.timestamp,
            "elapsed_s":         state.elapsed_s,
            "failure_mode":      state.failure_mode,
            "predicted_failure": label,
            "dominant_state":    dominant,
            "vote_distribution": str(vote_dist),
            "margin":            margin,
            "window_full":       state.window_full,
            "window_size":       len(self._buffer),
        }
        self._csv_writer.writerow(row)
        self._csv_fh.flush()

        return state

    def reset(self) -> None:
        """
        Clear buffer manually (not called automatically on episode boundary).
        Only use if explicit isolation between episodes is required.
        """
        self._buffer.clear()

    def close(self) -> None:
        """Flush and close the CSV file handle."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None

    @property
    def size(self) -> int:
        """Current buffer fill level."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """True once 10 predictions have been seen."""
        return len(self._buffer) >= self._size
