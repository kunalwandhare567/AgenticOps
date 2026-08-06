"""
backend/live_feed_simulator/world_scenario_engine.py
=====================================================
Markov-chain based real-world failure mode sequencer.

Instead of hardcoding which failure modes to run, this engine uses a
probabilistic transition matrix to pick the NEXT failure mode after
each episode completes — mimicking how real infrastructure failures
cascade (e.g. a memory leak often precedes CPU saturation, which can
trigger a cascading failure).

Usage:
    from Simulator.live_feed_simulator.world_scenario_engine import WorldScenarioEngine

    engine = WorldScenarioEngine(seed=None)          # random seed
    for mode, ep_idx in engine.generate_session():   # infinite generator
        ...                                          # Ctrl+C to stop
"""
from __future__ import annotations

import numpy as np
from typing import Generator, Tuple

# =============================================================================
# Real-World Failure Transition Matrix
# =============================================================================
# Rows = current mode, Columns = next mode.
# Probabilities in each row sum to 1.0.
#
# Design rationale (domain knowledge):
#   - NONE (healthy): mostly stays healthy, but any mode can onset
#   - MEMORY_LEAK: heap pressure → CPU thrash or cascading collapse
#   - CPU_SATURATION: queue backup, retry storms, cascading common
#   - DB_SLOWDOWN: cache stampede (cache bypassed), retry storms
#   - CACHE_STAMPEDE: DB hammering → DB_SLOWDOWN or CASCADING
#   - CASCADING_FAILURE: resolves mostly to healthy but bad deploys happen
#   - BAD_DEPLOY: rolled back → healthy, or error storm persists
#   - DISK_IO_SATURATION: slow, typically resolves or triggers DB_SLOWDOWN
# =============================================================================

TRANSITION_MATRIX: dict[str, dict[str, float]] = {
    "NONE": {
        "NONE":               0.35,
        "MEMORY_LEAK":        0.12,
        "CPU_SATURATION":     0.10,
        "LATENCY_SPIKE":      0.09,
        "DB_SLOWDOWN":        0.08,
        "ERROR_STORM":        0.06,
        "BAD_DEPLOY":         0.05,
        "CACHE_STAMPEDE":     0.04,
        "QUEUE_BACKUP":       0.04,
        "RETRY_STORM":        0.03,
        "DEPENDENCY_TIMEOUT": 0.02,
        "DISK_IO_SATURATION": 0.01,
        "CASCADING_FAILURE":  0.01,
    },
    "MEMORY_LEAK": {
        "NONE":               0.10,
        "MEMORY_LEAK":        0.20,
        "CPU_SATURATION":     0.28,
        "LATENCY_SPIKE":      0.10,
        "CASCADING_FAILURE":  0.12,
        "QUEUE_BACKUP":       0.07,
        "DB_SLOWDOWN":        0.05,
        "ERROR_STORM":        0.04,
        "RETRY_STORM":        0.02,
        "BAD_DEPLOY":         0.01,
        "CACHE_STAMPEDE":     0.01,
        "DISK_IO_SATURATION": 0.00,
        "DEPENDENCY_TIMEOUT": 0.00,
    },
    "CPU_SATURATION": {
        "NONE":               0.12,
        "CPU_SATURATION":     0.15,
        "CASCADING_FAILURE":  0.25,
        "QUEUE_BACKUP":       0.18,
        "RETRY_STORM":        0.10,
        "LATENCY_SPIKE":      0.08,
        "ERROR_STORM":        0.07,
        "MEMORY_LEAK":        0.03,
        "DB_SLOWDOWN":        0.01,
        "DEPENDENCY_TIMEOUT": 0.01,
        "BAD_DEPLOY":         0.00,
        "CACHE_STAMPEDE":     0.00,
        "DISK_IO_SATURATION": 0.00,
    },
    "LATENCY_SPIKE": {
        "NONE":               0.20,
        "LATENCY_SPIKE":      0.18,
        "DB_SLOWDOWN":        0.15,
        "CASCADING_FAILURE":  0.12,
        "DEPENDENCY_TIMEOUT": 0.12,
        "QUEUE_BACKUP":       0.08,
        "CPU_SATURATION":     0.07,
        "RETRY_STORM":        0.05,
        "ERROR_STORM":        0.02,
        "MEMORY_LEAK":        0.01,
        "CACHE_STAMPEDE":     0.00,
        "BAD_DEPLOY":         0.00,
        "DISK_IO_SATURATION": 0.00,
    },
    "ERROR_STORM": {
        "NONE":               0.10,
        "ERROR_STORM":        0.15,
        "BAD_DEPLOY":         0.20,
        "CASCADING_FAILURE":  0.18,
        "RETRY_STORM":        0.15,
        "DEPENDENCY_TIMEOUT": 0.10,
        "CPU_SATURATION":     0.07,
        "QUEUE_BACKUP":       0.03,
        "MEMORY_LEAK":        0.01,
        "LATENCY_SPIKE":      0.01,
        "DB_SLOWDOWN":        0.00,
        "CACHE_STAMPEDE":     0.00,
        "DISK_IO_SATURATION": 0.00,
    },
    "DB_SLOWDOWN": {
        "NONE":               0.15,
        "DB_SLOWDOWN":        0.15,
        "CACHE_STAMPEDE":     0.22,
        "CASCADING_FAILURE":  0.18,
        "RETRY_STORM":        0.12,
        "LATENCY_SPIKE":      0.08,
        "DEPENDENCY_TIMEOUT": 0.05,
        "QUEUE_BACKUP":       0.03,
        "ERROR_STORM":        0.01,
        "CPU_SATURATION":     0.01,
        "MEMORY_LEAK":        0.00,
        "BAD_DEPLOY":         0.00,
        "DISK_IO_SATURATION": 0.00,
    },
    "CACHE_STAMPEDE": {
        "NONE":               0.15,
        "CACHE_STAMPEDE":     0.12,
        "DB_SLOWDOWN":        0.22,
        "CASCADING_FAILURE":  0.18,
        "LATENCY_SPIKE":      0.12,
        "RETRY_STORM":        0.08,
        "CPU_SATURATION":     0.07,
        "QUEUE_BACKUP":       0.04,
        "ERROR_STORM":        0.01,
        "DEPENDENCY_TIMEOUT": 0.01,
        "MEMORY_LEAK":        0.00,
        "BAD_DEPLOY":         0.00,
        "DISK_IO_SATURATION": 0.00,
    },
    "QUEUE_BACKUP": {
        "NONE":               0.18,
        "QUEUE_BACKUP":       0.20,
        "CPU_SATURATION":     0.18,
        "CASCADING_FAILURE":  0.15,
        "RETRY_STORM":        0.12,
        "LATENCY_SPIKE":      0.08,
        "DB_SLOWDOWN":        0.05,
        "ERROR_STORM":        0.02,
        "MEMORY_LEAK":        0.01,
        "DEPENDENCY_TIMEOUT": 0.01,
        "CACHE_STAMPEDE":     0.00,
        "BAD_DEPLOY":         0.00,
        "DISK_IO_SATURATION": 0.00,
    },
    "DEPENDENCY_TIMEOUT": {
        "NONE":               0.20,
        "DEPENDENCY_TIMEOUT": 0.15,
        "RETRY_STORM":        0.18,
        "CASCADING_FAILURE":  0.15,
        "LATENCY_SPIKE":      0.12,
        "ERROR_STORM":        0.10,
        "QUEUE_BACKUP":       0.05,
        "CPU_SATURATION":     0.03,
        "DB_SLOWDOWN":        0.01,
        "BAD_DEPLOY":         0.01,
        "MEMORY_LEAK":        0.00,
        "CACHE_STAMPEDE":     0.00,
        "DISK_IO_SATURATION": 0.00,
    },
    "BAD_DEPLOY": {
        "NONE":               0.35,
        "BAD_DEPLOY":         0.10,
        "ERROR_STORM":        0.20,
        "CASCADING_FAILURE":  0.15,
        "DEPENDENCY_TIMEOUT": 0.08,
        "RETRY_STORM":        0.07,
        "CPU_SATURATION":     0.03,
        "LATENCY_SPIKE":      0.01,
        "DB_SLOWDOWN":        0.01,
        "MEMORY_LEAK":        0.00,
        "QUEUE_BACKUP":       0.00,
        "CACHE_STAMPEDE":     0.00,
        "DISK_IO_SATURATION": 0.00,
    },
    "RETRY_STORM": {
        "NONE":               0.20,
        "RETRY_STORM":        0.18,
        "CASCADING_FAILURE":  0.18,
        "CPU_SATURATION":     0.12,
        "QUEUE_BACKUP":       0.10,
        "ERROR_STORM":        0.10,
        "DEPENDENCY_TIMEOUT": 0.07,
        "LATENCY_SPIKE":      0.03,
        "DB_SLOWDOWN":        0.01,
        "MEMORY_LEAK":        0.01,
        "CACHE_STAMPEDE":     0.00,
        "BAD_DEPLOY":         0.00,
        "DISK_IO_SATURATION": 0.00,
    },
    "DISK_IO_SATURATION": {
        "NONE":               0.30,
        "DISK_IO_SATURATION": 0.20,
        "DB_SLOWDOWN":        0.18,
        "LATENCY_SPIKE":      0.12,
        "CASCADING_FAILURE":  0.08,
        "CPU_SATURATION":     0.06,
        "QUEUE_BACKUP":       0.04,
        "MEMORY_LEAK":        0.01,
        "RETRY_STORM":        0.01,
        "ERROR_STORM":        0.00,
        "BAD_DEPLOY":         0.00,
        "CACHE_STAMPEDE":     0.00,
        "DEPENDENCY_TIMEOUT": 0.00,
    },
    "CASCADING_FAILURE": {
        "NONE":               0.40,
        "CASCADING_FAILURE":  0.08,
        "BAD_DEPLOY":         0.15,
        "CPU_SATURATION":     0.12,
        "ERROR_STORM":        0.10,
        "MEMORY_LEAK":        0.07,
        "RETRY_STORM":        0.05,
        "QUEUE_BACKUP":       0.02,
        "DB_SLOWDOWN":        0.01,
        "LATENCY_SPIKE":      0.00,
        "CACHE_STAMPEDE":     0.00,
        "DEPENDENCY_TIMEOUT": 0.00,
        "DISK_IO_SATURATION": 0.00,
    },
}

# All failure mode keys in a fixed order (for numpy indexing)
_ALL_MODES = sorted(TRANSITION_MATRIX.keys())


def _validate_matrix() -> None:
    """Assert all rows sum to 1.0 ± tolerance."""
    for mode, row in TRANSITION_MATRIX.items():
        total = sum(row.values())
        assert abs(total - 1.0) < 1e-9, (
            f"TRANSITION_MATRIX row '{mode}' sums to {total:.6f}, expected 1.0"
        )
        # Ensure all 13 modes are present
        missing = set(_ALL_MODES) - set(row.keys())
        assert not missing, f"Row '{mode}' missing keys: {missing}"


_validate_matrix()


# =============================================================================
# WorldScenarioEngine
# =============================================================================

class WorldScenarioEngine:
    """
    Infinite real-world failure mode sequencer using a Markov transition matrix.

    Each call to next_mode() probabilistically selects the next failure mode
    based on the current mode, mimicking real infrastructure failure cascades.

    Args:
        seed:         Optional numpy random seed for reproducibility.
        start_mode:   Initial mode (default: "NONE" — healthy start).
    """

    def __init__(self, seed: int | None = None, start_mode: str = "NONE") -> None:
        self._rng = np.random.default_rng(seed)
        self._current_mode: str = start_mode.upper()
        self._episode_count: int = 0
        self._transition_history: list[tuple[str, str]] = []

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def next_mode(self) -> str:
        """
        Sample the next failure mode from the transition distribution.

        Returns:
            Next failure mode string (e.g. "MEMORY_LEAK", "CPU_SATURATION").
        """
        row = TRANSITION_MATRIX[self._current_mode]
        modes = list(row.keys())
        probs = list(row.values())

        # numpy.random.choice requires a 1-D array of probabilities
        chosen = str(self._rng.choice(modes, p=probs))

        self._transition_history.append((self._current_mode, chosen))
        prev = self._current_mode
        self._current_mode = chosen
        self._episode_count += 1

        return chosen

    def generate_session(
        self,
        max_episodes: int | None = None,
    ) -> Generator[Tuple[str, int], None, None]:
        """
        Infinite generator that yields (failure_mode, episode_index) tuples.

        Each yielded item represents one complete episode to simulate.
        The first episode always starts from NONE (healthy) and transitions
        from there.

        Args:
            max_episodes:  Stop after this many episodes (None = infinite).

        Yields:
            (failure_mode: str, episode_index: int)
        """
        ep = 0
        while max_episodes is None or ep < max_episodes:
            mode = self.next_mode()
            yield mode, ep
            ep += 1

    def transition_summary(self) -> str:
        """Return a compact readable chain of transitions so far."""
        if not self._transition_history:
            return "(no transitions yet)"
        parts = [self._transition_history[0][0]]
        for _, dst in self._transition_history:
            parts.append(dst)
        return " → ".join(parts[-10:])  # last 10 transitions
