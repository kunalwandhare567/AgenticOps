"""app_simulator/physics/base_scenario.py — Abstract base for all 13 scenarios."""
from __future__ import annotations
from abc import ABC, abstractmethod
from .distributions import Dist
from ..state import SimulatorState


class BaseScenario(ABC):
    """
    Contract every failure-mode scenario must fulfil.

    apply() is called ONCE per simulation tick.
    It receives the current SimulatorState, mutates metric fields in-place,
    and returns the same state object.
    """
    failure_mode: str = "UNKNOWN"

    @abstractmethod
    def apply(self, state: SimulatorState, dist: Dist) -> SimulatorState:
        """Apply failure physics for one tick. Mutate state, return it."""
        ...
