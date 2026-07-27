"""
app_simulator/physics/__init__.py
===================================
Scenario registry — maps failure mode name → scenario class.
Use load_scenario(mode_name) to get an instantiated scenario.
"""
from .base_scenario import BaseScenario
from .distributions import Dist
from .none import NoneScenario
from .memory_leak import MemoryLeakScenario
from .cpu_saturation import CpuSaturationScenario
from .latency_spike import LatencySpikeScenario
from .error_storm import ErrorStormScenario
from .db_slowdown import DbSlowdownScenario
from .cache_stampede import CacheStampedeScenario
from .queue_backup import QueueBackupScenario
from .dependency_timeout import DependencyTimeoutScenario
from .bad_deployment import BadDeploymentScenario
from .retry_storm import RetryStormScenario
from .disk_io_saturation import DiskIoSaturationScenario
from .cascading_failure import CascadingFailureScenario

SCENARIO_REGISTRY: dict[str, type] = {
    "NONE":               NoneScenario,
    "MEMORY_LEAK":        MemoryLeakScenario,
    "CPU_SATURATION":     CpuSaturationScenario,
    "LATENCY_SPIKE":      LatencySpikeScenario,
    "ERROR_STORM":        ErrorStormScenario,
    "DB_SLOWDOWN":        DbSlowdownScenario,
    "CACHE_STAMPEDE":     CacheStampedeScenario,
    "QUEUE_BACKUP":       QueueBackupScenario,
    "DEPENDENCY_TIMEOUT": DependencyTimeoutScenario,
    "BAD_DEPLOY":         BadDeploymentScenario,
    "RETRY_STORM":        RetryStormScenario,
    "DISK_IO_SATURATION": DiskIoSaturationScenario,
    "CASCADING_FAILURE":  CascadingFailureScenario,
}

__all__ = [
    "BaseScenario", "Dist", "SCENARIO_REGISTRY", "load_scenario",
]


def load_scenario(failure_mode: str) -> BaseScenario:
    """
    Instantiate and return the scenario for the given failure mode.

    Args:
        failure_mode: One of ALL_MODES (case-insensitive).

    Returns:
        An instantiated BaseScenario subclass.

    Raises:
        ValueError if failure_mode is not in SCENARIO_REGISTRY.
    """
    key = failure_mode.strip().upper()
    if key not in SCENARIO_REGISTRY:
        raise ValueError(
            f"Unknown failure mode: {failure_mode!r}. "
            f"Valid modes: {sorted(SCENARIO_REGISTRY)}"
        )
    return SCENARIO_REGISTRY[key]()
