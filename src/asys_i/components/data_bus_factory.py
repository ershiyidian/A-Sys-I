# src/asys_i/components/data_bus_factory.py
"""
Core Philosophy: Config-Driven, Separation, Graceful Degradation.
Factory function to create the appropriate DataBus instance based on configuration.
"""
import logging
from typing import Optional

from asys_i.common.types import DataBusType
from asys_i.components.data_bus_consumer import PythonQueueBus
from asys_i.components.data_bus_interface import BaseDataBus, NoOpDataBus
from asys_i.monitoring.monitor_interface import BaseMonitor, NoOpMonitor
from asys_i.orchestration.config_loader import MasterConfig

# Conditional import
try:
    from asys_i.components.data_bus_hpc import CppShardedSPMCBus
    from asys_i.hpc import CPP_EXTENSION_AVAILABLE

    HPC_BUS_AVAILABLE = True  # or check CPP_EXTENSION_AVAILABLE
except ImportError as e:
    HPC_BUS_AVAILABLE = False
    CppShardedSPMCBus = None  # type: ignore
    logging.warning(f"HPC DataBus not available: {e}")

log = logging.getLogger(__name__)


def create_data_bus(
    config: MasterConfig, monitor: Optional[BaseMonitor]
) -> BaseDataBus:
    """
    Factory: Creates a BaseDataBus instance based on config.data_bus.type.
    """
    bus_type = config.data_bus.type
    # bus_config = config.data_bus # Comment out or remove, CppShardedSPMCBus now takes MasterConfig
    # Ensure we always have a monitor, even if it's NoOp
    effective_monitor = (
        monitor if monitor is not None else NoOpMonitor(config.monitor, config.project)
    )

    log.info(f"Factory creating DataBus of type: {bus_type}")

    if bus_type == DataBusType.CPP_SHARDED_SPMC:
        if not HPC_BUS_AVAILABLE or CppShardedSPMCBus is None:
            log.error(
                "CppShardedSPMCBus selected but not available (missing C++ extension/dependencies?). "
                "Falling back to NoOpDataBus. Run `pip install -e .[hpc]` and check build."
            )
            # Graceful Degradation / Design for Failure
            return NoOpDataBus(
                config.data_bus, effective_monitor
            )  # Pass specific data_bus config
        return CppShardedSPMCBus(
            config, effective_monitor
        )  # Pass the whole MasterConfig

    elif bus_type == DataBusType.PYTHON_QUEUE:
        return PythonQueueBus(
            config.data_bus, effective_monitor
        )  # Pass specific data_bus config

    else:
        log.error(f"Unknown data bus type: {bus_type}. Falling back to NoOpDataBus.")
        return NoOpDataBus(
            config.data_bus, effective_monitor
        )  # Pass specific data_bus config
