# src/asys_i/components/data_bus_factory.py (CONFIRMED FROM YOUR LAST INPUT - GOOD)
"""
Factory to create the appropriate DataBus instance based on configuration.
Ensures that HPC dependencies are met or fails hard.
"""
import logging

from asys_i.common.types import DataBusType
from asys_i.components.data_bus_consumer import PythonQueueBus
from asys_i.components.data_bus_interface import BaseDataBus # NoOpDataBus removed from direct use here
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MasterConfig
from asys_i.hpc import CPP_EXTENSION_AVAILABLE, check_hpc_prerequisites # check_hpc raises if not available

# Conditional import for the new CppShardedSPMCBus
if CPP_EXTENSION_AVAILABLE:
    from asys_i.components.data_bus_hpc import CppShardedSPMCBus
else:
    CppShardedSPMCBus = None # type: ignore

log = logging.getLogger(__name__)

def create_data_bus(
    config: MasterConfig, monitor: BaseMonitor
) -> BaseDataBus:
    """
    Factory: Creates a BaseDataBus instance.
    - For HPC, requires the C++ extension to be compiled and loadable.
    - If HPC dependencies are not met by check_hpc_prerequisites(), it will raise an ImportError.
    """
    bus_type = config.data_bus.type
    log.info(f"Factory creating DataBus of type: {bus_type}")

    if bus_type == DataBusType.CPP_SHARDED_SPMC:
        check_hpc_prerequisites() # This will raise if the C++ extension is not available
        if CppShardedSPMCBus is None: # Should be caught by check_hpc, but defense-in-depth
            raise ImportError(
                "CppShardedSPMCBus selected, but the C++ extension module could not be imported "
                "even after prerequisite check. This indicates a deeper issue with the environment or build."
            )
        return CppShardedSPMCBus(config, monitor)

    elif bus_type == DataBusType.PYTHON_QUEUE:
        return PythonQueueBus(config.data_bus, monitor) # Pass specific DataBusConfig

    else:
        # This case should ideally be caught by Pydantic validation of DataBusType enum during config load.
        log.error(f"Unknown or unsupported data bus type: {bus_type}. This is a critical configuration error.")
        raise ValueError(
            f"Invalid data_bus.type '{bus_type}' in configuration. "
            f"Supported types are: {[e.value for e in DataBusType]}."
        )

