 # src/asys_i/monitoring/monitor_factory.py
 """
 Core Philosophy: Config-Driven, Separation, Graceful Degradation.
 Factory function to create the appropriate Monitor instance based on configuration.
 Low Coupling: Decouples the rest of the system from concrete Monitor class names.
 """
import logging

from asys_i.common.types import MonitorType
from asys_i.orchestration.config_loader import MasterConfig
from asys_i.monitoring.monitor_interface import BaseMonitor, NoOpMonitor
from asys_i.monitoring.monitor_consumer import LoggingCSVTensorBoardMonitor
# Use try-except for conditional import of HPC module
try:
     from asys_i.monitoring.monitor_hpc import PrometheusMonitor
     HPC_MONITOR_AVAILABLE = True
except ImportError as e:
      HPC_MONITOR_AVAILABLE = False
      PrometheusMonitor = None # type: ignore
      logging.warning(f"HPC Monitor not available: {e}")


log = logging.getLogger(__name__)

def create_monitor(config: MasterConfig) -> BaseMonitor:
     """
     Factory: Creates a BaseMonitor instance based on config.monitor.type.
     """
     monitor_type = config.monitor.type
     monitor_config = config.monitor
     project_config = config.project
     
     log.info(f"Factory creating Monitor of type: {monitor_type}")

     if monitor_type == MonitorType.PROMETHEUS:
          if not HPC_MONITOR_AVAILABLE or PrometheusMonitor is None:
               log.error(
                    "PrometheusMonitor selected but not available (missing dependencies?). "
                     "Falling back to NoOpMonitor. Install with `pip install .[hpc]`."
               )
               # Graceful Degradation / Design for Failure
               return NoOpMonitor(monitor_config, project_config) 
          return PrometheusMonitor(monitor_config, project_config)
     
     elif monitor_type == MonitorType.CSV_TENSORBOARD:
          return LoggingCSVTensorBoardMonitor(monitor_config, project_config)
          
     elif monitor_type == MonitorType.LOGGING_ONLY:
           # TODO: Implement a pure logging monitor if needed, or reuse CSV without CSV/TB
           log.warning("LOGGING_ONLY monitor not fully implemented, using CSV_TENSORBOARD")
           return LoggingCSVTensorBoardMonitor(monitor_config, project_config)

     elif monitor_type == MonitorType.NONE:
         return NoOpMonitor(monitor_config, project_config)
         
     else:
          # Design for failure
          log.error(f"Unknown monitor type: {monitor_type}. Falling back to NoOpMonitor.")
          return NoOpMonitor(monitor_config, project_config)

