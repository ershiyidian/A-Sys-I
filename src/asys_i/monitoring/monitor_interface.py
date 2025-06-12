# src/asys_i/monitoring/monitor_interface.py
"""
Core Philosophy: Observability-First, Separation.
Defines the abstract interface for all monitoring implementations.
Low Coupling: Components depend on this interface, not concrete implementations.
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from asys_i.common.types import ComponentID
from asys_i.orchestration.config_loader import MonitorConfig, ProjectConfig

log = logging.getLogger(__name__)

class BaseMonitor(ABC):
    """
    Abstract Base Class for Monitoring systems.
    All monitoring implementations (Prometheus, CSV, Logging) must implement this.
    """
    def __init__(self,
                 monitor_config: MonitorConfig,
                 project_config: ProjectConfig):
        self.monitor_config = monitor_config
        self.project_config = project_config
        # In-memory cache for watchdog checks
        self._last_heartbeats: Dict[ComponentID, float] = {}
        log.info(f"Initialized Monitor: {self.__class__.__name__}")

    @abstractmethod
    def log_metric(self,
                   name: str,
                   value: float,
                   step: Optional[int] = None,
                   tags: Optional[Dict[str, Any]] = None) -> None:
        """Log a single metric value."""
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self,
                    metrics: Dict[str, float],
                     step: Optional[int] = None,
                    tags: Optional[Dict[str, Any]] = None) -> None:
        """Log multiple metrics at once."""
        raise NotImplementedError
        
    @abstractmethod
    def log_hyperparams(self, 
                         params: Dict[str, Any],
                         metrics: Optional[Dict[str, float]] = None) -> None:
         """Log experiment hyperparameters."""
         raise NotImplementedError

    def heartbeat(self, component_id: ComponentID) -> None:
        """
        Record a heartbeat for a component. Updates internal timestamp.
        Concrete classes can override to push heartbeat metric too.
         """
        self._last_heartbeats[component_id] = time.time()
        log.debug(f"Heartbeat received from {component_id}")


    def get_heartbeats(self) -> Dict[ComponentID, float]:
        """Return a copy of the last seen timestamps for all components."""
        # Return copy to avoid external modification
        return self._last_heartbeats.copy()
        
    def register_component(self, component_id: ComponentID) -> None:
         """ Explicitly register a component to be watched, initializing its heartbeat"""
         if component_id not in self._last_heartbeats:
              log.info(f"Registering component for heartbeat monitoring: {component_id}")
              self.heartbeat(component_id) # Initial heartbeat


    @abstractmethod
    def shutdown(self) -> None:
        """Flush buffers, close connections, and release resources."""
        log.info(f"Shutting down Monitor: {self.__class__.__name__}")
        raise NotImplementedError

# --- No-Op Implementation for testing or disabling ---
class NoOpMonitor(BaseMonitor):
     """A monitor implementation that does nothing. For testing or disabling."""
     def __init__(self,  monitor_config: MonitorConfig, project_config: ProjectConfig):
         super().__init__(monitor_config, project_config)
         log.warning("NoOpMonitor initialized: No metrics will be logged!")
     def log_metric(self, name: str, value: float, step: Optional[int] = None, tags: Optional[Dict[str, Any]] = None) -> None: pass
     def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, tags: Optional[Dict[str, Any]] = None) -> None: pass
     def log_hyperparams(self, params: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> None: pass
     # heartbeat and get_heartbeats use base class in-memory impl, which is fine.
     def shutdown(self) -> None:
         log.info("NoOpMonitor shutdown (no action).")

