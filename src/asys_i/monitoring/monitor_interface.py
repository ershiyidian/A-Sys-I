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

    def __init__(
        self,
        monitor_config: MonitorConfig,
        project_config: ProjectConfig,
        shared_heartbeats_dict: dict,
    ):  # Actually a DictProxy
        self.monitor_config = monitor_config
        self.project_config = project_config
        self.shared_heartbeats = (
            shared_heartbeats_dict  # Stores {"last_seen": float, "metrics": dict}
        )
        log.info(f"Initialized Monitor: {self.__class__.__name__}")

    @abstractmethod
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single metric value."""
        raise NotImplementedError

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log multiple metrics at once."""
        raise NotImplementedError

    @abstractmethod
    def log_hyperparams(
        self, params: Dict[str, Any], metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log experiment hyperparameters."""
        raise NotImplementedError

    def heartbeat(self, component_id: ComponentID) -> None:
        """
        Record a heartbeat for a component. Updates internal timestamp.
        Concrete classes can override to push heartbeat metric too.
        """
        current_time = time.time()
        if component_id not in self.shared_heartbeats:
            self.shared_heartbeats[component_id] = {
                "last_seen": current_time,
                "metrics": {},
            }
        else:
            self.shared_heartbeats[component_id]["last_seen"] = current_time
        log.debug(f"Heartbeat received from {component_id} into shared dict.")

    def get_heartbeats(self) -> Dict[ComponentID, float]:
        """Return a copy of the last seen timestamps for all components from shared dict."""
        # Return copy to avoid external modification, extracting only last_seen
        return {
            cid: data.get("last_seen", 0.0)
            for cid, data in self.shared_heartbeats.items()
        }

    def register_component(self, component_id: ComponentID) -> None:
        """Explicitly register a component to be watched, initializing its heartbeat in shared dict."""
        if component_id not in self.shared_heartbeats:
            log.info(
                f"Registering component for heartbeat monitoring (shared): {component_id}"
            )
            # Initialize with current time and empty metrics
            self.shared_heartbeats[component_id] = {
                "last_seen": time.time(),
                "metrics": {},
            }
        else:
            # If already exists, just update its heartbeat as a refresh
            self.heartbeat(component_id)

    def update_metrics(self, component_id: ComponentID, metrics: dict) -> None:
        """Update additional metrics for a component in the shared heartbeat structure."""
        if component_id not in self.shared_heartbeats:
            self.register_component(component_id)  # Ensure it's registered

        # Ensure "metrics" sub-dictionary exists
        if "metrics" not in self.shared_heartbeats[component_id]:
            self.shared_heartbeats[component_id]["metrics"] = {}

        self.shared_heartbeats[component_id]["metrics"].update(metrics)

    def get_metrics(self, component_id: ComponentID) -> dict:
        """Retrieve additional metrics for a component from the shared heartbeat structure."""
        component_data = self.shared_heartbeats.get(component_id, {})
        return component_data.get("metrics", {}).copy()  # Return a copy

    @abstractmethod
    def shutdown(self) -> None:
        """Flush buffers, close connections, and release resources."""
        log.info(f"Shutting down Monitor: {self.__class__.__name__}")
        # Subclasses should implement specific cleanup


# --- No-Op Implementation for testing or disabling ---
class NoOpMonitor(BaseMonitor):
    """A monitor implementation that does nothing. For testing or disabling."""

    def __init__(
        self,
        monitor_config: MonitorConfig,
        project_config: ProjectConfig,
        shared_heartbeats_dict: dict,
    ):
        super().__init__(monitor_config, project_config, shared_heartbeats_dict)
        log.warning("NoOpMonitor initialized: No metrics will be logged!")

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def log_hyperparams(
        self, params: Dict[str, Any], metrics: Optional[Dict[str, float]] = None
    ) -> None:
        pass

    # heartbeat and get_heartbeats use base class in-memory impl, which is fine.
    def shutdown(self) -> None:
        log.info("NoOpMonitor shutdown (no action).")
