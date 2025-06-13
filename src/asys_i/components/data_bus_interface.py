# src/asys_i/components/data_bus_interface.py
"""
Core Philosophy: Separation, Predictability.
Defines the abstract interface for the data transport layer between producer (Hooker)
and consumers (Trainer, Archiver).
Low Coupling: Components depend only on this interface.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from asys_i.common.types import ActivationPacket, ConsumerID
from asys_i.monitoring.monitor_interface import BaseMonitor, NoOpMonitor
from asys_i.orchestration.config_loader import DataBusConfig

log = logging.getLogger(__name__)


class BaseDataBus(ABC):
    """
    Abstract Base Class for Data Bus implementations.
    Defines the contract for pushing and pulling activation data.
    """

    def __init__(self, config: DataBusConfig, monitor: BaseMonitor):
        self.config = config
        self.monitor = monitor
        self._is_ready = False
        log.info(
            f"Initialized DataBus: {self.__class__.__name__} with buffer size {config.buffer_size_per_shard}, shards {config.num_shards}"
        )

    @abstractmethod
    def push(self, packet: ActivationPacket, timeout: Optional[float] = None) -> bool:
        """
        Pushes a single packet onto the bus.
        Args:
            packet: The ActivationPacket to send.
            timeout: Max time to wait if buffer is full. Defaults to config.
        Returns:
             True if successful, False if buffer full / timeout (backpressure signal).
        """
        raise NotImplementedError

    @abstractmethod
    def pull_batch(
        self,
        consumer_id: ConsumerID,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[ActivationPacket]:
        """
        Pulls a batch of packets for a specific consumer.
        Args:
             consumer_id: ID of the consumer (to manage shard subscription).
             batch_size: Max number of packets to pull. Defaults to config.
             timeout: Max time to wait for at least one packet. Defaults to config.
        Returns:
             A list of ActivationPacket (can be empty if timeout, or less than batch_size).
        """
        raise NotImplementedError

    @abstractmethod
    def register_consumer(
        self, consumer_id: ConsumerID, layer_indices: List[int]
    ) -> None:
        """Register a consumer and map it to specific shards based on layer indices (HPC)."""
        raise NotImplementedError

    def is_ready(self) -> bool:
        """Check if the bus is ready for operation."""
        return self._is_ready

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics (e.g., queue depth, drop count)."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanly shut down the bus, releasing resources (shared memory, queues)."""
        log.info(f"Shutting down DataBus: {self.__class__.__name__}")
        self._is_ready = False
        # Subclasses should implement specific cleanup


# --- No-Op Implementation ---
class NoOpDataBus(BaseDataBus):
    """A DataBus that does nothing. Pushes always succeed but drop data, pulls always return empty."""

    def __init__(self, config: DataBusConfig, monitor: BaseMonitor):
        # Ensure we have a monitor, even if it's NoOp
        super().__init__(config, monitor if monitor else NoOpMonitor(None, None))
        self._is_ready = True
        log.warning("NoOpDataBus initialized: All data will be dropped!")

    def push(self, packet: ActivationPacket, timeout: Optional[float] = None) -> bool:
        self.monitor.log_metric("data_bus_noop_drop_count", 1)
        return True  # Pretend success, but data is dropped

    def pull_batch(
        self,
        consumer_id: ConsumerID,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[ActivationPacket]:
        return []

    def register_consumer(
        self, consumer_id: ConsumerID, layer_indices: List[int]
    ) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {"depth": 0, "drops": "N/A"}

    def shutdown(self) -> None:
        super().shutdown()
        log.info("NoOpDataBus shutdown (no action).")
