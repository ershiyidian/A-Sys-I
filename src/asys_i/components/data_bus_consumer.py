 # src/asys_i/components/data_bus_consumer.py
 """
 Core Philosophy: Graceful Degradation.
 CONSUMER mode data bus implementation using Python's multiprocessing.Queue.
 Simple, cross-platform, but lower performance (GIL, serialization overhead).
 """
import logging
import multiprocessing
import queue  # Import for queue.Empty and queue.Full exceptions
import time
from typing import List, Dict, Any, Optional

from asys_i.common.types import ActivationPacket, ConsumerID, LayerIndex
from asys_i.orchestration.config_loader import DataBusConfig
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.components.data_bus_interface import BaseDataBus

log = logging.getLogger(__name__)

class PythonQueueBus(BaseDataBus):
    """
     CONSUMER Mode Data Bus:
     - Uses a single multiprocessing.Queue.
     - Does NOT support sharding (num_shards is ignored).
     - Packet data (torch.Tensor) is serialized/deserialized, high overhead.
     """
    def __init__(self,
                  config: DataBusConfig,
                  monitor: BaseMonitor):
         super().__init__(config, monitor)
         # PythonQueue doesn't support sharding well for SPMC/MPMC easily
         if config.num_shards > 1:
              log.warning(f"PythonQueueBus does not support sharding. Ignoring num_shards={config.num_shards}.")
         
         # Use a Manager().Queue() if the bus object itself needs to be passed across processes
         # Or just standard Queue if created in main and inherited. Assume latter.
         self.queue: multiprocessing.Queue[ActivationPacket] = multiprocessing.Queue(
              maxsize=config.buffer_size_per_shard
          )
         self._is_ready = True
         self._push_count = 0
         self._pull_count = 0
         self._drop_count = 0
         log.info(f"PythonQueueBus initialized with maxsize={config.buffer_size_per_shard}")

    def push(self,
              packet: ActivationPacket,
              timeout: Optional[float] = None) -> bool:
         if not self._is_ready: return False
         effective_timeout = timeout if timeout is not None else self.config.push_timeout_sec
         tags = {"shard": 0, "bus_type": "python_queue"}
         try:
             # detach and move tensor to CPU if it's not already. Queue pickle might handle this, but explicit is better
             if isinstance(packet['data'], torch.Tensor) and packet['data'].is_cuda:
                  packet['data'] = packet['data'].detach().cpu()
                  
             start_time = time.perf_counter()
             self.queue.put(packet, block=True, timeout=effective_timeout)
             latency = (time.perf_counter() - start_time) * 1000
             self.monitor.log_metric("data_bus_push_latency_ms", latency, tags=tags)
             self.monitor.log_metric("data_bus_push_count", 1, tags=tags)
             self.monitor.log_metric("data_bus_queue_depth", self.queue.qsize(), tags=tags)
             self._push_count +=1
             return True
         except queue.Full:
              self.monitor.log_metric("data_bus_drop_count", 1, tags=tags)
              self.monitor.log_metric("data_bus_queue_depth", self.queue.qsize(), tags=tags) # log depth when full
              self._drop_count +=1
              log.debug("DataBus Queue is full. Dropping packet.")
              return False # Backpressure signal
         except Exception as e:
              log.error(f"Error pushing to PythonQueueBus: {e}")
              self.monitor.log_metric("data_bus_error_count", 1, tags={**tags, "type": "push"})
              return False

    def pull_batch(self,
                    consumer_id: ConsumerID, # consumer_id ignored here, all consumers pull from one queue
                    batch_size: Optional[int] = None,
                    timeout: Optional[float] = None) -> List[ActivationPacket]:
         if not self._is_ready: return []
         effective_batch_size = batch_size if batch_size is not None else self.config.pull_batch_size_max
         effective_timeout = timeout if timeout is not None else self.config.pull_timeout_sec
         batch: List[ActivationPacket] = []
         tags = {"consumer": consumer_id, "shard": 0, "bus_type": "python_queue"}
         
         start_time = time.time()
         
         try:
             # First get: block up to timeout
             packet = self.queue.get(block=True, timeout=effective_timeout)
             batch.append(packet)
             
             # Subsequent gets: non-blocking, grab whatever is available quickly
             while len(batch) < effective_batch_size:
                  try:
                       # Check if timeout exceeded during batch build
                       if time.time() - start_time > effective_timeout : break
                       packet = self.queue.get(block=False) # get_nowait()
                       batch.append(packet)
                  except queue.Empty:
                       break # Queue is now empty, return what we have
         except queue.Empty:
               # Timeout occurred on the first get
               pass 
         except Exception as e:
              log.error(f"Error pulling from PythonQueueBus by {consumer_id}: {e}")
              self.monitor.log_metric("data_bus_error_count", 1, tags={**tags, "type": "pull"})
              
         if batch:
              latency = (time.time() - start_time) * 1000
              self.monitor.log_metric("data_bus_pull_latency_ms", latency , tags=tags)
              self.monitor.log_metric("data_bus_pull_count", len(batch), tags=tags)
              self.monitor.log_metric("data_bus_queue_depth", self.queue.qsize(), tags=tags) # depth after pull
              self._pull_count += len(batch)
              
         return batch
         
    def register_consumer(self, consumer_id: ConsumerID, layer_indices: List[LayerIndex]) -> None:
        log.debug(f"Consumer {consumer_id} registered (no-op for PythonQueueBus).")
        # No shard mapping needed for single queue

    def get_stats(self) -> Dict[str, Any]:
        depth = 0
        try:
             depth = self.queue.qsize()
        except NotImplementedError:
             depth = -1 # qsize not implemented on all platforms (macOS)
        return {
             "type": "PythonQueue",
             "depth": depth,
             "max_size": self.config.buffer_size_per_shard,
             "push_count": self._push_count,
             "pull_count": self._pull_count,
             "drop_count": self._drop_count,
             }

    def shutdown(self) -> None:
        super().shutdown()
        try:
             # Drain queue to help processes terminate
             while not self.queue.empty():
                  self.queue.get_nowait()
        except (queue.Empty, OSError, EOFError):
              pass
        try:
             self.queue.close()
             self.queue.join_thread() # Allow background thread to exit
        except Exception as e:
             log.warning(f"Error closing queue: {e}")
        log.info("PythonQueueBus resources released.")
        import torch # Ensure torch is imported if tensors are involved

