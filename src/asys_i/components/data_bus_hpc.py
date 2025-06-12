  # src/asys_i/components/data_bus_hpc.py
 """
 Core Philosophy: Predictability (Performance).
 HPC mode data bus implementation: Python wrapper for C++ shared-memory ring buffer.
 Achieves high throughput via zero-copy (shared mem) and lock-free mechanisms, avoiding GIL.
 """
import logging
import time
import multiprocessing.shared_memory
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple

# HPC specific conditional imports
try:
     # This is the name defined in PYBIND11_MODULE
     # import asys_i.hpc.cpp_ringbuffer_bindings as cpp_backend 
     # Simulate backend for structure
     class MockCppQueue:
         def __init__(self, size): self._q = []; self._size=size
         def push(self, item, timeout): 
              if len(self._q) < self._size: self._q.append(item); return True
              return False
         def pop(self, timeout): 
             return self._q.pop(0) if self._q else None
         def size(self): return len(self._q)
             
     class MockCppBackend:
          def SPMCQueue(self, size): return MockCppQueue(size)
     cpp_backend = MockCppBackend()
     CPP_AVAILABLE = True # SIMULATION
except ImportError:
     cpp_backend = None
     CPP_AVAILABLE = False
     
from asys_i.common.types import ActivationPacket, ConsumerID, LayerIndex, TensorRef, RunProfile
from asys_i.orchestration.config_loader import DataBusConfig
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.hpc import check_hpc_prerequisites, CPP_EXTENSION_AVAILABLE

log = logging.getLogger(__name__)

# Define a structure matching the C++ packet metadata if necessary
# PacketMeta = Dict[str, Any] 

class SharedMemoryManager:
     """ Manages tensor storage in shared memory """
     def __init__(self, name:str, size_bytes: int, monitor: BaseMonitor):
          self.monitor = monitor
          self.name = name
          try:
               # Create new block
              self.shm = multiprocessing.shared_memory.SharedMemory(name=name, create=True, size=size_bytes)
              log.info(f"Created shared memory '{name}' of size {size_bytes/1e9:.2f} GB")
              self.owner = True
          except FileExistsError:
               # Attach to existing block (e.g., worker processes)
               self.shm = multiprocessing.shared_memory.SharedMemory(name=name, create=False)
               log.info(f"Attached to existing shared memory '{name}'")
               self.owner = False
          # TODO: Implement sophisticated allocation/free list/ref counting
          self._next_offset = 0
          self.size_bytes = size_bytes
          
     def put_tensor(self, tensor: torch.Tensor) -> Optional[TensorRef]:
         """ Copy tensor to shared memory and return ref: 'offset:size:dtype:shape' """
         # Must be contiguous CPU tensor
         tensor_cpu = tensor.detach().contiguous().cpu()
         tensor_bytes = tensor_cpu.numpy().nbytes
         
         if self._next_offset + tensor_bytes > self.size_bytes:
             self.monitor.log_metric("data_bus_shm_full_count", 1)
             log.error("Shared memory full!")
             # TODO: handle wrapping or reallocation
             return None
         
         # View SHM buffer as numpy array and copy
         shm_array = np.ndarray(tensor_cpu.shape, dtype=tensor_cpu.numpy().dtype, buffer=self.shm.buf[self._next_offset : self._next_offset + tensor_bytes])
         shm_array[:] = tensor_cpu.numpy()[:]
         
         ref = f"{self._next_offset}:{tensor_bytes}:{str(tensor.dtype)}:{'-'.join(map(str, tensor.shape))}"
         self._next_offset += tensor_bytes
         self.monitor.log_metric("data_bus_shm_usage_bytes", self._next_offset)
         return ref

     def get_tensor(self, ref: TensorRef) -> torch.Tensor:
         """ Reconstruct tensor from shared memory reference """
         try:
             offset_s, size_s, dtype_s, shape_s = ref.split(':')
             offset, size = int(offset_s), int(size_s)
             # Map torch dtype str to numpy dtype
             torch_dtype = eval(dtype_s.replace('torch.', 'torch.')) # e.g. "torch.float32"
             np_dtype = torch.empty(0, dtype=torch_dtype).numpy().dtype
             shape = tuple(map(int, shape_s.split('-')))
             
             # Create numpy view on the buffer
             shm_array = np.ndarray(shape, dtype=np_dtype, buffer=self.shm.buf[offset : offset + size])
             # Create torch tensor from numpy view (zero-copy on CPU)
             tensor = torch.from_numpy(shm_array) # .clone() # clone if modification is expected
             return tensor
         except Exception as e:
              log.error(f"Failed to reconstruct tensor from ref '{ref}': {e}")
              raise

     def close(self):
          self.shm.close()
          if self.owner:
               try:
                   self.shm.unlink() # Release memory block only if we created it
                   log.info(f"Unlinked shared memory '{self.name}'")
               except FileNotFoundError:
                    pass # already unlinked
               
class CppShardedSPMCBus(BaseDataBus):
     """
      HPC Mode Data Bus:
      - Uses multiple C++ SPMC lock-free ring buffers (shards).
      - Tensor data stored in Python shared_memory, only metadata+TensorRef in C++ queue.
      - Data assigned to shard based on layer_idx % num_shards.
      - Consumers subscribe to specific shards.
      """
     def __init__(self,
                   config: DataBusConfig,
                   monitor: BaseMonitor):
          # Fail fast if C++ extension is not available
          # if not CPP_EXTENSION_AVAILABLE or cpp_backend is None:
          #     check_hpc_prerequisites() # Raises ImportError
          if config.run_profile != RunProfile.HPC:
               log.warning("CppShardedSPMCBus initialized but run_profile is not HPC!")
               
          super().__init__(config, monitor)
          self.num_shards = config.num_shards
          self.shards: List[Any] = []
          self.consumer_map: Dict[ConsumerID, List[int]] = {} # Map consumer_id to list of shard indices
          self.consumer_shard_cursor: Dict[ConsumerID, int] = {} # For round-robin pulling

          shm_name = f"asys_i_shm_{os.getpid()}_{int(time.time())}"
          shm_size = int(config.shared_memory_size_gb * 1e9)
          self.memory_manager = SharedMemoryManager(shm_name, shm_size, monitor)

          log.info(f"Initializing {self.num_shards} C++ SPMC shards with capacity {config.buffer_size_per_shard} each.")
          for i in range(self.num_shards):
                # Pass the buffer size to C++ constructor
               # queue = cpp_backend.SPMCQueue(config.buffer_size_per_shard) 
               queue = MockCppQueue(config.buffer_size_per_shard) # SIMULATION
               self.shards.append(queue)
          
          self._is_ready = True
          log.warning("!!! CppShardedSPMCBus is RUNNING IN SIMULATION MODE (Mock C++ Backend) !!!")


     def _get_shard_idx(self, layer_idx: LayerIndex) -> int:
         return layer_idx % self.num_shards

     def push(self,
               packet: ActivationPacket,
               timeout: Optional[float] = None) -> bool:
          if not self._is_ready: return False
          
          # 1. Store tensor in shared memory
          if not isinstance(packet['data'], torch.Tensor):
               log.error(f"HPC Bus expects torch.Tensor data, got {type(packet['data'])}")
               return False
          tensor_ref = self.memory_manager.put_tensor(packet['data'])
          if tensor_ref is None:
                # Shared memory full is a form of backpressure
                self.monitor.log_metric("data_bus_drop_count", 1, tags={"reason": "shm_full"})
                return False 
                
          # 2. Create metadata packet for C++ queue
          meta_packet = packet.copy()
          meta_packet['data'] = tensor_ref # Replace Tensor with TensorRef
           
          # 3. Select shard and push
          shard_idx = self._get_shard_idx(packet['layer_idx'])
          shard = self.shards[shard_idx]
          effective_timeout = timeout if timeout is not None else self.config.push_timeout_sec
          tags = {"shard": shard_idx, "bus_type": "cpp_sharded"}

          start_time = time.perf_counter()
          # success = shard.push(meta_packet, effective_timeout) # Actual C++ call
          success = shard.push(meta_packet, effective_timeout) # SIMULATION

          if success:
              latency = (time.perf_counter() - start_time) * 1000
              self.monitor.log_metric("data_bus_push_latency_ms", latency, tags=tags)
              self.monitor.log_metric("data_bus_push_count", 1, tags=tags)
              self.monitor.log_metric("data_bus_queue_depth", shard.size(), tags=tags) # SIMULATION
              return True
          else:
               self.monitor.log_metric("data_bus_drop_count", 1, tags={**tags, "reason": "queue_full"})
               self.monitor.log_metric("data_bus_queue_depth", shard.size(), tags=tags) # SIMULATION
               log.debug(f"DataBus Shard {shard_idx} is full. Dropping packet.")
               return False # Backpressure signal

     def pull_batch(self,
                     consumer_id: ConsumerID,
                     batch_size: Optional[int] = None,
                     timeout: Optional[float] = None) -> List[ActivationPacket]:
         if not self._is_ready: return []
         if consumer_id not in self.consumer_map:
              log.error(f"Consumer {consumer_id} not registered. Cannot pull.")
              return []

         effective_batch_size = batch_size if batch_size is not None else self.config.pull_batch_size_max
         effective_timeout = timeout if timeout is not None else self.config.pull_timeout_sec
         batch: List[ActivationPacket] = []
         meta_batch : List[ActivationPacket] = []
         tags = {"consumer": consumer_id, "bus_type": "cpp_sharded"}
         
         assigned_shards_idx = self.consumer_map[consumer_id]
         if not assigned_shards_idx: return []
             
         start_time = time.time()
         
         # Round-robin over assigned shards to ensure fairness
         start_cursor = self.consumer_shard_cursor.get(consumer_id, 0)
         
         packets_per_shard = max(1, effective_batch_size // len(assigned_shards_idx))
         
         shards_checked = 0
         current_cursor = start_cursor
         while len(meta_batch) < effective_batch_size and shards_checked < len(assigned_shards_idx):
             if time.time() - start_time > effective_timeout: break
                 
             shard_idx = assigned_shards_idx[current_cursor % len(assigned_shards_idx)]
             shard = self.shards[shard_idx]
             
             pulled_from_shard = 0
             while pulled_from_shard < packets_per_shard and len(meta_batch) < effective_batch_size:
                   # Use very short timeout for pop, main timeout is controlled by loop
                   # meta_packet = shard.pop(timeout=0.001) # Actual C++ call
                   meta_packet = shard.pop(timeout=0.001) if shard.size() > 0 else None # SIMULATION
                   if meta_packet:
                         meta_batch.append(meta_packet)
                         pulled_from_shard +=1
                   else:
                         break # Shard empty or timeout
             
             current_cursor += 1
             shards_checked +=1 # Ensure we don't loop forever if all shards empty
         
         # Update cursor for next pull
         self.consumer_shard_cursor[consumer_id] = current_cursor % len(assigned_shards_idx)

         # Reconstruct Tensors
         for meta_packet in meta_batch:
              try:
                    tensor_ref = meta_packet['data']
                    if isinstance(tensor_ref, TensorRef):
                         tensor = self.memory_manager.get_tensor(tensor_ref)
                         full_packet = meta_packet.copy()
                         full_packet['data'] = tensor
                         batch.append(full_packet)
                         # TODO: Signal memory manager to free/decrement ref count for tensor_ref
                    else:
                         log.error(f"Packet data is not a TensorRef: {type(tensor_ref)}")
              except Exception as e:
                   log.error(f"Failed to reconstruct tensor for consumer {consumer_id}: {e}")
                   self.monitor.log_metric("data_bus_error_count", 1, tags={**tags, "type": "reconstruct"})
     
         if batch:
              latency = (time.time() - start_time) * 1000
              self.monitor.log_metric("data_bus_pull_latency_ms", latency , tags=tags)
              self.monitor.log_metric("data_bus_pull_count", len(batch), tags=tags)
              # Monitor depth per shard is tricky here, maybe log push/pull rates per shard instead
              
         return batch

     def register_consumer(self, consumer_id: ConsumerID, layer_indices: List[LayerIndex]) -> None:
         """ Assign shards to a consumer based on the layers it's responsible for."""
         assigned_shards = sorted(list(set(self._get_shard_idx(idx) for idx in layer_indices)))
         self.consumer_map[consumer_id] = assigned_shards
         self.consumer_shard_cursor[consumer_id] = 0
         log.info(f"Consumer {consumer_id} registered for layers {layer_indices}, assigned to shards: {assigned_shards}")
         self.monitor.log_metric("data_bus_registered_consumers", len(self.consumer_map))


     def get_stats(self) -> Dict[str, Any]:
           # Aggregated stats, more detailed via monitor
           total_depth = sum(s.size() for s in self.shards) # SIMULATION
           return {
                "type": "CppShardedSPMC",
                "total_depth": total_depth,
                "num_shards": self.num_shards,
                "shm_usage_gb": self.memory_manager._next_offset / 1e9,
                 "registered_consumers": list(self.consumer_map.keys()),
           }

     def shutdown(self) -> None:
         super().shutdown()
         # Signal C++ queues to shut down if necessary
         self.shards.clear()
         self.memory_manager.close() # Close and unlink SHM
         log.info("CppShardedSPMCBus resources released.")
         import os # required for getpid

