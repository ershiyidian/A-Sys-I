# src/asys_i/components/activation_hooker.py
"""
Core Philosophy: Separation (Observer), Predictability, Graceful Degradation.
Attaches hooks to the host model, captures activations, preprocesses them,
and pushes them to the DataBus. Handles sampling and backpressure.
This is the "sensor" of the Telescope.
"""
import logging
import time
import random
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from functools import partial
from typing import List, Callable, Dict, Any, Optional, Tuple

from asys_i.common.types import (
     LayerIndex, GlobalStep, RunProfile, ActivationPacket, create_activation_packet,
 )
from asys_i.orchestration.config_loader import MasterConfig
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.hpc.gpu_kernels import preprocess_tensor_on_gpu, CUDA_AVAILABLE

log = logging.getLogger(__name__)

# Add log_once capabilities
_LOG_CACHE = set()
def log_once(level, msg, *args, **kwargs):
     if msg not in _LOG_CACHE:
         log.log(level, msg, *args, **kwargs)
         _LOG_CACHE.add(msg)

class ActivationHooker:
    """
    Manages attaching/detaching hooks and processing activation data.
    """
    def __init__(self,
                 model: nn.Module,
                 config: MasterConfig,
                 data_bus: BaseDataBus,
                 monitor: BaseMonitor):
        self.model = model
        self.hook_config = config.hook
        self.hardware_config = config.hardware
        self.profile = config.run_profile
        self.data_bus = data_bus
        self.monitor = monitor
        self._handles: List[RemovableHandle] = []
        self._attached = False
        self._current_sampling_rate = self.hook_config.sampling_rate
        self._global_step_ref: Optional[Callable[[], GlobalStep]] = None
        self._last_backpressure_time = 0.0
        
        # HPC: Low priority stream for async processing and copy
        self.cuda_stream: Optional[torch.cuda.Stream] = None
        self.pinned_memory_pool: List[torch.Tensor] = [] # Simple pool management
        if self.profile == RunProfile.HPC and CUDA_AVAILABLE:
             # priority=-1 is lower than default stream (0)
            self.cuda_stream = torch.cuda.Stream(priority=-1) 
            log.info("HPC mode: Initialized low-priority CUDA stream for hook processing.")
        
        log.info(f"ActivationHooker initialized. Profile: {self.profile}, Sampling: {self._current_sampling_rate:.2f}, Layers: {self.hook_config.layers_to_hook}")

    def _should_sample(self) -> bool:
        if self._current_sampling_rate >= 0.999:
            return True
        if self._current_sampling_rate <= 0.001:
             return False
        return random.random() < self._current_sampling_rate

    def _handle_backpressure(self, success: bool):
        """Dynamically adjusts sampling rate based on DataBus push success."""
        tags = {"profile": self.profile}
        if not success:
             now = time.time()
             # Apply backpressure damping
             if now - self._last_backpressure_time > self.hook_config.backpressure_debounce_sec:
                 old_rate = self._current_sampling_rate
                 # Exponential backoff
                 self._current_sampling_rate = max(self.hook_config.min_sampling_rate, old_rate * 0.7)
                 log.warning(f"BACKPRESSURE: DataBus full. Reducing sampling rate: {old_rate:.3f} -> {self._current_sampling_rate:.3f}")
                 self.monitor.log_metric("hook_backpressure_event_count", 1, tags=tags)
                 self.monitor.log_metric("hook_sampling_rate", self._current_sampling_rate, tags=tags)
                 self._last_backpressure_time = now
        # Optional: slowly increase rate if successful (Addictive Increase Multiplicative Decrease - AIMD)
        # elif self._current_sampling_rate < self.hook_config.sampling_rate:
             # self._current_sampling_rate = min(self.hook_config.sampling_rate, self._current_sampling_rate + 0.01)
             # self.monitor.log_metric("hook_sampling_rate", self._current_sampling_rate, tags=tags)
             pass


    def _find_module_by_index(self, layer_idx: LayerIndex) -> Optional[nn.Module]:
         """ Heuristic to find the module corresponding to layer_idx. NEEDS ADAPTATION per model arch."""
         try:
              # Common transformer structures: model.layers, model.h, blocks etc.
              if hasattr(self.model, 'layers'): return self.model.layers[layer_idx] # type: ignore
              if hasattr(self.model, 'h'): return self.model.h[layer_idx] # type: ignore
              if hasattr(self.model, 'blocks'): return self.model.blocks[layer_idx] # type: ignore
              # MOCK model access
              if hasattr(self.model, 'layers') and isinstance(self.model.layers, nn.ModuleList):
                   return self.model.layers[layer_idx]
              # Fallback/Advanced: use model.named_modules() and match string name like 'layers.5.mlp'
              log_once(logging.WARNING, f"Hooker: Cannot find standard layer list, attempting direct access for layer {layer_idx}. Define hook_point name mapping.")
              # This is just a placeholder for more robust module finding based on `hook_point` config
              modules = list(self.model.children())
              if layer_idx < len(modules):
                   return modules[layer_idx]
         except IndexError:
              log.error(f"Layer index {layer_idx} out of bounds for model.")
         except Exception as e:
              log.error(f"Error finding module for layer {layer_idx}: {e}")
         return None
         
    def _extract_tensor(self, output: Any) -> Optional[torch.Tensor]:
         """ Extracts the primary tensor from hook output, handling tuples/dicts """
         if isinstance(output, torch.Tensor):
             return output
         # Common HuggingFace / TRL outputs
         if isinstance(output, tuple):
             if output and isinstance(output[0], torch.Tensor):
                  return output[0] # Usually the first element
             # Handle BaseModelOutputWithPast etc.
             for item in output:
                  if isinstance(item, torch.Tensor) and item.ndim >= 2: # look for batch x dim
                       return item
         if isinstance(output, dict) and 'hidden_states' in output:
               # This needs care, hidden_states might be a tuple of all layers
              pass
         log_once(logging.DEBUG, f"Could not extract primary tensor from hook output type: {type(output)}")
         return None

    # --- The Core Hook Function ---
    def _create_hook_fn(self, layer_idx: LayerIndex):
        """Creates the closure function to be registered as hook."""
        
        def hook(module: nn.Module, input: Tuple, output: Any):
             start_time_ns = time.time_ns()
             tags = {"layer": layer_idx, "profile": self.profile}
             
             if not self._attached or self._global_step_ref is None or not self.data_bus.is_ready():
                 return # Avoid processing if detaching or not ready
             
             self.monitor.log_metric("hook_trigger_count", 1, tags=tags)
             
             if not self._should_sample():
                  self.monitor.log_metric("hook_sample_skip_count", 1, tags=tags)
                  return

             try:
                tensor = self._extract_tensor(output)
                if tensor is None:
                    self.monitor.log_metric("hook_error_count", 1, tags={**tags, "reason": "extract_fail"})
                    return
                    
                # Detach from graph IMMEDIATELY to avoid interfering with backprop
                # Flatten batch and sequence dims: (batch, seq, dim) -> (batch*seq, dim)
                # TODO: make flattening configurable, keep metadata (batch_idx, seq_pos)
                tensor_detached = tensor.detach().view(-1, tensor.shape[-1])
                if tensor_detached.shape[0] == 0: return # Empty tensor

                step = self._global_step_ref()
                packet_data: Any = None
                meta: Dict[str, Any] = {}
                success = False

                # --- HPC Path: GPU Preprocessing & Async Copy ---
                if self.profile == RunProfile.HPC and self.cuda_stream and tensor_detached.is_cuda:
                     # ALL work on the low-priority stream
                     with torch.cuda.stream(self.cuda_stream):
                         # 1. GPU Preprocessing (quantize, compress)
                         processed_tensor, meta = preprocess_tensor_on_gpu(
                              tensor_detached, self.hook_config, self.monitor, layer_idx
                         )
                         # 2. Async copy to CPU Pinned Memory for DataBus (if not using GPU Direct RDMA/SHM)
                         # For simplicity, assume DataBus expects CPU tensor or handles pinned memory
                         # Simplification: DataBusHPC expects CPU tensor to put in shared memory
                         # Create/reuse pinned memory
                         # pinned_buffer = torch.empty(processed_tensor.shape, dtype=processed_tensor.dtype, pin_memory=True)
                         # pinned_buffer.copy_(processed_tensor, non_blocking=True)
                         # packet_data = pinned_buffer # Pass pinned buffer
                         
                         # Even simpler: Sync copy to CPU, DataBusHPC copies to ShMem
                         # Let the CppDataBus handle the copy to shared memory for now.
                         # If processing moves tensor to CPU, ensure it is.
                         packet_data = processed_tensor # Pass GPU or CPU tensor
                         if packet_data.is_cuda:
                              # Use non_blocking copy if DataBus can handle async event
                              packet_data = packet_data.to(device='cpu', non_blocking=True) 
                              
                         # Create packet *after* tensor processing
                         packet = create_activation_packet(layer_idx, step, packet_data, self.profile, meta)
                         
                         # CRITICAL: DataBus PUSH must happen within stream context or after sync
                         # OR DataBus must handle synchronization. Assume Push is fast CPU op.
                         # self.cuda_stream.synchronize() # Ensure copy/process finished before push!
                         # Pushing TensorRef is CPU bound, so sync needed if DataBus uses tensor content NOW
                         push_start = time.perf_counter()
                         success = self.data_bus.push(packet)
                         push_latency = (time.perf_counter() - push_start) * 1000
                         self.monitor.log_metric("hook_push_latency_ms", push_latency, tags=tags)

                 # --- CONSUMER Path: Simple CPU Copy ---
                 else:
                      # Simple detach and move to CPU
                      cpu_tensor = tensor_detached.to(device='cpu', non_blocking=True)
                      packet_data = cpu_tensor
                      packet = create_activation_packet(layer_idx, step, packet_data, self.profile, meta)
                      push_start = time.perf_counter()
                      success = self.data_bus.push(packet)
                      push_latency = (time.perf_counter() - push_start) * 1000
                      self.monitor.log_metric("hook_push_latency_ms", push_latency, tags=tags)
                
                self._handle_backpressure(success)
                if success:
                     self.monitor.log_metric("hook_packet_success_count", 1, tags=tags)
                     # Calculate throughput bytes
                     if isinstance(packet_data, torch.Tensor):
                          nbytes = packet_data.nelement() * packet_data.element_size()
                          self.monitor.log_metric("hook_throughput_bytes", nbytes, tags=tags)
                else:
                     self.monitor.log_metric("hook_packet_drop_count", 1, tags=tags) # Backpressure drop

             except Exception as e:
                  log.exception(f"Error in hook function for layer {layer_idx}:")
                  self.monitor.log_metric("hook_error_count", 1, tags={**tags, "reason": "exception"})
                  # Design for failure: hook error must not crash the host model
             finally:
                  # If HPC, sync stream before measuring total latency
                  # if self.cuda_stream: self.cuda_stream.synchronize()
                  latency_ms = (time.time_ns() - start_time_ns) / 1_000_000
                  self.monitor.log_metric("hook_total_latency_ms", latency_ms, tags=tags)
        return hook

    def attach(self, global_step_ref: Callable[[], GlobalStep]):
        if self._attached:
            log.warning("Hooks are already attached.")
            return
        self._global_step_ref = global_step_ref
        self._current_sampling_rate = self.hook_config.sampling_rate # Reset rate
        self.monitor.log_metric("hook_sampling_rate", self._current_sampling_rate, tags={"profile": self.profile})

        for layer_idx in self.hook_config.layers_to_hook:
            module = self._find_module_by_index(layer_idx)
            if module:
                 hook_fn = self._create_hook_fn(layer_idx)
                 # Use register_forward_hook
                 handle = module.register_forward_hook(hook_fn)
                 self._handles.append(handle)
                 log.info(f"Attached forward hook to layer {layer_idx} ({module.__class__.__name__})")
            else:
                 log.error(f"Could not find module for layer index {layer_idx}. Hook not attached.")
                 self.monitor.log_metric("hook_attach_error_count", 1, tags={"layer": layer_idx})
        
        if self._handles:
             self._attached = True
             log.info(f"Successfully attached {len(self._handles)} hooks.")
        else:
             log.error("No hooks were attached!")

    def detach(self):
        if not self._attached:
            return
        log.info(f"Detaching {len(self._handles)} hooks...")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._attached = False
        self._global_step_ref = None
        if self.cuda_stream: # Ensure all work on stream is finished
             self.cuda_stream.synchronize()

    def shutdown(self):
         self.detach()
         # cleanup resources, e.g., pinned memory pool
         self.pinned_memory_pool.clear()
         log.info("ActivationHooker shut down.")

