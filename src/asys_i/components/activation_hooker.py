# src/asys_i/components/activation_hooker.py (REVISED FROM YOUR LAST INPUT - WITH PINNED MEMORY)
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from asys_i.common.types import (
    GlobalStep,
    RunProfile,
    ActivationPacket,
    create_activation_packet, # Python-side packet creation for Consumer
    TensorRef, # Python-side type hint for what CppBus returns
    # DTYPE_TO_CODE_MAP, # Not directly used by Hooker, but by CppBus/Trainer
    # calculate_checksum, # Done by CppBus or Consumer packet factory
)
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.hpc.gpu_kernels import CUDA_AVAILABLE, preprocess_tensor_on_gpu
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MasterConfig, HookConfig # Ensure HookConfig used by self.hook_config is correct

log = logging.getLogger(__name__)

class ActivationHooker:
    def __init__(
        self,
        model: nn.Module,
        config: MasterConfig,
        data_bus: BaseDataBus,
        monitor: BaseMonitor,
    ):
        self.model = model
        self.hook_config: HookConfig = config.hook # Explicitly type hint
        self.hardware_config = config.hardware
        self.profile = config.run_profile
        self.data_bus = data_bus
        self.monitor = monitor
        self._handles: Dict[str, RemovableHandle] = {}
        self._attached = False
        self._current_sampling_rate = self.hook_config.sampling_rate
        self._global_step_ref: Optional[Callable[[], GlobalStep]] = None
        self._last_backpressure_time = 0.0
        self._layer_name_to_idx: Dict[str, int] = {} # FQN path -> numeric_idx

        self.cuda_stream: Optional[torch.cuda.Stream] = None
        self.pinned_memory_buffer: Optional[torch.Tensor] = None # For HPC async copy

        if self.profile == RunProfile.HPC and CUDA_AVAILABLE:
            self.cuda_stream = torch.cuda.Stream(priority=-1)
            log.info("HPC mode: Initialized low-priority CUDA stream for hook processing.")
            # Pre-allocate pinned memory if a max size can be estimated.
            # This is a simplification; a pool or dynamic allocation might be better.
            # Max tensor size: PPO batch_size * sequence_length (if applicable) * d_model * dtype_bytes
            # For now, let's assume d_in from sae_model config is a good proxy for max feature dim
            # and ppo.batch_size is the batch dimension.
            if isinstance(config.sae_model.d_in, int) and config.ppo.batch_size > 0:
                # Simplistic estimation, assuming flattened activations (batch*seq, features)
                # A more accurate estimation needs to consider max sequence length of host model.
                # Let's assume max items = ppo.batch_size * (some_max_seq_len, e.g., 512)
                # For now, just use ppo.batch_size * d_in * (typical float bytes)
                # This is a placeholder for a more robust estimation.
                # The actual size of the tensor being hooked could be (batch, seq, features).
                # We use a fairly large buffer as an example.
                # Max items in a flattened batch (e.g. if seq_len is large)
                # This estimate is rough and should be refined based on actual model usage.
                # A good estimate: config.ppo.batch_size * max_sequence_length_of_host_model
                # Here, using a fixed large number of elements as an example.
                max_elements_estimate = config.ppo.batch_size * 2048 * config.sae_model.d_in # batch * (example_max_seq_len_times_features)
                if max_elements_estimate > 0:
                     # Determine dtype from hardware_config for buffer
                    buffer_dtype_str = config.hardware.dtype
                    try:
                        from asys_i.common.types import get_torch_dtype_from_str as common_get_dtype
                        buffer_dtype = common_get_dtype(buffer_dtype_str)
                        # Allocate a large enough buffer; size it based on d_in if available
                        # This creates a 1D buffer, will be reshaped.
                        # Max size for one very large activation batch.
                        # Max shape could be (ppo_batch_size * max_seq_len, d_in)
                        # A simple large pinned buffer:
                        # Example: 16 batch * 512 seq_len * 768 d_in * 2 bytes (bf16) ~ 12MB
                        # This needs to be configurable or dynamically sized based on actual hooked tensor shapes.
                        # For now, let's make a reasonably sized buffer:
                        # Heuristic: assume it can hold a few batches of (PPO_BATCH_SIZE, config.sae_model.d_in)
                        # This is still very approximate.
                        # True max size depends on the largest activation tensor encountered.
                        # Let's allocate a buffer of, say, 64MB.
                        pinned_buffer_size_bytes = 64 * 1024 * 1024
                        elements_for_pinned_buffer = pinned_buffer_size_bytes // buffer_dtype.itemsize

                        self.pinned_memory_buffer = torch.empty(elements_for_pinned_buffer, dtype=buffer_dtype, pin_memory=True)
                        log.info(f"HPC mode: Pre-allocated pinned memory buffer of {pinned_buffer_size_bytes / 1e6:.2f}MB for async GPU->CPU copy.")
                    except Exception as e:
                        log.warning(f"HPC mode: Failed to pre-allocate pinned memory buffer: {e}. Async copy might be slower.")
                        self.pinned_memory_buffer = None


        log.info(f"ActivationHooker initialized. Profile: {self.profile.value}, Sampling: {self._current_sampling_rate:.2f}, Layers: {list(self.hook_config.layers_to_hook.keys())}")

    def _should_sample(self) -> bool:
        # (Same as your reconstructed version)
        if self._current_sampling_rate >= 0.999: return True
        if self._current_sampling_rate <= 0.001: return False
        return random.random() < self._current_sampling_rate

    def _handle_backpressure(self, success: bool):
        # (Same as your reconstructed version, ensure hook_config has these fields)
        tags = {"profile": self.profile.value}
        if not success:
            now = time.time()
            # Check if backpressure_debounce_sec and min_sampling_rate are in hook_config
            debounce_sec = getattr(self.hook_config, 'backpressure_debounce_sec', 2.0)
            min_rate = getattr(self.hook_config, 'min_sampling_rate', 0.01)

            if (now - self._last_backpressure_time > debounce_sec):
                old_rate = self._current_sampling_rate
                self._current_sampling_rate = max(min_rate, old_rate * 0.7)
                log.warning(f"BACKPRESSURE: DataBus full. Reducing sampling rate: {old_rate:.3f} -> {self._current_sampling_rate:.3f}")
                self.monitor.log_metric("hook_backpressure_event_count", 1, tags=tags)
                self.monitor.log_metric("hook_sampling_rate", self._current_sampling_rate, tags=tags)
                self._last_backpressure_time = now

    def _get_module_by_fqn(self, fqn_path: str) -> Optional[nn.Module]:
        # (Same as your reconstructed version)
        try:
            module = self.model.get_submodule(fqn_path)
            return module
        except AttributeError:
            log.error(f"Module not found at FQN path: {fqn_path} in model {type(self.model)}")
            return None
        except Exception as e:
            log.error(f"Error accessing module at FQN path {fqn_path}: {e}")
            return None

    def _extract_tensor(self, output: Any) -> Optional[torch.Tensor]:
        # (Same as your reconstructed version, can be expanded for more model output types)
        if isinstance(output, torch.Tensor): return output
        if isinstance(output, tuple):
            if output and isinstance(output[0], torch.Tensor): return output[0]
            for item in output:
                if isinstance(item, torch.Tensor) and item.ndim >= 2: return item
        log.warning(f"Could not extract primary tensor from hook output type: {type(output)}. Check model output structure or refine _extract_tensor.")
        return None

    def _create_hook_fn(self, layer_name_fqn: str, layer_idx_numeric: int):
        # (Largely same as your reconstructed, with pinned memory optimization)
        def hook(module: nn.Module, inputs: Tuple, output: Any):
            start_time_ns = time.time_ns()
            tags = {"layer_name": layer_name_fqn, "profile": self.profile.value, "layer_idx_numeric": layer_idx_numeric}

            if not self._attached or self._global_step_ref is None or not self.data_bus.is_ready(): return
            self.monitor.log_metric("hook_trigger_count", 1, tags=tags)
            if not self._should_sample():
                self.monitor.log_metric("hook_sample_skip_count", 1, tags=tags)
                return

            try:
                tensor = self._extract_tensor(output)
                if tensor is None:
                    self.monitor.log_metric("hook_error_count", 1, tags={**tags, "reason": "extract_fail"})
                    return

                # Detach and flatten: (batch, seq, dim) -> (batch*seq, dim) or (batch, dim)
                # This assumes features are in the last dimension.
                original_shape = tensor.shape
                if tensor.ndim > 2: # e.g. (batch, seq_len, features)
                    tensor_detached = tensor.detach().reshape(-1, original_shape[-1])
                elif tensor.ndim == 2: # (batch, features)
                    tensor_detached = tensor.detach()
                else: # 1D or scalar, less common for activations we care about
                    log.warning(f"Hooked tensor for {layer_name_fqn} has low ndim ({tensor.ndim}). May not be typical activation.")
                    tensor_detached = tensor.detach().view(-1,1) if tensor.ndim == 1 else tensor.detach().view(1,1)


                if tensor_detached.shape[0] == 0: return # Empty tensor

                step = self._global_step_ref()
                meta_custom: Dict[str, Any] = {"original_shape": str(original_shape)}
                success = False
                packet_data_for_bus: torch.Tensor # This will be what's passed to data_bus.push

                if self.profile == RunProfile.HPC and self.cuda_stream and tensor_detached.is_cuda:
                    with torch.cuda.stream(self.cuda_stream):
                        processed_tensor_gpu, meta_gpu_preprocess = preprocess_tensor_on_gpu(
                            tensor_detached, self.hook_config, self.monitor, layer_idx_numeric
                        )
                        meta_custom.update(meta_gpu_preprocess)
                        
                        # Async copy to CPU pinned memory
                        if self.pinned_memory_buffer is not None and \
                           self.pinned_memory_buffer.numel() >= processed_tensor_gpu.numel() and \
                           self.pinned_memory_buffer.dtype == processed_tensor_gpu.dtype:
                            
                            # Reshape a slice of the pinned buffer to match the tensor
                            pinned_slice = self.pinned_memory_buffer.narrow(0, 0, processed_tensor_gpu.numel()).view_as(processed_tensor_gpu)
                            pinned_slice.copy_(processed_tensor_gpu, non_blocking=True)
                            packet_data_for_bus = pinned_slice # Pass the CPU-pinned tensor
                        else:
                            if self.pinned_memory_buffer is None:
                                log.warning_once("HPC hook: Pinned memory buffer not allocated. Falling back to sync copy.")
                            elif self.pinned_memory_buffer.dtype != processed_tensor_gpu.dtype:
                                log.warning_once(f"HPC hook: Pinned memory dtype ({self.pinned_memory_buffer.dtype}) mismatch with tensor ({processed_tensor_gpu.dtype}). Sync copy.")
                            else: # Buffer too small
                                log.warning_once(f"HPC hook: Pinned memory buffer too small ({self.pinned_memory_buffer.numel()}) for tensor ({processed_tensor_gpu.numel()}). Sync copy.")
                            packet_data_for_bus = processed_tensor_gpu.cpu() # Fallback to synchronous copy

                    # Packet creation and push happen outside the CUDA stream context,
                    # but after the copy to pinned_slice (which is on CPU) is initiated.
                    # The CppDataBus.push call will then operate on this CPU tensor.
                    # If pinned_slice was used, CUDA stream sync might be needed before C++ accesses its data
                    # if the C++ side reads immediately without its own sync.
                    # However, CppDataBus.push(numpy_array) implies CPU data access.
                    # The copy_ non_blocking needs a stream.synchronize() before CPU can safely access data.
                    self.cuda_stream.synchronize() # Ensure GPU-CPU copy is complete

                    packet = ActivationPacket( # For HPC, this packet is passed to CppBus.push
                        layer_name=layer_name_fqn,
                        layer_idx_numeric=layer_idx_numeric,
                        global_step=step,
                        data=packet_data_for_bus, # This is the CPU tensor (pinned or not)
                        profile=self.profile,
                        timestamp_ns=time.time_ns(),
                        meta=meta_custom
                    )
                    push_start_hpc = time.perf_counter()
                    success = self.data_bus.push(packet) # CppBus.push expects torch.Tensor, converts to numpy
                    push_latency_hpc = (time.perf_counter() - push_start_hpc) * 1000
                    self.monitor.log_metric("hook_push_latency_ms", push_latency_hpc, tags=tags)

                else: # CONSUMER Path
                    packet_data_for_bus = tensor_detached.cpu() # Ensure CPU for PythonQueue
                    packet = create_activation_packet( # Uses the common factory
                        layer_name=layer_name_fqn,
                        layer_idx_numeric=layer_idx_numeric,
                        global_step=step,
                        tensor_data=packet_data_for_bus,
                        profile=self.profile,
                        meta=meta_custom
                    )
                    push_start_consumer = time.perf_counter()
                    success = self.data_bus.push(packet)
                    push_latency_consumer = (time.perf_counter() - push_start_consumer) * 1000
                    self.monitor.log_metric("hook_push_latency_ms", push_latency_consumer, tags=tags)

                self._handle_backpressure(success)
                if success:
                    self.monitor.log_metric("hook_packet_success_count", 1, tags=tags)
                    # Use packet_data_for_bus for nbytes calculation as it's the one pushed
                    nbytes = packet_data_for_bus.nelement() * packet_data_for_bus.element_size()
                    self.monitor.log_metric("hook_throughput_bytes", nbytes, tags=tags)
                else:
                    self.monitor.log_metric("hook_packet_drop_count", 1, tags=tags)

            except Exception as e:
                log.exception(f"Error in hook function for layer '{layer_name_fqn}':")
                self.monitor.log_metric("hook_error_count", 1, tags={**tags, "reason": f"exception_{type(e).__name__}"})
            finally:
                latency_ms = (time.time_ns() - start_time_ns) / 1_000_000
                self.monitor.log_metric("hook_total_latency_ms", latency_ms, tags=tags)
        return hook

    def attach(self, global_step_ref: Callable[[], GlobalStep]):
        # (Same as your reconstructed version, using FQN from config.hook.layers_to_hook)
        if self._attached: log.warning("Hooks already attached."); return
        self._global_step_ref = global_step_ref
        self._current_sampling_rate = self.hook_config.sampling_rate
        self.monitor.log_metric("hook_sampling_rate", self._current_sampling_rate, tags={"profile": self.profile.value})
        self._layer_name_to_idx.clear()

        numeric_idx_counter = 0
        if not self.hook_config.layers_to_hook:
            log.warning("ActivationHooker: hook.layers_to_hook is empty in config. No hooks will be attached.")
            return

        for friendly_name, fqn_path in self.hook_config.layers_to_hook.items():
            module_to_hook = self._get_module_by_fqn(fqn_path)
            if module_to_hook:
                self._layer_name_to_idx[fqn_path] = numeric_idx_counter # Map FQN to a running numeric index
                hook_fn = self._create_hook_fn(fqn_path, numeric_idx_counter)
                handle = module_to_hook.register_forward_hook(hook_fn)
                self._handles[fqn_path] = handle # Store by FQN
                log.info(f"Attached forward hook to layer '{friendly_name}' (FQN: {fqn_path}, Index: {numeric_idx_counter}) -> {module_to_hook.__class__.__name__}")
                numeric_idx_counter += 1
            else:
                # This should be a critical error if a configured FQN is not found.
                # Consider raising an exception here to stop pipeline setup.
                msg = f"CRITICAL: Module for FQN path '{fqn_path}' (Friendly name: '{friendly_name}') not found. Hooking failed."
                log.error(msg)
                self.monitor.log_metric("hook_attach_error_count", 1, tags={"layer_fqn": fqn_path, "reason": "module_not_found"})
                # raise ValueError(msg) # Optional: make this fatal

        if self._handles:
            self._attached = True
            log.info(f"Successfully attached {len(self._handles)} hooks.")
        elif self.hook_config.layers_to_hook: # If layers were configured but none attached
            log.error("No hooks were attached despite configuration! Check FQN paths and model structure.")
            # This should also be a critical error.
            # raise RuntimeError("Failed to attach any configured hooks.")


    def detach(self):
        # (Same as your reconstructed version)
        if not self._attached: return
        log.info(f"Detaching {len(self._handles)} hooks...")
        for fqn_path, handle in self._handles.items():
            try: handle.remove()
            except Exception as e: log.warning(f"Error removing hook for {fqn_path}: {e}")
        self._handles.clear()
        self._attached = False
        self._global_step_ref = None
        if self.cuda_stream: self.cuda_stream.synchronize()
        log.info("All hooks detached.")

    def shutdown(self):
        self.detach()
        self.pinned_memory_buffer = None # Release pinned memory reference
        if CUDA_AVAILABLE and torch.cuda.is_available(): # Check again, might have changed
            torch.cuda.empty_cache() # Help release CUDA memory
        log.info("ActivationHooker shut down.")

