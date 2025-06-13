import logging

import threading

from typing import List, Dict, Optional, Tuple

  

import torch

import torch.nn as nn

  

logger = logging.getLogger(__name__)

  

class PinnedMemoryPool:

    """

    A simple, thread-safe pool for managing a pre-allocated pinned memory buffer.

  

    Warning:

    This implementation uses a simple pointer-bumping allocation strategy. It is

    efficient for scenarios where activations are processed sequentially and the

    pool is reset after each model forward pass. It is NOT a general-purpose

    memory allocator and can suffer from fragmentation if not used as intended.

    For best performance, the pool size should be large enough to hold all

    captured activations from a single forward pass to avoid fallback to

    slower, synchronous copies.

  

    This class IS thread-safe through the use of a lock.

    """

    def __init__(self, pool_size_bytes: int, device: torch.device):

        self.pool_size_bytes = pool_size_bytes

        self.device = device

        self.pool: Optional[torch.Tensor] = None

        self.allocated_bytes = 0

        self._lock = threading.Lock() # Add a lock for thread safety

        if self.pool_size_bytes > 0:

            logger.info(f"Allocating pinned memory pool of {pool_size_bytes / (1024**2):.2f} MB.")

            try:

                self.pool = torch.empty(self.pool_size_bytes, dtype=torch.uint8, pin_memory=True)

            except Exception as e:

                logger.error(f"Failed to allocate pinned memory pool: {e}. Will fall back to sync copies.", exc_info=True)

                self.pool = None

                self.pool_size_bytes = 0

    def get_buffer(self, size_bytes: int) -> Optional[torch.Tensor]:

        """

        Gets a buffer of a specific size from the pool. Thread-safe.

        Returns None if the pool is disabled or not enough contiguous space is available.

        """

        if self.pool is None or size_bytes == 0:

            return None

        with self._lock:

            if self.allocated_bytes + size_bytes > self.pool_size_bytes:

                logger.warning(

                    f"Pinned memory pool full. Requested {size_bytes / (1024**2):.2f} MB, "

                    f"but only { (self.pool_size_bytes - self.allocated_bytes) / (1024**2):.2f} MB "

                    f"available. Falling back to synchronous copy for this tensor."

                )

                return None

            buffer_slice = self.pool[self.allocated_bytes : self.allocated_bytes + size_bytes]

            self.allocated_bytes += size_bytes

            return buffer_slice

  

    def reset(self):

        """Resets the allocation pointer. Thread-safe."""

        with self._lock:

            self.allocated_bytes = 0

  
  

class ActivationHooker:

    """

    Attaches hooks to specified PyTorch module layers to capture activations.

    In HPC mode, it uses an asynchronous GPU -> CPU copy via a pinned memory pool.

    """

    def __init__(self,

                 model: nn.Module,

                 target_modules: List[str],

                 bus, # The data bus instance (e.g., CppShardedSPMCBus)

                 pinned_memory_buffer_size_mb: int):

        self.model = model

        self.target_modules = target_modules

        self.bus = bus

        self.device = next(model.parameters()).device

  

        pool_size_bytes = pinned_memory_buffer_size_mb * 1024 * 1024

        self.memory_pool = PinnedMemoryPool(pool_size_bytes, self.device)

        self.hooks: Dict[str, nn.Module.register_forward_hook] = {}

        self._attach_hooks()

        # A list to hold non-blocking copy events and their associated data

        self.copy_events: List[Tuple[torch.cuda.Event, torch.Tensor, str]] = []

  

    def _attach_hooks(self):

        """Finds target modules and attaches forward hooks to them."""

        for name, module in self.model.named_modules():

            if name in self.target_modules:

                self.hooks[name] = module.register_forward_hook(self._create_hook_fn(name))

                logger.info(f"Attached hook to module: {name}")

  

    def _create_hook_fn(self, module_name: str):

        """Creates a closure for the hook function to capture the module name."""

        def hook(_module, _input, output):

            # Models can return a single tensor or a tuple of tensors/other data.

            # We assume the first element of a tuple is the primary activation.

            tensor_to_capture = output[0] if isinstance(output, tuple) else output

            if not isinstance(tensor_to_capture, torch.Tensor):

                logger.warning(f"Hook on '{module_name}' received a non-tensor output of type {type(tensor_to_capture)}. Skipping.")

                return

  

            self._process_tensor(tensor_to_capture.detach(), module_name)

        return hook

  

    def _process_tensor(self, tensor: torch.Tensor, module_name: str):

        """Processes a captured tensor, using async copy if possible."""

        tensor_bytes = tensor.numel() * tensor.element_size()

        # Try to get a buffer from our pinned memory pool

        pinned_buffer = self.memory_pool.get_buffer(tensor_bytes)

  

        if pinned_buffer is not None:

            # Asynchronous copy path

            # Reshape buffer to match tensor view and then copy

            buffer_view = pinned_buffer.view(tensor.dtype).view(tensor.shape)

            buffer_view.copy_(tensor, non_blocking=True)

            # Create a CUDA event to track when the async copy is complete

            event = torch.cuda.Event()

            event.record()

            self.copy_events.append((event, buffer_view, module_name))

        else:

            # Fallback to synchronous copy

            cpu_tensor = tensor.to('cpu', non_blocking=False)

            self.bus.push(cpu_tensor, module_name)

  

    def synchronize_and_push(self):

        """

        Must be called after the model's forward pass. This method synchronizes

        all pending asynchronous copies and pushes the completed tensors to the data bus.

        """

        if not self.copy_events:

            return

  

        logger.debug(f"Synchronizing {len(self.copy_events)} async H2D copy events.")

        for event, buffer_view, module_name in self.copy_events:

            event.synchronize()  # Blocks until the copy for this event is complete

            # The tensor is now guaranteed to be ready on the CPU-pinned memory.

            # The .cpu() call is effectively a no-op but clarifies intent and returns

            # a tensor without the pinned memory property.

            self.bus.push(buffer_view.cpu(), module_name)

        self.copy_events.clear()

        self.memory_pool.reset() # Make the entire pool available for the next iteration

  

    def remove_hooks(self):

        """Removes all attached hooks from the model."""

        for handle in self.hooks.values():

            handle.remove()

        self.hooks.clear()

        logger.info("All hooks have been removed.")
