import atexit
import logging
import time
from typing import Optional, List, Tuple

import numpy as np
import torch

from asys_i.common.types import DTYPE_TO_CODE_MAP
from asys_i.hpc.cpp_ringbuffer import CppRingBuffer  # C++ extension

logger = logging.getLogger(__name__)

class CppShardedSPMCBus:
    """
    A sharded Single-Producer-Multiple-Consumer (SPMC) bus for high-performance
    data transfer using C++ shared memory ring buffers.
    """

    def __init__(self,
                 num_shards: int,
                 shard_size_bytes: int,
                 meta_queue_name: str,
                 shm_name_prefix: str,
                 spin_lock_timeout_us: int = 1000):
        self.num_shards = num_shards
        self.shard_size_bytes = shard_size_bytes
        self.meta_queue_name = meta_queue_name
        self.shm_name_prefix = shm_name_prefix
        self.spin_lock_timeout_us = spin_lock_timeout_us

        self._producer: Optional[CppRingBuffer.Producer] = None
        # Flag to prevent multiple shutdown calls
        self._is_shutdown = False
        # Register cleanup function to be called on interpreter exit
        atexit.register(self.shutdown)

    def initialize_producer(self):
        """Initializes and returns the producer instance."""
        if self._producer is not None:
            logger.warning("Producer is already initialized.")
            return

        logger.info(f"Initializing SPMC producer with {self.num_shards} shards...")
        try:
            self._producer = CppRingBuffer.Producer(
                self.num_shards,
                self.shard_size_bytes,
                self.meta_queue_name.encode('utf-8'),
                self.shm_name_prefix.encode('utf-8'),
                self.spin_lock_timeout_us
            )
            logger.info("SPMC producer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize C++ SPMC Producer: {e}")
            raise

    def push(self, tensor: torch.Tensor, metadata: str):
        """Pushes a tensor and metadata to the bus."""
        if self._producer is None:
            raise RuntimeError("Producer has not been initialized. Call initialize_producer() first.")
        if self._is_shutdown:
            raise RuntimeError("Cannot push to a shutdown bus.")
            
        tensor_key = f"torch.{tensor.dtype}".split('.')[-1]
        dtype_str = f"torch.{tensor_key}"
        
        if dtype_str not in DTYPE_TO_CODE_MAP:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")
        
        dtype_code = DTYPE_TO_CODE_MAP[dtype_str]
        shape = list(tensor.shape)
        
        # The C++ extension expects a contiguous tensor's data pointer
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        try:
            self._producer.push(
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                dtype_code,
                shape,
                metadata
            )
        except Exception as e:
            logger.error(f"Error during producer push: {e}")
            # Depending on the error, we might want to handle it differently
            # e.g., if the bus is full, we could log and drop, or block.
            # The C++ implementation currently might throw if a lock fails.
            raise

    def shutdown(self):
        """
        Gracefully shuts down the producer, releasing IPC resources.
        This method is idempotent and registered with atexit.
        """
        if self._is_shutdown:
            return
            
        # Set flag immediately to prevent race conditions
        self._is_shutdown = True

        if self._producer is not None:
            logger.info("Shutting down SPMC producer and releasing IPC resources...")
            try:
                # The C++ destructor handles the cleanup.
                # Setting the producer to None will trigger it.
                self._producer = None
                logger.info("SPMC producer shutdown complete.")
            except Exception as e:
                # Log error, but don't re-raise as this is often called
                # during shutdown sequences where exceptions can be problematic.
                logger.error(f"An error occurred during producer shutdown: {e}", exc_info=True)
    
    def __enter__(self):
        self.initialize_producer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # The shutdown is guaranteed to be called by atexit,
        # but we call it here for prompt cleanup when used as a context manager.
        self.shutdown()

# Note: The Consumer class would have a similar structure with atexit registration
# for its own cleanup logic if it were managed in Python.
# Since it's part of a separate process, its own lifecycle management is key.
