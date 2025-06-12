"""
Core Philosophy: Observability-First, Config-Driven.
Defines the fundamental data structures and types used across the A-Sys-I system.
Ensures type consistency and clarity (High Readability).
High Cohesion: All core types in one place.
"""
import time
from typing import Any, Dict, Optional, TypedDict, Union
from enum import Enum
import torch

# Type Aliases for clarity
LayerIndex = int
GlobalStep = int
WorkerID = str
ComponentID = str
# HPC: Reference (e.g., offset, key) to tensor in shared memory
TensorRef = str 
# Unique ID for a consumer group (e.g. "trainer_0", "archiver") to manage shard subscription
ConsumerID = str 

class RunProfile(str, Enum):
    """Defines the execution mode."""
    HPC = "HPC"
    CONSUMER = "CONSUMER"
    BENCHMARK = "BENCHMARK"

    def __str__(self) -> str:
        return self.value

class MonitorType(str, Enum):
     """Defines the monitoring backend type."""
     PROMETHEUS = "prometheus"
     CSV_TENSORBOARD = "csv_tensorboard"
     LOGGING_ONLY = "logging"
     NONE = "none"
     
     def __str__(self) -> str:
        return self.value

class DataBusType(str, Enum):
    """Defines the DataBus implementation type"""
    CPP_SHARDED_SPMC = "cpp_sharded_spmc" # HPC
    PYTHON_QUEUE = "python_queue" # CONSUMER
    # e.g., REDIS = "redis"
    
    def __str__(self) -> str:
        return self.value

# Core Data Structure: The unit of information transfer
# Using TypedDict for structural typing and clarity
class ActivationPacket(TypedDict):
    """
    The fundamental data packet transferred from Hooker to Consumers via DataBus.
    Represents activations captured from a specific layer at a specific step.
    """
    layer_idx: LayerIndex
    global_step: GlobalStep
     # The actual activation data or a reference to it in shared memory
    data: Union[torch.Tensor, TensorRef] 
    # Profile this packet was generated under (useful for consumers)
    profile: RunProfile 
    timestamp_ns: int  # Nanoseconds timestamp for precise latency measurement
    sequence_id: Optional[int] # e.g., PPO generation sequence ID
    token_position: Optional[int] # Position of token generating this activation
    batch_element_idx: Optional[int] # Which element in the host model batch
     # Any other metadata, e.g., quantization scale, compression ratio, original shape/dtype
    meta: Dict[str, Any]

def create_activation_packet(
        layer_idx: LayerIndex,
        global_step: GlobalStep,
        data: Union[torch.Tensor, TensorRef],
        profile: RunProfile,
        meta: Optional[Dict[str, Any]] = None,
         **kwargs: Any
) -> ActivationPacket:
    """Helper factory for creating packets with timestamp."""
    return ActivationPacket(
         layer_idx=layer_idx,
         global_step=global_step,
         data=data,
         profile=profile,
         timestamp_ns=time.time_ns(),
         sequence_id=kwargs.get("sequence_id"),
         token_position=kwargs.get("token_position"),
         batch_element_idx=kwargs.get("batch_element_idx"),
         meta=meta if meta is not None else {},
    )

# --- Add init file ---
