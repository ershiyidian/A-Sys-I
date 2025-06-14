// src/asys_i/components/data_bus_hpc.py (REWRITTEN)
import logging
import multiprocessing
import multiprocessing.shared_memory
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from asys_i.common.types import (
    ActivationPacket, ComponentID, TensorRef, RunProfile, calculate_checksum,
    DTYPE_TO_CODE_MAP, CODE_TO_DTYPE_MAP
)
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.hpc import c_ext_wrapper, SHM_NAME_PREFIX, MQ_NAME_PREFIX
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MasterConfig

log = logging.getLogger(__name__)

class CppShardedSPMCBus(BaseDataBus):
    def __init__(self, master_config: MasterConfig, monitor: BaseMonitor):
        super().__init__(master_config.data_bus, monitor)
        self.master_config = master_config
        
        instance_id = f"{os.getpid()}_{int(time.time_ns() / 1000)}"
        self.tensor_shm_name = f"{SHM_NAME_PREFIX}tensor_{instance_id}"
        self.mq_name = f"{MQ_NAME_PREFIX}mq_{instance_id}"
        
        log.info(f"Creating HPC DataBus. SHM: {self.tensor_shm_name}, MQ: {self.mq_name}")
        self.cpp_manager = c_ext_wrapper.ShmManager(
            self.tensor_shm_name, self.mq_name,
            int(self.config.shared_memory_size_gb * 1024 * 1024 * 1024),
            self.config.buffer_size_per_shard, create=True
        )
        if not self.cpp_manager.is_valid():
            raise RuntimeError("Failed to initialize C++ ShmManager.")

        self._tensor_data_shm_obj = multiprocessing.shared_memory.SharedMemory(name=self.tensor_shm_name, create=False)
        
        self.consumer_map: Dict[ComponentID, int] = {} # consumer_id -> consumer_shm_id
        self._is_ready = True
        log.info("CppShardedSPMCBus initialized and ready.")

    def push(self, packet: ActivationPacket, timeout: Optional[float] = None) -> bool:
        if not self._is_ready or not isinstance(packet.data, torch.Tensor): return False

        tensor = packet.data
        # ActivationHooker is responsible for moving data to CPU (preferably pinned memory)
        # before pushing to this bus. This assertion makes the expectation explicit.
        assert not tensor.is_cuda, "Input tensor to CppShardedSPMCBus.push() should be on CPU."
        tensor_cpu_numpy = tensor.contiguous().cpu().numpy() # .cpu() is no-op if already on CPU. .contiguous() is safeguard.
        
        checksum = calculate_checksum(tensor_cpu_numpy.tobytes()) if self.config.use_checksum else 0
        dtype_code_val = DTYPE_TO_CODE_MAP.get(tensor.dtype)
        if dtype_code_val is None:
            log.error(f"Unsupported dtype for HPC bus: {tensor.dtype}"); return False
        
        dtype_code = c_ext_wrapper.TorchDtypeCode(dtype_code_val)

        success = self.cpp_manager.push(
            tensor_numpy_array=tensor_cpu_numpy, dtype_code=dtype_code,
            checksum=checksum, layer_idx_numeric=packet.layer_idx_numeric,
            global_step=packet.global_step
        )

        tags = {"bus_type": "cpp_spmc"}
        if not success:
            self.monitor.log_metric("data_bus_drop_count", 1, {**tags, "reason": "cpp_push_fail"})
            return False
        
        self.monitor.log_metric("data_bus_push_count", 1, tags=tags)
        return True
        
    def pull_batch(self, consumer_id: ComponentID, batch_size: Optional[int] = None, timeout: Optional[float] = None) -> List[ActivationPacket]:
        if not self._is_ready: return []
            
        effective_batch_size = batch_size or self.config.pull_batch_size_max
        cpp_metadata_list = self.cpp_manager.pull_batch(effective_batch_size)
        if not cpp_metadata_list: return []

        reconstructed_packets: List[ActivationPacket] = []
        for cpp_meta in cpp_metadata_list:
            try:
                tensor_ref = TensorRef(
                    shm_data_offset=cpp_meta.shm_data_offset,
                    data_size_bytes=cpp_meta.data_size_bytes,
                    dtype_code=int(cpp_meta.dtype_code),
                    ndim=cpp_meta.ndim, shape=tuple(cpp_meta.shape),
                    checksum=cpp_meta.checksum
                )
                packet = ActivationPacket(
                    layer_name=f"layer_idx_{cpp_meta.layer_idx_numeric}",
                    layer_idx_numeric=cpp_meta.layer_idx_numeric,
                    global_step=cpp_meta.global_step,
                    data=tensor_ref, profile=self.master_config.run_profile,
                    timestamp_ns=cpp_meta.timestamp_ns, meta={}
                )
                reconstructed_packets.append(packet)
            except Exception as e:
                log.error(f"Failed to process C++ PacketMetadata: {e}")
                self.monitor.log_metric("data_bus_packet_conversion_error", 1)
        
        self.monitor.log_metric("data_bus_pull_count", len(reconstructed_packets), {"consumer_id": consumer_id})
        return reconstructed_packets

    def register_consumer(self, consumer_id: ComponentID, layer_indices_numeric: List[int]) -> None:
        consumer_shm_id = self.cpp_manager.register_consumer()
        if consumer_shm_id == -1:
            raise RuntimeError(f"Failed to register consumer {consumer_id}: max consumers reached.")
        self.consumer_map[consumer_id] = consumer_shm_id
        log.info(f"Consumer {consumer_id} registered with SHM ID: {consumer_shm_id}")

    def get_stats(self) -> Dict[str, Any]:
        return {"type": "CppShardedSPMCBus", "shm_name": self.tensor_shm_name, "mq_name": self.mq_name}

    def shutdown(self) -> None:
        if not self._is_ready: return
        super().shutdown()
        if hasattr(self, '_tensor_data_shm_obj'): self._tensor_data_shm_obj.close()
        if hasattr(self, 'cpp_manager'): del self.cpp_manager
        log.info("CppShardedSPMCBus resources released.")

    def __getstate__(self):
        return {
            'master_config_dict': self.master_config.model_dump(),
            'monitor': self.monitor,
            'tensor_shm_name': self.tensor_shm_name,
            'mq_name': self.mq_name,
        }

    def __setstate__(self, state):
        self.master_config = MasterConfig.model_validate(state['master_config_dict'])
        self.config = self.master_config.data_bus
        self.monitor = state['monitor']
        self.tensor_shm_name = state['tensor_shm_name']
        self.mq_name = state['mq_name']

        try:
            self.cpp_manager = c_ext_wrapper.ShmManager(
                self.tensor_shm_name, self.mq_name, 0, 0, create=False
            )
            if not self.cpp_manager.is_valid():
                raise RuntimeError("Worker failed to re-initialize C++ ShmManager.")

            self._tensor_data_shm_obj_worker = multiprocessing.shared_memory.SharedMemory(name=self.tensor_shm_name, create=False)
            self._tensor_data_shm_view_for_pull = self._tensor_data_shm_obj_worker.buf
        except Exception as e:
            log.critical(f"Worker ({os.getpid()}) failed during CppShardedSPMCBus unpickling: {e}", exc_info=True)
            self._is_ready = False
            raise

        self.consumer_map = {}
        self._is_ready = True
        log.info(f"CppShardedSPMCBus re-initialized in worker process ({os.getpid()}).")
