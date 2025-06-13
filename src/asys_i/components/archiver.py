# src/asys_i/components/archiver.py (REVISED FROM YOUR LAST INPUT)
import asyncio
import logging
import multiprocessing
import os
import time
from typing import Dict, List, Optional, Any, Tuple # Ensure Tuple for shape
from collections import defaultdict # For self.buffer

import numpy as np
import torch

try:
    import aiofiles
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AIO_AVAILABLE = True
except ImportError:
    PYARROW_AIO_AVAILABLE = False
    pa, pq, aiofiles = None, None, None
    logging.warning("Archiver: pyarrow or aiofiles not installed. Parquet archiving will be disabled.")

from asys_i.common.types import (
    ActivationPacket,
    ComponentID,
    LayerIndex, # Numeric layer index
    TensorRef,
    RunProfile,
    CODE_TO_DTYPE_MAP, # For reconstructing dtype from C++ code
    calculate_checksum,
)
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.hpc.resource_manager import bind_current_process
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MasterConfig
from asys_i.components.managers.base_manager import BaseWorkerManager

from asys_i.hpc import CPP_EXTENSION_AVAILABLE # To check for CppShardedSPMCBus
if CPP_EXTENSION_AVAILABLE:
    from asys_i.components.data_bus_hpc import CppShardedSPMCBus

log = logging.getLogger(__name__)

def archiver_worker_process_target(
    worker_id: ComponentID,
    config: MasterConfig,
    data_bus: BaseDataBus,
    monitor: BaseMonitor,
    stop_event: multiprocessing.Event,
    assigned_numeric_layer_indices: List[LayerIndex], # Archiver gets numeric indices it's responsible for
    shm_info: Optional[Tuple[str, str, int]] = None, # (tensor_shm_name, mq_name, tensor_shm_size_bytes) for HPC
):
    # Seed setting
    torch.manual_seed(config.project.seed + hash(worker_id) % (2**32))
    np.random.seed(config.project.seed + hash(worker_id) % (2**32))

    local_shm_buffer_view: Optional[memoryview] = None
    local_shm_obj_for_cleanup: Optional[multiprocessing.shared_memory.SharedMemory] = None

    if config.run_profile == RunProfile.HPC and isinstance(data_bus, CppShardedSPMCBus):
        try:
            if hasattr(data_bus, '_tensor_data_shm_obj_worker') and \
               data_bus._tensor_data_shm_obj_worker is not None and \
               hasattr(data_bus, '_tensor_data_shm_view_for_pull') and \
               data_bus._tensor_data_shm_view_for_pull is not None:
                
                local_shm_obj_for_cleanup = data_bus._tensor_data_shm_obj_worker
                local_shm_buffer_view = data_bus._tensor_data_shm_view_for_pull
                log.info(f"ArchiverWorker {worker_id} using SHM view from its DataBus instance for SHM: {data_bus.tensor_shm_name}")
            else:
                raise RuntimeError(f"Worker {worker_id} CppDataBus instance missing SHM view.")
        except Exception as e:
            log.critical(f"ArchiverWorker {worker_id}: Error obtaining SHM view for HPC mode: {e}")
            raise

    worker = ArchiverWorker(
        worker_id, config, data_bus, monitor, stop_event, assigned_numeric_layer_indices, local_shm_buffer_view
    )
    try:
        worker.run()
    except Exception: # General catch
        log.exception(f"ArchiverWorker {worker_id} CRASHED in run():")
        monitor.log_metric("worker_crash_count", 1, tags={"component": worker_id, "type": "archiver", "stage":"run"})
    finally:
        try:
            worker.shutdown()
        except Exception:
            log.exception(f"ArchiverWorker {worker_id} CRASHED in shutdown():")
            monitor.log_metric("worker_crash_count", 1, tags={"component": worker_id, "type": "archiver", "stage":"shutdown"})

        if local_shm_obj_for_cleanup:
            try: local_shm_obj_for_cleanup.close()
            except Exception as e: log.error(f"ArchiverWorker {worker_id} error closing SHM: {e}")
        log.info(f"ArchiverWorker {worker_id} process finished.")


class ArchiverWorker:
    def __init__(
        self,
        worker_id: ComponentID,
        config: MasterConfig,
        data_bus: BaseDataBus,
        monitor: BaseMonitor,
        stop_event: multiprocessing.Event,
        assigned_numeric_layer_indices: List[LayerIndex],
        shm_buffer_view_for_read: Optional[memoryview] = None,
    ):
        self.worker_id = worker_id
        self.config = config
        self.archiver_config = config.archiver
        self.data_bus = data_bus
        self.monitor = monitor
        self.stop_event = stop_event
        self.assigned_numeric_layer_indices = assigned_numeric_layer_indices
        self.shm_buffer_view_for_read = shm_buffer_view_for_read

        self.buffer: Dict[LayerIndex, List[ActivationPacket]] = defaultdict(list)
        self.last_flush_time = time.time()
        self.save_dir = os.path.join(config.project.output_dir, "archived_activations_parquet")
        os.makedirs(self.save_dir, exist_ok=True)
        self.heartbeat_interval = self.archiver_config.heartbeat_interval_sec
        self._packet_count = 0
        self._bytes_written = 0
        self.tags = {"worker_id": self.worker_id}

        if not self.archiver_config.enabled:
            log.warning(f"ArchiverWorker {worker_id} is configured as disabled.")
        elif not PYARROW_AIO_AVAILABLE:
            log.error(f"ArchiverWorker {worker_id}: PyArrow/aiofiles unavailable. Archiving forced disable.")
            self.archiver_config.enabled = False

    def _setup(self):
        if not self.archiver_config.enabled: return
        log.info(f"ArchiverWorker {self.worker_id} setting up for numeric layer indices: {self.assigned_numeric_layer_indices}")
        bind_current_process(self.config, self.worker_id, self.monitor)
        self.data_bus.register_consumer(self.worker_id, self.assigned_numeric_layer_indices)
        self.monitor.register_component(self.worker_id)
        self.monitor.heartbeat(self.worker_id)

    def _reconstruct_tensor_from_ref(self, ref: TensorRef) -> torch.Tensor: # Same as SAETrainer's
        if self.config.run_profile != RunProfile.HPC or self.shm_buffer_view_for_read is None:
            raise RuntimeError(f"ArchiverWorker {self.worker_id}: Tensor reconstruction from ref called inappropriately.")

        torch_dtype = CODE_TO_DTYPE_MAP.get(ref.dtype_code)
        if torch_dtype is None: raise ValueError(f"Invalid dtype_code {ref.dtype_code}.")
        
        np_dtype_for_view = np.dtype('bfloat16') if torch_dtype == torch.bfloat16 else np.dtype(torch_dtype.name)
        item_bytes = np_dtype_for_view.itemsize
        expected_elements = ref.data_size_bytes // item_bytes
        if ref.data_size_bytes % item_bytes != 0:
            raise ValueError(f"TensorRef size {ref.data_size_bytes} not multiple of itemsize {item_bytes}")
        
        elements_from_shape = np.prod(ref.shape).item() if ref.shape else 0
        if elements_from_shape != expected_elements:
            raise ValueError(f"Shape {ref.shape} numel ({elements_from_shape}) mismatch with size/itemsize derived numel ({expected_elements})")

        tensor_shm_slice = self.shm_buffer_view_for_read[ref.shm_data_offset : ref.shm_data_offset + ref.data_size_bytes]
        numpy_array = np.frombuffer(tensor_shm_slice, dtype=np_dtype_for_view, count=expected_elements).reshape(ref.shape)
        tensor = torch.from_numpy(numpy_array)

        if self.config.data_bus.use_checksum:
            if calculate_checksum(tensor_shm_slice) != ref.checksum:
                self.monitor.log_metric("archiver_checksum_error_count", 1, self.tags)
                raise ValueError(f"Archiver {self.worker_id}: Checksum mismatch for tensor at SHM offset {ref.shm_data_offset}")
        return tensor

    async def _write_buffer_async(self, layer_idx_numeric: LayerIndex, packets: List[ActivationPacket]):
        # (Same async Parquet writing logic as your reconstructed version, ensure schema matches packet fields)
        if not packets or not PYARROW_AIO_AVAILABLE: return
        tags_layer = {**self.tags, "layer_numeric_idx": layer_idx_numeric}
        start_time_write = time.perf_counter()

        # Prepare data for Pa.Table
        field_data = defaultdict(list)
        total_bytes_in_batch = 0

        for p in packets:
            tensor_to_write: Optional[torch.Tensor] = None
            if isinstance(p.data, torch.Tensor):
                tensor_to_write = p.data.cpu() # Ensure CPU for numpy conversion
            elif isinstance(p.data, TensorRef) and self.config.run_profile == RunProfile.HPC:
                try:
                    tensor_to_write = self._reconstruct_tensor_from_ref(p.data).cpu()
                except Exception as e:
                    log.error(f"{self.worker_id} archiver failed to reconstruct tensor from ref for packet {p}: {e}")
                    self.monitor.log_metric("archiver_reconstruct_error", 1, tags_layer)
                    continue
            if tensor_to_write is None: continue

            np_array = tensor_to_write.numpy()
            total_bytes_in_batch += np_array.nbytes

            field_data["global_step"].append(p.global_step)
            field_data["timestamp_ns"].append(p.timestamp_ns)
            field_data["layer_numeric_idx"].append(p.layer_idx_numeric) # Changed from layer_name to numeric
            field_data["layer_name_fqn"].append(p.layer_name) # Keep the original FQN/friendly name
            field_data["activation_blob"].append(np_array.tobytes())
            field_data["shape"].append(str(np_array.shape))
            field_data["dtype"].append(str(np_array.dtype)) # NumPy dtype string
            field_data["meta_json"].append(str(p.meta)) # Assuming meta is simple dict

        if not field_data["global_step"]: return # No valid packets processed

        try:
            # Create PyArrow Table
            pa_table_dict = {}
            pa_table_dict["global_step"] = pa.array(field_data["global_step"], type=pa.int64())
            pa_table_dict["timestamp_ns"] = pa.array(field_data["timestamp_ns"], type=pa.int64())
            pa_table_dict["layer_numeric_idx"] = pa.array(field_data["layer_numeric_idx"], type=pa.int32())
            pa_table_dict["layer_name_fqn"] = pa.array(field_data["layer_name_fqn"], type=pa.string())
            pa_table_dict["activation_blob"] = pa.array(field_data["activation_blob"], type=pa.binary())
            pa_table_dict["shape"] = pa.array(field_data["shape"], type=pa.string())
            pa_table_dict["dtype"] = pa.array(field_data["dtype"], type=pa.string())
            pa_table_dict["meta_json"] = pa.array(field_data["meta_json"], type=pa.string())
            
            table = pa.Table.from_pydict(pa_table_dict)
            
            # Filename generation
            # Use the first packet's layer_name_fqn if available and consistent for the batch
            batch_layer_name_part = field_data["layer_name_fqn"][0].replace(".", "_").replace("/", "_") if field_data["layer_name_fqn"] else f"numericidx_{layer_idx_numeric}"
            file_ts = int(time.time()*1000)
            filename = f"activations_{batch_layer_name_part}_steps_{field_data['global_step'][0]}-{field_data['global_step'][-1]}_{file_ts}.parquet"
            filepath = os.path.join(self.save_dir, filename)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: pq.write_table(table, filepath, compression=self.archiver_config.compression))

            write_latency_ms = (time.perf_counter() - start_time_write) * 1000
            self.monitor.log_metric("archiver_write_latency_ms", write_latency_ms, tags_layer)
            self.monitor.log_metric("archiver_write_packet_count", len(packets), tags_layer) # Corrected metric name
            self.monitor.log_metric("archiver_write_byte_count", total_bytes_in_batch, tags_layer) # Corrected metric name
            self._bytes_written += total_bytes_in_batch
            log.info(f"Archiver {self.worker_id}: Wrote {filepath} ({total_bytes_in_batch/1e6:.2f} MB)")

        except Exception as e: # Catch errors during table creation or write
            log.exception(f"Error creating/writing parquet file for layer index {layer_idx_numeric}:")
            self.monitor.log_metric("archiver_error_count", 1, {**tags_layer, "type": "write_parquet", "error": str(type(e).__name__)})


    async def _flush_buffers(self):
        # (Same as your reconstructed version)
        tasks = []
        buffers_to_flush = self.buffer
        self.buffer = defaultdict(list)
        for numeric_idx, pkts in buffers_to_flush.items():
            if pkts: tasks.append(self._write_buffer_async(numeric_idx, pkts))
        if tasks:
            try: await asyncio.gather(*tasks)
            except Exception as e:
                log.exception(f"Archiver {self.worker_id}: Error in _flush_buffers gather: {e}")
                self.monitor.log_metric("archiver_error_count", 1, {**self.tags, "type": "flush_gather"})
        self.last_flush_time = time.time()

    async def run_async_loop(self):
        # (Same main loop structure as your reconstructed version)
        self._setup()
        if not self.archiver_config.enabled:
            log.info(f"ArchiverWorker {self.worker_id} exiting as archiver is disabled."); return
        last_heartbeat_ts = time.time()
        log.info(f"ArchiverWorker {self.worker_id} async loop started.")

        while not self.stop_event.is_set():
            try:
                now = time.time()
                if now - last_heartbeat_ts > self.heartbeat_interval:
                    self.monitor.heartbeat(self.worker_id); last_heartbeat_ts = now

                pulled_packets = self.data_bus.pull_batch(self.worker_id, batch_size=self.archiver_config.batch_size, timeout=0.5) # Shorter timeout for responsiveness

                if not pulled_packets:
                    await asyncio.sleep(0.05) # Small sleep if no data
                    if (now - self.last_flush_time > self.archiver_config.flush_interval_sec) and any(self.buffer.values()):
                        await self._flush_buffers()
                    continue

                self._packet_count += len(pulled_packets)
                self.monitor.log_metric("archiver_pull_packet_count", len(pulled_packets), self.tags) # Corrected metric name

                current_total_items_in_buffer = sum(len(lst) for lst in self.buffer.values())
                for packet in pulled_packets:
                    self.buffer[packet.layer_idx_numeric].append(packet) # Buffer by numeric index
                    current_total_items_in_buffer +=1
                
                time_to_flush = (now - self.last_flush_time > self.archiver_config.flush_interval_sec)
                size_to_flush = current_total_items_in_buffer >= self.archiver_config.batch_size
                if (time_to_flush or size_to_flush) and any(self.buffer.values()):
                    log.debug(f"Archiver {self.worker_id} flushing: TimeCond={time_to_flush}, SizeCond={size_to_flush}, Items={current_total_items_in_buffer}")
                    await self._flush_buffers()
            except asyncio.CancelledError: log.info(f"ArchiverWorker {self.worker_id} async loop cancelled."); break
            except Exception as e:
                log.exception(f"Error in ArchiverWorker {self.worker_id} loop:")
                self.monitor.log_metric("archiver_error_count", 1, {**self.tags, "type": "loop_exception", "error": str(type(e).__name__)})
                await asyncio.sleep(1) # Shorter backoff

        log.info(f"ArchiverWorker {self.worker_id} stopping. Flushing final buffers...")
        if any(self.buffer.values()): await self._flush_buffers()


    def run(self): # Entry point for the process
        # (Same as your reconstructed version)
        if not self.archiver_config.enabled:
            log.warning(f"Archiver {self.worker_id} disabled. Exiting run method."); return
        try: asyncio.run(self.run_async_loop())
        except KeyboardInterrupt: log.info(f"ArchiverWorker {self.worker_id} received KeyboardInterrupt.")
        except Exception as e:
            log.exception(f"Critical error in ArchiverWorker {self.worker_id} run method (outside async):")
            self.monitor.log_metric("archiver_error_count", 1, {**self.tags, "type": "critical_run_method", "error": str(type(e).__name__)})

    def shutdown(self):
        # (Same as your reconstructed version)
        log.info(f"ArchiverWorker {self.worker_id} shutdown. Total packets processed: {self._packet_count}, Total bytes written: {self._bytes_written/1e6:.2f} MB")
        self.shm_buffer_view_for_read = None # Release memoryview


class ArchiverManager(BaseWorkerManager):
    def __init__(
        self,
        config: MasterConfig,
        data_bus: BaseDataBus,
        monitor: BaseMonitor,
        stop_event: multiprocessing.Event,
        activation_hooker: ActivationHooker, # Get FQN->NumIdx map from Hooker
    ):
        super().__init__(config, data_bus, monitor, stop_event, "archiver_worker")
        self.archiver_config = config.archiver
        self.fqn_to_numeric_idx_map = activation_hooker._layer_name_to_idx
        
        self.shm_info_for_workers: Optional[Tuple[str, str, int]] = None # (tensor_shm_name, mq_name, tensor_shm_size_bytes)
        if config.run_profile == RunProfile.HPC and isinstance(data_bus, CppShardedSPMCBus):
             self.shm_info_for_workers = (
                data_bus.tensor_shm_name,
                data_bus.mq_name,
                int(data_bus.config.shared_memory_size_gb * 1024 * 1024 * 1024)
            )

        if not self.archiver_config.enabled:
            log.warning("ArchiverManager initialized, but Archiver is disabled by configuration.")
        elif not PYARROW_AIO_AVAILABLE:
            log.error("ArchiverManager: PyArrow/aiofiles not available. Archiving will be force-disabled.")
            self.archiver_config.enabled = False

    def _assign_layers_to_workers(self) -> Dict[int, List[LayerIndex]]: # Returns worker_proc_idx -> list_of_numeric_layer_indices
        # (Same as your reconstructed - one archiver worker gets all numeric_idx)
        all_numeric_indices = list(self.fqn_to_numeric_idx_map.values())
        if not all_numeric_indices and self.archiver_config.enabled:
             log.warning("ArchiverManager: No layers found from Hooker's map. Archiver will be idle even if enabled.")
        # Assign all numeric indices to worker 0 (single archiver worker model)
        return {0: all_numeric_indices} if all_numeric_indices else {}


    def _create_worker_process(self, worker_id_str: ComponentID, assigned_numeric_indices: List[LayerIndex]) -> multiprocessing.Process:
        # (Passes shm_info_for_workers for HPC now)
        process = multiprocessing.Process(
            target=archiver_worker_process_target,
            args=(
                worker_id_str,
                self.config,
                self.data_bus,
                self.monitor,
                self.stop_event,
                assigned_numeric_indices,
                self.shm_info_for_workers if self.config.run_profile == RunProfile.HPC else None,
            ),
            daemon=True,
            name=worker_id_str,
        )
        return process

    def initialize_workers(self):
        # (Largely same as your reconstructed version, ensures only 1 worker for archiver)
        if not self.archiver_config.enabled: return

        worker_layer_assignments = self._assign_layers_to_workers()
        # Archiver manager currently supports only one worker (worker_process_idx=0)
        assigned_indices_for_worker_0 = worker_layer_assignments.get(0, [])

        if assigned_indices_for_worker_0:
            worker_id_str = f"{self.worker_name_prefix}_0"
            log.info(f"Initializing ArchiverWorker: {worker_id_str} for numeric layer indices: {assigned_indices_for_worker_0}")
            process = self._create_worker_process(worker_id_str, assigned_indices_for_worker_0)
            self.workers[worker_id_str] = process
            self.worker_assignments[worker_id_str] = assigned_indices_for_worker_0
            self.monitor.register_component(worker_id_str)
        else:
            log.warning("ArchiverManager: No numeric layer indices assigned to the archiver worker (possibly no layers hooked). It will be idle.")

    # start_all, stop_all, get_worker_ids, restart_worker inherited from BaseWorkerManager.
