# src/asys_i/components/archiver.py
"""
Core Philosophy: Separation, Design for Failure.
Worker process responsible for consuming ActivationPackets from DataBus
 and persisting them to storage (disk/object storage).
 High Cohesion: Purely data archiving.
"""
import asyncio
import logging
import multiprocessing
import os
import time
from typing import Dict, List

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
    logging.warning("pyarrow or aiofiles not installed. Archiver will be disabled.")

from asys_i.common.types import (
    ActivationPacket,
    ComponentID,
    LayerIndex,
)
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.hpc.resource_manager import bind_current_process
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.monitoring.watchdog import RestartableManager
from asys_i.orchestration.config_loader import MasterConfig

log = logging.getLogger(__name__)


# Use this as the Process target function to ensure setup/teardown
def archiver_worker_process_target(
    worker_id: ComponentID,
    config: MasterConfig,
    data_bus: BaseDataBus,  # Must be process-safe/pickleable
    monitor: BaseMonitor,  # Must be process-safe/pickleable
    stop_event: multiprocessing.Event,
    all_layer_indices: List[LayerIndex],
):
    worker = ArchiverWorker(
        worker_id, config, data_bus, monitor, stop_event, all_layer_indices
    )
    try:
        worker.run()
    except Exception:
        log.exception(f"ArchiverWorker {worker_id} crashed:")
        monitor.log_metric(
            "worker_crash_count", 1, tags={"component": worker_id, "type": "archiver"}
        )
        # Let watchdog handle restart
    finally:
        worker.shutdown()
        log.info(f"ArchiverWorker {worker_id} finished.")


class ArchiverWorker:
    """
    A single worker process for archiving activation data.
    Runs an asyncio loop internally for non-blocking IO.
    """

    def __init__(
        self,
        worker_id: ComponentID,
        config: MasterConfig,
        data_bus: BaseDataBus,
        monitor: BaseMonitor,
        stop_event: multiprocessing.Event,
        all_layer_indices: List[LayerIndex],  # Archiver subscribes to all layers
    ):
        self.worker_id = worker_id
        self.config = config
        self.archiver_config = config.archiver
        self.data_bus = data_bus
        self.monitor = monitor
        self.stop_event = stop_event
        self.all_layer_indices = all_layer_indices
        self.device = config.hardware.device
        self.buffer: Dict[LayerIndex, List[ActivationPacket]] = {}
        self.last_flush_time = time.time()
        self.save_dir = os.path.join(config.project.output_dir, "archived_activations")
        self.heartbeat_interval = self.archiver_config.heartbeat_interval_sec
        self._packet_count = 0
        self._bytes_written = 0
        os.makedirs(self.save_dir, exist_ok=True)
        if not PYARROW_AIO_AVAILABLE:
            log.error(
                f"Worker {worker_id}: PyArrow/aiofiles unavailable. Archiving disabled."
            )

    def _setup(self):
        log.info(f"ArchiverWorker {self.worker_id} setting up.")
        bind_current_process(self.config, self.worker_id, self.monitor)
        # Register with data bus to receive data from ALL shards corresponding to layers
        self.data_bus.register_consumer(self.worker_id, self.all_layer_indices)
        self.monitor.register_component(self.worker_id)
        self.monitor.heartbeat(self.worker_id)

    async def _write_buffer_async(
        self, layer_idx: LayerIndex, packets: List[ActivationPacket]
    ):
        if not packets or not PYARROW_AIO_AVAILABLE:
            return
        tags = {"component": self.worker_id, "layer": layer_idx}
        start_time = time.perf_counter()
        log.debug(
            f"Archiver {self.worker_id}: Writing {len(packets)} packets for layer {layer_idx}"
        )
        try:
            # Convert to Arrow Table
            data_list = []
            meta_list = []
            steps = []
            timestamps = []
            total_bytes = 0

            for p in packets:
                tensor = p["data"]
                if isinstance(tensor, torch.Tensor):
                    np_array = tensor.detach().cpu().numpy()
                    data_list.append(np_array)
                    total_bytes += np_array.nbytes
                else:  # Handle TensorRef if necessary, though pull_batch should reconstruct
                    log.warning("Archiver received TensorRef, skipping")
                    continue
                meta_list.append(str(p.get("meta", {})))
                steps.append(p["global_step"])
                timestamps.append(p["timestamp_ns"])

            if not data_list:
                return

            # Create a structured array or list of arrays for Arrow
            # Flatten tensors for simpler storage or use fixed_size_list array
            # Simple approach: store as binary blob or flatten
            _ = [t.flatten() for t in data_list]  # flattened_tensors
            shapes = [str(t.shape) for t in data_list]
            dtypes = [str(data_list[0].dtype)] * len(data_list)  # Assume same dtype

            table = pa.table(
                {
                    "global_step": pa.array(steps, type=pa.int64()),
                    "timestamp_ns": pa.array(timestamps, type=pa.int64()),
                    "layer_idx": pa.array([layer_idx] * len(steps), type=pa.int32()),
                    # 'activation': pa.array(flattened_tensors), # Needs careful type handling
                    "activation_blob": pa.array(
                        [t.tobytes() for t in data_list], type=pa.binary()
                    ),
                    "shape": pa.array(shapes, type=pa.string()),
                    "dtype": pa.array(dtypes, type=pa.string()),
                    "meta": pa.array(meta_list, type=pa.string()),
                }
            )

            filename = f"layer_{layer_idx}_step_{steps[0]}-{steps[-1]}_{int(time.time()*1000)}.parquet"
            filepath = os.path.join(self.save_dir, filename)

            # Use a thread pool for blocking Parquet write or just block the async loop
            # pq.write_table is blocking. Run in executor for true async.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,  # default thread pool
                lambda: pq.write_table(
                    table, filepath, compression=self.archiver_config.compression
                ),
            )
            # async with aiofiles.open(filepath, mode='wb') as f:
            #      # Need async parquet writer, which doesn't exist easily.

            write_latency = (time.perf_counter() - start_time) * 1000
            self.monitor.log_metric(
                "archiver_write_latency_ms", write_latency, tags=tags
            )
            self.monitor.log_metric("archiver_write_packets", len(packets), tags=tags)
            self.monitor.log_metric("archiver_write_bytes", total_bytes, tags=tags)
            self._bytes_written += total_bytes
            log.info(
                f"Archiver {self.worker_id}: Wrote {filepath} ({total_bytes/1e6:.2f} MB)"
            )

        except Exception:
            log.exception(f"Error writing parquet file for layer {layer_idx}:")
            self.monitor.log_metric(
                "archiver_error_count", 1, tags={**tags, "type": "write"}
            )

    async def _flush_buffers(self):
        tasks = []
        buffers_to_flush = self.buffer
        self.buffer = {}  # Clear buffer before writing
        for layer_idx, packets in buffers_to_flush.items():
            if packets:
                tasks.append(self._write_buffer_async(layer_idx, packets))
        if tasks:
            try:
                await asyncio.gather(*tasks)
            except Exception:
                log.exception(
                    f"Archiver {self.worker_id}: Error during asyncio.gather in _flush_buffers"
                )
                self.monitor.log_metric(
                    "archiver_error_count", 1, tags={"component": self.worker_id, "type": "flush_gather"}
                )
        self.last_flush_time = time.time()

    async def run_async_loop(self):
        self._setup()
        last_heartbeat = time.time()
        log.info(f"ArchiverWorker {self.worker_id} async loop started.")

        while not self.stop_event.is_set():
            try:
                # Check heartbeat
                now = time.time()
                if now - last_heartbeat > self.heartbeat_interval:
                    self.monitor.heartbeat(self.worker_id)
                    last_heartbeat = now

                # Pull data (blocking call relative to process, but OK)
                # Use small timeout to allow stop_event check
                batch = self.data_bus.pull_batch(self.worker_id, timeout=1.0)

                if not batch:
                    await asyncio.sleep(0.1)  # Avoid busy-wait if bus is empty
                    # Check flush time even if no new data
                    if (
                        now - self.last_flush_time
                        > self.archiver_config.flush_interval_sec
                    ):
                        await self._flush_buffers()
                    continue

                self._packet_count += len(batch)
                self.monitor.log_metric(
                    "archiver_pull_count",
                    len(batch),
                    tags={"component": self.worker_id},
                )

                # Add to buffer
                buffer_size = 0
                for packet in batch:
                    layer = packet["layer_idx"]
                    if layer not in self.buffer:
                        self.buffer[layer] = []
                    self.buffer[layer].append(packet)
                    buffer_size += 1  # Simplified count

                # Check flush conditions
                time_to_flush = (
                    now - self.last_flush_time > self.archiver_config.flush_interval_sec
                )
                size_to_flush = buffer_size >= self.archiver_config.batch_size
                if time_to_flush or size_to_flush:
                    log.debug(
                        f"Archiver {self.worker_id} flushing buffers (Time: {time_to_flush}, Size: {size_to_flush}, Count: {buffer_size})."
                    )
                    await self._flush_buffers()

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception(f"Error in ArchiverWorker {self.worker_id} loop:")
                self.monitor.log_metric(
                    "archiver_error_count",
                    1,
                    tags={"component": self.worker_id, "type": "loop"},
                )
                await asyncio.sleep(5)  # Backoff on error

        log.info(f"ArchiverWorker {self.worker_id} stopping. Flushing final buffers...")
        await self._flush_buffers()  # Final flush

    def run(self):
        # Entry point for the process
        if not self.archiver_config.enabled or not PYARROW_AIO_AVAILABLE:
            log.warning(
                f"Archiver {self.worker_id} disabled by config or missing deps. Exiting."
            )
            return
        try:
            asyncio.run(self.run_async_loop())
        except KeyboardInterrupt:
            pass
        except Exception: # Catch any other potential error during asyncio.run
            log.exception(f"Critical error in ArchiverWorker {self.worker_id} run method (outside async loop):")
            self.monitor.log_metric("archiver_error_count", 1, tags={"component": self.worker_id, "type": "critical_run"})


    def shutdown(self):
        # Called from finally block in target function
        log.info(
            f"ArchiverWorker {self.worker_id} shutdown. Total packets: {self._packet_count}, Bytes: {self._bytes_written/1e6:.2f} MB"
        )


# --- MANAGER ---
class ArchiverManager(RestartableManager):
    """Manages the lifecycle of the ArchiverWorker process(es). Implements RestartableManager for Watchdog."""

    def __init__(
        self,
        config: MasterConfig,
        data_bus: BaseDataBus,
        monitor: BaseMonitor,
        stop_event: multiprocessing.Event,
        all_layer_indices: List[LayerIndex],
    ):
        self.config = config
        self.data_bus = data_bus
        self.monitor = monitor
        self.stop_event = stop_event
        self.all_layer_indices = all_layer_indices
        self.workers: Dict[ComponentID, multiprocessing.Process] = {}
        self.enabled = config.archiver.enabled and PYARROW_AIO_AVAILABLE
        if not self.enabled:
            log.warning("ArchiverManager is disabled.")

    def _create_and_register_worker(
        self, worker_id: ComponentID
    ) -> multiprocessing.Process:
        process = multiprocessing.Process(
            target=archiver_worker_process_target,
            args=(
                worker_id,
                self.config,
                self.data_bus,
                self.monitor,
                self.stop_event,
                self.all_layer_indices,
            ),
            daemon=True,
            name=worker_id,
        )
        self.workers[worker_id] = process
        self.monitor.register_component(worker_id)
        return process

    def initialize(self):
        if not self.enabled:
            return
        # For simplicity, one archiver worker. Can extend to num_workers.
        worker_id = "archiver_worker_0"
        log.info(f"Initializing ArchiverWorker: {worker_id}")
        self._create_and_register_worker(worker_id)

    def start_all(self):
        if not self.enabled:
            return
        log.info("Starting ArchiverManager workers...")
        for worker_id, process in self.workers.items():
            if not process.is_alive():
                process.start()
                log.info(f"Started process {process.pid} for {worker_id}")

    def stop_all(self, timeout: float = 30.0):
        if not self.enabled:
            return
        log.info("Stopping ArchiverManager workers...")
        self.stop_event.set()
        for worker_id, process in self.workers.items():
            if process.is_alive():
                process.join(timeout)
                if process.is_alive():
                    log.error(
                        f"Worker {worker_id} (PID {process.pid}) did not terminate, forcing."
                    )
                    process.terminate()
        self.workers.clear()

    def get_worker_ids(self) -> List[ComponentID]:
        # Filter out dead processes
        return [wid for wid, proc in self.workers.items() if proc.is_alive()]

    def restart_worker(self, worker_id: ComponentID) -> bool:
        if not self.enabled:
            return False
        if worker_id not in self.workers:
            log.error(f"Cannot restart unknown archiver worker: {worker_id}")
            return False
        old_process = self.workers[worker_id]
        log.warning(
            f"Restarting archiver worker {worker_id} (old PID: {old_process.pid})..."
        )

        if old_process.is_alive():
            old_process.terminate()  # Force stop for restart
            old_process.join(timeout=5.0)

        del self.workers[worker_id]
        new_process = self._create_and_register_worker(worker_id)  # Re-use ID
        new_process.start()
        log.info(
            f"Restarted archiver worker {worker_id} with new PID: {new_process.pid}"
        )
        return True
