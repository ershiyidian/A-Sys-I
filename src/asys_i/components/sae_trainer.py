# src/asys_i/components/sae_trainer.py (REVISED FROM YOUR LAST INPUT - NUMPY DTYPE FIX)
import logging
import multiprocessing
import os
import platform # For OS specific checks if any (though not directly used here now)
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Tuple # Ensure Tuple for shape

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from asys_i.common.types import (
    ActivationPacket,
    ComponentID,
    GlobalStep,
    LayerIndex, # Numeric layer index
    TensorRef,
    RunProfile,
    CODE_TO_DTYPE_MAP, # For reconstructing dtype from C++ code
    calculate_checksum,
)
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.components.sae_model import SparseAutoencoder
from asys_i.hpc.resource_manager import bind_current_process
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MasterConfig
from asys_i.components.managers.base_manager import BaseWorkerManager

# Import CppShardedSPMCBus for type checking for SHM info
from asys_i.hpc import CPP_EXTENSION_AVAILABLE
if CPP_EXTENSION_AVAILABLE:
    from asys_i.components.data_bus_hpc import CppShardedSPMCBus # Used for isinstance check

log = logging.getLogger(__name__)

def trainer_worker_process_target(
    worker_id: ComponentID,
    config: MasterConfig,
    data_bus: BaseDataBus,
    monitor: BaseMonitor,
    stop_event: multiprocessing.Event,
    assigned_layers_map: Dict[str, LayerIndex], # FQN_path (key for user) -> numeric_idx (key for model dict)
    shm_info: Optional[Tuple[str, int, int]] = None, # (tensor_shm_name, mq_name, tensor_shm_size_bytes)
                                                    # Note: CppDataBus __getstate__ now only passes names.
                                                    # This target needs to know how to re-open if it's not the original CppBus object.
                                                    # Or, better, CppBus __setstate__ re-establishes connection.
                                                    # Let's assume data_bus passed here for worker is already "live" via __setstate__.
                                                    # The shm_info is then for data_bus to provide access to its raw buffer.
):
    if config.hardware.device != "cpu" and torch.cuda.is_available():
        try: torch.cuda.init()
        except RuntimeError as e:
            log.warning(f"SAETrainerWorker {worker_id}: CUDA init failed ({e}). Defaulting to CPU.")
            # Modifying config object in a child process doesn't affect parent or other children.
            # The effective device for this worker will be CPU.
            config = config.model_copy(update={"hardware": {"device": "cpu"}})


    torch.manual_seed(config.project.seed + hash(worker_id) % (2**32))
    np.random.seed(config.project.seed + hash(worker_id) % (2**32))

    # For HPC mode, the worker needs to map the tensor data SHM segment for reading.
    # The data_bus object passed here (if CppShardedSPMCBus) should provide a way
    # to get this mapping or the necessary SHM view.
    # Let's assume CppShardedSPMCBus.__setstate__ prepares a direct SHM buffer view for workers.
    
    # This local_shm_buffer_view is what the worker uses for zero-copy reads.
    # It's obtained from the data_bus instance specific to this worker process.
    local_shm_buffer_view: Optional[memoryview] = None
    local_shm_obj_for_cleanup: Optional[multiprocessing.shared_memory.SharedMemory] = None

    if config.run_profile == RunProfile.HPC and isinstance(data_bus, CppShardedSPMCBus):
        try:
            # The worker's data_bus instance (after unpickling) should have re-established
            # its connection to the SHM and can provide a view or the raw object.
            # Let's assume CppShardedSPMCBus has a method like get_worker_shm_view()
            # or its _tensor_data_shm_obj_worker and _tensor_data_shm_view_for_pull are accessible.
            if hasattr(data_bus, '_tensor_data_shm_obj_worker') and \
               data_bus._tensor_data_shm_obj_worker is not None and \
               hasattr(data_bus, '_tensor_data_shm_view_for_pull') and \
               data_bus._tensor_data_shm_view_for_pull is not None:
                
                local_shm_obj_for_cleanup = data_bus._tensor_data_shm_obj_worker # Keep ref for close()
                local_shm_buffer_view = data_bus._tensor_data_shm_view_for_pull
                log.info(f"SAETrainerWorker {worker_id} using SHM view from its DataBus instance for SHM: {data_bus.tensor_shm_name}")
            else:
                log.error(f"SAETrainerWorker {worker_id}: CppShardedSPMCBus instance in worker does not have a valid SHM view. Zero-copy will fail.")
                # This is a critical setup failure for HPC.
                raise RuntimeError(f"Worker {worker_id} could not obtain SHM view from its DataBus.")

        except Exception as e:
            log.critical(f"SAETrainerWorker {worker_id}: Error obtaining SHM view for HPC mode: {e}")
            raise


    worker = SAETrainerWorker(
        worker_id, config, data_bus, monitor, stop_event, assigned_layers_map, local_shm_buffer_view
    )
    try:
        worker.run()
    except Exception: # General catch for logging before process exit
        log.exception(f"SAETrainerWorker {worker_id} CRASHED in run():")
        monitor.log_metric("worker_crash_count", 1, tags={"component": worker_id, "type": "trainer", "stage": "run"})
    finally:
        try:
            worker.shutdown()
        except Exception:
            log.exception(f"SAETrainerWorker {worker_id} CRASHED in shutdown():")
            monitor.log_metric("worker_crash_count", 1, tags={"component": worker_id, "type": "trainer", "stage": "shutdown"})
        
        if local_shm_obj_for_cleanup:
            try:
                local_shm_obj_for_cleanup.close()
                log.info(f"SAETrainerWorker {worker_id} closed its SHM object.")
            except Exception as e:
                log.error(f"SAETrainerWorker {worker_id} error closing its SHM object: {e}")
        log.info(f"SAETrainerWorker {worker_id} process finished.")


class SAETrainerWorker:
    def __init__(
        self,
        worker_id: ComponentID,
        config: MasterConfig,
        data_bus: BaseDataBus,
        monitor: BaseMonitor,
        stop_event: multiprocessing.Event,
        assigned_layers_map: Dict[str, LayerIndex], # FQN -> numeric_idx
        shm_buffer_view_for_read: Optional[memoryview] = None,
    ):
        self.worker_id = worker_id
        self.config = config
        self.trainer_config = config.sae_trainer
        self.model_config = config.sae_model # d_in should be int here
        self.data_bus = data_bus
        self.monitor = monitor
        self.stop_event = stop_event
        self.assigned_layers_map = assigned_layers_map
        self.device = config.hardware.device
        self.shm_buffer_view_for_read = shm_buffer_view_for_read

        self.models: Dict[LayerIndex, SparseAutoencoder] = {} # Keyed by numeric_idx
        self.optimizers: Dict[LayerIndex, optim.Optimizer] = {}
        self.steps: Dict[LayerIndex, GlobalStep] = defaultdict(int)
        self.checkpoint_dir = os.path.join(config.project.checkpoint_dir, "sae_models") # More specific
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.heartbeat_interval = self.trainer_config.heartbeat_interval_sec
        self.tags = {"worker_id": self.worker_id} # Standardize tag key

    def _setup(self):
        log.info(f"SAETrainerWorker {self.worker_id} setting up for layers (FQN->NumIdx): {self.assigned_layers_map}")
        bind_current_process(self.config, self.worker_id, self.monitor)
        numeric_indices_for_bus_registration = list(self.assigned_layers_map.values())
        self.data_bus.register_consumer(self.worker_id, numeric_indices_for_bus_registration)
        self.monitor.register_component(self.worker_id)

        OptClass = getattr(optim, self.trainer_config.optimizer, optim.AdamW)

        for fqn_path, numeric_idx in self.assigned_layers_map.items():
            log.debug(f"{self.worker_id}: Initializing SAE for layer FQN '{fqn_path}' (NumIdx {numeric_idx})")
            if not isinstance(self.model_config.d_in, int): # Should have been resolved by pipeline
                raise ValueError(f"SAE d_in is not an integer ({self.model_config.d_in}) during worker setup for {fqn_path}.")
            
            current_sae_config = self.model_config.model_copy() # Make a copy to avoid modifying shared config
            # TODO: If per-layer SAE configs are introduced, load them here based on fqn_path or numeric_idx

            model = SparseAutoencoder(current_sae_config)
            model.to(self.device)
            optimizer = OptClass(
                model.parameters(),
                lr=self.trainer_config.learning_rate,
                betas=(self.trainer_config.adam_beta1, self.trainer_config.adam_beta2),
                weight_decay=self.trainer_config.weight_decay,
            )
            self.models[numeric_idx] = model
            self.optimizers[numeric_idx] = optimizer
        self.monitor.heartbeat(self.worker_id)

    def _reconstruct_tensor_from_ref(self, ref: TensorRef) -> torch.Tensor:
        if self.config.run_profile != RunProfile.HPC or self.shm_buffer_view_for_read is None:
            raise RuntimeError(f"SAETrainerWorker {self.worker_id}: Tensor reconstruction from ref called inappropriately (not HPC or no SHM view).")

        torch_dtype = CODE_TO_DTYPE_MAP.get(ref.dtype_code)
        if torch_dtype is None:
            raise ValueError(f"Invalid dtype_code {ref.dtype_code} in TensorRef.")

        # Handle NumPy dtype compatibility, especially for bfloat16
        if torch_dtype == torch.bfloat16:
            np_dtype_for_view = np.dtype('bfloat16') # Requires NumPy 1.21+ and compatible build
        else:
            np_dtype_for_view = np.dtype(torch_dtype.name)

        item_bytes = np_dtype_for_view.itemsize
        expected_elements = ref.data_size_bytes // item_bytes
        if ref.data_size_bytes % item_bytes != 0:
            raise ValueError(f"TensorRef data_size_bytes {ref.data_size_bytes} not multiple of item_bytes {item_bytes} for dtype {torch_dtype}")

        # Calculate total elements from shape to verify against expected_elements
        total_elements_from_shape = 1
        for dim_size in ref.shape:
            total_elements_from_shape *= dim_size
        
        if total_elements_from_shape != expected_elements:
             raise ValueError(f"Mismatch: elements from shape ({total_elements_from_shape}) vs elements from size/itemsize ({expected_elements}). Ref: {ref}")


        tensor_shm_slice = self.shm_buffer_view_for_read[ref.shm_data_offset : ref.shm_data_offset + ref.data_size_bytes]
        numpy_array = np.frombuffer(tensor_shm_slice, dtype=np_dtype_for_view, count=expected_elements).reshape(ref.shape)
        tensor = torch.from_numpy(numpy_array) # Zero-copy

        if self.config.data_bus.use_checksum:
            if calculate_checksum(tensor_shm_slice) != ref.checksum: # Checksum the raw memoryview slice
                self.monitor.log_metric("trainer_checksum_error_count", 1, {**self.tags, "layer_numeric_idx": ref.dtype_code }) # Layer index not in ref easily
                raise ValueError(f"SAETrainer {self.worker_id}: Checksum mismatch for tensor at SHM offset {ref.shm_data_offset}")
        return tensor

    def _prepare_batch(self, packets: List[ActivationPacket]) -> Dict[LayerIndex, torch.Tensor]:
        # (Largely same as your reconstructed version, ensure it uses numeric_idx from packet)
        grouped_tensors: Dict[LayerIndex, List[torch.Tensor]] = defaultdict(list)
        for packet in packets:
            tensor_data: Optional[torch.Tensor] = None
            if isinstance(packet.data, torch.Tensor): # Consumer mode
                tensor_data = packet.data
            elif isinstance(packet.data, TensorRef) and self.config.run_profile == RunProfile.HPC:
                try:
                    tensor_data = self._reconstruct_tensor_from_ref(packet.data)
                except Exception as e:
                    log.error(f"{self.worker_id} failed to reconstruct tensor from ref {packet.data}: {e}")
                    self.monitor.log_metric("trainer_reconstruct_error", 1, {**self.tags, "layer_numeric_idx": packet.layer_idx_numeric})
                    continue
            else: # Should not happen with typed DataBus
                log.warning(f"{self.worker_id} received unexpected data type in packet: {type(packet.data)}")
                self.monitor.log_metric("trainer_data_error_count", 1, self.tags)
                continue
            
            if tensor_data is not None:
                grouped_tensors[packet.layer_idx_numeric].append(tensor_data)

        final_batches: Dict[LayerIndex, torch.Tensor] = {}
        for numeric_idx, tensor_list in grouped_tensors.items():
            if numeric_idx in self.models: # Ensure this worker is responsible
                if not tensor_list: continue
                try:
                    # TODO: Optimization: pre-allocate large GPU buffer and copy into slices,
                    # instead of torch.cat which does new allocation + copies.
                    # For now, torch.cat is simpler.
                    batch_tensor = torch.cat(tensor_list, dim=0).to(self.device, non_blocking=True)
                    final_batches[numeric_idx] = batch_tensor
                except Exception as e:
                    log.error(f"{self.worker_id}: Error stacking tensors for layer index {numeric_idx}: {e}")
                    self.monitor.log_metric("trainer_batch_error_count", 1, {**self.tags, "layer_numeric_idx": numeric_idx})
        return final_batches

    def _save_checkpoint(self, layer_idx_numeric: LayerIndex, step: GlobalStep):
        # (Same as your reconstructed version, using fqn_for_filename from assigned_layers_map)
        fqn_for_filename = f"numeric_idx_{layer_idx_numeric}" # Fallback
        for fqn, idx_val in self.assigned_layers_map.items():
            if idx_val == layer_idx_numeric:
                fqn_for_filename = fqn.replace(".","_") # Sanitize FQN for filename
                break
        
        path = os.path.join(self.checkpoint_dir, f"sae_model_{fqn_for_filename}_step_{step}.pt")
        try:
            self.models[layer_idx_numeric].save_weights(path)
            log.info(f"{self.worker_id}: Saved checkpoint for layer NumIdx {layer_idx_numeric} (FQN part: {fqn_for_filename}) step {step} to {path}")
            self.monitor.log_metric("trainer_checkpoint_count", 1, {**self.tags, "layer_numeric_idx": layer_idx_numeric})
        except Exception as e:
            log.error(f"{self.worker_id}: Failed to save checkpoint for layer NumIdx {layer_idx_numeric}: {e}")
            self.monitor.log_metric("trainer_error_count", 1, {**self.tags, "layer_numeric_idx": layer_idx_numeric, "type": "checkpoint"})

    def _train_step(self, layer_idx_numeric: LayerIndex, batch_tensor: torch.Tensor):
        # (Same as your reconstructed version, includes periodic W_dec normalization)
        model = self.models[layer_idx_numeric]
        optimizer = self.optimizers[layer_idx_numeric]
        current_step = self.steps[layer_idx_numeric] # Use local step for this SAE
        tags_train = {**self.tags, "layer_numeric_idx": layer_idx_numeric}

        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True) # More memory efficient
        output = model(batch_tensor)
        loss = output["loss"]
        loss.backward()
        # Optional: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if current_step > 0 and current_step % self.trainer_config.normalize_weights_interval == 0:
            with torch.no_grad():
                model.W_dec.data = F.normalize(model.W_dec.data, p=2, dim=0)
            self.monitor.log_metric("sae_decoder_weights_normalized", 1, tags=tags_train)

        self.steps[layer_idx_numeric] += 1
        latency_ms = (time.perf_counter() - start_time) * 1000

        metrics_to_log = {k: v.detach().item() for k, v in output.items() if isinstance(v, torch.Tensor) and v.ndim == 0}
        metrics_to_log["train_step_latency_ms"] = latency_ms
        metrics_to_log["learning_rate"] = optimizer.param_groups[0]["lr"]
        # Log with the SAE's own step count for its learning curve
        self.monitor.log_metrics(metrics_to_log, step=current_step, tags=tags_train)

        if current_step > 0 and current_step % self.trainer_config.save_interval_steps == 0:
            self._save_checkpoint(layer_idx_numeric, current_step)

    def run(self):
        # (Same main loop structure as your reconstructed version)
        self._setup()
        last_heartbeat = time.time()
        log.info(f"SAETrainerWorker {self.worker_id} training loop started. Device: {self.device}")

        while not self.stop_event.is_set():
            try:
                now = time.time()
                if now - last_heartbeat > self.heartbeat_interval:
                    self.monitor.heartbeat(self.worker_id)
                    last_heartbeat = now

                batch_packets = self.data_bus.pull_batch(
                    self.worker_id,
                    batch_size=self.trainer_config.batch_size,
                    timeout=1.0,
                )
                if not batch_packets: time.sleep(0.01); continue # Reduced sleep

                self.monitor.log_metric("trainer_pull_packet_count", len(batch_packets), self.tags)
                batches_by_layer = self._prepare_batch(batch_packets)

                for layer_idx_numeric, tensor_batch in batches_by_layer.items():
                    if tensor_batch.numel() == 0: # Skip empty batches
                        log.warning(f"Skipping empty tensor batch for layer_idx_numeric {layer_idx_numeric}")
                        continue
                    try:
                        self._train_step(layer_idx_numeric, tensor_batch)
                    except Exception: # Catch per-layer training error
                        log.exception(f"{self.worker_id}: Error during training step for layer index {layer_idx_numeric}:")
                        self.monitor.log_metric("trainer_error_count", 1, {**self.tags, "layer_numeric_idx": layer_idx_numeric, "type": "train_step"})
            except Exception: # Catch main loop error
                log.exception(f"Error in SAETrainerWorker {self.worker_id} main loop:")
                self.monitor.log_metric("trainer_error_count", 1, {**self.tags, "type": "loop"})
                time.sleep(self.config.monitor.heartbeat_check_interval_sec / 3) # Backoff

        log.info(f"SAETrainerWorker {self.worker_id} stopping. Performing final saves...")
        for numeric_idx, step_count in self.steps.items():
            if numeric_idx in self.models: # Ensure model still exists
                self._save_checkpoint(numeric_idx, step_count)

    def shutdown(self):
        # (Same as your reconstructed version)
        log.info(f"SAETrainerWorker {self.worker_id} shutting down. Releasing resources.")
        self.models.clear()
        self.optimizers.clear()
        self.shm_buffer_view_for_read = None # Release memoryview
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class SAETrainerManager(BaseWorkerManager):
    def __init__(
        self,
        config: MasterConfig,
        data_bus: BaseDataBus,
        monitor: BaseMonitor,
        stop_event: multiprocessing.Event,
        # layers_to_hook_config: Dict[str, str], # FQN map from HookConfig
        activation_hooker: ActivationHooker, # Pass hooker to get its FQN->NumIdx map
    ):
        super().__init__(config, data_bus, monitor, stop_event, "trainer_worker")
        self.trainer_config = config.sae_trainer
        # This map is critical: FQN (user-facing unique ID) -> numeric_idx (internal array index)
        # It's generated by ActivationHooker.attach() based on iteration order.
        self.fqn_to_numeric_idx_map: Dict[str, LayerIndex] = activation_hooker._layer_name_to_idx
        
        self.shm_info_for_workers: Optional[Tuple[str, str, int]] = None
        if config.run_profile == RunProfile.HPC and isinstance(data_bus, CppShardedSPMCBus):
            self.shm_info_for_workers = (
                data_bus.tensor_shm_name,
                data_bus.mq_name,
                int(data_bus.config.shared_memory_size_gb * 1024 * 1024 * 1024)
            )

    def _assign_layers_to_workers(self) -> Dict[int, Dict[str, LayerIndex]]:
        # (Same as your reconstructed version - assigns FQN->NumIdx map portions to workers)
        num_workers = self.trainer_config.num_workers
        assignments: Dict[int, Dict[str, LayerIndex]] = defaultdict(dict) # worker_process_idx -> {fqn: num_idx}
        
        fqn_paths_sorted = sorted(self.fqn_to_numeric_idx_map.keys()) # Consistent assignment order
        if not fqn_paths_sorted:
            log.warning("SAETrainerManager: No layers found in fqn_to_numeric_idx_map from Hooker. No trainers will be assigned work.")
            return {}

        for i, fqn_path in enumerate(fqn_paths_sorted):
            worker_process_idx_for_layer = i % num_workers
            numeric_idx = self.fqn_to_numeric_idx_map[fqn_path]
            assignments[worker_process_idx_for_layer][fqn_path] = numeric_idx
        
        log.info(f"SAE Trainer layer assignments (WorkerProcIdx -> {{FQN: NumIdx}}): {dict(assignments)}")
        return dict(assignments)

    def _create_worker_process(
        self, worker_id_str: ComponentID, assigned_layers_map_for_worker: Dict[str, LayerIndex]
    ) -> multiprocessing.Process:
        # (Passes shm_info_for_workers for HPC now)
        process = multiprocessing.Process(
            target=trainer_worker_process_target,
            args=(
                worker_id_str,
                self.config,
                self.data_bus, # This DataBus object will be pickled
                self.monitor,
                self.stop_event,
                assigned_layers_map_for_worker,
                self.shm_info_for_workers if self.config.run_profile == RunProfile.HPC else None,
            ),
            daemon=True, # Ensure processes exit if main crashes (though graceful shutdown is preferred)
            name=worker_id_str,
        )
        return process

    def initialize_workers(self):
        # (Largely same as your reconstructed version)
        worker_layer_assignments = self._assign_layers_to_workers()
        if not worker_layer_assignments and self.trainer_config.num_workers > 0 and self.fqn_to_numeric_idx_map:
            log.error("SAETrainerManager: Layer assignment resulted in no work for any worker, but layers and workers were configured. Check assignment logic.")
            return

        for worker_process_idx in range(self.trainer_config.num_workers):
            worker_id_str = f"{self.worker_name_prefix}_{worker_process_idx}" # e.g., trainer_worker_0
            layers_map_for_this_worker = worker_layer_assignments.get(worker_process_idx, {})
            
            if layers_map_for_this_worker:
                log.info(f"Initializing SAETrainerWorker: {worker_id_str} for layers (FQN->NumIdx) {layers_map_for_this_worker}")
                process = self._create_worker_process(worker_id_str, layers_map_for_this_worker)
                self.workers[worker_id_str] = process
                # Store the assignment map for this worker, needed for restart
                self.worker_assignments[worker_id_str] = layers_map_for_this_worker 
                self.monitor.register_component(worker_id_str)
            else:
                log.warning(f"SAE Trainer Worker {worker_id_str} has no layers assigned. It will be idle.")

    # start_all, stop_all, get_worker_ids, restart_worker are inherited from BaseWorkerManager.
    # The BaseWorkerManager._create_worker_process is now abstract, so this class's
    # _create_worker_process will be used by restart_worker via self.
