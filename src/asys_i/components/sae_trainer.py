# src/asys_i/components/sae_trainer.py
"""
 Core Philosophy: Separation, Design for Failure, Predictability.
 Manages and executes the training of Sparse Autoencoders in separate processes.
 High Cohesion: SAE Training logic and worker management.
"""
import logging
import multiprocessing
import time
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional
import torch
import torch.optim as optim
import numpy as np

from asys_i.common.types import (
     ActivationPacket, ComponentID, LayerIndex, RunProfile, GlobalStep
 )
from asys_i.orchestration.config_loader import MasterConfig
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.components.sae_model import SparseAutoencoder
from asys_i.monitoring.watchdog import RestartableManager
from asys_i.hpc.resource_manager import bind_current_process

log = logging.getLogger(__name__)

# Process target function
def trainer_worker_process_target(
       worker_id: ComponentID,
       config: MasterConfig,
       data_bus: BaseDataBus,
       monitor: BaseMonitor,
       stop_event: multiprocessing.Event,
       assigned_layer_indices: List[LayerIndex],
 ):
      # Ensure GPU is visible/initialized correctly in child process
      torch.cuda.init()
      # Set seeds for reproducibility in child process
      torch.manual_seed(config.project.get("seed", 42) + hash(worker_id) % (2**32))
      np.random.seed(config.project.get("seed", 42) + hash(worker_id) % (2**32))
      
      worker = SAETrainerWorker(worker_id, config, data_bus, monitor, stop_event, assigned_layer_indices)
      try:
          worker.run()
      except Exception as e:
          log.exception(f"SAETrainerWorker {worker_id} crashed:")
          monitor.log_metric("worker_crash_count", 1, tags={"component": worker_id, "type": "trainer"})
      finally:
           worker.shutdown()
           log.info(f"SAETrainerWorker {worker_id} finished.")


class SAETrainerWorker:
    """ A single worker process training one or more SAEs."""
    def __init__(self,
                  worker_id: ComponentID,
                  config: MasterConfig,
                  data_bus: BaseDataBus,
                  monitor: BaseMonitor,
                  stop_event: multiprocessing.Event,
                  assigned_layer_indices: List[LayerIndex]
                  ):
         self.worker_id = worker_id
         self.config = config
         self.trainer_config = config.sae_trainer
         self.model_config = config.sae_model
         self.data_bus = data_bus
         self.monitor = monitor
         self.stop_event = stop_event
         self.layer_indices = assigned_layer_indices
         self.device = config.hardware.device
         
         self.models: Dict[LayerIndex, SparseAutoencoder] = {}
         self.optimizers: Dict[LayerIndex, optim.Optimizer] = {}
         self.schedulers: Dict[LayerIndex, Any] = {} # Optional
         self.steps: Dict[LayerIndex, GlobalStep] = defaultdict(int)
         self.checkpoint_dir = os.path.join(config.project.checkpoint_dir, "sae")
         os.makedirs(self.checkpoint_dir, exist_ok=True)
         
         self.heartbeat_interval = self.trainer_config.heartbeat_interval_sec
         self.tags = {"worker": self.worker_id}

    def _setup(self):
        log.info(f"SAETrainerWorker {self.worker_id} setting up for layers: {self.layer_indices}")
        bind_current_process(self.config, self.worker_id, self.monitor)
        self.data_bus.register_consumer(self.worker_id, self.layer_indices)
        self.monitor.register_component(self.worker_id)
        
        OptClass = getattr(optim, self.trainer_config.optimizer, optim.AdamW)
        
        for layer_idx in self.layer_indices:
             log.debug(f"{self.worker_id}: Initializing SAE and Optimizer for layer {layer_idx}")
             model = SparseAutoencoder(self.model_config)
             model.to(self.device)
             # TODO: Load checkpoint if exists
             optimizer = OptClass(
                  model.parameters(),
                  lr=self.trainer_config.learning_rate,
                  betas=(self.trainer_config.adam_beta1, self.trainer_config.adam_beta2),
                  weight_decay=self.trainer_config.weight_decay
             )
             self.models[layer_idx] = model
             self.optimizers[layer_idx] = optimizer
             # TODO: Init scheduler
        self.monitor.heartbeat(self.worker_id)


    def _prepare_batch(self, packets: List[ActivationPacket]) -> Dict[LayerIndex, torch.Tensor]:
         """ Group packets by layer and stack tensors on device """
         grouped_tensors: Dict[LayerIndex, List[torch.Tensor]] = defaultdict(list)
         for packet in packets:
             if isinstance(packet['data'], torch.Tensor):
                grouped_tensors[packet['layer_idx']].append(packet['data'])
             else:
                 log.warning(f"{self.worker_id} received non-tensor data, skipping.")
                 self.monitor.log_metric("trainer_data_error_count", 1, tags=self.tags)

         final_batches: Dict[LayerIndex, torch.Tensor] = {}
         for layer_idx, tensor_list in grouped_tensors.items():
              if layer_idx in self.models: # Only process layers this worker manages
                  try:
                     # Ensure all tensors have same shape before stacking
                     # Stack vs Cat: stack adds a dim, cat concatenates along existing dim. Use cat for batch.
                      batch_tensor = torch.cat(tensor_list, dim=0).to(self.device, non_blocking=True)
                      final_batches[layer_idx] = batch_tensor
                  except Exception as e:
                       log.error(f"{self.worker_id}: Error stacking tensors for layer {layer_idx}: {e}")
                       self.monitor.log_metric("trainer_batch_error_count", 1, tags={**self.tags, "layer": layer_idx})
         return final_batches
         
    def _save_checkpoint(self, layer_idx: LayerIndex, step: GlobalStep):
         path = os.path.join(self.checkpoint_dir, f"sae_layer_{layer_idx}_step_{step}.pt")
         try:
             self.models[layer_idx].save_weights(path)
             log.info(f"{self.worker_id}: Saved checkpoint for layer {layer_idx} step {step} to {path}")
             self.monitor.log_metric("trainer_checkpoint_count", 1, tags={**self.tags, "layer": layer_idx})
         except Exception as e:
              log.error(f"{self.worker_id}: Failed to save checkpoint for layer {layer_idx}: {e}")
              self.monitor.log_metric("trainer_error_count", 1, tags={**self.tags, "layer": layer_idx, "type": "checkpoint"})

    def _train_step(self, layer_idx: LayerIndex, batch_tensor: torch.Tensor):
          model = self.models[layer_idx]
          optimizer = self.optimizers[layer_idx]
          step = self.steps[layer_idx]
          tags = {**self.tags, "layer": layer_idx}

          start_time = time.perf_counter()
          optimizer.zero_grad()
          
          output = model(batch_tensor)
          loss = output["loss"]
          
          loss.backward()
          # Optional: gradient clipping
          optimizer.step()
          # Optional: scheduler.step()
          
          self.steps[layer_idx] += 1
          latency = (time.perf_counter() - start_time) * 1000

          # Prepare metrics (detach tensors)
          metrics = {k: v.detach().item() for k, v in output.items() if isinstance(v, torch.Tensor) and v.ndim == 0}
          metrics["train_step_latency_ms"] = latency
          metrics["learning_rate"] = optimizer.param_groups[0]['lr']
          self.monitor.log_metrics(metrics, step=step, tags=tags)
          
          if step > 0 and step % self.trainer_config.save_interval_steps == 0:
               self._save_checkpoint(layer_idx, step)
          # TODO: Dead neuron check/revival logic

    def run(self):
         self._setup()
         last_heartbeat = time.time()
         log.info(f"SAETrainerWorker {self.worker_id} training loop started.")
         
         while not self.stop_event.is_set():
              try:
                    now = time.time()
                    if now - last_heartbeat > self.heartbeat_interval:
                         self.monitor.heartbeat(self.worker_id)
                         last_heartbeat = now
                         
                    # Use timeout to periodically check stop_event
                    batch_packets = self.data_bus.pull_batch(
                         self.worker_id, 
                         batch_size=self.trainer_config.batch_size, 
                         timeout=1.0
                    )

                    if not batch_packets:
                         time.sleep(0.05) # Small sleep if no data
                         continue
                    
                    self.monitor.log_metric("trainer_pull_count", len(batch_packets), tags=self.tags)
                    batches = self._prepare_batch(batch_packets)
                    
                    for layer_idx, tensor_batch in batches.items():
                          try:
                              self._train_step(layer_idx, tensor_batch)
                          except Exception as e:
                               # Error in one layer shouldn't stop others
                               log.exception(f"{self.worker_id}: Error during training step for layer {layer_idx}:")
                               self.monitor.log_metric("trainer_error_count", 1, tags={**self.tags, "layer": layer_idx, "type": "train_step"})

              except Exception as e:
                   log.exception(f"Error in SAETrainerWorker {self.worker_id} loop:")
                   self.monitor.log_metric("trainer_error_count", 1, tags={**self.tags, "type": "loop"})
                   time.sleep(5) # Backoff

         log.info(f"SAETrainerWorker {self.worker_id} stopping.")
         # Final save?

    def shutdown(self):
         log.info(f"SAETrainerWorker {self.worker_id} shutdown.")
         # free resources
         self.models.clear()
         self.optimizers.clear()
         if torch.cuda.is_available():
              torch.cuda.empty_cache()

# --- MANAGER ---
class SAETrainerManager(RestartableManager):
      """ Manages lifecycle of SAETrainerWorker processes. Implements RestartableManager. """
      def __init__(self,
                   config: MasterConfig,
                   data_bus: BaseDataBus,
                   monitor: BaseMonitor,
                   stop_event: multiprocessing.Event,
                   all_layer_indices: List[LayerIndex]):
           self.config = config
           self.trainer_config = config.sae_trainer
           self.data_bus = data_bus
           self.monitor = monitor
           self.stop_event = stop_event
           self.all_layer_indices = sorted(all_layer_indices)
           self.workers: Dict[ComponentID, multiprocessing.Process] = {}
           self.worker_assignments: Dict[ComponentID, List[LayerIndex]] = {}
           log.info(f"SAETrainerManager initialized for {self.trainer_config.num_workers} workers, layers: {self.all_layer_indices}")

      def _assign_layers(self) -> Dict[int, List[LayerIndex]]:
           """ Distributes layers among workers """
           num_workers = self.trainer_config.num_workers
           assignments: Dict[int, List[LayerIndex]] = defaultdict(list)
           # Simple round-robin
           for i, layer_idx in enumerate(self.all_layer_indices):
                assignments[i % num_workers].append(layer_idx)
           log.info(f"Layer assignments: {dict(assignments)}")
           return dict(assignments)

      def _create_and_register_worker(self, worker_id: ComponentID, layers: List[LayerIndex]) -> multiprocessing.Process:
           process = multiprocessing.Process(
                target=trainer_worker_process_target,
                args=(
                     worker_id, self.config, self.data_bus, self.monitor,
                     self.stop_event, layers
                ),
                daemon=True,
                name=worker_id
           )
           self.workers[worker_id] = process
           self.worker_assignments[worker_id] = layers
           self.monitor.register_component(worker_id)
           return process

      def initialize(self):
           assignments = self._assign_layers()
           for i in range(self.trainer_config.num_workers):
                worker_id = f"trainer_worker_{i}"
                layers = assignments.get(i, [])
                if layers:
                    log.info(f"Initializing SAETrainerWorker: {worker_id} for layers {layers}")
                    self._create_and_register_worker(worker_id, layers)
                else:
                     log.warning(f"Worker {worker_id} has no layers assigned.")
                     
      # --- Rest is identical in structure to ArchiverManager ---
      def start_all(self):
          log.info("Starting SAETrainerManager workers...")
          for worker_id, process in self.workers.items():
               if not process.is_alive():
                   process.start()
                   log.info(f"Started process {process.pid} for {worker_id}")
      
      def stop_all(self, timeout: float = 60.0): # Give more time for checkpoint save
          log.info("Stopping SAETrainerManager workers...")
          self.stop_event.set()
          for worker_id, process in self.workers.items():
               if process.is_alive():
                    process.join(timeout)
                    if process.is_alive():
                         log.error(f"Worker {worker_id} (PID {process.pid}) did not terminate, forcing.")
                         process.terminate()
          self.workers.clear()
          self.worker_assignments.clear()

      def get_worker_ids(self) -> List[ComponentID]:
           return [wid for wid, proc in self.workers.items() if proc.is_alive()]

      def restart_worker(self, worker_id: ComponentID) -> bool:
          if worker_id not in self.workers:
               log.error(f"Cannot restart unknown trainer worker: {worker_id}")
               return False
          old_process = self.workers[worker_id]
          assigned_layers = self.worker_assignments.get(worker_id, [])
          log.warning(f"Restarting trainer worker {worker_id} (old PID: {old_process.pid}) for layers {assigned_layers}...")
          
          if old_process.is_alive():
               old_process.terminate()
               old_process.join(timeout=10.0)
          
          del self.workers[worker_id]
          del self.worker_assignments[worker_id]
          # Re-create using original layers
          new_process = self._create_and_register_worker(worker_id, assigned_layers)
          new_process.start()
          log.info(f"Restarted trainer worker {worker_id} with new PID: {new_process.pid}")
          return True

