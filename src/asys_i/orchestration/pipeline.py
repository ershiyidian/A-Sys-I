# src/asys_i/orchestration/pipeline.py
"""
Core Philosophy: ALL.
(UPGRADED) The central conductor of the A-Sys-I system.
Instantiates, connects, starts, and stops all components in the correct order.
Handles 'auto' config resolution at runtime.
High Cohesion: System lifecycle management.
"""
import logging
import time
import multiprocessing
import threading
import os
from typing import Dict, Any, Optional
import psutil # For CPU core detection
import torch # For model dimension detection

from asys_i.common.types import RunProfile, ComponentID
from asys_i.orchestration.config_loader import MasterConfig, load_config
# Interfaces & Factories
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.monitoring.monitor_factory import create_monitor
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.components.data_bus_factory import create_data_bus
# Components
from asys_i.components.ppo_host import PPOHostProcess
from asys_i.components.activation_hooker import ActivationHooker
from asys_i.components.sae_trainer import SAETrainerManager
from asys_i.components.archiver import ArchiverManager
from asys_i.monitoring.watchdog import Watchdog

log = logging.getLogger(__name__)

# Shared dictionary for heartbeats across processes for watchdog thread
# Must be created in the main process before forking/spawning
HEARTBEAT_MANAGER = multiprocessing.Manager()
SHARED_HEARTBEATS = HEARTBEAT_MANAGER.dict()

# Monkey-patch BaseMonitor to use shared dict
# A bit hacky, but keeps interface clean and prevents circular dependencies if Monitor directly imported Managers
def shared_heartbeat(self, component_id: ComponentID):
      SHARED_HEARTBEATS[component_id] = time.time()
      # log.debug(f"Heartbeat from {component_id} via shared dict") # Noisy
      # Call original logic too if it exists and does more (e.g. log metric)
      if hasattr(self, '_original_heartbeat'):
           self._original_heartbeat(component_id)

def get_shared_heartbeats(self) -> Dict[ComponentID, float]:
       return SHARED_HEARTBEATS.copy()
       
def register_shared_component(self, component_id: ComponentID):
      if component_id not in SHARED_HEARTBEATS:
          log.info(f"Registering component for heartbeat monitoring: {component_id}")
          self.heartbeat(component_id)


class ExperimentPipeline:
    """ Orchestrates the entire experiment lifecycle. """
    def __init__(self, config: MasterConfig):
        self.config = config
        self.monitor: Optional[BaseMonitor] = None
        self.data_bus: Optional[BaseDataBus] = None
        self.ppo_host: Optional[PPOHostProcess] = None
        self.hooker: Optional[ActivationHooker] = None
        self.trainer_manager: Optional[SAETrainerManager] = None
        self.archiver_manager: Optional[ArchiverManager] = None
        self.watchdog: Optional[Watchdog] = None
        # Use multiprocessing.Event for cross-process signalling
        self.stop_event = multiprocessing.Event() 
        self._is_setup = False
        self._is_shutting_down = False
        
        # Apply patch for shared heartbeats BEFORE creating monitor
        if not hasattr(BaseMonitor, '_original_heartbeat'):
             BaseMonitor._original_heartbeat = BaseMonitor.heartbeat # type: ignore
        BaseMonitor.heartbeat = shared_heartbeat # type: ignore
        BaseMonitor.get_heartbeats = get_shared_heartbeats # type: ignore
        BaseMonitor.register_component = register_shared_component # type: ignore
        log.info("BaseMonitor patched for shared heartbeats across processes.")

    def _resolve_auto_configs(self):
        """
        UPGRADE: Intelligently resolves 'auto' settings after initial components
        are created but before dependent components are.
        """
        log.info("Resolving 'auto' configurations...")
        
        # 1. Auto-detect d_in for SAE
        if self.config.sae_model.d_in == "auto":
            assert self.ppo_host, "PPOHost must be initialized before auto-detecting d_in"
            host_model = self.ppo_host.get_model()
            detected_dim = -1
            try:
                # Heuristic to get hidden size. Adapts to different model types.
                # Prioritize standard HuggingFace config attribute
                if hasattr(host_model, 'config') and hasattr(host_model.config, 'hidden_size'):
                    detected_dim = host_model.config.hidden_size
                elif hasattr(host_model, 'config') and hasattr(host_model.config, 'd_model'):
                    detected_dim = host_model.config.d_model
                # Fallback: inspect model layers (e.g., first Linear layer's input features)
                # This is a heuristic and might need refinement for complex models
                elif hasattr(host_model, 'layers') and isinstance(host_model.layers, torch.nn.ModuleList) and len(host_model.layers) > 0:
                     first_layer_module = host_model.layers[0] # Try to find a common layer block
                     # Look for a linear layer within it
                     first_linear = next((m for m in first_layer_module.modules() if isinstance(m, torch.nn.Linear)), None)
                     if first_linear:
                          detected_dim = first_linear.in_features
                else: # Generic search
                    first_linear = next((m for m in host_model.modules() if isinstance(m, torch.nn.Linear)), None)
                    if first_linear:
                        detected_dim = first_linear.in_features
                    else:
                        raise RuntimeError("Cannot automatically determine model hidden dimension (d_in). No 'hidden_size', 'd_model', or common Linear layers found.")
                
                if detected_dim <= 0:
                     raise RuntimeError(f"Detected d_in value {detected_dim} is invalid.")

                log.info(f"Auto-detected host model d_in: {detected_dim}. Updating SAE config.")
                self.config.sae_model.d_in = detected_dim
                self.monitor.log_metric("sae_model_d_in_auto_detected", detected_dim)
            except Exception as e:
                log.error(f"Failed to auto-detect d_in: {e}. Please specify `sae_model.d_in` in config if 'auto' fails.")
                raise

        # 2. Auto-generate resource_manager CPU map (Linux only, for HPC)
        res_config = self.config.resource_manager
        if (self.config.run_profile == RunProfile.HPC and 
            res_config.apply_bindings and 
            res_config.allocation_strategy == "auto" and 
            not res_config.cpu_affinity_map): # Only auto-generate if no map provided
            
            if not os.name == 'posix' or not hasattr(psutil, 'cpu_count'):
                log.warning("Auto CPU affinity requires Linux and psutil. Skipping auto-binding.")
                self.monitor.log_metric("resource_binding_auto_skip", 1, tags={"reason": "not_linux_or_psutil"})
                return

            log.info("Auto-generating CPU affinity map...")
            num_logical_cores = psutil.cpu_count(logical=True)
            if num_logical_cores is None or num_logical_cores == 0:
                log.error("Could not determine number of logical CPU cores. Cannot auto-assign.")
                self.monitor.log_metric("resource_binding_auto_error", 1, tags={"reason": "no_cpu_count"})
                return
            
            num_trainers = self.config.sae_trainer.num_workers
            num_archivers = 1 if self.config.archiver.enabled else 0 # Assume one archiver worker
            
            # Heuristic for core allocation:
            # Try to reserve 1-2 cores for system/main pipeline, 1 for archiver if enabled, rest for trainers.
            # This is a basic strategy; for NUMA-aware systems, more sophisticated logic is needed.
            
            # Start with all available cores
            remaining_cores = list(range(num_logical_cores))
            core_map = {}
            
            # Assign Host Process (main pipeline thread runs here)
            host_min_cores = 1
            host_actual_cores = min(host_min_cores, len(remaining_cores))
            if host_actual_cores > 0:
                core_map["host_process"] = remaining_cores[:host_actual_cores]
                remaining_cores = remaining_cores[host_actual_cores:]
                log.debug(f"Assigned {host_actual_cores} cores to host_process: {core_map['host_process']}")

            # Assign Archiver Worker
            if num_archivers > 0 and remaining_cores:
                archiver_min_cores = 1
                archiver_actual_cores = min(archiver_min_cores, len(remaining_cores))
                if archiver_actual_cores > 0:
                    core_map["archiver_worker_0"] = remaining_cores[:archiver_actual_cores]
                    remaining_cores = remaining_cores[archiver_actual_cores:]
                    log.debug(f"Assigned {archiver_actual_cores} cores to archiver_worker_0: {core_map['archiver_worker_0']}")

            # Assign Trainer Workers
            if remaining_cores:
                cores_per_trainer = max(1, len(remaining_cores) // num_trainers)
                if cores_per_trainer == 0: # Not enough cores for at least 1 per trainer
                     log.warning(f"Not enough remaining cores ({len(remaining_cores)}) to assign at least 1 core per trainer. Trainers may share cores.")
                     cores_per_trainer = 1 # Each trainer gets at least 1, will overlap
                     
                for i in range(num_trainers):
                    worker_id = f"trainer_worker_{i}"
                    start_idx = i * cores_per_trainer
                    if start_idx >= len(remaining_cores):
                         break # No more cores to assign
                    
                    assigned_cores = remaining_cores[start_idx : min(start_idx + cores_per_trainer, len(remaining_cores))]
                    if assigned_cores:
                        core_map[worker_id] = assigned_cores
                        log.debug(f"Assigned {len(assigned_cores)} cores to {worker_id}: {assigned_cores}")
                    else:
                        log.warning(f"No cores could be assigned to {worker_id}. It will run on unpinned cores.")
            else:
                 log.warning("No remaining cores for SAE trainer workers. They will run on unpinned cores.")
                 
            self.config.resource_manager.cpu_affinity_map = core_map
            self.monitor.log_metric("resource_binding_auto_success", 1, tags={"allocated_cores": len(core_map)})
            log.info(f"Generated CPU affinity map: {core_map}")
            
        else:
             log.info("CPU affinity map either not requested for auto-generation or manually specified.")

    def setup(self):
        if self._is_setup:
            log.warning("Pipeline setup already called.")
            return
        log.info(">>> PIPELINE SETUP STARTED <<<")
        start_time = time.time()
        
        try:
             # ORDER IS NOW CRITICAL FOR AUTO-CONFIG RESOLUTION
             # 1. Monitor (dependency for all others)
             self.monitor = create_monitor(self.config)
             # Log full config (including "auto" values that will be resolved)
             # The actual resolved values will be logged when each component binds/initializes
             self.monitor.log_hyperparams(self.config.dict()) 

             # 2. Data Bus (dependency for hooker, trainer, archiver)
             self.data_bus = create_data_bus(self.config, self.monitor)

             # 3. Host Process (MUST be created before auto-config resolution for d_in)
             self.ppo_host = PPOHostProcess(self.config, self.monitor, self.stop_event)

             # 4. *** INTELLIGENCE UPGRADE STEP: RESOLVE "AUTO" CONFIGS ***
             self._resolve_auto_configs()

             # 5. Hooker (now uses potentially updated d_in from config)
             self.hooker = ActivationHooker(
                  self.ppo_host.get_model(), self.config, self.data_bus, self.monitor
             )
             
             all_layers = self.config.hook.layers_to_hook
             if not all_layers:
                 log.warning("No layers configured to hook. SAE training and archiving will be idle.")

             # 6. Trainer Manager (now uses fully resolved config, including d_in and cpu_affinity)
             self.trainer_manager = SAETrainerManager(
                  self.config, self.data_bus, self.monitor, self.stop_event, all_layers
             )
             self.trainer_manager.initialize()

             # 7. Archiver Manager (now uses fully resolved config, including cpu_affinity)
             self.archiver_manager = ArchiverManager(
                   self.config, self.data_bus, self.monitor, self.stop_event, all_layers
             )
             self.archiver_manager.initialize()

             # 8. Watchdog (needs monitor and managers)
             managed = {}
             if self.trainer_manager: managed["trainer"] = self.trainer_manager
             if self.archiver_manager: managed["archiver"] = self.archiver_manager
             self.watchdog = Watchdog(
                  self.monitor, self.config.monitor, managed_components=managed
             )
             # Register host with monitor so watchdog *could* see it (though it's not managed for restart)
             self.monitor.register_component(self.ppo_host.component_id)

             self._is_setup = True
             duration = time.time() - start_time
             log.info(f">>> PIPELINE SETUP COMPLETE ({duration:.2f}s) <<<")
             self.monitor.log_metric("pipeline_setup_duration_sec", duration)

        except Exception as e:
             log.exception("Error during pipeline setup:")
             if self.monitor: self.monitor.log_metric("pipeline_error_count", 1, tags={"stage": "setup"})
             self.shutdown() # Clean up anything partially created
             raise

    def run(self):
         if not self._is_setup:
             raise RuntimeError("Pipeline.setup() must be called before run().")
         if self.stop_event.is_set():
              log.warning("Run called but stop_event is already set, perhaps from previous shutdown.")
              return
         log.info(">>> PIPELINE RUN STARTED <<<")
         start_time = time.time()
         try:
             assert self.watchdog and self.trainer_manager and self.archiver_manager and self.ppo_host
             
             self.watchdog.start()
             self.trainer_manager.start_all()
             self.archiver_manager.start_all()
             
             # --- Blocking Call ---
             # Host loop runs in the main thread/process
             self.ppo_host.run_training_loop(self.hooker) 
             # ---------------------
             
             # If loop finishes normally, signal stop
             self.stop_event.set()
             
             duration = time.time() - start_time
             log.info(f">>> PIPELINE RUN FINISHED ({duration:.2f}s) <<<")
             self.monitor.log_metric("pipeline_run_duration_sec", duration)
             
         except Exception as e:
              log.exception("Error during pipeline run:")
              if self.monitor: self.monitor.log_metric("pipeline_error_count", 1, tags={"stage": "run"})
              self.stop_event.set() # Ensure shutdown occurs
              raise # Re-raise for script handler
        

    def shutdown(self):
         if self._is_shutting_down:
              log.warning("Pipeline shutdown already in progress.")
              return
         self._is_shutting_down = True
         log.info(">>> PIPELINE SHUTDOWN STARTED <<<")
         start_time = time.time()
         self.stop_event.set() # Ensure all components receive stop signal

         # Shutdown in reverse order of dependency/startup
         try:
            if self.ppo_host:       self.ppo_host.shutdown()
            if self.hooker:         self.hooker.shutdown() # detach
            if self.archiver_manager: self.archiver_manager.stop_all()
            if self.trainer_manager:  self.trainer_manager.stop_all()
            if self.watchdog and self.watchdog.is_alive(): self.watchdog.stop()
            # Give workers time to finish/flush after stop_event
            time.sleep(2) 
            if self.data_bus:       self.data_bus.shutdown() # Release SHM!
            if self.monitor:        self.monitor.shutdown() # Flush metrics!
            SHARED_HEARTBEATS.clear()
            # HEARTBEAT_MANAGER.shutdown() # Be careful shutting down manager, might kill other shared objects
            
            duration = time.time() - start_time
            log.info(f">>> PIPELINE SHUTDOWN COMPLETE ({duration:.2f}s) <<<")
         except Exception as e:
              log.exception("Error during pipeline shutdown:")
              # Monitor might be down, rely on logging
              pass # Suppress further exceptions during shutdown

         self._is_setup = False
         self._is_shutting_down = False

