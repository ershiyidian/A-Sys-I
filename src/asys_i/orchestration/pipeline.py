# src/asys_i/orchestration/pipeline.py (REVISED BASED ON NEW COMPONENTS AND ROBUSTNESS)
import logging
import multiprocessing
import os
import platform # For OS-specific logic, e.g. in _resolve_auto_configs
import signal
import time
from multiprocessing.managers import SyncManager # Using standard SyncManager
from typing import Any, Dict, List, Optional, cast

import psutil # For CPU core detection
import torch

from asys_i.common.types import ComponentID, RunProfile, LayerIndex # LayerIndex for internal numeric mapping
from asys_i.components.activation_hooker import ActivationHooker
from asys_i.components.archiver import ArchiverManager
from asys_i.components.data_bus_factory import create_data_bus
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.components.ppo_host import PPOHostProcess
from asys_i.components.sae_trainer import SAETrainerManager
from asys_i.monitoring.monitor_factory import create_monitor
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.monitoring.watchdog import Watchdog
from asys_i.orchestration.config_loader import MasterConfig
from asys_i.scripts.cleanup_shm import main as cleanup_shm_cmd # For programmatic call if needed

# Import CppShardedSPMCBus for type checking during auto-config
from asys_i.hpc import CPP_EXTENSION_AVAILABLE, SHM_NAME_PREFIX as GLOBAL_SHM_PREFIX
if CPP_EXTENSION_AVAILABLE:
    from asys_i.components.data_bus_hpc import CppShardedSPMCBus

log = logging.getLogger(__name__)

# --- Multiprocessing Manager Setup ---
# It's generally safer to define the manager and register types at the module level
# if it's going to be used by various parts of the application being pickled.
class PipelineSyncManager(SyncManager):
    pass

# Register basic dict proxy if needed, though manager.dict() is standard.
# PipelineSyncManager.register('SharedDict', dict, exposed=['__contains__', '__delitem__', '__getitem__', '__setitem__', 'get', 'items', 'keys', 'update', 'values'])


class ExperimentPipeline:
    def __init__(self, config: MasterConfig):
        self.config = config
        self.sync_manager: Optional[PipelineSyncManager] = None
        self.shared_heartbeats_dict: Optional[Dict[Any, Any]] = None # Proxy from manager
        self.monitor: Optional[BaseMonitor] = None
        self.data_bus: Optional[BaseDataBus] = None
        self.ppo_host: Optional[PPOHostProcess] = None
        self.hooker: Optional[ActivationHooker] = None
        self.trainer_manager: Optional[SAETrainerManager] = None
        self.archiver_manager: Optional[ArchiverManager] = None
        self.watchdog: Optional[Watchdog] = None
        self.stop_event = multiprocessing.Event() # For signaling components to stop
        self._is_setup = False
        self._is_shutting_down = False # To prevent re-entrant shutdown

    def _resolve_auto_configs(self):
        # (Largely same as your reconstructed version, with minor safety checks)
        log.info("Resolving 'auto' configurations...")
        if self.monitor is None: # Should not happen if setup order is correct
            log.error("Monitor not initialized before _resolve_auto_configs. Aborting auto-config.")
            raise RuntimeError("Monitor must be initialized before resolving auto-configurations.")

        # 1. Auto-detect d_in for SAE
        if self.config.sae_model.d_in == "auto":
            if self.ppo_host is None:
                raise RuntimeError("PPOHost must be initialized for d_in auto-detection.")
            
            host_model_for_inspection = self.ppo_host.get_model()
            detected_dim: Any = -1 # Use Any to allow for None from getattr
            try:
                if hasattr(host_model_for_inspection, "config"):
                    model_internal_config = host_model_for_inspection.config
                    detected_dim = getattr(model_internal_config, "hidden_size", None)
                    if detected_dim is None:
                        detected_dim = getattr(model_internal_config, "d_model", -1)
                
                if not isinstance(detected_dim, int) or detected_dim <= 0: # Check after potential None
                    log.info("Could not find hidden_size/d_model in model.config. Inspecting layers for d_in...")
                    # Fallback: inspect model layers (more fragile)
                    first_linear_layer = next((m for m in host_model_for_inspection.modules() if isinstance(m, torch.nn.Linear)), None)
                    if first_linear_layer:
                        log.warning(
                            "Using fallback heuristic for 'sae_model.d_in': detected from the first torch.nn.Linear layer's 'in_features'. "
                            "This heuristic can be unreliable for complex model architectures. "
                            "If the detected dimension (%s) is incorrect or questionable, "
                            "please explicitly set 'sae_model.d_in' in your configuration file.",
                            first_linear_layer.in_features
                        )
                        detected_dim = first_linear_layer.in_features
                        log.info(f"Used heuristic: d_in={detected_dim} from first Linear layer input features.")
                    else: # No Linear layers found or other issue
                        raise ValueError("Cannot find any Linear layer to infer d_in.")

                if not isinstance(detected_dim, int) or detected_dim <= 0:
                    raise ValueError(f"Auto-detected d_in is invalid: {detected_dim}. Must be a positive integer.")
                
                log.info(f"Auto-detected host model d_in: {detected_dim}. Updating SAE config.")
                self.config.sae_model.d_in = detected_dim # Pydantic will validate assignment
                self.monitor.log_metric("sae_model_d_in_auto_detected", float(detected_dim))
            except Exception as e:
                log.error(f"Failed to auto-detect d_in: {e}. Please specify 'sae_model.d_in' explicitly in config.")
                raise # This is a critical failure for SAE model setup

        # 2. Auto-generate resource_manager CPU map (HPC, Linux only)
        res_config = self.config.resource_manager
        if (self.config.run_profile == RunProfile.HPC and
            res_config.apply_bindings and
            res_config.allocation_strategy == "auto" and
            not res_config.cpu_affinity_map): # Only if no manual map provided

            if platform.system() != "Linux":
                log.warning("Auto CPU affinity map generation is Linux-only. Skipping.")
                return

            log.info("Auto-generating CPU affinity map (simple strategy)...")
            num_logical_cores = psutil.cpu_count(logical=True)
            if num_logical_cores is None or num_logical_cores == 0:
                log.error("Could not determine CPU core count. Cannot auto-assign affinities."); return

            core_map: Dict[ComponentID, List[int]] = {}
            available_cores_pool = list(range(num_logical_cores))
            
            def assign_cores(comp_id: ComponentID, num_cores_to_assign: int, preferred_cores: Optional[List[int]] = None) -> None:
                nonlocal available_cores_pool
                assigned: List[int] = []
                # Try preferred first if any overlap with available
                if preferred_cores:
                    for core in preferred_cores:
                        if core in available_cores_pool and len(assigned) < num_cores_to_assign:
                            assigned.append(core)
                            available_cores_pool.remove(core)
                # Fill remaining from general pool
                while len(assigned) < num_cores_to_assign and available_cores_pool:
                    assigned.append(available_cores_pool.pop(0)) # Take from start of pool
                
                if assigned: core_map[comp_id] = assigned
                else: log.warning(f"Could not assign any cores to {comp_id} (requested {num_cores_to_assign})")

            # Simple assignment: Host first, then archiver, then spread trainers
            assign_cores("host_process", min(2, len(available_cores_pool))) # Give host up to 2 cores
            if self.config.archiver.enabled and len(available_cores_pool) > 0:
                assign_cores("archiver_worker_0", min(1, len(available_cores_pool)))
            
            num_trainers = self.config.sae_trainer.num_workers
            if num_trainers > 0 and len(available_cores_pool) > 0:
                cores_per_trainer = max(1, len(available_cores_pool) // num_trainers)
                for i in range(num_trainers):
                    if not available_cores_pool: break
                    assign_cores(f"trainer_worker_{i}", min(cores_per_trainer, len(available_cores_pool)))
            
            if core_map:
                self.config.resource_manager.cpu_affinity_map = core_map # Pydantic validates
                log.info(f"Auto-generated CPU affinity map: {core_map}")
                self.monitor.log_metric("resource_binding_auto_map_generated", 1.0, tags={"map_size": len(core_map)})
            else:
                log.warning("Auto CPU affinity map generation resulted in an empty map. No bindings will be applied via this map.")


    def setup(self) -> None:
        if self._is_setup: log.warning("Pipeline setup already called."); return
        log.info(">>> PIPELINE SETUP STARTED <<<")
        setup_start_time = time.time()

        # Ensure stop_event is clear at the start of setup
        self.stop_event.clear()

        try:
            # 0. Initialize SyncManager for shared objects (e.g., heartbeats dict)
            # This manager needs to be started before creating any proxies from it.
            self.sync_manager = PipelineSyncManager()
            self.sync_manager.start(lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)) # Make manager process ignore SIGINT
            log.info(f"PipelineSyncManager started (PID: {self.sync_manager._process.pid}).") # type: ignore
            self.shared_heartbeats_dict = self.sync_manager.dict() # type: ignore # Get a proxy

            # 1. Monitor (critical first dependency, uses shared_heartbeats_dict)
            self.monitor = create_monitor(self.config, self.shared_heartbeats_dict)
            self.monitor.log_hyperparams(self.config.model_dump(mode='json')) # Log full config

            # 2. Data Bus (HPC one might create SHM segments)
            try:
                self.data_bus = create_data_bus(self.config, self.monitor)
            except ImportError as e: # If CppShardedSPMCBus fails to load
                log.critical(f"CRITICAL: Failed to create DataBus due to missing dependencies: {e}. Pipeline cannot start.")
                self.monitor.log_metric("pipeline_error_count", 1, tags={"stage": "setup", "error": "DataBusImportError"})
                raise # This is fatal.
            except Exception as e: # Other DataBus creation errors
                log.critical(f"CRITICAL: Failed to create DataBus: {e}")
                self.monitor.log_metric("pipeline_error_count", 1, tags={"stage": "setup", "error": "DataBusCreationError"})
                raise

            # 3. Host Process (needed for d_in auto-detection if 'auto')
            self.ppo_host = PPOHostProcess(self.config, self.monitor, self.stop_event)
            self.monitor.register_component(self.ppo_host.component_id) # Register with monitor

            # 4. *** RESOLVE "AUTO" CONFIGURATIONS *** (e.g., d_in, CPU map)
            self._resolve_auto_configs()

            # 5. Hooker (uses resolved d_in, model from PPOHost)
            self.hooker = ActivationHooker(
                self.ppo_host.get_model(), self.config, self.data_bus, self.monitor
            )
            # Hooker's attach() will populate its _layer_name_to_idx map (FQN -> numeric_idx)

            # 6. Trainer Manager (needs FQN->NumIdx map from Hooker for assignments)
            self.trainer_manager = SAETrainerManager(
                self.config, self.data_bus, self.monitor, self.stop_event,
                self.hooker # Pass the hooker instance
            )
            self.trainer_manager.initialize_workers() # Creates worker processes

            # 7. Archiver Manager (also needs FQN->NumIdx map from Hooker)
            self.archiver_manager = ArchiverManager(
                self.config, self.data_bus, self.monitor, self.stop_event,
                self.hooker # Pass the hooker instance
            )
            if self.config.archiver.enabled:
                self.archiver_manager.initialize_workers()
            else:
                log.info("Archiver is disabled. ArchiverManager initialized but no workers created.")


            # 8. Watchdog (monitors components managed by TrainerManager and ArchiverManager)
            managed_components_for_watchdog: Dict[str, Any] = {}
            if self.trainer_manager and self.config.sae_trainer.num_workers > 0 : # Only if trainers are active
                managed_components_for_watchdog["trainer_manager"] = self.trainer_manager
            if self.archiver_manager and self.config.archiver.enabled:
                managed_components_for_watchdog["archiver_manager"] = self.archiver_manager
            
            self.watchdog = Watchdog(
                self.monitor,
                self.config.monitor, # Pass MonitorConfig specifically
                managed_components=managed_components_for_watchdog,
            )

            self._is_setup = True
            duration_setup = time.time() - setup_start_time
            log.info(f">>> PIPELINE SETUP COMPLETE ({duration_setup:.2f}s) <<<")
            self.monitor.log_metric("pipeline_setup_duration_sec", duration_setup)

        except Exception as e_setup: # Catch all exceptions during setup
            log.critical(f"FATAL ERROR DURING PIPELINE SETUP: {e_setup}", exc_info=True)
            if self.monitor: # Try to log, monitor might not be fully up
                self.monitor.log_metric("pipeline_error_count", 1, tags={"stage": "setup_fatal", "error": str(type(e_setup).__name__)})
            # Attempt a partial shutdown of whatever was initialized
            self._is_shutting_down = True # Prevent re-entry from signal handler if setup fails badly
            self._perform_shutdown_sequence()
            raise # Re-raise to stop application

    def _signal_handler(self, sig, frame):
        log.warning(f"!!! Signal {signal.Signals(sig).name} received by pipeline (PID {os.getpid()}). Initiating graceful shutdown... !!!")
        if not self.stop_event.is_set():
            self.stop_event.set() # Signal all components
        
        # If shutdown is already in progress (e.g. from another signal or error), don't re-enter
        if self._is_shutting_down:
            log.info("Shutdown already in progress, signal ignored for re-entry prevention.")
            return
        
        # Trigger shutdown sequence. The main run loop's finally block will also call it,
        # but this ensures it happens even if stuck outside the main loop.
        # Setting a flag is often enough if components check stop_event.
        # A more forceful approach might be to directly call _perform_shutdown_sequence
        # but that can be risky if called from a signal handler directly due to re-entrancy.
        # For now, setting stop_event is the primary mechanism.
        # The main thread's `finally` block in `run()` will call `self.shutdown()`.

    def run(self) -> None:
        if not self._is_setup:
            raise RuntimeError("Pipeline.setup() must be called before run().")
        if self.stop_event.is_set(): # If stop_event was set during setup or before run
            log.warning("Pipeline run() called, but stop_event is already set. Exiting.")
            self.shutdown() # Ensure cleanup if setup partially completed then failed
            return

        log.info(f">>> PIPELINE RUN STARTED (PID {os.getpid()}) <<<")
        run_start_time = time.time()
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Assert components that are critical for run()
            if not (self.watchdog and self.trainer_manager and self.ppo_host and self.hooker):
                 raise RuntimeError("Critical components (Watchdog, TrainerManager, PPOHost, Hooker) not initialized before run.")
            if self.config.archiver.enabled and not self.archiver_manager:
                 raise RuntimeError("Archiver is enabled but ArchiverManager not initialized.")


            self.watchdog.start() # Start Watchdog thread
            if self.config.sae_trainer.num_workers > 0:
                self.trainer_manager.start_all() # Start SAETrainerWorker processes
            if self.config.archiver.enabled and self.archiver_manager:
                self.archiver_manager.start_all() # Start ArchiverWorker process(es)

            # This is the main blocking call for the experiment duration
            self.ppo_host.run_training_loop(self.hooker)

            # If run_training_loop finishes normally (not due to stop_event or exception):
            if not self.stop_event.is_set():
                log.info("PPO host training loop completed its max_steps. Initiating shutdown.")
                self.stop_event.set() # Signal all other components to stop

            run_duration = time.time() - run_start_time
            log.info(f">>> PIPELINE RUN FINISHED (Duration: {run_duration:.2f}s) <<<")
            if self.monitor:
                self.monitor.log_metric("pipeline_run_duration_sec", run_duration)

        except KeyboardInterrupt: # Specifically catch Ctrl+C if not handled by _signal_handler first
            log.warning(f"KeyboardInterrupt caught in pipeline.run() (PID {os.getpid()}). Forcing shutdown via stop_event.")
            if not self.stop_event.is_set(): self.stop_event.set()
        except Exception as e_run: # Catch any other unexpected errors from main PPO loop
            log.critical(f"FATAL ERROR DURING PIPELINE RUN: {e_run}", exc_info=True)
            if self.monitor:
                self.monitor.log_metric("pipeline_error_count", 1, tags={"stage": "run_fatal", "error": str(type(e_run).__name__)})
            if not self.stop_event.is_set(): self.stop_event.set() # Ensure components are signaled
        finally:
            log.info(f"Pipeline run() exiting 'try' block (PID {os.getpid()}). Proceeding to shutdown sequence.")
            self.shutdown() # This calls _perform_shutdown_sequence if not already shutting down

            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)

    def _perform_shutdown_sequence(self):
        """The actual component shutdown logic, to be called once."""
        log.info(f">>> PIPELINE SHUTDOWN SEQUENCE STARTED (PID {os.getpid()}) <<<")
        shutdown_start_time = time.time()

        # 1. Signal PPO Host first (if it's in its loop and not already stopped by event)
        if self.ppo_host and not self.stop_event.is_set(): # If stop_event not already set by PPO loop end or signal
            log.debug("Ensuring PPOHost's internal stop event is set for shutdown.")
            # PPOHost should primarily react to self.stop_event passed to it.
            # self.ppo_host._stop_event.set() # This is redundant if it uses the shared one.

        # 2. Stop Watchdog first to prevent it from trying to restart components during shutdown
        if self.watchdog and self.watchdog.is_alive():
            log.debug("Stopping Watchdog...")
            self.watchdog.stop(timeout=5.0) # Give it a short timeout
            if self.watchdog.is_alive(): log.error("Watchdog thread did not terminate cleanly.")
        
        # 3. Stop Worker Managers (they handle their worker processes)
        # Archiver first, then Trainer, as Trainers might depend on DataBus longer
        if self.archiver_manager and self.config.archiver.enabled:
            log.debug("Stopping ArchiverManager and its workers...")
            self.archiver_manager.stop_all() # Uses BaseWorkerManager stop_all logic
        if self.trainer_manager and self.config.sae_trainer.num_workers > 0:
            log.debug("Stopping SAETrainerManager and its workers...")
            self.trainer_manager.stop_all() # Uses BaseWorkerManager stop_all logic

        # 4. Shutdown PPO Host (releases model, etc.) - ensure it's called after workers that might use its model via hooks
        if self.ppo_host:
            log.debug("Shutting down PPOHost process logic...")
            self.ppo_host.shutdown() # Internal cleanup for PPOHost
        
        # 5. Detach Hooker (after PPO host and consumers are stopping/stopped)
        if self.hooker:
            log.debug("Shutting down ActivationHooker (detaching hooks)...")
            self.hooker.shutdown()

        # 6. Shutdown DataBus (releases C++ resources like SHM, MQ)
        # This should be done after all producers (Hooker) and consumers (Trainers, Archivers) are stopped.
        if self.data_bus:
            log.debug("Shutting down DataBus...")
            self.data_bus.shutdown()

        # 7. Shutdown Monitor (flushes any remaining metrics)
        if self.monitor:
            log.debug("Shutting down Monitor...")
            self.monitor.shutdown()

        # 8. Shutdown the multiprocessing SyncManager
        if self.sync_manager:
            log.debug("Shutting down PipelineSyncManager...")
            try:
                # Check if manager process is alive. SyncManager has a _process attribute.
                if hasattr(self.sync_manager, '_process') and self.sync_manager._process and self.sync_manager._process.is_alive(): # type: ignore
                    self.sync_manager.shutdown()
                    log.info("PipelineSyncManager shut down.")
                elif not hasattr(self.sync_manager, '_process'):
                     log.warning("PipelineSyncManager does not have _process attribute, attempting shutdown anyway.")
                     self.sync_manager.shutdown()
                else: # Process not alive or None
                    log.info("PipelineSyncManager process already stopped or not started.")
            except Exception as e_mgr_shutdown:
                log.error(f"Exception during PipelineSyncManager shutdown: {e_mgr_shutdown}")
            self.sync_manager = None
            self.shared_heartbeats_dict = None
        
        # 9. Final SHM cleanup for segments created by *this* pipeline instance (HPC only)
        # This is a best-effort cleanup for the SHM segments this main process initiated.
        # Orphaned segments from crashed *workers* might need the standalone cleanup script.
        if self.config.run_profile == RunProfile.HPC and CPP_EXTENSION_AVAILABLE:
            # If data_bus was CppShardedSPMCBus, it would have specific SHM names
            # Its shutdown should handle unlinking. This is more of a fallback.
            log.info("Performing final check for SHM segments associated with this pipeline instance...")
            # The CppShardedSPMCBus destructor (via self.data_bus.shutdown -> del self.cpp_manager)
            # is the primary mechanism for unlinking the SHM it created.
            # The standalone `asys-i-cleanup` script is for orphaned segments from crashes.
            # No direct action here beyond what CppDataBus.shutdown does.
            # We could list SHM segments here for logging if needed.
            # For example, find segments matching self.data_bus.tensor_shm_name if it exists.

        shutdown_duration = time.time() - shutdown_start_time
        log.info(f">>> PIPELINE SHUTDOWN SEQUENCE COMPLETE (Duration: {shutdown_duration:.2f}s) <<<")


    def shutdown(self) -> None:
        if self._is_shutting_down:
            log.info(f"Shutdown already in progress for pipeline (PID {os.getpid()}). Ignoring re-entrant call.")
            return
        self._is_shutting_down = True # Set flag *before* starting sequence

        self._perform_shutdown_sequence()
        
        self._is_setup = False # Mark as no longer setup
        self._is_shutting_down = False # Clear flag after completion
