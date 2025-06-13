# src/asys_i/components/managers/base_manager.py (REVISED FROM YOUR LAST INPUT)
import logging
import multiprocessing
import os
import signal # Correct import for os.kill signals
import time   # For sleep during forced termination
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from asys_i.common.types import ComponentID
from asys_i.components.data_bus_interface import BaseDataBus
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MasterConfig

log = logging.getLogger(__name__)

class RestartableManager(ABC): # Protocol definition
    @abstractmethod
    def restart_worker(self, worker_id: ComponentID) -> bool: ...
    @abstractmethod
    def get_worker_ids(self) -> list[ComponentID]: ...

class BaseWorkerManager(RestartableManager):
    def __init__(
        self,
        config: MasterConfig,
        data_bus: BaseDataBus, # Not directly used by base, but often needed by subclasses
        monitor: BaseMonitor,
        stop_event: multiprocessing.Event,
        worker_name_prefix: str,
    ):
        self.config = config
        self.data_bus = data_bus
        self.monitor = monitor
        self.stop_event = stop_event
        self.worker_name_prefix = worker_name_prefix
        self.workers: Dict[ComponentID, multiprocessing.Process] = {}
        self.worker_assignments: Dict[ComponentID, Any] = {} # Stores what each worker is assigned
        log.info(f"BaseWorkerManager initialized for prefix: '{self.worker_name_prefix}'")

    @abstractmethod
    def _create_worker_process(self, worker_id: ComponentID, assignment: Any) -> multiprocessing.Process:
        raise NotImplementedError

    @abstractmethod
    def initialize_workers(self):
        raise NotImplementedError

    def start_all(self):
        log.info(f"Starting {self.worker_name_prefix} workers...")
        for worker_id, process_obj in self.workers.items(): # Changed 'process' to 'process_obj'
            if not process_obj.is_alive():
                try:
                    process_obj.start()
                    log.info(f"Started process {process_obj.pid} for {worker_id}")
                except Exception as e:
                    log.error(f"Failed to start worker {worker_id} (PID {process_obj.pid if process_obj.pid else 'N/A'}): {e}")
                    self.monitor.log_metric("worker_start_error", 1, tags={"component": worker_id, "prefix": self.worker_name_prefix})
            else:
                log.warning(f"Worker {worker_id} (PID {process_obj.pid}) already running.")

    def stop_all(self, timeout_graceful: float = 15.0, timeout_force_sigint: float = 10.0, timeout_force_sigterm: float = 5.0):
        log.info(f"Stopping {self.worker_name_prefix} workers...")
        if not self.stop_event.is_set():
            self.stop_event.set()
            log.debug(f"{self.worker_name_prefix} manager explicitly set stop_event for its workers.")

        # Iterate over a copy of worker IDs in case dict changes during iteration (e.g. restart)
        worker_ids_to_stop = list(self.workers.keys())

        for worker_id in worker_ids_to_stop:
            process_obj = self.workers.get(worker_id)
            if process_obj and process_obj.is_alive():
                log.info(f"Attempting graceful shutdown for {worker_id} (PID {process_obj.pid}). Waiting up to {timeout_graceful}s.")
                process_obj.join(timeout_graceful)

                if process_obj.is_alive():
                    log.warning(f"Worker {worker_id} (PID {process_obj.pid}) did not terminate gracefully. Sending SIGINT. Waiting up to {timeout_force_sigint}s.")
                    try:
                        if process_obj.pid: os.kill(process_obj.pid, signal.SIGINT)
                        process_obj.join(timeout_force_sigint)
                    except Exception as e: # Catch errors like "No such process" if it died meanwhile
                        log.error(f"Error sending SIGINT to {worker_id} (PID {process_obj.pid}): {e}")
                
                if process_obj.is_alive():
                    log.error(f"Worker {worker_id} (PID {process_obj.pid}) still alive after SIGINT. Sending SIGTERM. Waiting up to {timeout_force_sigterm}s.")
                    try:
                        process_obj.terminate() # Sends SIGTERM
                        process_obj.join(timeout_force_sigterm)
                    except Exception as e:
                         log.error(f"Error sending SIGTERM to {worker_id} (PID {process_obj.pid}): {e}")

                if process_obj.is_alive():
                    log.critical(f"Worker {worker_id} (PID {process_obj.pid}) FAILED TO TERMINATE after all attempts.")
                    self.monitor.log_metric("worker_terminate_failed", 1, tags={"component": worker_id, "prefix": self.worker_name_prefix})
                else:
                    log.info(f"Worker {worker_id} (Old PID {process_obj.pid}) terminated successfully.")

                # Always try to close the process object to free resources
                try:
                    process_obj.close()
                    log.debug(f"Closed process object for {worker_id}")
                except Exception as e:
                    log.warning(f"Error closing process object for {worker_id} (PID {process_obj.pid if process_obj.pid else 'N/A'}): {e}")
            
            # Clean up from manager's tracking dicts
            if worker_id in self.workers: del self.workers[worker_id]
            if worker_id in self.worker_assignments: del self.worker_assignments[worker_id]
        
        log.info(f"All {self.worker_name_prefix} workers have been processed for shutdown.")


    def get_worker_ids(self) -> List[ComponentID]:
        # Return IDs of currently managed workers that are (or should be) alive
        return [wid for wid, proc_obj in self.workers.items() if proc_obj.is_alive()]

    def restart_worker(self, worker_id: ComponentID) -> bool:
        if worker_id not in self.worker_assignments: # Check assignment first
            log.error(f"Cannot restart worker: {worker_id}. No assignment found for manager {self.worker_name_prefix}.")
            return False
        
        assignment = self.worker_assignments[worker_id]
        old_process = self.workers.get(worker_id)
        old_pid_str = f"Old PID: {old_process.pid}" if old_process and old_process.pid else "Old process/PID not found"

        log.warning(f"Restarting worker {worker_id} ({old_pid_str}) with assignment: {assignment}...")

        if old_process and old_process.is_alive():
            log.info(f"Terminating old process for worker {worker_id} (PID {old_process.pid})...")
            old_process.terminate()
            old_process.join(timeout=10.0)
            if old_process.is_alive():
                log.error(f"Old process for {worker_id} (PID {old_process.pid}) could not be terminated. Restart aborted.")
                return False
            try: old_process.close()
            except Exception: pass # Ignore errors closing already terminated process

        # Remove old process from active tracking, assignment is kept.
        if worker_id in self.workers:
            del self.workers[worker_id]

        try:
            # Create and start the new process
            # The stop_event passed to the new worker should be the manager's original stop_event.
            # If the old worker died due to stop_event, a new one shouldn't be started by Watchdog.
            # Watchdog should ideally check stop_event before calling restart.
            if self.stop_event.is_set():
                log.warning(f"Manager's stop_event is set. Suppressing restart of worker {worker_id}.")
                return False

            new_process = self._create_worker_process(worker_id, assignment) # Re-uses worker_id and its assignment
            self.workers[worker_id] = new_process # Store new process under the same ID
            
            new_process.start()
            # Short delay to allow process to initialize before checking is_alive or PID
            time.sleep(0.1)
            if not new_process.is_alive() or new_process.pid is None:
                log.error(f"New process for worker {worker_id} failed to start or has no PID. Exit code: {new_process.exitcode}")
                # Clean up the failed process entry
                if worker_id in self.workers: del self.workers[worker_id]
                return False

            self.monitor.register_component(worker_id) # Re-register with monitor
            self.monitor.heartbeat(worker_id)         # Provide an initial heartbeat
            log.info(f"Successfully restarted worker {worker_id} with new PID: {new_process.pid}")
            return True
        except Exception as e:
            log.error(f"Failed to create or start new process for worker {worker_id} during restart: {e}")
            if worker_id in self.workers: # Clean up if it was added but failed to start
                del self.workers[worker_id]
            return False
