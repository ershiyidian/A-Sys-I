# src/asys_i/monitoring/watchdog.py
"""
Core Philosophy: Design for Failure, Observability-First.
Monitors component heartbeats and triggers restart actions for failed components.
 Runs in a separate thread.
 """
import logging
import threading
import time
from typing import Dict, Protocol, Any

from asys_i.common.types import ComponentID
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MonitorConfig
 # Avoid circular dependency: Use Protocol for type hinting managers instead of direct import
 # from asys_i.components.sae_trainer import SAETrainerManager 
 
log = logging.getLogger(__name__)

# Define structural types (Protocols) for managers that can restart components
class RestartableManager(Protocol):
     def restart_worker(self, worker_id: ComponentID) -> bool:
          ...
     def get_worker_ids(self) -> list[ComponentID]:
         ...

class Watchdog(threading.Thread):
     """
     Monitors heartbeats via BaseMonitor and calls restart logic on managers.
     """
     def __init__(self,
                  monitor: BaseMonitor,
                  config: MonitorConfig,
                  managed_components: Dict[str, RestartableManager],
                  component_id: ComponentID = "watchdog"
                   ):
         super().__init__(daemon=True) # Daemon thread exits when main thread exits
         self.monitor = monitor
         self.config = config
         self.managed_components = managed_components
         self._stop_event = threading.Event()
         self.check_interval = config.heartbeat_check_interval_sec
         self.timeout = config.component_timeout_sec
         self.component_id = component_id
         self._monitored_ids : set[ComponentID] = set()
         log.info(f"Watchdog initialized. Check interval: {self.check_interval}s, Timeout: {self.timeout}s")

     def _refresh_monitored_ids(self):
         """ Collect all IDs from all registered managers """
         current_ids = set()
         for name, manager in self.managed_components.items():
             try:
                ids = manager.get_worker_ids()
                current_ids.update(ids)
             except Exception as e:
                 log.error(f"Error getting worker IDs from manager {name}: {e}")
         # Register any new IDs with monitor
         for cid in current_ids - self._monitored_ids:
              self.monitor.register_component(cid)
         self._monitored_ids = current_ids
         log.debug(f"Watchdog monitoring components: {self._monitored_ids}")


     def run(self):
         log.info("Watchdog thread started.")
         self.monitor.register_component(self.component_id)
         # Initial heartbeat
         self.monitor.heartbeat(self.component_id)
         
         # Initial wait
         self._stop_event.wait(self.check_interval) 

         while not self._stop_event.is_set():
             try:
                 self.monitor.heartbeat(self.component_id) # Watchdog is alive
                 self._refresh_monitored_ids() # Check if new workers were added/restarted
                 heartbeats = self.monitor.get_heartbeats()
                 now = time.time()
                 
                 components_to_check = self._monitored_ids # Only check managed components

                 for component_id in components_to_check:
                      last_seen = heartbeats.get(component_id)
                      if last_seen is None:
                           log.warning(f"Component {component_id} is monitored but never sent a heartbeat!")
                           # Optionally: trigger immediate restart or register it
                           # self.monitor.register_component(component_id) 
                           continue
                           
                      time_since_last_seen = now - last_seen
                      if time_since_last_seen > self.timeout:
                           log.error(
                                f"WATCHDOG TIMEOUT: Component '{component_id}' last seen "
                                f"{time_since_last_seen:.1f}s ago (timeout={self.timeout}s). Attempting restart."
                           )
                           self.monitor.log_metric("watchdog_timeout_count", 1, tags={"component": component_id})
                           self._attempt_restart(component_id)
                           # Reset heartbeat after restart attempt to give it time
                           self.monitor.heartbeat(component_id) 
                      else:
                           log.debug(f"Component {component_id} OK (last seen {time_since_last_seen:.1f}s ago)")
             
             except Exception as e:
                  log.exception("Exception in Watchdog loop:")
                  self.monitor.log_metric("watchdog_error_count", 1)
                  # Continue running despite error

             # Wait for next check or stop event
             self._stop_event.wait(self.check_interval) 
         
         log.info("Watchdog thread stopping.")

     def _attempt_restart(self, component_id: ComponentID):
          restarted = False
          # Find which manager owns this component_id
          for manager_name, manager in self.managed_components.items():
               try:
                    if component_id in manager.get_worker_ids():
                         log.warning(f"Requesting manager '{manager_name}' to restart '{component_id}'")
                         success = manager.restart_worker(component_id)
                         self.monitor.log_metric("watchdog_restart_attempt", 1, tags={"component": component_id, "manager": manager_name, "success": str(success)})
                         if success:
                             log.info(f"Manager '{manager_name}' reported successful restart for '{component_id}'.")
                             restarted = True
                             # important: refresh IDs so manager.get_worker_ids reflects change
                             self._refresh_monitored_ids() 
                             break
                         else:
                              log.error(f"Manager '{manager_name}' failed to restart '{component_id}'.")
               except Exception as e:
                    log.exception(f"Error during restart attempt for {component_id} via manager {manager_name}:")

          if not restarted:
               log.error(f"Could not find a manager to restart component {component_id} or restart failed.")
               self.monitor.log_metric("watchdog_restart_failed", 1, tags={"component": component_id})


     def stop(self, timeout:float=10.0):
         log.info("Signaling Watchdog to stop.")
         self._stop_event.set()
         self.join(timeout)
         if self.is_alive():
              log.error("Watchdog thread did not terminate cleanly.")

