# src/asys_i/hpc/resource_manager.py
"""
Core Philosophy: Predictability (Performance), Separation.
Manages CPU affinity and NUMA node binding for processes in HPC mode to
minimize context switching and memory latency. Linux only.
"""
import logging
import os
import platform
import psutil
# import subprocess
from typing import List, Optional

from asys_i.orchestration.config_loader import ResourceManagerConfig, MasterConfig
from asys_i.common.types import ComponentID, RunProfile
from asys_i.monitoring.monitor_interface import BaseMonitor

log = logging.getLogger(__name__)

_IS_LINUX = platform.system() == "Linux"

def bind_current_process(
    config: MasterConfig,
    component_id: ComponentID,
    monitor: BaseMonitor
    ) -> None:
    """
    Binds the *current* process (os.getpid()) to specific CPU cores and NUMA node
    based on the configuration map for the given component_id.
    Should be called by a worker process itself at the start of its run() method.
    """
    resource_config = config.resource_manager
    tags = {"component": component_id, "pid": os.getpid()}

    if config.run_profile != RunProfile.HPC or not resource_config.apply_bindings:
        return # Binding only active in HPC mode and if enabled
        
    if not _IS_LINUX:
        log.warning(f"CPU/NUMA binding requested on non-Linux system ({platform.system()}) for {component_id}. Skipping.")
        monitor.log_metric("resource_binding_skip_count", 1, tags={**tags, "reason": "not_linux"})
        return

    pid = os.getpid()
    try:
        p = psutil.Process(pid)
        
        # 1. CPU Affinity
        cores: Optional[List[int]] = resource_config.cpu_affinity_map.get(component_id)
        if cores:
            available_cores = psutil.cpu_count(logical=True)
            valid_cores = [c for c in cores if 0 <= c < available_cores]
            if len(valid_cores) != len(cores):
                 log.warning(f"Invalid core IDs specified for {component_id}: {cores}. Using valid: {valid_cores}")
                 monitor.log_metric("resource_binding_error_count", 1, tags={**tags, "reason": "invalid_cores"})
            if valid_cores:
                 p.cpu_affinity(valid_cores)
                 log.info(f"Bound process {pid} ({component_id}) to CPU cores: {p.cpu_affinity()}")
                 monitor.log_metric("resource_binding_success_count", 1, tags={**tags, "type": "cpu"})
            else:
                  log.warning(f"No valid cores found to bind for {component_id}.")
        else:
             log.debug(f"No CPU affinity map found for {component_id}.")
             
        # 2. NUMA Binding (more complex, psutil doesn't directly support numactl)
        numa_node: Optional[int] = resource_config.numa_node_map.get(component_id)
        if numa_node is not None:
             # Option A: Use `numactl` wrapper when launching process (preferred)
             # Option B: Use ctypes to call set_mempolicy (complex)
             # Option C: subprocess call - less reliable for already running process
             # For now, just log intent. Actual binding often needs `numactl -m X -N X python ...`
             log.warning(f"NUMA binding ({numa_node}) for {pid} ({component_id}) requested. "
                          "psutil binding is CPU-only. Use `numactl` to launch for memory/node binding.")
             monitor.log_metric("resource_binding_skip_count", 1, tags={**tags, "reason": "numa_unsupported"})
             # Example:
             # cmd = ['numactl', f'--membind={numa_node}', f'--cpunodebind={numa_node}', '--', 'echo', 'pid', str(pid)]
             # subprocess.run(cmd, check=False) # Does not affect the running process itself this way

    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        log.error(f"Error binding process {pid} ({component_id}): {e}")
        monitor.log_metric("resource_binding_error_count", 1, tags={**tags, "reason": str(type(e))})
    except Exception as e:
         log.exception(f"Unexpected error during resource binding for {component_id}:")
         monitor.log_metric("resource_binding_error_count", 1, tags={**tags, "reason": "exception"})

