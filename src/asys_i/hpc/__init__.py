# src/asys_i/hpc/__init__.py
# Check if HPC components are available
import importlib.util
import logging

log = logging.getLogger(__name__)

_cpp_spec = importlib.util.find_spec("asys_i.hpc.cpp_ringbuffer_bindings")
CPP_EXTENSION_AVAILABLE = _cpp_spec is not None

_prometheus_spec = importlib.util.find_spec("prometheus_client")
PROMETHEUS_AVAILABLE = _prometheus_spec is not None

# Add checks for nvcomp, transformer-engine etc.
# NVCOMP_AVAILABLE = ...

HPC_DEPENDENCIES_AVAILABLE = CPP_EXTENSION_AVAILABLE and PROMETHEUS_AVAILABLE # Add others

if not HPC_DEPENDENCIES_AVAILABLE:
     log.warning(
        "HPC dependencies not fully met. "
        f"CPP_EXTENSION_AVAILABLE={CPP_EXTENSION_AVAILABLE}, "
         f"PROMETHEUS_AVAILABLE={PROMETHEUS_AVAILABLE}. "
         "Running in HPC mode will raise errors."
      )

def check_hpc_prerequisites():
     if not HPC_DEPENDENCIES_AVAILABLE:
          raise ImportError(
               "HPC dependencies (C++ extension, Prometheus, etc.) are not installed or compiled. "
               "Please run 'pip install -e .[hpc]' and ensure build environment is set up."
          )

