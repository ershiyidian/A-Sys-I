# src/asys_i/hpc/__init__.py (CONFIRMED FROM YOUR LAST INPUT - GOOD)
import importlib.util
import logging

log = logging.getLogger(__name__)

# Module name is 'c_ext_wrapper' as defined in bindings.cpp and CMakeLists.txt.
# It's installed into the asys_i.hpc package.
_cpp_ext_module_name = "asys_i.hpc.c_ext_wrapper"
_cpp_ext_spec = importlib.util.find_spec(_cpp_ext_module_name)
CPP_EXTENSION_AVAILABLE = _cpp_ext_spec is not None

c_ext_wrapper = None # Placeholder for the imported module

if CPP_EXTENSION_AVAILABLE:
    try:
        # Import the module dynamically
        c_ext_wrapper = importlib.import_module(_cpp_ext_module_name)
        if hasattr(c_ext_wrapper, 'verify_cpp_module'):
            log.info(f"Successfully loaded A-Sys-I C++ extension: {c_ext_wrapper.verify_cpp_module()}")
        else:
            log.info(f"Successfully loaded A-Sys-I C++ extension ({_cpp_ext_module_name}), but verify_cpp_module() not found.")
            # This might be okay if verify_cpp_module is optional or named differently
    except ImportError as e:
        log.error(f"Found C++ extension spec for '{_cpp_ext_module_name}' but failed to import it: {e}")
        CPP_EXTENSION_AVAILABLE = False
    except Exception as e_load: # Catch other potential errors during import or verify call
        log.error(f"Error during C++ extension loading or verification for '{_cpp_ext_module_name}': {e_load}")
        CPP_EXTENSION_AVAILABLE = False
else:
    log.warning(
        f"A-Sys-I C++ extension ('{_cpp_ext_module_name}') not found. "
        "HPC DataBus (CppShardedSPMCBus) will be unavailable. "
        "Ensure the project was installed with 'pip install .[hpc]' (or 'pip install .[all]') "
        "and that C++ compilation (CMake, C++ Compiler, Boost development libraries) succeeded. "
        "Check the build logs for errors related to C++ compilation or linking."
    )

# Global constant for SHM naming, used by CppDataBus and cleanup script
SHM_NAME_PREFIX = "asys_i_tensor_shm_" # For tensor data SHM
MQ_NAME_PREFIX = "asys_i_mq_"        # For metadata message queues


def check_hpc_prerequisites():
    """
    Checks if all prerequisites for running in HPC mode are met.
    Raises ImportError if C++ extension dependencies are missing.
    Raises ImportError if Prometheus client (another HPC dep) is missing.
    """
    if not CPP_EXTENSION_AVAILABLE:
        raise ImportError(
            f"A-Sys-I C++ extension ('{_cpp_ext_module_name}') is not available or failed to load. "
            "This is required for CppShardedSPMCBus in HPC mode. "
            "Please ensure A-Sys-I was installed with the '[hpc]' option (e.g., 'pip install .[hpc]') "
            "and that all C++ build dependencies (CMake, C++ Compiler, Boost development libraries) are met "
            "and the compilation was successful. Review build logs for details."
        )
    try:
        import prometheus_client # noqa: F401 imported but unused
    except ImportError:
        raise ImportError(
            "The 'prometheus-client' library is not installed. This is required for "
            "PrometheusMonitor in HPC mode. Please install it via 'pip install prometheus-client' "
            "or by installing A-Sys-I with the '[hpc]' option (e.g., 'pip install .[hpc]')."
        )
    log.debug("HPC prerequisites (C++ extension, Prometheus client) met.")

