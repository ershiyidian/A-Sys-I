# Known Issues

## 1. Critical: Missing `cpp_ringbuffer` C++ Component for HPC Mode

**Date Discovered:** 2025-06-13
**Severity:** Critical

### Description:
The C++ component `cpp_ringbuffer`, intended to implement the `CppShardedSPMCBus` for the High-Performance Computing (HPC) mode, is missing from the repository. The directory `src/asys_i/hpc/cpp_ringbuffer/` currently only contains a `README.md` file, with no C++ source code (`.cpp`, `.h`, `.cu`) or `CMakeLists.txt` build definition file.

The project's `pyproject.toml` includes build dependencies like `cmake`, `ninja`, and `pybind11`, indicating an intention to compile C++ extensions. However, without the actual source code for `cpp_ringbuffer`, this C++ data bus cannot be built.

### Implication:
The HPC mode, as described in `README.md` and other documentation (which details a zero-copy, shared memory, C++ based ring buffer for high performance), is **not functional as designed**. The system will likely fall back to a Python-based data bus or fail if configured explicitly for `CppShardedSPMCBus` without the component being available. This significantly impacts the stated goals of predictable high performance and low latency in HPC mode.

### Affected Components:
- `src/asys_i/components/data_bus_hpc.py` (which would contain/use `CppShardedSPMCBus`)
- `src/asys_i/components/data_bus_factory.py` (which attempts to select the data bus based on configuration)
- Overall HPC mode performance and functionality.

### Possible Resolutions:
1.  **Implement/Recover Source Code:** The original C++ source code and associated `CMakeLists.txt` for `cpp_ringbuffer` need to be implemented or recovered if they exist elsewhere.
2.  **Revise HPC Design:** If the C++ component is not forthcoming, the project documentation (`README.md`, configs) and the HPC mode design need to be revised:
    *   Clearly state that the C++ shared memory bus is not currently available.
    *   Potentially adapt the HPC mode to use an optimized Python alternative if feasible, though this may not meet the original performance SLAs.
    *   Adjust performance expectations and SLA documentation for HPC mode.
3.  **Remove HPC C++ References:** If HPC mode is to be pursued with pure Python components for the time being, remove misleading build dependencies for C++ from `pyproject.toml` or clarify their purpose if intended for other future extensions.

### Workaround:
Currently, only CONSUMER mode or a Python-based data bus in HPC mode (if such a fallback exists and is configured) can be considered functional for data transfer.
