 # C++ Lock-Free Ring Buffer
 
 Placeholder for C++ core implementation and Pybind11 bindings.
 
 Files expected:
 - `spmc_queue.hpp`: Header-only template class `SPMCRingBuffer<T>` using `std::atomic`.
 - `memory_manager.hpp`: Manages allocation/deallocation within a shared memory segment for Tensors.
 - `bindings.cpp`: pybind11 module definition. Binds `SPMCRingBuffer` and memory manager functions. Defines C++ struct equivalent of `ActivationPacket` metadata.
 - `CMakeLists.txt`: Build configuration.
 
 **Purpose**:
 Provide a high-throughput, low-latency, Single-Producer-Multiple-Consumer (SPMC) queue based on shared memory.
 1.  Tensor data is written directly to a large shared memory block managed by `memory_manager`.
 2.  Only metadata and a reference (`TensorRef`: offset/size in shared memory) is pushed onto the `SPMCRingBuffer`.
 3.  C++ layer avoids Python GIL and serialization overhead.
 4.  Uses `std::memory_order_acquire` / `release` for synchronization.
 5.  Cache line alignment (`alignas(64)`) to prevent false sharing.
 
 To be integrated via `pyproject.toml` build system.
 The Python interface is `asys_i.components.data_bus_hpc.py`.
 
