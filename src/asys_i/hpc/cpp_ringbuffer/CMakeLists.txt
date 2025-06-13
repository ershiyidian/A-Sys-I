# src/asys_i/hpc/cpp_ringbuffer/CMakeLists.txt
# This defines the core static library that the pybind11 module will link against.

add_library(asys_i_core
    shm_manager.cpp
)

# Include directories needed to build the library
target_include_directories(asys_i_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    # Add path to moodycamel::ConcurrentQueue header
    "${CMAKE_CURRENT_SOURCE_DIR}/moodycamel"
    # Add Boost include directory
    ${Boost_INCLUDE_DIRS}
)

# Link against Boost libraries
target_link_libraries(asys_i_core PRIVATE Boost::interprocess)
