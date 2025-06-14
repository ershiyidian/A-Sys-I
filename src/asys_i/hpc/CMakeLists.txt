cmake_minimum_required(VERSION 3.16)

project(cpp_ringbuffer CXX)

  

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

  

# --- User-configurable Boost Handling ---

# By default, we fetch Boost ourselves to simplify setup. The user can override

# this by setting -DASYS_I_USE_SYSTEM_BOOST=ON during CMake configuration.

option(ASYS_I_USE_SYSTEM_BOOST "Use system-provided Boost installation instead of FetchContent" OFF)

  

if(ASYS_I_USE_SYSTEM_BOOST)

    message(STATUS "Attempting to find system-provided Boost...")

    # Add a specific version requirement for clarity. Boost.Interprocess is stable,

    # so a wide range of versions should work.

    find_package(Boost 1.70.0 QUIET)

    if(NOT Boost_FOUND)

        message(FATAL_ERROR "ASYS_I_USE_SYSTEM_BOOST is ON, but Boost could not be found on the system. Please install Boost development libraries (e.g., 'libboost-dev' on Debian/Ubuntu) or turn this option OFF.")

    endif()

    message(STATUS "Found System Boost: ${Boost_VERSION_STRING}")

    set(BOOST_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})

else()

    message(STATUS "Using FetchContent to provide Boost dependency...")

    include(FetchContent)

  

    set(BOOST_VERSION "1.83.0")

    set(BOOST_URL "https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}.tar.gz")

    set(BOOST_SHA256 "af58c49a03e1ecd62c058a03c3937418285572074320485906a13b19808385d8")

  

    FetchContent_Declare(

      boost

      URL ${BOOST_URL}

      URL_HASH SHA256=${BOOST_SHA256}

    )

    FetchContent_GetProperties(boost)

    if(NOT boost_POPULATED)

      message(STATUS "Fetching Boost ${BOOST_VERSION}... (This may take a while)")

      FetchContent_Populate(boost)

      set(BOOST_INCLUDE_DIRS ${boost_SOURCE_DIR})

      message(STATUS "Boost source directory: ${boost_SOURCE_DIR}")

    else()

      message(STATUS "Boost already populated.")

      set(BOOST_INCLUDE_DIRS ${boost_SOURCE_DIR})

    endif()

endif()

# --- End of Boost Handling ---

  

# Find Pybind11, which is expected to be installed via pip.

find_package(pybind11 CONFIG REQUIRED)

  

# Source files for the C++ extension

set(SOURCES

    src/bindings.cpp

    cpp_ringbuffer/shm_manager.cpp

)

  

pybind11_add_module(

    c_ext_wrapper

    ${SOURCES}

)

  

# Link libraries and include directories

target_include_directories(c_ext_wrapper PRIVATE

    ${pybind11_INCLUDE_DIRS}

    ${BOOST_INCLUDE_DIRS}

    cpp_ringbuffer  # For shm_manager.h and torch_dtype_codes.h
    src             # For torch_dtype_pybind_bindings.inc

)

  

# Boost.Interprocess is header-only, but it requires linking against the

# real-time library (-lrt) on Linux systems for some synchronization primitives.

if (UNIX AND NOT APPLE)

    target_link_libraries(c_ext_wrapper PRIVATE rt)

endif()

  

# Set properties for the output file to be a standard Python extension

set_target_properties(c_ext_wrapper PROPERTIES

    PREFIX ""

    OUTPUT_NAME "c_ext_wrapper"

)
