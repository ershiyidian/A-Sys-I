#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path

def parse_dtype_map_from_file(py_file_path: Path) -> dict:
    """
    Parses the DTYPE_TO_CODE_MAP dictionary from a .py file using regex.
    This avoids importing the project module during the build process,
    making the script more robust and independent of the build environment's sys.path.
    """
    if not py_file_path.exists():
        print(f"Error: Source Python file not found at {py_file_path}", file=sys.stderr)
        sys.exit(1)

    content = py_file_path.read_text(encoding='utf-8')
    # This regex is intentionally simple. It is brittle and assumes a specific
    # coding style, but it is more robust than a build-time import.
    # It looks for a variable assignment `DTYPE_TO_CODE_MAP = { ... }`
    match = re.search(r"DTYPE_TO_CODE_MAP\s*=\s*\{([^}]+)\}", content, re.DOTALL)
    if not match:
        print(f"Error: Could not find DTYPE_TO_CODE_MAP dictionary in {py_file_path}", file=sys.stderr)
        sys.exit(1)

    dict_content = match.group(1)
    # This regex extracts 'key': value pairs. It handles single or double quotes
    # around the key and expects an integer value.
    item_pattern = re.compile(r"['\"](torch\.[a-zA-Z0-9_]+)['\"]\s*:\s*(\d+)")
    dtype_map = {}
    for line in dict_content.splitlines():
        item_match = item_pattern.search(line)
        if item_match:
            dtype_map[item_match.group(1)] = int(item_match.group(2))

    if not dtype_map:
        print(f"Error: DTYPE_TO_CODE_MAP dictionary was found, but no valid entries could be parsed.", file=sys.stderr)
        sys.exit(1)
    return dtype_map

def generate_header_content(dtype_map: dict) -> str:
    """Generates the C++ header file content from the parsed dtype map."""
    header = """//
// Created by A-Sys-I automated script. DO NOT EDIT.
// Source: src/asys_i/common/types.py
//

#ifndef TORCH_DTYPE_CODES_H
#define TORCH_DTYPE_CODES_H

#include <cstdint>
#include <stdexcept>
#include <string>

namespace asys_i {
namespace hpc {

// Enum defining codes for various Torch dtypes.
// This is auto-generated and must be kept in sync with python's DTYPE_TO_CODE_MAP.
enum class TorchDtypeCode : uint8_t {
"""
    # Sort by code to ensure a stable, deterministic output for the header file
    sorted_dtypes = sorted(dtype_map.items(), key=lambda item: item[1])
    for (dtype_str, code) in sorted_dtypes:
        # 'torch.float32' -> 'float32'
        enum_name = dtype_str.split('.')[-1]
        header += f"    {enum_name} = {code},\n"
    header += """};

// Helper function to get a string representation for a TorchDtypeCode
inline std::string to_string(TorchDtypeCode code) {
    switch (code) {
"""

    for (dtype_str, code) in sorted_dtypes:
        enum_name = dtype_str.split('.')[-1]
        header += f'        case TorchDtypeCode::{enum_name}: return "torch.{enum_name}";\n'

    header += """        default:
            throw std::runtime_error("Unknown TorchDtypeCode: " + std::to_string(static_cast<int>(code)));
    }
}

// Helper function to get the size in bytes for a TorchDtypeCode
inline size_t get_dtype_size_bytes(TorchDtypeCode code) {
    switch (code) {
"""
    # Mapping from Python dtype string (short form like 'float32') to C++ size expression
    # This map should cover all dtypes present in DTYPE_TO_CODE_MAP
    dtype_shortname_to_size_expr = {
        "float32": "sizeof(float)",
        "float64": "sizeof(double)",  # torch.double
        "double": "sizeof(double)",   # Alias for float64
        "complex64": "sizeof(float) * 2",
        "complex128": "sizeof(double) * 2", # torch.cdouble
        "cdouble": "sizeof(double) * 2",  # Alias for complex128
        "float16": "2",  # sizeof(uint16_t) effectively, but hardcoded for simplicity as no std::float16_t yet
        "half": "2",     # Alias for float16
        "bfloat16": "2", # sizeof(uint16_t) effectively
        "uint8": "sizeof(uint8_t)",
        "int8": "sizeof(int8_t)",
        "int16": "sizeof(int16_t)",   # torch.short
        "short": "sizeof(int16_t)",   # Alias for int16
        "int32": "sizeof(int32_t)",   # torch.int
        "int": "sizeof(int32_t)",     # Alias for int32
        "int64": "sizeof(int64_t)",   # torch.long
        "long": "sizeof(int64_t)",    # Alias for int64
        "bool": "sizeof(bool)"        # Or sizeof(uint8_t) if specific packing is needed
    }

    for (dtype_str, _) in sorted_dtypes:
        enum_name = dtype_str.split('.')[-1] # e.g. "float32" from "torch.float32"
        size_expr = dtype_shortname_to_size_expr.get(enum_name)
        if size_expr is None:
            # This case should ideally not be reached if dtype_shortname_to_size_expr is comprehensive
            # and DTYPE_TO_CODE_MAP only contains supported types.
            # However, as a fallback, one might add an error or a default size.
            # For now, we assume all dtypes in DTYPE_TO_CODE_MAP will be in our mapping.
            # If a new dtype is added to Python map, it must be added here too.
            # Consider raising an error during script execution if a mapping is missing.
            print(f"Warning: No size mapping for dtype '{enum_name}' (from '{dtype_str}') in generate_cpp_header.py. It will lead to C++ compilation error.", file=sys.stderr)
            header += f'        // case TorchDtypeCode::{enum_name}: /* ERROR: No size mapping in script */ break;\n'
        else:
            header += f"        case TorchDtypeCode::{enum_name}: return {size_expr};\n"

    header += """        default:
            throw std::runtime_error("Unsupported or unknown TorchDtypeCode in get_dtype_size_bytes: " + std::to_string(static_cast<int>(code)));
    }
}

} // namespace hpc
} // namespace asys_i

#endif // TORCH_DTYPE_CODES_H
"""
    return header

def generate_pybind_enum_registration_content(dtype_map: dict) -> str:
    """Generates the C++ pybind11 enum registration code content."""
    content = """//
// Created by A-Sys-I automated script. DO NOT EDIT.
// Source: src/asys_i/common/types.py (via generate_cpp_header.py)
//

#ifndef TORCH_DTYPE_PYBIND_BINDINGS_INC
#define TORCH_DTYPE_PYBIND_BINDINGS_INC

#include <pybind11/pybind11.h>
#include "torch_dtype_codes.h" // Generated header with TorchDtypeCode enum

namespace py = pybind11;

namespace asys_i {
namespace hpc {

// Static function to register the TorchDtypeCode enum with pybind11
static void register_torch_dtype_enum(py::module_ &m) {
    py::enum_<TorchDtypeCode>(m, "TorchDtypeCode", "Enum for Torch dtypes")
"""
    # Sort by code to ensure a stable, deterministic output
    sorted_dtypes = sorted(dtype_map.items(), key=lambda item: item[1])
    for (dtype_str, _) in sorted_dtypes:
        # 'torch.float32' -> 'float32'
        enum_name_cpp = dtype_str.split('.')[-1]
        # 'float32' -> 'FLOAT32'
        enum_name_python = enum_name_cpp.upper()
        content += f'        .value("{enum_name_python}", TorchDtypeCode::{enum_name_cpp})\n'

    content += """        .export_values();
}

} // namespace hpc
} // namespace asys_i

#endif // TORCH_DTYPE_PYBIND_BINDINGS_INC
"""
    return content

def write_if_changed(target_path: Path, new_content: str):
    """Writes the new_content to target_path only if it differs from existing content."""
    os.makedirs(target_path.parent, exist_ok=True)
    existing_content = ""
    if target_path.exists():
        with open(target_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()

    if new_content != existing_content:
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"File '{target_path.name}' generated/updated successfully at {target_path.parent}.")
        return True
    else:
        print(f"File '{target_path.name}' is already up-to-date. No changes made.")
        return False

def main():
    """Main function to parse the Python source and generate C++ files."""
    # Assume this script is in project_root/scripts/
    project_root = Path(__file__).parent.parent
    source_py_path = project_root / 'src' / 'asys_i' / 'common' / 'types.py'

    print(f"Parsing dtype map from: {source_py_path}")
    dtype_map = parse_dtype_map_from_file(source_py_path)

    # Generate C++ header for TorchDtypeCode enum
    target_header_path = project_root / 'src' / 'asys_i' / 'hpc' / 'cpp_ringbuffer' / 'torch_dtype_codes.h'
    print(f"Processing C++ header file: {target_header_path.name}")
    header_content = generate_header_content(dtype_map)
    try:
        write_if_changed(target_header_path, header_content)
    except IOError as e:
        print(f"Error writing to file {target_header_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate C++ include for pybind11 enum registration
    target_pybind_path = project_root / 'src' / 'asys_i' / 'hpc' / 'src' / 'torch_dtype_pybind_bindings.inc'
    print(f"Processing C++ pybind include file: {target_pybind_path.name}")
    pybind_content = generate_pybind_enum_registration_content(dtype_map)
    try:
        write_if_changed(target_pybind_path, pybind_content)
    except IOError as e:
        print(f"Error writing to file {target_pybind_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
