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

        header += f"    {enum_name} = {code},\n"

    header += """};

  

// Helper function to get a string representation for a TorchDtypeCode

inline std::string to_string(TorchDtypeCode code) {

    switch (code) {

"""

  

    for (dtype_str, code) in sorted_dtypes:

        enum_name = dtype_str.split('.')[-1]

        header += f'        case TorchDtypeCode::{enum_name}: return "torch.{enum_name}";\n'

  

    header += """        default:

            throw std::runtime_error("Unknown TorchDtypeCode: " + std::to_string(static_cast<int>(code)));

    }

}

  

} // namespace hpc

} // namespace asys_i

  

#endif // TORCH_DTYPE_CODES_H

"""

    return header

  

def main():

    """Main function to parse the Python source and generate the C++ header."""

    # Assume this script is in project_root/scripts/

    project_root = Path(__file__).parent.parent

    source_py_path = project_root / 'src' / 'asys_i' / 'common' / 'types.py'

    target_header_path = project_root / 'src' / 'asys_i' / 'hpc' / 'cpp_ringbuffer' / 'torch_dtype_codes.h'

    print(f"Parsing dtype map from: {source_py_path}")

    dtype_map = parse_dtype_map_from_file(source_py_path)

    print(f"Generating C++ header file at: {target_header_path}")

    content = generate_header_content(dtype_map)

    os.makedirs(target_header_path.parent, exist_ok=True)

    try:

        # Only write the file if the content has changed to avoid unnecessary

        # recompilations of the C++ extension.

        existing_content = ""

        if os.path.exists(target_header_path):

            with open(target_header_path, 'r', encoding='utf-8') as f:

                existing_content = f.read()

  

        if content != existing_content:

            with open(target_header_path, 'w', encoding='utf-8') as f:

                f.write(content)

            print("Header file generated/updated successfully.")

        else:

            print("Header file is already up-to-date. No changes made.")

  

    except IOError as e:

        print(f"Error writing to file {target_header_path}: {e}", file=sys.stderr)

        sys.exit(1)

  

if __name__ == "__main__":

    main()
