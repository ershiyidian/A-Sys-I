import torch
import enum

# This map is the single source of truth for dtype-to-code mapping.
# The C++ enum `TorchDtypeCode` is generated from this map at build time.
# See: scripts/generate_cpp_header.py
DTYPE_TO_CODE_MAP = {
    'torch.float32': 0,
    'torch.float64': 1,
    'torch.complex64': 2,
    'torch.complex128': 3,
    'torch.float16': 4,
    'torch.bfloat16': 5,
    'torch.uint8': 6,
    'torch.int8': 7,
    'torch.int16': 8,
    'torch.int32': 9,
    'torch.int64': 10,
    'torch.bool': 11,
}

# Reverse mapping for convenience
CODE_TO_DTYPE_MAP = {v: getattr(torch, k.split('.')[-1]) for k, v in DTYPE_TO_CODE_MAP.items()}

# Generic type for component identifiers
ComponentID = str

class DataBusType(enum.Enum):
    CPP_SHARDED_SPMC = "cpp_sharded_spmc"
    PYTHON_QUEUE = "python_queue"
