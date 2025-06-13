// src/asys_i/hpc/cpp_ringbuffer/torch_dtype_codes.h (NEW FILE)
#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>

namespace asys_i_hpc {

// Enum to represent common torch dtypes numerically.
// This MUST BE KEPT IN SYNC with Python's DTYPE_TO_CODE_MAP.
enum class TorchDtypeCode : uint16_t {
    UNKNOWN = 0,
    FLOAT32 = 1,
    FLOAT16 = 2,
    BFLOAT16 = 3,
    INT64 = 4,
    INT32 = 5,
    INT16 = 6,
    INT8 = 7,
    UINT8 = 8,
};

// Helper to get byte size from dtype code. Crucial for SHM calculations.
inline uint16_t get_dtype_size_bytes(TorchDtypeCode code) {
    switch (code) {
        case TorchDtypeCode::FLOAT32: return 4;
        case TorchDtypeCode::FLOAT16: return 2;
        case TorchDtypeCode::BFLOAT16: return 2;
        case TorchDtypeCode::INT64: return 8;
        case TorchDtypeCode::INT32: return 4;
        case TorchDtypeCode::INT16: return 2;
        case TorchDtypeCode::INT8: return 1;
        case TorchDtypeCode::UINT8: return 1;
        default: throw std::runtime_error("Unknown dtype code for size calculation");
    }
}

} // namespace asys_i_hpc
