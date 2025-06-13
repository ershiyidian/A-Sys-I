// src/asys_i/hpc/cpp_ringbuffer/shm_manager.h (REWRITTEN FOR ROBUSTNESS)
#pragma once

#include <string>
#include <atomic>
#include <vector>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include "torch_dtype_codes.h"

namespace bip = boost::interprocess;

constexpr size_t MAX_DIMS = 8;
constexpr size_t MAX_CONSUMERS = 64; // Max number of concurrent consumer processes

// Metadata packet stored in the process-safe message queue.
struct PacketMetadata {
    uint64_t shm_data_offset;
    uint64_t data_size_bytes;
    asys_i_hpc::TorchDtypeCode dtype_code;
    uint16_t ndim;
    uint64_t shape[MAX_DIMS];
    uint32_t checksum;
    int32_t layer_idx_numeric;
    int64_t global_step;
    int64_t timestamp_ns;
};

// Header at the beginning of the tensor data shared memory segment.
struct TensorDataShmHeader {
    std::atomic<uint64_t> write_head;
    std::atomic<uint64_t> read_tails[MAX_CONSUMERS];
    uint64_t capacity;
    uint32_t magic;
    uint32_t version;
    std::atomic<uint8_t> num_registered_consumers;
};

class ShmManager {
public:
    ShmManager(const std::string& tensor_shm_name,
               const std::string& mq_name,
               uint64_t tensor_shm_size_bytes,
               uint64_t max_mq_messages,
               bool create);
    ~ShmManager();

    bool push(const char* tensor_data_ptr,
              uint64_t tensor_data_size,
              asys_i_hpc::TorchDtypeCode dtype_code,
              const std::vector<uint64_t>& shape_vec,
              uint32_t checksum,
              int32_t layer_idx_numeric,
              int64_t global_step);

    size_t pull_batch(std::vector<PacketMetadata>& metadata_batch_vec, size_t max_batch_size);

    char* get_tensor_data_shm_ptr() const;
    bool is_valid() const;

    int register_consumer();
    void update_consumer_tail(int consumer_id, uint64_t new_tail_offset);

private:
    bool allocate_space_in_tensor_shm(uint64_t required_size, uint64_t& allocated_offset);
    uint64_t get_slowest_consumer_tail() const;

    std::string tensor_shm_name_;
    std::string mq_name_;
    bip::managed_shared_memory tensor_data_segment_;
    std::unique_ptr<bip::message_queue> metadata_mq_;

    bool owner_;

    TensorDataShmHeader* tensor_data_header_ = nullptr;
    char* tensor_data_region_ = nullptr;

    static constexpr uint32_t TENSOR_SHM_MAGIC = 0xDA7A0002; // Version bump
    static constexpr uint32_t TENSOR_SHM_VERSION = 2;
};
