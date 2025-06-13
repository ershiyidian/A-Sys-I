// src/asys_i/hpc/cpp_ringbuffer/shm_manager.cpp (REWRITTEN)
#include "shm_manager.h"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <cstring>

ShmManager::ShmManager(const std::string& tensor_shm_name,
                       const std::string& mq_name,
                       uint64_t tensor_shm_size_bytes,
                       uint64_t max_mq_messages,
                       bool create)
    : tensor_shm_name_(tensor_shm_name), mq_name_(mq_name), owner_(create) {

    try {
        if (create) {
            bip::shared_memory_object::remove(tensor_shm_name_.c_str());
            bip::message_queue::remove(mq_name_.c_str());

            tensor_data_segment_ = bip::managed_shared_memory(bip::create_only, tensor_shm_name_.c_str(), tensor_shm_size_bytes);
            metadata_mq_ = std::make_unique<bip::message_queue>(bip::create_only, mq_name_.c_str(), max_mq_messages, sizeof(PacketMetadata));

            tensor_data_header_ = tensor_data_segment_.construct<TensorDataShmHeader>("TensorDataShmHeader")();
            tensor_data_header_->write_head.store(0, std::memory_order_relaxed);
            for (size_t i = 0; i < MAX_CONSUMERS; ++i) {
                tensor_data_header_->read_tails[i].store(0, std::memory_order_relaxed);
            }
            tensor_data_header_->magic = TENSOR_SHM_MAGIC;
            tensor_data_header_->version = TENSOR_SHM_VERSION;
            tensor_data_header_->num_registered_consumers.store(0, std::memory_order_relaxed);

            tensor_data_region_ = static_cast<char*>(tensor_data_segment_.get_address()) + sizeof(TensorDataShmHeader);
            tensor_data_header_->capacity = tensor_shm_size_bytes - sizeof(TensorDataShmHeader);

        } else {
            tensor_data_segment_ = bip::managed_shared_memory(bip::open_only, tensor_shm_name_.c_str());
            metadata_mq_ = std::make_unique<bip::message_queue>(bip::open_only, mq_name_.c_str());

            auto header_res = tensor_data_segment_.find<TensorDataShmHeader>("TensorDataShmHeader");
            if (!header_res.first || header_res.first->magic != TENSOR_SHM_MAGIC) {
                throw std::runtime_error("Invalid or non-existent TensorDataShmHeader found in SHM: " + tensor_shm_name_);
            }
            tensor_data_header_ = header_res.first;
            tensor_data_region_ = static_cast<char*>(tensor_data_segment_.get_address()) + sizeof(TensorDataShmHeader);
        }
    } catch (const bip::interprocess_exception& ex) {
        std::cerr << "Boost IPC Exception in ShmManager: " << ex.what() << std::endl;
        throw;
    }
}

ShmManager::~ShmManager() {
    if (owner_) {
        bip::shared_memory_object::remove(tensor_shm_name_.c_str());
        bip::message_queue::remove(mq_name_.c_str());
    }
}

uint64_t ShmManager::get_slowest_consumer_tail() const {
    uint8_t consumer_count = tensor_data_header_->num_registered_consumers.load(std::memory_order_acquire);
    if (consumer_count == 0) {
        // If no consumers, the entire buffer is free up to the write head's own position
        return tensor_data_header_->write_head.load(std::memory_order_acquire);
    }

    uint64_t slowest_tail = -1; // Equivalent to max uint64_t
    for (uint8_t i = 0; i < consumer_count; ++i) {
        uint64_t current_tail = tensor_data_header_->read_tails[i].load(std::memory_order_acquire);
        if (slowest_tail == (uint64_t)-1 || current_tail < slowest_tail) {
            slowest_tail = current_tail;
        }
    }
    return slowest_tail;
}


bool ShmManager::allocate_space_in_tensor_shm(uint64_t required_size, uint64_t& allocated_offset) {
    if (!is_valid() || required_size > tensor_data_header_->capacity) return false;

    uint64_t current_head = tensor_data_header_->write_head.load(std::memory_order_acquire);
    uint64_t slowest_tail = get_slowest_consumer_tail();

    uint64_t available_space;
    if (current_head >= slowest_tail) {
        available_space = tensor_data_header_->capacity - (current_head - slowest_tail);
    } else { // Head has wrapped around
        available_space = slowest_tail - current_head;
    }

    if (required_size >= available_space) { // Use >= because we need 1 byte free for ambiguity
        return false; // Not enough space
    }

    uint64_t offset = current_head % tensor_data_header_->capacity;
    if (offset + required_size <= tensor_data_header_->capacity) {
        // Fits without wrapping
        allocated_offset = offset;
        tensor_data_header_->write_head.fetch_add(required_size, std::memory_order_release);
    } else {
        // Doesn't fit at end, check if it fits at the beginning
        if (required_size >= slowest_tail) { // Check against the original value of slowest_tail
            return false; // Not enough space even after wrapping
        }
        // "Lose" the space at the end and wrap around
        uint64_t new_head = current_head + (tensor_data_header_->capacity - offset) + required_size;
        allocated_offset = 0;
        tensor_data_header_->write_head.store(new_head, std::memory_order_release);
    }
    return true;
}

bool ShmManager::push(const char* tensor_data_ptr,
                      uint64_t tensor_data_size,
                      asys_i_hpc::TorchDtypeCode dtype_code,
                      const std::vector<uint64_t>& shape_vec,
                      uint32_t checksum,
                      int32_t layer_idx_numeric,
                      int64_t global_step) {
    if (!is_valid()) return false;

    uint64_t shm_offset;
    if (!allocate_space_in_tensor_shm(tensor_data_size, shm_offset)) {
        return false; // Backpressure signal
    }

    std::memcpy(tensor_data_region_ + shm_offset, tensor_data_ptr, tensor_data_size);

    PacketMetadata meta;
    meta.shm_data_offset = shm_offset;
    meta.data_size_bytes = tensor_data_size;
    meta.dtype_code = dtype_code;
    meta.ndim = static_cast<uint16_t>(shape_vec.size());
    if (meta.ndim > MAX_DIMS) {
        std::cerr << "ShmManager::push: Tensor ndim " << meta.ndim << " exceeds MAX_DIMS " << MAX_DIMS << std::endl;
        return false;
    }
    std::copy(shape_vec.begin(), shape_vec.end(), meta.shape);
    meta.checksum = checksum;
    meta.layer_idx_numeric = layer_idx_numeric;
    meta.global_step = global_step;
    meta.timestamp_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now()
    ).time_since_epoch().count();

    if (!metadata_mq_->try_send(&meta, sizeof(PacketMetadata), 0)) {
        // MQ is full. The allocated SHM space is now orphaned.
        // A more complex system might try to "reclaim" it, but for now this is a data drop.
        return false;
    }
    return true;
}

size_t ShmManager::pull_batch(std::vector<PacketMetadata>& metadata_batch_vec, size_t max_batch_size) {
    if (!is_valid()) return 0;
    metadata_batch_vec.resize(max_batch_size); // Ensure vector has space

    size_t received_count = 0;
    unsigned int priority;
    bip::message_queue::size_type recvd_size;

    for (size_t i = 0; i < max_batch_size; ++i) {
        if (metadata_mq_->try_receive(&metadata_batch_vec[i], sizeof(PacketMetadata), recvd_size, priority)) {
            if (recvd_size == sizeof(PacketMetadata)) {
                received_count++;
            }
        } else {
            break; // Queue is empty
        }
    }
    metadata_batch_vec.resize(received_count);
    return received_count;
}

char* ShmManager::get_tensor_data_shm_ptr() const {
    return tensor_data_region_;
}

bool ShmManager::is_valid() const {
    return tensor_data_header_ != nullptr && metadata_mq_ && tensor_data_region_ != nullptr;
}

int ShmManager::register_consumer() {
    if (!is_valid()) return -1;
    uint8_t consumer_id = tensor_data_header_->num_registered_consumers.fetch_add(1, std::memory_order_release);
    if (consumer_id >= MAX_CONSUMERS) {
        tensor_data_header_->num_registered_consumers.fetch_sub(1, std::memory_order_release); // Rollback
        return -1; // Max consumers reached
    }
    // Initialize this consumer's tail to the current write head
    tensor_data_header_->read_tails[consumer_id].store(
        tensor_data_header_->write_head.load(std::memory_order_acquire),
        std::memory_order_release
    );
    return static_cast<int>(consumer_id);
}

void ShmManager::update_consumer_tail(int consumer_id, uint64_t new_tail_offset) {
    if (is_valid() && consumer_id >= 0 && consumer_id < MAX_CONSUMERS) {
        tensor_data_header_->read_tails[consumer_id].store(new_tail_offset, std::memory_order_release);
    }
}
