// src/asys_i/hpc/src/bindings.cpp (REWRITTEN)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_ringbuffer/shm_manager.h"
#include "torch_dtype_pybind_bindings.inc" // Generated enum bindings

namespace py = pybind11;

class ShmManagerWrapper {
public:
    ShmManagerWrapper(const std::string& tensor_shm_name,
                      const std::string& mq_name,
                      uint64_t tensor_shm_size_bytes,
                      uint64_t max_mq_messages,
                      bool create) {
        py::gil_scoped_release release_gil;
        manager_ = std::make_unique<ShmManager>(tensor_shm_name, mq_name, tensor_shm_size_bytes, max_mq_messages, create);
    }

    bool push(py::array tensor_np,
              asys_i_hpc::TorchDtypeCode dtype_code,
              uint32_t checksum,
              int32_t layer_idx_numeric,
              int64_t global_step) {
        py::buffer_info buf_info = tensor_np.request();
        if (buf_info.ptr == nullptr) {
            throw std::runtime_error("NumPy array buffer is null.");
        }
        const char* data_ptr = static_cast<const char*>(buf_info.ptr);
        uint64_t data_size_bytes = buf_info.size * buf_info.itemsize;

        if (buf_info.itemsize != asys_i_hpc::get_dtype_size_bytes(dtype_code)) {
            throw std::runtime_error("Itemsize mismatch between NumPy array and specified dtype code.");
        }

        std::vector<uint64_t> shape_vec;
        for (py::ssize_t i = 0; i < buf_info.ndim; ++i) {
            shape_vec.push_back(static_cast<uint64_t>(buf_info.shape[i]));
        }

        py::gil_scoped_release release_gil;
        return manager_->push(data_ptr, data_size_bytes, dtype_code, shape_vec, checksum, layer_idx_numeric, global_step);
    }

    std::vector<PacketMetadata> pull_batch(size_t max_batch_size) {
        std::vector<PacketMetadata> batch_vec;
        batch_vec.reserve(max_batch_size);
        py::gil_scoped_release release_gil;
        manager_->pull_batch(batch_vec, max_batch_size);
        return batch_vec;
    }

    uintptr_t get_tensor_data_shm_ptr_as_int() {
        return reinterpret_cast<uintptr_t>(manager_->get_tensor_data_shm_ptr());
    }

    bool is_valid() {
        return manager_ ? manager_->is_valid() : false;
    }
    
    int register_consumer() {
        py::gil_scoped_release release_gil;
        return manager_->register_consumer();
    }

    void update_consumer_tail(int consumer_id, uint64_t new_tail_offset) {
        py::gil_scoped_release release_gil;
        manager_->update_consumer_tail(consumer_id, new_tail_offset);
    }

private:
    std::unique_ptr<ShmManager> manager_;
};

PYBIND11_MODULE(c_ext_wrapper, m) {
    m.doc() = "C++ extension for A-Sys-I High-Performance Data Bus (v0.3.0)";

    // Register the TorchDtypeCode enum using the generated function
    asys_i::hpc::register_torch_dtype_enum(m);

    py::class_<PacketMetadata>(m, "PacketMetadata")
        .def_readonly("shm_data_offset", &PacketMetadata::shm_data_offset)
        .def_readonly("data_size_bytes", &PacketMetadata::data_size_bytes)
        .def_readonly("dtype_code", &PacketMetadata::dtype_code)
        .def_readonly("ndim", &PacketMetadata::ndim)
        .def_property_readonly("shape", [](const PacketMetadata &p) {
            py::list s;
            for (uint16_t i = 0; i < p.ndim; ++i) { s.append(p.shape[i]); }
            return py::tuple(s);
         })
        .def_readonly("checksum", &PacketMetadata::checksum)
        .def_readonly("layer_idx_numeric", &PacketMetadata::layer_idx_numeric)
        .def_readonly("global_step", &PacketMetadata::global_step)
        .def_readonly("timestamp_ns", &PacketMetadata::timestamp_ns);

    py::class_<ShmManagerWrapper>(m, "ShmManager")
        .def(py::init<const std::string&, const std::string&, uint64_t, uint64_t, bool>())
        .def("push", &ShmManagerWrapper::push, py::arg("tensor_numpy_array"), py::arg("dtype_code"), py::arg("checksum"), py::arg("layer_idx_numeric"), py::arg("global_step"))
        .def("pull_batch", &ShmManagerWrapper::pull_batch, py::arg("max_batch_size"))
        .def("get_tensor_data_shm_ptr_as_int", &ShmManagerWrapper::get_tensor_data_shm_ptr_as_int)
        .def("is_valid", &ShmManagerWrapper::is_valid)
        .def("register_consumer", &ShmManagerWrapper::register_consumer)
        .def("update_consumer_tail", &ShmManagerWrapper::update_consumer_tail, py::arg("consumer_id"), py::arg("new_tail_offset"));
}
