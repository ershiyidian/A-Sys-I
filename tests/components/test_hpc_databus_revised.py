# tests/components/test_hpc_databus_revised.py
import pytest
import torch
import numpy as np
import os
import time
import multiprocessing
from unittest.mock import MagicMock

from asys_i.hpc import CPP_EXTENSION_AVAILABLE, SHM_NAME_PREFIX, MQ_NAME_PREFIX
if CPP_EXTENSION_AVAILABLE:
    from asys_i.hpc import c_ext_wrapper # The pybind11 module
    from asys_i.components.data_bus_hpc import CppShardedSPMCBus
    from asys_i.common.types import (
        ActivationPacket, RunProfile, TensorRef,
        DTYPE_TO_CODE_MAP, CODE_TO_DTYPE_MAP,
        get_str_from_torch_dtype, calculate_checksum
    )
else: # Define placeholders if C++ ext not available so file can be parsed
    c_ext_wrapper = None 
    CppShardedSPMCBus = None
    ActivationPacket, RunProfile, TensorRef = None, None, None
    DTYPE_TO_CODE_MAP, CODE_TO_DTYPE_MAP = {}, {}
    get_str_from_torch_dtype = lambda x: ""
    calculate_checksum = lambda x: 0


from asys_i.orchestration.config_loader import MasterConfig

# Skip all tests in this file if C++ extension is not available
pytestmark = pytest.mark.skipif(not CPP_EXTENSION_AVAILABLE, reason="HPC C++ extension ('c_ext_wrapper') not available or failed to load.")

# --- Fixture for CppShardedSPMCBus Instance ---
@pytest.fixture
def hpc_bus_fixture(hpc_config_minimal: MasterConfig, mock_monitor_for_tests: MagicMock):
    # Ensure unique SHM and MQ names for each test run
    instance_id_test = f"pytest_{os.getpid()}_{int(time.time_ns() / 1000)}"
    hpc_config_minimal.data_bus.shared_memory_size_gb = 0.02 # 20MB is enough for these tests
    hpc_config_minimal.data_bus.buffer_size_per_shard = 50   # Max MQ messages

    bus: Optional[CppShardedSPMCBus] = None
    tensor_shm_name_created = f"{SHM_NAME_PREFIX}tensor_{instance_id_test}"
    mq_name_created = f"{MQ_NAME_PREFIX}mq_{instance_id_test}"

    # Override names in config for the bus instance (if bus uses them directly, it does)
    # This is a bit of a hack; ideally bus generates names and exposes them.
    # The CppShardedSPMCBus now generates names internally, so this direct override isn't needed
    # if the fixture can access bus.tensor_shm_name and bus.mq_name after creation.

    try:
        bus = CppShardedSPMCBus(hpc_config_minimal, mock_monitor_for_tests)
        # Update names for cleanup based on what the bus actually created
        tensor_shm_name_created = bus.tensor_shm_name
        mq_name_created = bus.mq_name
        yield bus
    finally:
        if bus:
            bus.shutdown() # This should trigger C++ ShmManager destructor to unlink/remove

        # Additional explicit cleanup as a safeguard, especially if C++ part fails
        # Try to clean Python's SharedMemory mapping if it exists
        try:
            py_shm = multiprocessing.shared_memory.SharedMemory(name=tensor_shm_name_created)
            py_shm.close()
            py_shm.unlink()
            print(f"Test fixture explicitly unlinked Python SHM: {tensor_shm_name_created}")
        except FileNotFoundError:
            pass # Already cleaned or never fully created by Python side
        except Exception as e_py_shm_clean:
            print(f"Error during test fixture Python SHM cleanup for {tensor_shm_name_created}: {e_py_shm_clean}")
        
        # For Boost IPC objects, direct removal from Python is hard.
        # The C++ destructor in ShmManager is primary.
        # If issues, manual `ipcrm` or `/dev/shm` checks might be needed locally.
        # We can try to call the static remove methods from Boost via a C++ helper if truly stuck,
        # but that's beyond typical pytest fixtures.
        if c_ext_wrapper and hasattr(c_ext_wrapper, 'ShmManager'): # Check if C++ module loaded
            # Conceptual: If we had a static C++ cleanup exposed via pybind:
            # c_ext_wrapper.cleanup_ipc_resource(tensor_shm_name_created, "shm")
            # c_ext_wrapper.cleanup_ipc_resource(mq_name_created, "mq")
            pass


def test_hpc_bus_basic_creation_and_validity(hpc_bus_fixture: CppShardedSPMCBus):
    assert hpc_bus_fixture is not None
    assert hpc_bus_fixture._is_ready
    assert hpc_bus_fixture.cpp_manager is not None
    assert hpc_bus_fixture.cpp_manager.is_valid()
    
    # Check if SHM segment was created by C++ (Python maps it)
    assert hasattr(hpc_bus_fixture, '_tensor_data_shm_obj')
    assert hpc_bus_fixture._tensor_data_shm_obj is not None
    assert hpc_bus_fixture._tensor_data_shm_obj.name == hpc_bus_fixture.tensor_shm_name


def test_hpc_bus_push_one_pull_one_tensor(hpc_bus_fixture: CppShardedSPMCBus, hpc_config_minimal: MasterConfig):
    bus = hpc_bus_fixture
    test_dtype = torch.float32
    original_tensor = torch.arange(20, dtype=test_dtype).reshape(4, 5)
    
    # Create an ActivationPacket (Python side)
    # The `data` field is the raw torch.Tensor for push()
    packet_to_push = ActivationPacket(
        layer_name="hpc.test.layer", layer_idx_numeric=7, global_step=101,
        data=original_tensor, profile=RunProfile.HPC,
        timestamp_ns=time.time_ns(), meta={"test_info": "push_one"}
    )

    push_success = bus.push(packet_to_push)
    assert push_success, "bus.push() failed"
    bus.monitor.log_metric.assert_any_call("data_bus_push_count", 1, tags=ANY)

    # Register a consumer and pull
    consumer_id = "test_consumer_hpc_1"
    bus.register_consumer(consumer_id, [7]) # Register for numeric layer index 7

    pulled_packets = bus.pull_batch(consumer_id, batch_size=5)
    assert len(pulled_packets) == 1, "Did not pull exactly one packet"
    bus.monitor.log_metric.assert_any_call("data_bus_pull_count", 1, tags=ANY)
    
    pulled_py_packet: ActivationPacket = pulled_packets[0]
    assert isinstance(pulled_py_packet.data, TensorRef), "Pulled packet data is not a TensorRef"
    
    # --- Reconstruct tensor from TensorRef to verify ---
    tensor_ref: TensorRef = pulled_py_packet.data
    
    # Get the SHM view from the bus instance (as a worker would in __setstate__)
    # This assumes bus._tensor_data_shm_obj.buf is the correct memoryview
    shm_view_for_test = bus._tensor_data_shm_obj.buf # type: ignore

    reconstructed_tensor = SAETrainerWorker._reconstruct_tensor_from_ref( # type: ignore
        # Mock a SAETrainerWorker instance just enough for this method
        MagicMock(
            config=hpc_config_minimal, # Needs config for use_checksum
            shm_buffer_view_for_read=shm_view_for_test,
            monitor=bus.monitor, # Use the bus's monitor for checksum error logging
            tags={}, worker_id="test_reconstructor"
        ),
        tensor_ref
    )
    
    assert torch.allclose(reconstructed_tensor, original_tensor), "Reconstructed tensor data mismatch"
    assert pulled_py_packet.layer_idx_numeric == packet_to_push.layer_idx_numeric
    assert pulled_py_packet.global_step == packet_to_push.global_step
    
    if hpc_config_minimal.data_bus.use_checksum:
        expected_checksum = calculate_checksum(original_tensor)
        assert tensor_ref.checksum == expected_checksum, "Checksum mismatch"

def test_hpc_bus_dtype_and_shape_preservation(hpc_bus_fixture: CppShardedSPMCBus, hpc_config_minimal: MasterConfig):
    bus = hpc_bus_fixture
    # Test with a different dtype and more complex shape
    test_dtype = torch.bfloat16 # Requires CUDA or recent CPU for full support
    if hpc_config_minimal.hardware.device == "cpu" and not hasattr(torch.ops.aten, 'empty_like_bfloat16'):
         # Skip if CPU doesn't support bfloat16 well for torch.tensor creation
         # A better check might be needed depending on PyTorch version.
         try:
             torch.tensor(1.0, dtype=torch.bfloat16)
         except Exception:
            pytest.skip("torch.bfloat16 not well supported on this CPU PyTorch build for tensor creation.")

    original_tensor = torch.randn(2, 3, 4, 2, dtype=test_dtype) # 4D tensor

    packet_to_push = ActivationPacket(
        layer_name="hpc.bf16.layer", layer_idx_numeric=8, global_step=102,
        data=original_tensor, profile=RunProfile.HPC,
        timestamp_ns=time.time_ns(), meta={"dtype_shape_test": True}
    )
    push_success = bus.push(packet_to_push)
    assert push_success

    bus.register_consumer("consumer_bf16", [8])
    pulled_packets = bus.pull_batch("consumer_bf16", 1)
    assert len(pulled_packets) == 1
    
    pulled_py_packet = pulled_packets[0]
    tensor_ref: TensorRef = pulled_py_packet.data
    
    # Verify dtype_code and shape in TensorRef
    expected_dtype_code = DTYPE_TO_CODE_MAP.get(original_tensor.dtype)
    assert expected_dtype_code is not None, f"Test dtype {original_tensor.dtype} not in DTYPE_TO_CODE_MAP"
    assert tensor_ref.dtype_code == expected_dtype_code
    assert tensor_ref.ndim == original_tensor.ndim
    assert tuple(tensor_ref.shape) == original_tensor.shape # pybind converts C array to tuple
    assert tensor_ref.data_size_bytes == original_tensor.nelement() * original_tensor.element_size()

    # Reconstruct and verify (using the same mock worker approach)
    shm_view_for_test = bus._tensor_data_shm_obj.buf # type: ignore
    reconstructed_tensor = SAETrainerWorker._reconstruct_tensor_from_ref( # type: ignore
        MagicMock(config=hpc_config_minimal, shm_buffer_view_for_read=shm_view_for_test, monitor=bus.monitor, tags={}, worker_id="bf16_reconstructor"),
        tensor_ref
    )
    assert reconstructed_tensor.dtype == original_tensor.dtype
    assert reconstructed_tensor.shape == original_tensor.shape
    assert torch.allclose(reconstructed_tensor.float(), original_tensor.float()), "BF16 tensor data mismatch (compared as float)" # Compare as float32 due to bfloat16 precision

def test_hpc_bus_multiprocess_pickle_unpickle_worker_access(hpc_config_minimal: MasterConfig, mock_monitor_for_tests):
    # This test verifies that CppShardedSPMCBus can be pickled, sent to another process,
    # unpickled, and then used by that worker process to pull data pushed by the main process.

    # 1. Main process creates the bus (and SHM/MQ)
    main_bus = CppShardedSPMCBus(hpc_config_minimal, mock_monitor_for_tests)
    assert main_bus.cpp_manager.is_valid(), "Main bus C++ manager invalid after creation"

    # 2. Data to be pushed by main, pulled by worker
    original_tensor = torch.tensor([[10., 20.], [30., 40.]], dtype=torch.float32)
    packet_main_push = ActivationPacket("mp_layer", 9, 201, original_tensor, RunProfile.HPC, time.time_ns(), {})

    # 3. Worker process target function
    def worker_process_fn(pickled_bus_state, result_queue):
        try:
            # Create a new MasterConfig for the worker (or pickle the relevant parts)
            # For simplicity, re-create; a real app might pass more context.
            # The key is that __setstate__ uses the *names* from pickled_bus_state to reconnect.
            worker_config = MasterConfig.model_validate(pickled_bus_state['master_config_dict'])
            worker_monitor = pickled_bus_state['monitor'] # Assume monitor is pickleable

            worker_bus = CppShardedSPMCBus.__new__(CppShardedSPMCBus) # Create uninitialized instance
            worker_bus.__setstate__(pickled_bus_state) # Unpickle

            assert worker_bus._is_ready, "Worker bus not ready after unpickle"
            # CppManager in worker is different instance, but should point to same SHM/MQ
            assert worker_bus.cpp_manager is not None and worker_bus.cpp_manager.is_valid(), "Worker bus C++ manager invalid"
            
            # Worker registers and pulls
            worker_bus.register_consumer("worker_consumer_mp", [9])
            
            # Wait a bit for main to push if needed (though push happens before worker starts here)
            time.sleep(0.2)
            pulled_by_worker = worker_bus.pull_batch("worker_consumer_mp", 1)
            
            if not pulled_by_worker:
                result_queue.put(ValueError("Worker pulled no packets"))
                return

            # Reconstruct (simplified for test, real worker has _reconstruct_tensor_from_ref)
            ref_in_worker: TensorRef = pulled_by_worker[0].data
            
            # Worker needs its own SHM view. This is established in CppDataBus.__setstate__
            # via _tensor_data_shm_obj_worker and _tensor_data_shm_view_for_pull
            assert hasattr(worker_bus, '_tensor_data_shm_view_for_pull')
            shm_view_in_worker = worker_bus._tensor_data_shm_view_for_pull
            
            dtype_w = CODE_TO_DTYPE_MAP[ref_in_worker.dtype_code]
            np_dtype_w = np.dtype('bfloat16') if dtype_w == torch.bfloat16 else np.dtype(dtype_w.name)
            itemsize_w = np_dtype_w.itemsize
            numel_w = ref_in_worker.data_size_bytes // itemsize_w

            slice_w = shm_view_in_worker[ref_in_worker.shm_data_offset : ref_in_worker.shm_data_offset + ref_in_worker.data_size_bytes]
            arr_w = np.frombuffer(slice_w, dtype=np_dtype_w, count=numel_w).reshape(ref_in_worker.shape)
            tensor_w = torch.from_numpy(arr_w.copy()) # Copy for safety in queue

            result_queue.put(tensor_w)
            worker_bus.shutdown() # Worker cleans up its C++ manager and SHM view object

        except Exception as e_worker:
            result_queue.put(e_worker)


    # 4. Prepare for multiprocessing
    ctx = multiprocessing.get_context("spawn") # Use spawn for cleaner separation
    result_queue = ctx.Queue()
    
    # Pickle the bus state (this calls __getstate__)
    # Ensure __getstate__ only returns pickleable info (names, config), not live C++ objects
    pickled_main_bus_state = main_bus.__getstate__()

    worker_proc = ctx.Process(target=worker_process_fn, args=(pickled_main_bus_state, result_queue))
    worker_proc.start()

    # 5. Main process pushes data after worker has started (or before, order depends on test goal)
    time.sleep(0.1) # Give worker a moment to start and potentially register
    push_ok_main = main_bus.push(packet_main_push)
    assert push_ok_main, "Main process failed to push packet"

    # 6. Get result from worker
    worker_proc.join(timeout=5.0) # Wait for worker to finish
    assert not worker_proc.is_alive(), "Worker process did not terminate"
    assert worker_proc.exitcode == 0, f"Worker process exited with error code {worker_proc.exitcode}"

    try:
        result_from_worker = result_queue.get(timeout=1.0)
    except Exception: # queue.Empty or other
        pytest.fail("Worker did not put a result in the queue.")

    if isinstance(result_from_worker, Exception):
        raise result_from_worker # Propagate exception from worker

    assert isinstance(result_from_worker, torch.Tensor), "Worker did not return a tensor"
    assert torch.allclose(result_from_worker, original_tensor), "Tensor mismatch between main push and worker pull"

    # 7. Main process shuts down its bus (this should unlink SHM/MQ as it was creator)
    main_bus.shutdown()
