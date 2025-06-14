# tests/components/test_hpc_databus_wrapper.py
import pytest
import torch
import numpy as np
import os
import time
import multiprocessing
from unittest.mock import MagicMock

from asys_i.hpc import CPP_EXTENSION_AVAILABLE, SHM_NAME_PREFIX
if CPP_EXTENSION_AVAILABLE:
    from asys_i.components.data_bus_hpc import CppShardedSPMCBus
    from asys_i.common.types import ActivationPacket, RunProfile, TensorRef, get_str_from_torch_dtype, calculate_checksum
    from asys_i.common.types import DTYPE_TO_CODE_MAP # Added for the new test

from asys_i.orchestration.config_loader import MasterConfig

# Skip all tests in this file if C++ extension is not available
pytestmark = pytest.mark.skipif(not CPP_EXTENSION_AVAILABLE, reason="HPC C++ extension not available")

@pytest.fixture
def hpc_bus_instance(hpc_config: MasterConfig, tmp_path):
    # Ensure a unique SHM name for each test run to avoid collisions
    hpc_config.data_bus.shared_memory_size_gb = 0.01 # Small for test
    mock_monitor = MagicMock()
    
    bus = None
    shm_name_for_cleanup = ""
    try:
        bus = CppShardedSPMCBus(hpc_config, mock_monitor)
        shm_name_for_cleanup = bus.shm_name
        yield bus
    finally:
        if bus:
            bus.shutdown()
        # Explicitly clean up the shared memory segment if it exists
        if shm_name_for_cleanup:
            try:
                shm = multiprocessing.shared_memory.SharedMemory(name=shm_name_for_cleanup)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass # Already cleaned
            except Exception as e:
                print(f"Error during test SHM cleanup: {e}")


def test_hpc_bus_creation_and_shutdown(hpc_bus_instance: CppShardedSPMCBus):
    assert hpc_bus_instance is not None
    assert hpc_bus_instance.manager is not None # C++ manager object
    assert hpc_bus_instance._is_ready
    # Check if SHM segment was created (indirectly)
    try:
        shm = multiprocessing.shared_memory.SharedMemory(name=hpc_bus_instance.shm_name, create=False)
        shm.close()
    except FileNotFoundError:
        pytest.fail(f"SHM segment {hpc_bus_instance.shm_name} not found after bus creation.")

    hpc_bus_instance.shutdown()
    assert not hpc_bus_instance._is_ready
    # Check if SHM segment was unlinked
    with pytest.raises(FileNotFoundError):
        multiprocessing.shared_memory.SharedMemory(name=hpc_bus_instance.shm_name, create=False)


def test_hpc_bus_push_pull_simple(hpc_bus_instance: CppShardedSPMCBus, hpc_config: MasterConfig):
    tensor_data = torch.randn(2, 10, dtype=torch.float32) # d_in=10 from hpc_config
    packet_to_push = ActivationPacket(
        layer_name="test_layer.0",
        layer_idx=0,
        global_step=1,
        data=tensor_data,
        profile=RunProfile.HPC,
        timestamp_ns=time.time_ns(),
        meta={"info": "test_packet"}
    )
    
    success = hpc_bus_instance.push(packet_to_push)
    assert success

    # Register a dummy consumer
    consumer_id = "test_consumer_1"
    hpc_bus_instance.register_consumer(consumer_id, [0]) # Subscribes to layer_idx 0

    pulled_packets = hpc_bus_instance.pull_batch(consumer_id, batch_size=1)
    assert len(pulled_packets) == 1
    
    pulled_packet = pulled_packets[0]
    assert isinstance(pulled_packet.data, TensorRef) # Data should be a TensorRef
    
    # Manually reconstruct the tensor from TensorRef for verification
    # This simulates what a worker process would do
    tensor_ref: TensorRef = pulled_packet.data
    
    # Access the SHM buffer directly for test (worker would use its shm_view)
    shm_test_view = multiprocessing.shared_memory.SharedMemory(name=hpc_bus_instance.shm_name)
    
    from asys_i.common.types import get_torch_dtype_from_str # local import for clarity
    dtype_from_ref = get_torch_dtype_from_str(tensor_ref.dtype_str)
    np_dtype = getattr(np, dtype_from_ref.name)

    num_elements = tensor_ref.size // np.dtype(np_dtype).itemsize
    
    reconstructed_np_array = np.frombuffer(
        shm_test_view.buf, dtype=np_dtype, count=num_elements, offset=tensor_ref.offset
    ).reshape(tensor_ref.shape)
    reconstructed_tensor = torch.from_numpy(reconstructed_np_array.copy()) # copy to avoid modifying SHM in test

    shm_test_view.close()

    assert torch.allclose(reconstructed_tensor, tensor_data)
    assert pulled_packet.layer_idx == packet_to_push.layer_idx
    assert pulled_packet.global_step == packet_to_push.global_step
    
    if hpc_config.data_bus.use_checksum:
         assert calculate_checksum(tensor_data) == tensor_ref.checksum


def test_hpc_bus_checksum_error(hpc_bus_instance: CppShardedSPMCBus, hpc_config: MasterConfig):
    if not hpc_config.data_bus.use_checksum:
        pytest.skip("Checksum not enabled in config for this test.")

    tensor_data = torch.ones(2, 10, dtype=torch.float32)
    # Calculate correct checksum
    correct_checksum = calculate_checksum(tensor_data)
    
    # Push with a deliberately incorrect checksum via the C++ extension's push method
    # This requires mocking or directly calling the C++ part if possible,
    # or modifying the packet data before Python push if checksum is calculated there.
    # For now, let's assume the C++ side takes the checksum.
    # We will simulate the C++ PacketMetadata creation with a bad checksum.

    # This test is tricky because the Python `hpc_bus_instance.push` calculates checksum.
    # To test consumer-side checksum failure, we'd need to corrupt SHM or
    # have the C++ layer produce a bad TensorRef.

    # Alternative: test the `calculate_checksum` and `TensorRef` creation.
    ref = TensorRef(offset=0, size=tensor_data.nelement()*tensor_data.element_size(),
                    dtype_str=get_str_from_torch_dtype(tensor_data.dtype),
                    shape=tuple(tensor_data.shape), checksum=correct_checksum + 1) # Bad checksum

    shm_test_view = multiprocessing.shared_memory.SharedMemory(name=hpc_bus_instance.shm_name)
    # Simulate writing the tensor data to SHM at offset 0 for this ref
    np.frombuffer(shm_test_view.buf, dtype=np.float32, count=tensor_data.nelement(), offset=0)[:] = tensor_data.numpy().flatten()
    
    # This specific test setup requires the SAETrainerWorker's _reconstruct_tensor_from_ref
    # or similar logic that actually performs the checksum.
    # We can mock this part or make it a more integrated test.
    # For now, this is a conceptual placeholder for how checksum validation would be tested.
    
    mock_sae_worker = MagicMock() # Simulate a worker
    mock_sae_worker.shm_buffer_view = shm_test_view.buf
    mock_sae_worker.config = hpc_config # Give it config for checksum setting
    mock_sae_worker.monitor = MagicMock()
    mock_sae_worker.tags = {}

    from asys_i.components.sae_trainer import SAETrainerWorker # Local import
    
    with pytest.raises(ValueError, match="Checksum mismatch"):
        SAETrainerWorker._reconstruct_tensor_from_ref(mock_sae_worker, ref) # Call as method of mocked worker
    
    mock_sae_worker.monitor.log_metric.assert_any_call("trainer_checksum_error_count", 1, tags=pytest.ANY)
    shm_test_view.close()


def test_torch_dtype_code_synchronization():
    if not CPP_EXTENSION_AVAILABLE:
        pytest.skip("HPC C++ extension not available, cannot test TorchDtypeCode enum.")

    try:
        # Attempt to import the C++ extension module directly or its members
        # This path might vary based on how pybind11 names and packages the module.
        # Given CMake module name is 'c_ext_wrapper', it might be in 'asys_i.hpc'.
        from asys_i.hpc.c_ext_wrapper import TorchDtypeCode
    except ImportError:
        try:
            # Fallback if it's not directly in asys_i.hpc (e.g. if it's a top level module)
            # This depends on how `pip install -e .` makes it available.
            # Let's assume the build system places it such that it can be imported.
            # The CMakeLists.txt sets OUTPUT_NAME "c_ext_wrapper", no specific package path there.
            # Hatchling might place it inside asys_i/hpc/.
            # If this still fails, the test won't run, but the logic is what we want to add.
            import c_ext_wrapper # Try direct import if it's top-level in site-packages
            TorchDtypeCode = c_ext_wrapper.TorchDtypeCode
        except ImportError:
            pytest.fail("Failed to import TorchDtypeCode from C++ extension 'c_ext_wrapper'. "
                        "Ensure it's built and importable. Module might be in asys_i.hpc.c_ext_wrapper or similar.")


    assert len(DTYPE_TO_CODE_MAP) > 0, "DTYPE_TO_CODE_MAP is empty, cannot verify synchronization."

    # Check that all items in DTYPE_TO_CODE_MAP are present in the C++ enum and match
    for dtype_str, expected_code in DTYPE_TO_CODE_MAP.items():
        # 'torch.float32' -> 'FLOAT32'
        python_enum_name = dtype_str.split('.')[-1].upper()

        try:
            actual_enum_member = getattr(TorchDtypeCode, python_enum_name)
        except AttributeError:
            pytest.fail(f"TorchDtypeCode enum in C++ extension is missing member '{python_enum_name}' "
                        f"corresponding to '{dtype_str}' from DTYPE_TO_CODE_MAP.")

        assert int(actual_enum_member) == expected_code, \
            f"Mismatch for {dtype_str}: Python's DTYPE_TO_CODE_MAP code is {expected_code}, " \
            f"but C++ TorchDtypeCode.{python_enum_name} is {int(actual_enum_member)}."

        # Check reverse mapping: C++ code to enum member
        assert TorchDtypeCode(expected_code) == actual_enum_member, \
            f"Mismatch for code {expected_code} ({dtype_str}): C++ TorchDtypeCode({expected_code}) " \
            f"is not TorchDtypeCode.{python_enum_name}."

    # Check that the C++ enum does not have extra members not in DTYPE_TO_CODE_MAP
    # (This assumes DTYPE_TO_CODE_MAP is the single source of truth)
    py_enum_names_from_map = {dtype_str.split('.')[-1].upper() for dtype_str in DTYPE_TO_CODE_MAP.keys()}

    # getattr(TorchDtypeCode, '__members__') gives a dict like {'FLOAT32': <TorchDtypeCode.FLOAT32: 0>, ...}
    cpp_enum_members = TorchDtypeCode.__members__

    for cpp_enum_name_str, _ in cpp_enum_members.items():
        assert cpp_enum_name_str in py_enum_names_from_map, \
            f"C++ TorchDtypeCode has member '{cpp_enum_name_str}' which is not derived from DTYPE_TO_CODE_MAP."
