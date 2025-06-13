# tests/components/test_activation_hooker_revised.py
import pytest
import torch
from unittest.mock import MagicMock, call, ANY # ANY for tags

from asys_i.components.activation_hooker import ActivationHooker
from asys_i.common.types import RunProfile, ActivationPacket
from asys_i.orchestration.config_loader import MasterConfig
from asys_i.hpc import CPP_EXTENSION_AVAILABLE # For conditional skip
from asys_i.hpc.gpu_kernels import CUDA_AVAILABLE as ACTUAL_CUDA_AVAILABILITY # Original state

# Use the complex FQN model for these tests
@pytest.fixture
def model_for_hooker_tests(mock_model_for_hooking_complex_fqn):
    return mock_model_for_hooking_complex_fqn


def test_hooker_initialization(consumer_config_minimal: MasterConfig, model_for_hooker_tests):
    mock_data_bus = MagicMock()
    mock_monitor = MagicMock() # Use fixture if more complex mocking needed
    hooker = ActivationHooker(model_for_hooker_tests, consumer_config_minimal, mock_data_bus, mock_monitor)
    assert hooker.profile == RunProfile.CONSUMER
    assert not hooker._attached
    assert hooker.hook_config is consumer_config_minimal.hook # Ensure correct sub-config is used

def test_hooker_attach_detach_valid_fqn(consumer_config_minimal: MasterConfig, model_for_hooker_tests):
    mock_data_bus = MagicMock()
    mock_monitor = MagicMock()
    
    # Valid FQN path from MockTestModel: first Linear layer in the first block of 'h'
    valid_fqn = "transformer.h.0.0" # Path to the first Linear(d_in, d_in*2)
    consumer_config_minimal.hook.layers_to_hook = {"test_block_0_linear_0": valid_fqn}
    
    hooker = ActivationHooker(model_for_hooker_tests, consumer_config_minimal, mock_data_bus, mock_monitor)
    mock_global_step_ref = MagicMock(return_value=1)

    hooker.attach(mock_global_step_ref)
    assert hooker._attached
    assert valid_fqn in hooker._handles # Stored by FQN
    assert len(hooker._handles) == 1
    
    target_module = model_for_hooker_tests.get_submodule(valid_fqn)
    assert len(target_module._forward_hooks) == 1 # PyTorch internal check

    hooker.detach()
    assert not hooker._attached
    assert len(hooker._handles) == 0
    assert len(target_module._forward_hooks) == 0


def test_hooker_attach_invalid_fqn_logs_error(consumer_config_minimal: MasterConfig, model_for_hooker_tests):
    mock_data_bus = MagicMock()
    mock_monitor = MagicMock()
    invalid_fqn = "transformer.h.non_existent_block.0"
    consumer_config_minimal.hook.layers_to_hook = {"bad_path": invalid_fqn}
    
    hooker = ActivationHooker(model_for_hooker_tests, consumer_config_minimal, mock_data_bus, mock_monitor)
    mock_global_step_ref = MagicMock(return_value=1)

    # In the revised Hooker, attach() might raise an error if FQNs are critical.
    # If it logs and continues, this test is fine. If it raises, use pytest.raises.
    # Current hooker.attach logs error and continues if _get_module_by_fqn returns None for a path.
    # If all paths are invalid, _handles remains empty.
    
    hooker.attach(mock_global_step_ref)
    # If attach becomes critical on FQN failure (recommended), then:
    # with pytest.raises(ValueError, match="Module for FQN path .* not found"):
    #    hooker.attach(mock_global_step_ref)
    
    # Assuming it logs and continues (current behavior if one FQN fails among others,
    # or if all fail and it doesn't raise for that):
    assert not hooker._attached # Because no valid hooks were made
    assert len(hooker._handles) == 0
    mock_monitor.log_metric.assert_any_call("hook_attach_error_count", 1, tags={"layer_fqn": invalid_fqn, "reason": "module_not_found"})


def test_hooker_consumer_mode_capture_and_packet(consumer_config_minimal: MasterConfig, model_for_hooker_tests):
    mock_data_bus = MagicMock()
    mock_monitor = MagicMock()
    
    # Hook the output of the embeddings layer for this test
    capture_fqn = "transformer.embeddings"
    consumer_config_minimal.hook.layers_to_hook = {"emb_output": capture_fqn}
    # Ensure d_in in config matches model for packet data shape checks
    consumer_config_minimal.sae_model.d_in = model_for_hooker_tests.d_in
    
    hooker = ActivationHooker(model_for_hooker_tests, consumer_config_minimal, mock_data_bus, mock_monitor)
    mock_global_step_ref = MagicMock(return_value=77)
    hooker.attach(mock_global_step_ref)

    batch_size = 3
    d_in_test = model_for_hooker_tests.d_in
    dummy_input = torch.randn(batch_size, d_in_test, device=consumer_config_minimal.hardware.device)
    model_for_hooker_tests.to(dummy_input.device) # Ensure model is on correct device

    # Manually get expected output of the hooked layer
    # Here, it's transformer.embeddings which is a Linear layer.
    # Its output is then passed through relu by the MockTestModel's forward.
    # The hook captures the output *of the Linear layer itself*.
    expected_hooked_tensor_manual = model_for_hooker_tests.transformer.embeddings(dummy_input)

    model_for_hooker_tests(dummy_input) # Trigger forward pass

    mock_data_bus.push.assert_called_once()
    args_pushed, _ = mock_data_bus.push.call_args
    packet: ActivationPacket = args_pushed[0]

    assert packet.layer_name == capture_fqn
    assert packet.layer_idx_numeric == 0 # First (and only) hooked layer gets numeric index 0
    assert packet.global_step == 77
    assert packet.profile == RunProfile.CONSUMER
    assert isinstance(packet.data, torch.Tensor)
    assert not packet.data.is_cuda # Consumer path moves to CPU

    # Hooked tensor is detached and reshaped/viewed: (batch*seq (1 here), features)
    # Original output of nn.Linear is (batch, features)
    expected_tensor_in_packet = expected_hooked_tensor_manual.detach().view(-1, d_in_test).cpu()
    assert torch.allclose(packet.data, expected_tensor_in_packet)
    assert packet.meta.get("original_shape") == str(tuple(expected_hooked_tensor_manual.shape))
    mock_monitor.log_metric.assert_any_call("hook_packet_success_count", 1, tags=ANY)


@pytest.mark.skipif(not ACTUAL_CUDA_AVAILABILITY, reason="Test requires CUDA for pinned memory path.")
@pytest.mark.skipif(not CPP_EXTENSION_AVAILABLE, reason="HPC C++ extension not available for DataBus interaction.")
def test_hooker_hpc_mode_capture_pinned_memory(hpc_config_minimal: MasterConfig, model_for_hooker_tests):
    # This test focuses on the pinned memory path in HPC mode.
    # It assumes a CppDataBus that can accept the CPU-pinned tensor.
    mock_hpc_data_bus = MagicMock() # Mock the C++ DataBus
    mock_monitor = MagicMock()
    
    capture_fqn_hpc = "output_projection"
    hpc_config_minimal.hook.layers_to_hook = {"final_proj": capture_fqn_hpc}
    hpc_config_minimal.sae_model.d_in = model_for_hooker_tests.d_in # Match d_in
    hpc_config_minimal.hardware.device = "cuda" # Ensure CUDA for HPC path

    hooker = ActivationHooker(model_for_hooker_tests, hpc_config_minimal, mock_hpc_data_bus, mock_monitor)
    # Ensure pinned memory buffer is allocated in hooker if conditions met
    if hooker.pinned_memory_buffer is None:
        pytest.skip("Hooker did not allocate pinned memory buffer, cannot test this path.")

    mock_global_step_ref = MagicMock(return_value=88)
    hooker.attach(mock_global_step_ref)

    batch_size = 2
    d_in_hpc = model_for_hooker_tests.d_in
    dummy_input_gpu = torch.randn(batch_size, d_in_hpc, device="cuda")
    model_for_hooker_tests.to("cuda")

    # Manually get expected output of the hooked layer (output_projection)
    # Simulate the forward pass up to the input of output_projection
    with torch.no_grad():
        x_emb = torch.relu(model_for_hooker_tests.transformer.embeddings(dummy_input_gpu))
        x_hidden = x_emb
        for block in model_for_hooker_tests.transformer.h:
            x_hidden = torch.relu(block(x_hidden))
        expected_hooked_tensor_manual_gpu = model_for_hooker_tests.output_projection(x_hidden)
    
    model_for_hooker_tests(dummy_input_gpu) # Trigger hooks

    mock_hpc_data_bus.push.assert_called_once()
    args_pushed_hpc, _ = mock_hpc_data_bus.push.call_args
    packet_hpc: ActivationPacket = args_pushed_hpc[0]

    assert packet_hpc.layer_name == capture_fqn_hpc
    assert packet_hpc.profile == RunProfile.HPC
    assert isinstance(packet_hpc.data, torch.Tensor)
    assert not packet_hpc.data.is_cuda # Data passed to CppBus is on CPU
    assert packet_hpc.data.is_pinned() # Key check for this test path

    expected_tensor_in_hpc_packet = expected_hooked_tensor_manual_gpu.detach().view(-1, d_in_hpc)
    assert torch.allclose(packet_hpc.data.cpu(), expected_tensor_in_hpc_packet.cpu()) # Compare on CPU
    mock_monitor.log_metric.assert_any_call("hook_packet_success_count", 1, tags=ANY)

