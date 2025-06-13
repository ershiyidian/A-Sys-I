import unittest
from unittest.mock import MagicMock, patch, ANY

import torch
import torch.nn as nn

from asys_i.components.activation_hooker import ActivationHooker
from asys_i.common.types import RunProfile, create_activation_packet, LayerIndex, GlobalStep
from asys_i.orchestration.config_loader import MasterConfig, HookConfig, HardwareConfig, ProjectConfig, SAEModelConfig, PPOConfig, DataBusConfig, MonitorConfig


# --- Mock Classes ---
class MockModel(nn.Module):
    def __init__(self, d_in=10, num_layers=2):
        super().__init__()
        self.d_in = d_in
        self.layers = nn.ModuleList([nn.Linear(d_in, d_in) for _ in range(num_layers)])
        # Add a direct attribute for a layer to test specific named module hooking if needed
        self.named_layer = nn.Linear(d_in, d_in)
        self._hook_called_for_layer = -1 # Helper to check which layer's hook was called

    def forward(self, x):
        # Simulate passing through specified layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Simple way to check if hook on this layer was called (for specific layer tests)
            # In a real scenario, the hook itself would signal this.
            if hasattr(self, f'_hook_for_layer_{i}_called'):
                 self._hook_called_for_layer = i
        x = self.named_layer(x)
        return x

class TestActivationHookerConsumerMode(unittest.TestCase):

    def setUp(self):
        # Basic Mock Configuration
        self.project_config = ProjectConfig()
        self.hardware_config = HardwareConfig(device="cpu", dtype="float32")
        self.hook_config = HookConfig(layers_to_hook=[0, 1], sampling_rate=1.0)
        self.sae_model_config = SAEModelConfig(d_in=10) # d_in matches MockModel

        # Create a dummy MasterConfig object
        # For DataBusConfig and MonitorConfig, provide the minimal required fields
        # or use their default factories if they have them.
        # Assuming DataBusType and MonitorType enums are accessible
        from asys_i.common.types import DataBusType, MonitorType
        self.data_bus_config = DataBusConfig(type=DataBusType.PYTHON_QUEUE, buffer_size_per_shard=100)
        self.monitor_config = MonitorConfig(type=MonitorType.LOGGING_ONLY)

        self.mock_master_config = MasterConfig(
            run_profile=RunProfile.CONSUMER,
            project=self.project_config,
            hardware=self.hardware_config,
            ppo=PPOConfig(), # Using default PPOConfig
            sae_model=self.sae_model_config,
            hook=self.hook_config,
            data_bus=self.data_bus_config.dict(),  # Pass as dict
            monitor=self.monitor_config.dict()   # Pass as dict
        )

        self.mock_model = MockModel(d_in=self.sae_model_config.d_in, num_layers=len(self.hook_config.layers_to_hook))
        self.mock_data_bus = MagicMock()
        self.mock_monitor = MagicMock()
        self.mock_global_step_ref = MagicMock(return_value=GlobalStep(123))

        self.hooker = ActivationHooker(
            model=self.mock_model,
            config=self.mock_master_config,
            data_bus=self.mock_data_bus,
            monitor=self.mock_monitor
        )

    def tearDown(self):
        if self.hooker and self.hooker._attached:
            self.hooker.detach()

    def test_01_initialization(self):
        """Test correct initialization of ActivationHooker."""
        self.assertIsNotNone(self.hooker)
        self.assertEqual(self.hooker.profile, RunProfile.CONSUMER)
        self.assertEqual(len(self.hooker._handles), 0)
        self.assertFalse(self.hooker._attached)

    def test_02_attach_hooks(self):
        """Test successful attachment of forward hooks."""
        self.hooker.attach(self.mock_global_step_ref)
        self.assertTrue(self.hooker._attached)
        # Expected handles: one for each layer in layers_to_hook
        self.assertEqual(len(self.hooker._handles), len(self.hook_config.layers_to_hook))

        # Check if hooks are indeed registered on the nn.Module
        # This is a bit of an internal check, but useful
        for i, layer_idx in enumerate(self.hook_config.layers_to_hook):
            module_to_hook = self.mock_model.layers[layer_idx]
            self.assertGreater(len(module_to_hook._forward_hooks), 0, f"No forward hook found on layer {layer_idx}")

    def test_03_hook_called_on_forward_pass(self):
        """Verify hook function is called during forward pass."""
        self.hooker.attach(self.mock_global_step_ref)

        # We need to spy on the actual hook function created by _create_hook_fn
        # To do this, we can patch the _create_hook_fn method itself
        # Or, more simply, check if data_bus.push was called, as that's the end goal of the hook.

        dummy_input = torch.randn(2, self.sae_model_config.d_in) # batch_size=2
        self.mock_model(dummy_input) # Execute forward pass

        # Check that data_bus.push was called for each hooked layer
        self.assertEqual(self.mock_data_bus.push.call_count, len(self.hook_config.layers_to_hook))

    def test_04_activation_capture_and_packet_creation(self):
        """Verify activations are captured and packet is structured correctly."""
        # For this test, let's hook only one layer to simplify assertion
        target_layer_idx = LayerIndex(0)
        self.hook_config.layers_to_hook = [target_layer_idx]
        self.hooker = ActivationHooker( # Re-initialize hooker with new config
            model=self.mock_model,
            config=self.mock_master_config,
            data_bus=self.mock_data_bus,
            monitor=self.mock_monitor
        )
        self.hooker.attach(self.mock_global_step_ref)

        dummy_input = torch.randn(3, self.sae_model_config.d_in) # batch_size=3

        # Get the expected activation from the target layer manually for comparison
        expected_activation = None
        temp_hook_handle = None
        def temp_hook(module, input, output):
            nonlocal expected_activation
            # Output of nn.Linear is just the tensor
            expected_activation = output.detach().clone()

        module_to_hook = self.mock_model.layers[target_layer_idx]
        temp_hook_handle = module_to_hook.register_forward_hook(temp_hook)

        self.mock_model(dummy_input) # Execute forward pass

        temp_hook_handle.remove() # Clean up temp hook
        self.assertIsNotNone(expected_activation)

        self.mock_data_bus.push.assert_called_once()

        # Check the packet content
        # ANY is used for timestamp_ns and potentially meta if complex
        pushed_packet = self.mock_data_bus.push.call_args[0][0]

        self.assertEqual(pushed_packet['layer_idx'], target_layer_idx)
        self.assertEqual(pushed_packet['global_step'], self.mock_global_step_ref())
        self.assertEqual(pushed_packet['profile'], RunProfile.CONSUMER)

        # Verify activation data (after potential flattening by the hook)
        # The hook flattens (batch_size, features) -> (batch_size, features)
        # or (batch_size, seq_len, features) -> (batch_size*seq_len, features)
        # Our mock model layers are simple Linear, so (batch_size, d_in)
        captured_data = pushed_packet['data']
        self.assertIsInstance(captured_data, torch.Tensor)

        # Expected activation is (batch_size, d_in)
        # Hook's tensor_detached is (batch_size*seq_len, features), here seq_len is 1 implicitly
        self.assertEqual(captured_data.shape, expected_activation.shape)
        self.assertTrue(torch.allclose(captured_data, expected_activation.view(-1, expected_activation.shape[-1])))
        self.assertEqual(pushed_packet['meta'], {}) # Default meta is empty for consumer path unless added

    def test_05_detach_hooks(self):
        """Test successful detachment of hooks."""
        self.hooker.attach(self.mock_global_step_ref)
        self.assertTrue(self.hooker._attached)
        self.assertGreater(len(self.hooker._handles), 0)

        self.hooker.detach()
        self.assertFalse(self.hooker._attached)
        self.assertEqual(len(self.hooker._handles), 0)

        # Verify hooks are removed from the nn.Module
        for layer_idx in self.hook_config.layers_to_hook:
            module_to_hook = self.mock_model.layers[layer_idx]
            self.assertEqual(len(module_to_hook._forward_hooks), 0, f"Forward hook still present on layer {layer_idx} after detach")

    def test_06_no_push_if_not_sampling(self):
        """Verify no push to databus if sampling rate is 0."""
        self.hook_config.sampling_rate = 0.0
        # Re-initialize hooker with new config
        self.hooker = ActivationHooker(
            model=self.mock_model,
            config=self.mock_master_config,
            data_bus=self.mock_data_bus,
            monitor=self.mock_monitor
        )
        self.hooker.attach(self.mock_global_step_ref)

        dummy_input = torch.randn(2, self.sae_model_config.d_in)
        self.mock_model(dummy_input)

        self.mock_data_bus.push.assert_not_called()
        self.mock_monitor.log_metric.assert_any_call("hook_sample_skip_count", 1, tags=ANY)


if __name__ == '__main__':
    unittest.main()
