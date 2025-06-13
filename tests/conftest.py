# tests/conftest.py (CONFIRMED/REFINED FROM YOUR LAST INPUT)
import pytest
import os
import shutil
import torch # For MockTestModel
from unittest.mock import MagicMock # For monitor mocks

from asys_i.orchestration.config_loader import load_config, MasterConfig
from asys_i.hpc import CPP_EXTENSION_AVAILABLE # To skip HPC tests if needed
from asys_i.common.types import RunProfile, DataBusType, MonitorType

# --- Global Test Output Directory ---
BASE_TEST_OUTPUT_DIR_NAME = "pytest_asys_i_outputs"

@pytest.fixture(scope="session")
def base_test_output_dir(tmp_path_factory):
    # Use pytest's tmp_path_factory for session-scoped temporary directory
    # This is cleaner than manual creation/deletion in project dir.
    # However, if you *want* outputs to persist outside pytest's tmp, use manual.
    # For CI/local runs where outputs might be inspected, manual can be better.
    # Let's use a manual path for inspectable outputs, but ensure it's cleaned.

    # dir_path = tmp_path_factory.mktemp(BASE_TEST_OUTPUT_DIR_NAME)
    # return str(dir_path)

    # Manual path for easier inspection after tests:
    dir_path = os.path.abspath(os.path.join(".", BASE_TEST_OUTPUT_DIR_NAME)) # In project root
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path) # Clean from previous runs
    os.makedirs(dir_path, exist_ok=True)
    print(f"Test output directory: {dir_path}")
    yield dir_path
    # Optional: comment out rmtree if you want to inspect outputs after tests
    # shutil.rmtree(dir_path)

@pytest.fixture
def mock_monitor_for_tests():
    monitor = MagicMock()
    monitor.log_metric = MagicMock()
    monitor.log_metrics = MagicMock()
    monitor.log_hyperparams = MagicMock()
    monitor.heartbeat = MagicMock()
    monitor.register_component = MagicMock()
    monitor.shutdown = MagicMock()
    return monitor


@pytest.fixture
def consumer_config_minimal(base_test_output_dir) -> MasterConfig:
    consumer_yaml_content = f"""
schema_version: "1.0"
run_profile: CONSUMER
project:
  name: "consumer-test-minimal"
  output_dir: "{os.path.join(base_test_output_dir, 'consumer_minimal/outputs')}"
  log_dir: "{os.path.join(base_test_output_dir, 'consumer_minimal/logs')}"
  checkpoint_dir: "{os.path.join(base_test_output_dir, 'consumer_minimal/checkpoints')}"
  seed: 101
hardware:
  device: "cpu" # Force CPU for most unit tests
  dtype: "float32"
hook:
  layers_to_hook: {{ "mock_layer_0_fqn": "layers.0" }} # Assuming MockTestModel structure
data_bus:
  type: PYTHON_QUEUE # Enum value
  buffer_size_per_shard: 100 # Small for testing
monitor:
  type: CSV_TENSORBOARD # Enum value
sae_model:
  d_in: 10 # Match MockTestModel d_in
  d_sae: 20
ppo:
  max_steps: 10
"""
    config_path = os.path.join(base_test_output_dir, "profile_consumer_minimal_test.yaml")
    with open(config_path, "w") as f: f.write(consumer_yaml_content)
    # No separate base needed if profile is self-contained for this test fixture
    return load_config(config_path, base_config_path=None)


@pytest.fixture
def hpc_config_minimal(base_test_output_dir) -> MasterConfig:
    if not CPP_EXTENSION_AVAILABLE:
        pytest.skip("HPC C++ extension not available, skipping HPC config fixture.")

    hpc_yaml_content = f"""
schema_version: "1.0"
run_profile: HPC
project:
  name: "hpc-test-minimal"
  output_dir: "{os.path.join(base_test_output_dir, 'hpc_minimal/outputs')}"
  log_dir: "{os.path.join(base_test_output_dir, 'hpc_minimal/logs')}"
  checkpoint_dir: "{os.path.join(base_test_output_dir, 'hpc_minimal/checkpoints')}"
  seed: 202
hardware:
  device: "cpu" # Force CPU to avoid GPU needs in basic C++ bus tests
  dtype: "float32"
hook:
  layers_to_hook: {{ "mock_hpc_layer_0_fqn": "layers.0" }}
data_bus:
  type: CPP_SHARDED_SPMC # Enum value
  shared_memory_size_gb: 0.05 # Very small for testing (50MB)
  buffer_size_per_shard: 100 # Max MQ messages
  use_checksum: true
monitor:
  type: PROMETHEUS # Enum value
sae_model:
  d_in: 10 # Match MockTestModel d_in
  d_sae: 20
ppo:
  max_steps: 10
"""
    config_path = os.path.join(base_test_output_dir, "profile_hpc_minimal_test.yaml")
    with open(config_path, "w") as f: f.write(hpc_yaml_content)
    return load_config(config_path, base_config_path=None)

# --- Mock Model for Hooking Tests ---
class MockTestModel(torch.nn.Module):
    def __init__(self, d_in=10, num_hidden_layers=1): # num_hidden_layers for FQN path depth
        super().__init__()
        self.d_in = d_in
        # Create a nested structure for testing FQN
        self.transformer = torch.nn.Sequential()
        self.transformer.add_module("embeddings", torch.nn.Linear(d_in, d_in))
        
        hidden_layers_modulelist = torch.nn.ModuleList()
        for i in range(num_hidden_layers):
            layer_block = torch.nn.Sequential(
                torch.nn.Linear(d_in, d_in * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(d_in * 2, d_in)
            )
            hidden_layers_modulelist.add_module(f"block_{i}", layer_block)
        
        self.transformer.add_module("h", hidden_layers_modulelist) # e.g., model.transformer.h.0.0 (for first Linear)
        self.output_projection = torch.nn.Linear(d_in, d_in)

    def forward(self, x):
        x = torch.relu(self.transformer.embeddings(x))
        for block in self.transformer.h:
            x = torch.relu(block(x)) # Pass through each block
        return self.output_projection(x)

@pytest.fixture
def mock_model_for_hooking_complex_fqn():
    # d_in must match what configs expect (e.g., sae_model.d_in = 10)
    return MockTestModel(d_in=10, num_hidden_layers=2)
