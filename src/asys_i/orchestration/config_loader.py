# src/asys_i/orchestration/config_loader.py
"""
Core Philosophy: Config-Driven.
(UPGRADED) Loads, merges, and validates configuration. Handles "auto" settings.
Provides type-safe configuration objects to the entire system.
High Cohesion: All configuration logic resides here.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Union, Type

import yaml
from deepmerge import always_merger  # pip install deepmerge
from pydantic import BaseModel, Field, root_validator, validator

# Import core types
from asys_i.common.types import (
    ComponentID,
    DataBusType,
    LayerIndex,
    MonitorType,
    RunProfile,
)

log = logging.getLogger(__name__)

# Define nested Pydantic models for each configuration section


class ProjectConfig(BaseModel):
    name: str = "a-sys-i-experiment"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"
    seed: int = 42  # For reproducibility


class HardwareConfig(BaseModel):
    device: str = "cuda:0"
    dtype: str = "float16"  # or bfloat16, float32
    use_mixed_precision: bool = True
    compile_model: bool = False  # torch.compile


class PPOConfig(BaseModel):
    # Placeholder for trl.PPOConfig
    model_name: str = "gpt2"
    reward_model_name: str = "reward-model"
    learning_rate: float = 1e-5
    batch_size: int = 16
    max_steps: int = 1000
    # ... more PPO params


class SAEModelConfig(BaseModel):
    # UPGRADE: d_in can now be 'auto'
    d_in: Union[int, str] = Field(
        "auto", description="Input dimension. 'auto' detects from host model."
    )
    d_sae: int = Field(
        768 * 4, description="SAE feature dimension (expansion factor * d_in)"
    )
    l1_coefficient: float = Field(1e-3, description="Sparsity penalty weight")
    bias_decay: float = 0.0
    decoder_bias_init_method: str = "geometric_median"  # or mean, zeros
    apply_decoder_bias: bool = True
    # TODO: define separate configs per layer if needed

    @validator("d_in")
    def d_in_must_be_positive_or_auto(cls, v: Union[int, str]) -> Union[int, str]:
        if isinstance(v, int) and v <= 0:
            raise ValueError("d_in must be a positive integer or 'auto'")
        if isinstance(v, str) and v != "auto":
            raise ValueError("d_in must be a positive integer or 'auto'")
        return v


class SAETrainerConfig(BaseModel):
    # Note: this applies to *each* SAE model instance per layer/worker
    learning_rate: float = 3e-4
    optimizer: str = "AdamW"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    batch_size: int = Field(4096, description="Batch size for SAE training step")
    num_workers: int = Field(
        1, description="Number of parallel SAETrainerWorker processes"
    )
    layers_per_worker: Union[str, int] = Field(
        "auto", description="Distribute layers among workers"
    )
    save_interval_steps: int = 10000
    feature_sparsity_probe_interval_steps: int = 1000
    dead_feature_threshold: float = 1e-8
    heartbeat_interval_sec: float = 10.0


class HookConfig(BaseModel):
    layers_to_hook: List[LayerIndex] = [2, 4, 6, 8]
    hook_point: str = "resid_post"  # e.g., resid_post, attn_out, mlp_out
    sampling_rate: float = Field(
        1.0, ge=0.0, le=1.0, description="Probability of sampling an activation"
    )
    # HPC only
    gpu_kernel_mode: str = "NONE"  # NONE, FP8, FP8_LZ4, TOP_K
    backpressure_debounce_sec: float = 2.0
    min_sampling_rate: float = 0.01


class DataBusConfig(BaseModel):
    type: DataBusType
    # Common
    buffer_size_per_shard: int = Field(
        2**18, description="Max packets in queue/buffer per shard"
    )
    pull_batch_size_max: int = 4096  # Max packets a consumer can pull at once
    push_timeout_sec: float = (
        0.001  # Small timeout for push to enable fast backpressure
    )
    pull_timeout_sec: float = 1.0  # Timeout for consumers waiting for data
    # HPC specific
    num_shards: int = Field(1, description="Number of parallel ring buffers (HPC only)")
    shared_memory_size_gb: float = Field(
        8.0, description="Size of shared memory for tensors (HPC)"
    )

    @validator("type", pre=True, always=True)
    def validate_type(cls, v: DataBusType, values: Dict[str, Any], config: Any, field: Any) -> DataBusType:
        # Dynamically set default based on profile if not provided? - Better in MasterConfig
        return v

    @root_validator
    def check_hpc_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        db_type = values.get("type")
        if db_type == DataBusType.PYTHON_QUEUE and values.get("num_shards", 1) > 1:
            log.warning(
                "PythonQueueBus does not support sharding. num_shards will be ignored."
            )
            values["num_shards"] = 1
        elif (
            db_type == DataBusType.CPP_SHARDED_SPMC and values.get("num_shards", 1) < 1
        ):
            raise ValueError("num_shards must be >= 1 for CppShardedSPMCBus")
        return values


class MonitorConfig(BaseModel):
    type: MonitorType
    # Prometheus
    prometheus_port: int = 8001
    # CSV/Tensorboard
    metrics_flush_interval_sec: float = 10.0
    heartbeat_check_interval_sec: float = 30.0
    component_timeout_sec: int = 90  # Watchdog timeout


class ArchiverConfig(BaseModel):
    enabled: bool = True
    output_format: str = "parquet"  # parquet, arrow, pt
    batch_size: int = 8192
    flush_interval_sec: float = 60.0
    compression: str = "zstd"
    heartbeat_interval_sec: float = 15.0
    # TODO: S3 / Object Storage Config


class ResourceManagerConfig(BaseModel):
    # UPGRADE: Added allocation_strategy
    allocation_strategy: str = Field(
        "auto",
        description="'auto' for heuristic-based core assignment, 'manual' to use map below.",
    )
    apply_bindings: bool = True
    cpu_affinity_map: Dict[ComponentID, List[int]] = Field(default_factory=dict)
    # Example: {"trainer_worker_0": 0, "host_process": 1}
    numa_node_map: Dict[ComponentID, int] = Field(default_factory=dict)

    # Validate allocation_strategy
    @validator("allocation_strategy")
    def check_allocation_strategy(cls, v: str) -> str:
        if v not in ["auto", "manual"]:
            raise ValueError("allocation_strategy must be 'auto' or 'manual'")
        return v


# The Master Configuration Object
class MasterConfig(BaseModel):
    """Aggregated, validated configuration for the entire A-Sys-I system."""

    run_profile: RunProfile = RunProfile.CONSUMER

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    sae_model: SAEModelConfig = Field(default_factory=SAEModelConfig)  # type: ignore[arg-type]
    sae_trainer: SAETrainerConfig = Field(default_factory=SAETrainerConfig)  # type: ignore[arg-type]
    hook: HookConfig = Field(default_factory=HookConfig)  # type: ignore[arg-type]
    data_bus: DataBusConfig
    monitor: MonitorConfig
    archiver: ArchiverConfig = Field(default_factory=ArchiverConfig)
    resource_manager: ResourceManagerConfig = Field(
        default_factory=ResourceManagerConfig  # type: ignore[arg-type]
    )

    @root_validator(pre=True)
    def set_defaults_based_on_profile(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set sensible defaults if specific types are not defined in yaml, based on profile."""
        profile = values.get("run_profile", RunProfile.CONSUMER)

        # Set DataBus type default if not present
        if "data_bus" not in values or "type" not in values["data_bus"]:
            values.setdefault("data_bus", {})
            default_db_type = (
                DataBusType.CPP_SHARDED_SPMC
                if profile == RunProfile.HPC
                else DataBusType.PYTHON_QUEUE
            )
            values["data_bus"].setdefault("type", default_db_type)
            log.info(
                f"Defaulting DataBus type to {default_db_type} for profile {profile}"
            )

        # Set Monitor type default if not present
        if "monitor" not in values or "type" not in values["monitor"]:
            values.setdefault("monitor", {})
            default_mon_type = (
                MonitorType.PROMETHEUS
                if profile == RunProfile.HPC
                else MonitorType.CSV_TENSORBOARD
            )
            values["monitor"].setdefault("type", default_mon_type)
            log.info(
                f"Defaulting Monitor type to {default_mon_type} for profile {profile}"
            )

        return values

    @root_validator
    def cross_validate_profile(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure configuration consistency across sections based on profile."""
        profile = values.get("run_profile")
        data_bus_config: Optional[DataBusConfig] = values.get("data_bus")
        monitor_config: Optional[MonitorConfig] = values.get("monitor")
        resource_config: Optional[ResourceManagerConfig] = values.get(
            "resource_manager"
        )
        hook_config: Optional[HookConfig] = values.get("hook")
        sae_model_config: Optional[SAEModelConfig] = values.get("sae_model")

        if profile == RunProfile.HPC:
            if data_bus_config and data_bus_config.type != DataBusType.CPP_SHARDED_SPMC:
                log.warning(
                    f"HPC profile active, but DataBus type is {data_bus_config.type}. Performance SLA may not be met."
                )
            if monitor_config and monitor_config.type != MonitorType.PROMETHEUS:
                log.warning(
                    f"HPC profile active, but Monitor type is {monitor_config.type}. High-resolution monitoring unavailable."
                )
            if (
                resource_config
                and resource_config.apply_bindings
                and resource_config.allocation_strategy == "manual"
                and not resource_config.cpu_affinity_map
            ):
                log.warning(
                    "HPC profile: resource_manager.apply_bindings is True and allocation_strategy is 'manual', but cpu_affinity_map is empty. Binding will not occur as expected."
                )
            if hook_config and hook_config.gpu_kernel_mode != "NONE":
                log.info(
                    f"HPC profile: GPU kernel preprocessing mode enabled: {hook_config.gpu_kernel_mode}"
                )
                # TODO: Check CUDA/kernel availability here

        elif profile == RunProfile.CONSUMER:
            if data_bus_config and data_bus_config.type == DataBusType.CPP_SHARDED_SPMC:
                raise ValueError(
                    "CONSUMER profile cannot use CppShardedSPMCBus. Set data_bus.type to PYTHON_QUEUE."
                )
            if monitor_config and monitor_config.type == MonitorType.PROMETHEUS:
                log.warning(
                    "CONSUMER profile: PrometheusMonitor selected, ensure prometheus-client is installed."
                )
            if (
                resource_config
                and resource_config.apply_bindings
                and resource_config.allocation_strategy == "manual"
            ):
                log.warning(
                    "CONSUMER profile: resource_manager.apply_bindings is True, but CPU/NUMA binding is primarily for HPC and might not work or be effective."
                )
            if hook_config and hook_config.gpu_kernel_mode != "NONE":
                raise ValueError(
                    f"CONSUMER profile cannot use GPU kernel mode {hook_config.gpu_kernel_mode}. Set to NONE."
                )
            if sae_model_config and sae_model_config.d_in == "auto":
                # In consumer mode, we might not always have the full host model readily available for auto-detection
                log.warning(
                    "CONSUMER profile with `sae_model.d_in: auto`. Ensure the host model is loaded for detection."
                )

        # Add check for layer_idx validity etc.
        return values

    class Config:
        use_enum_values = True  # serialize enums as strings


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        log.error(f"Error parsing YAML file {path}: {e}")
        raise


def load_config(
    config_path: str, base_config_path: Optional[str] = "configs/base.yaml"
) -> MasterConfig:
    """
    Loads, merges, and validates configuration.
    1. Loads base config (if exists).
    2. Loads specific config.
    3. Deep merges (specific overrides base).
    4. Validates via Pydantic MasterConfig.
    """
    log.info(f"Loading configuration from: {config_path}")
    merged_config: Dict[str, Any] = {}

    # 1. Load Base
    if base_config_path and os.path.exists(base_config_path):
        log.info(f"Loading base configuration from: {base_config_path}")
        base_config = _load_yaml(base_config_path)
        always_merger.merge(merged_config, base_config)
    elif base_config_path:
        log.warning(f"Base config path specified but not found: {base_config_path}")

    # 2. Load Specific & 3. Merge
    specific_config = _load_yaml(config_path)
    always_merger.merge(merged_config, specific_config)  # specific overrides base

    # 4. Validate
    try:
        config = MasterConfig.parse_obj(merged_config)
        log.info(
            f"Configuration loaded and validated successfully. Run Profile: {config.run_profile}"
        )
        # log.debug(f"Final config: {config.json(indent=2)}")
        return config
    except Exception as e:
        log.error(f"Configuration validation failed: {e}")
        raise


# Example Usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Create dummy files for test
    os.makedirs("configs", exist_ok=True)
    with open("configs/base.yaml", "w") as f:
        f.write("project:\n  name: base-test\n  log_level: DEBUG\n")
    with open("configs/test_consumer.yaml", "w") as f:
        f.write("run_profile: CONSUMER\nproject:\n  name: consumer-override\n")
    try:
        cfg = load_config("configs/test_consumer.yaml")
        print("\n--- VALID CONFIG ---")
        print(cfg.json(indent=2))
        assert cfg.project.name == "consumer-override"
        assert cfg.project.log_level == "DEBUG"  # from base
        assert cfg.run_profile == RunProfile.CONSUMER
        assert cfg.data_bus.type == DataBusType.PYTHON_QUEUE  # default from validator
        assert cfg.monitor.type == MonitorType.CSV_TENSORBOARD  # default from validator

    except Exception as e:
        print(f"\n--- Config Load Failed ---\n{e}")
    finally:
        os.remove("configs/base.yaml")
        os.remove("configs/test_consumer.yaml")
        os.rmdir("configs")
