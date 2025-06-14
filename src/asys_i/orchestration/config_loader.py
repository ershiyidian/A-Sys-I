# src/asys_i/orchestration/config_loader.py (CONFIRMED/SLIGHTLY REFINED FROM YOUR LAST INPUT)
import logging
import os
import platform # For OS-specific logic if needed in future validators
from typing import Any, Dict, List, Optional, Union, Type

import yaml
from deepmerge import always_merger # Ensure this is installed
from pydantic import BaseModel, Field, root_validator, validator

from asys_i.common.types import ( # Ensure these are up-to-date
    ComponentID,
    DataBusType,
    # LayerIndex, # Config now uses FQN strings for layers_to_hook
    MonitorType,
    RunProfile,
    TORCH_DTYPE_STR_MAP, # For validating dtype strings
)

log = logging.getLogger(__name__)

# --- Pydantic Models (Confirming structure based on your last input) ---
class ProjectConfig(BaseModel):
    name: str = "a-sys-i-experiment"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"
    seed: int = 42

    @validator('log_level')
    def log_level_must_be_valid(cls, v):
        if v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {v}")
        return v.upper()

class HardwareConfig(BaseModel):
    device: str = "cuda:0" # Can be "cpu", "cuda", "cuda:0", "mps" etc.
    dtype: str = "bfloat16" # Default for modern GPUs
    use_mixed_precision: bool = True # For AMP
    compile_model: bool = False # For torch.compile()

    @validator('dtype')
    def dtype_must_be_valid_torch_str(cls, v):
        if v.lower() not in TORCH_DTYPE_STR_MAP:
            raise ValueError(f"Unsupported hardware.dtype: {v}. Supported: {list(TORCH_DTYPE_STR_MAP.keys())}")
        return v.lower() # Store canonical lower-case

class PPOConfig(BaseModel): # Placeholder, to be expanded
    model_name: str = "gpt2"
    reward_model_name: Optional[str] = None
    learning_rate: float = 1.41e-5
    batch_size: int = Field(64, gt=0)
    max_steps: int = Field(1000, gt=0)

class SAEModelConfig(BaseModel):
    d_in: Union[int, str] = Field("auto", description="Input dim. 'auto' detects from host.")
    d_sae: int = Field(768 * 4, gt=0, description="SAE feature dim (e.g., 4x d_in)")
    l1_coefficient: float = Field(1e-3, ge=0.0, description="Sparsity penalty")
    # bias_decay: float = 0.0 # If needed
    # decoder_bias_init_method: str = "geometric_median" # If needed
    # apply_decoder_bias: bool = True # If needed

    @validator("d_in")
    def d_in_valid(cls, v: Union[int, str]) -> Union[int, str]:
        if isinstance(v, int) and v <= 0: raise ValueError("d_in must be positive int or 'auto'")
        if isinstance(v, str) and v.lower() != "auto": raise ValueError("d_in string must be 'auto'")
        return v.lower() if isinstance(v, str) else v

class SAETrainerConfig(BaseModel):
    learning_rate: float = Field(3e-4, gt=0)
    optimizer: str = "AdamW" # Must be a valid torch.optim class name
    adam_beta1: float = Field(0.9, ge=0, lt=1)
    adam_beta2: float = Field(0.999, ge=0, lt=1)
    weight_decay: float = Field(0.01, ge=0)
    batch_size: int = Field(4096, gt=0)
    num_workers: int = Field(1, ge=0) # ge=0 allows disabling trainers
    save_interval_steps: int = Field(10000, gt=0)
    normalize_weights_interval: int = Field(100, ge=1, description="Frequency of W_dec normalization")
    heartbeat_interval_sec: float = Field(10.0, gt=0)

    @validator('num_workers')
    def warn_if_no_workers(cls, v):
        if v == 0:
            log.warning("sae_trainer.num_workers is 0. No SAEs will be trained.")
        return v


class HookConfig(BaseModel):
    layers_to_hook: Dict[str, str] = Field(default_factory=dict, description="Map of friendly_name: FQN_path_to_module")
    sampling_rate: float = Field(1.0, ge=0.0, le=1.0)
    gpu_kernel_mode: str = "NONE" # TODO: Make enum: NONE, FP8, TOP_K (future)
    backpressure_debounce_sec: float = Field(2.0, ge=0)
    min_sampling_rate: float = Field(0.01, ge=0, le=1.0)

    @validator("min_sampling_rate")
    def min_rate_le_sampling_rate(cls, v, values):
        if 'sampling_rate' in values and v > values['sampling_rate']:
            raise ValueError("min_sampling_rate cannot be greater than sampling_rate")
        return v

class DataBusConfig(BaseModel):
    type: DataBusType
    buffer_size_per_shard: int = Field(2**18, gt=0, description="Max messages in MQ (HPC) or packets in Queue (Consumer)")
    pull_batch_size_max: int = Field(4096, gt=0)
    push_timeout_sec: float = Field(0.001, ge=0) # Timeout for C++ MQ try_send can be 0
    pull_timeout_sec: float = Field(1.0, ge=0)
    num_shards: int = Field(1, ge=1) # For CppShardedSPMCBus if it ever becomes sharded. MQ is single for now.
    shared_memory_size_gb: float = Field(1.0, gt=0, description="Size of tensor data SHM (HPC)") # Min 1GB for safety
    use_checksum: bool = Field(True, description="Enable Adler32 checksum for HPC data bus (some overhead)") # Default to True for safety

    @root_validator(skip_on_failure=True) # E501 line too long
    def check_sharding_and_shm(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        db_type = values.get("type")
        num_shards_val = values.get("num_shards", 1)
        if db_type == DataBusType.PYTHON_QUEUE and num_shards_val > 1:
            log.warning("PythonQueueBus does not support sharding. num_shards will be effectively 1.")
            # values["num_shards"] = 1 # No need to modify, just a warning.
        if db_type == DataBusType.CPP_SHARDED_SPMC:
            if values.get("shared_memory_size_gb", 0) <= 0:
                raise ValueError("shared_memory_size_gb must be positive for CppShardedSPMCBus.")
        return values

class MonitorConfig(BaseModel):
    type: MonitorType
    prometheus_port: int = Field(8001, ge=1024, le=65535)
    metrics_flush_interval_sec: float = Field(10.0, gt=0)
    heartbeat_check_interval_sec: float = Field(15.0, gt=0) # Reduced for faster detection
    component_timeout_sec: int = Field(60, gt=0) # Stricter default timeout
    enable_csv_logging: bool = True # New field
    enable_tensorboard_logging: bool = True # New field

    @validator('component_timeout_sec')
    def timeout_gt_check_interval(cls, v, values):
        if 'heartbeat_check_interval_sec' in values and v <= values['heartbeat_check_interval_sec']:
            raise ValueError("component_timeout_sec must be greater than heartbeat_check_interval_sec")
        return v

class ArchiverConfig(BaseModel):
    enabled: bool = True
    output_format: str = "parquet" # TODO: Make enum if more formats
    batch_size: int = Field(8192, gt=0)
    flush_interval_sec: float = Field(60.0, gt=0)
    compression: Optional[str] = "zstd" # Can be None, or zstd, snappy, gzip
    heartbeat_interval_sec: float = Field(15.0, gt=0)

class ResourceManagerConfig(BaseModel):
    allocation_strategy: str = Field("auto", pattern="^(auto|manual)$")
    apply_bindings: bool = False # Default to False, True is more intrusive
    cpu_affinity_map: Dict[ComponentID, List[int]] = Field(default_factory=dict)
    numa_node_map: Dict[ComponentID, int] = Field(default_factory=dict)

# --- Master Configuration ---
class MasterConfig(BaseModel):
    schema_version: str = "1.0"
    run_profile: RunProfile = RunProfile.CONSUMER

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    sae_model: SAEModelConfig = Field(default_factory=SAEModelConfig)
    sae_trainer: SAETrainerConfig = Field(default_factory=SAETrainerConfig)
    hook: HookConfig = Field(default_factory=HookConfig)
    data_bus: DataBusConfig # No default factory, must be defined in YAML or defaulted by profile
    monitor: MonitorConfig   # No default factory, must be defined in YAML or defaulted by profile
    archiver: ArchiverConfig = Field(default_factory=ArchiverConfig)
    resource_manager: ResourceManagerConfig = Field(default_factory=ResourceManagerConfig)

    @validator("schema_version")
    def check_schema_version(cls, v: str):
        if v != "1.0": # Example check
            raise ValueError(f"Unsupported schema_version: {v}. Expected '1.0'.")
        return v

    @root_validator(pre=True, skip_on_failure=True) # pre=True allows defaulting before individual field validation
    def set_defaults_based_on_profile(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure run_profile is valid or default it
        profile_val_str = values.get("run_profile", RunProfile.CONSUMER.value)
        try:
            profile = RunProfile(profile_val_str)
        except ValueError:
            log.warning(f"Invalid run_profile string '{profile_val_str}' in config, defaulting to CONSUMER.")
            profile = RunProfile.CONSUMER
        values["run_profile"] = profile.value # Store the string value back

        # Default DataBus config if not fully specified
        if "data_bus" not in values or not isinstance(values.get("data_bus"), dict):
            values["data_bus"] = {} # Ensure it's a dict
        if "type" not in values["data_bus"]:
            default_db_type = DataBusType.CPP_SHARDED_SPMC if profile == RunProfile.HPC else DataBusType.PYTHON_QUEUE
            values["data_bus"]["type"] = default_db_type.value
            log.debug(f"Auto-set data_bus.type to {default_db_type.value} for profile {profile.value}")

        # Default Monitor config if not fully specified
        if "monitor" not in values or not isinstance(values.get("monitor"), dict):
            values["monitor"] = {}
        if "type" not in values["monitor"]:
            default_mon_type = MonitorType.PROMETHEUS if profile == RunProfile.HPC else MonitorType.CSV_TENSORBOARD
            values["monitor"]["type"] = default_mon_type.value
            log.debug(f"Auto-set monitor.type to {default_mon_type.value} for profile {profile.value}")
        return values

    @root_validator(skip_on_failure=True) # Runs after individual field validation
    def cross_validate_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Now that individual sections are validated Pydantic models, we can access them safely.
        profile: RunProfile = values.get("run_profile") # Already validated enum by now
        data_bus_cfg: DataBusConfig = values.get("data_bus")
        hook_cfg: HookConfig = values.get("hook")
        # hardware_cfg: HardwareConfig = values.get("hardware") # For CUDA checks

        if profile == RunProfile.CONSUMER:
            if data_bus_cfg.type == DataBusType.CPP_SHARDED_SPMC:
                raise ValueError("CONSUMER profile cannot use CppShardedSPMCBus. Set data_bus.type to PYTHON_QUEUE.")
            if hook_cfg.gpu_kernel_mode != "NONE": # TODO: Make gpu_kernel_mode an Enum
                raise ValueError("CONSUMER profile cannot use GPU kernels. Set hook.gpu_kernel_mode to NONE.")
            if values.get("resource_manager", {}).get("apply_bindings", False):
                log.warning("Resource bindings (CPU/NUMA) are enabled in CONSUMER profile. This typically has no effect or is not supported.")

        if profile == RunProfile.HPC:
            if data_bus_cfg.type != DataBusType.CPP_SHARDED_SPMC:
                # This is a strong recommendation, not a fatal error if user explicitly sets PythonQueue for HPC debug
                log.warning(
                    f"HPC profile is active, but DataBus type is {data_bus_cfg.type}. "
                    "Expected CPP_SHARDED_SPMC for optimal performance."
                )
            if hook_cfg.gpu_kernel_mode != "NONE" and not torch.cuda.is_available():
                 raise ValueError(f"HPC profile has GPU kernel mode '{hook_cfg.gpu_kernel_mode}' but CUDA is not available.")
            if not values.get("hardware", {}).get("device", "cpu").startswith("cuda"):
                 log.warning(f"HPC profile typically runs on CUDA, but hardware.device is set to '{values.get('hardware', {}).get('device', 'cpu')}'.")

        return values

    class Config:
        use_enum_values = True # Serialize enums as their string values when dumping
        validate_assignment = True # Raise error if fields are assigned invalid types after init

def _load_yaml_file(path: str) -> Dict[str, Any]:
    # (Same as your reconstructed version)
    if not os.path.exists(path): raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with open(path, "r") as f: return yaml.safe_load(f) or {}
    except yaml.YAMLError as e: log.error(f"Error parsing YAML file {path}: {e}"); raise

def load_config(
    profile_config_path: str, base_config_path: Optional[str] = None
) -> MasterConfig:
    # (Same as your reconstructed version, ensure base_config_path default is sensible or None)
    # If base_config_path is None, it won't try to load it.
    # It's often good to have a fixed default like "configs/base.yaml" if it's standard.
    # For flexibility, let's keep it as is. If None, no base is loaded unless profile_config_path *is* base.
    
    log.info(f"Loading configuration from profile: {profile_config_path}")
    merged_data: Dict[str, Any] = {}

    # Default base_config_path if not provided and not trying to load base as profile
    effective_base_path = base_config_path
    if effective_base_path is None and profile_config_path.lower() != "configs/base.yaml":
        # Heuristic: if a profile is loaded, assume a standard base.yaml exists
        # Or, make it explicit in the calling script.
        # For now, let's assume explicit base_config_path or no base.
        pass


    if effective_base_path and os.path.exists(effective_base_path):
        log.info(f"Loading base configuration from: {effective_base_path}")
        base_data = _load_yaml_file(effective_base_path)
        always_merger.merge(merged_data, base_data) # Base first
    elif effective_base_path: # Path given but not found
        log.warning(f"Base config file specified but not found: {effective_base_path}. Proceeding without it.")


    profile_data = _load_yaml_file(profile_config_path)
    always_merger.merge(merged_data, profile_data) # Profile overrides base

    try:
        config_object = MasterConfig.model_validate(merged_data) # Use model_validate for dicts
        log.info(f"Configuration loaded and validated. Run Profile: {config_object.run_profile.value}")
        # For debugging: log.debug(f"Final config state: {config_object.model_dump_json(indent=2)}")
        return config_object
    except Exception as e: # Catch Pydantic ValidationError specifically if possible
        log.error(f"Pydantic Configuration validation failed: {e}")
        log.error(f"Problematic merged config data before Pydantic parsing: {merged_data}")
        raise
