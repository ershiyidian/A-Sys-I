import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, validator

class BusConfigHPC(BaseModel):
    type: Literal['CppShardedSPMCBus']
    meta_queue_name: str = Field(..., description="Unique name for the metadata message queue.")
    shm_name_prefix: str = Field(..., description="Prefix for shared memory segment names.")
    num_shards: int = Field(4, gt=0, description="Number of parallel shards for the bus.")
    shared_memory_size_gb: int = Field(32, gt=0, description="Total size of shared memory in GB.")
    spin_lock_timeout_us: int = Field(1000, ge=0, description="Timeout in microseconds for spin locks.")

class HookConfigHPC(BaseModel):
    type: Literal['ActivationHooker']
    target_modules: List[str]
    pinned_memory_buffer_size_mb: int = Field(1024, gt=0)

class ObserverConfigHPC(BaseModel):
    type: Literal['HDF5Observer']
    hdf5_chunk_size: int = Field(1024, gt=0)
    hdf5_compression: Optional[str] = 'gzip'

class HPCProfile(BaseModel):
    mode: Literal['HPC']
    output_dir: str
    bus: BusConfigHPC
    hook: HookConfigHPC
    observer: ObserverConfigHPC

    @validator('output_dir', pre=True, always=True)
    def output_dir_from_env(cls, v, values):
        """
        Allows overriding the output_dir from an environment variable.
        This validator runs before Pydantic's type validation.
        """
        env_var = 'ASYS_I_OUTPUT_DIR'
        env_value = os.getenv(env_var)
        if env_value:
            print(f"Used '{env_var}' to set 'output_dir': '{env_value}'")
            return env_value
        # If the env var is not set, Pydantic will use the value from the yaml file.
        # 'v' is the value from the config file.
        return v
