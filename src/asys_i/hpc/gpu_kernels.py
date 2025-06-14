# src/asys_i/hpc/gpu_kernels.py
"""
Core Philosophy: Predictability (Performance).
Encapsulates GPU-accelerated preprocessing for activation tensors in HPC mode.
(Quantization, Compression, Sparsification)
"""
import logging
from typing import Any, Dict, Tuple

import torch

from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import HookConfig

log = logging.getLogger(__name__)

# Check availability of advanced libraries
try:
    # import transformer_engine.pytorch as te
    # import nvcomp
    TRANSFORMER_ENGINE_AVAILABLE = False  # Placeholder
    NVCOMP_AVAILABLE = False  # Placeholder
    log.warning(
        "GPU Kernels: transformer_engine and nvcomp not imported (simulation mode)."
    )
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    NVCOMP_AVAILABLE = False
    log.warning(
        "GPU Kernels: transformer_engine or nvcomp not found. Advanced modes disabled."
    )

CUDA_AVAILABLE = torch.cuda.is_available()


def preprocess_tensor_on_gpu(
    tensor: torch.Tensor, config: HookConfig, monitor: BaseMonitor, layer_idx: int
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Applies GPU-based preprocessing according to config.gpu_kernel_mode.
    Runs within the context of a low-priority CUDA stream in ActivationHooker.
    Should be non-blocking relative to the main model stream.
    """
    mode = config.gpu_kernel_mode
    meta: Dict[str, Any] = {"preprocess_mode": mode}
    tags = {"layer": layer_idx, "mode": mode}

    if not CUDA_AVAILABLE:
        log.warning_once("GPU kernel requested but CUDA not available. Skipping.")
        monitor.log_metric(
            "gpu_kernel_error_count", 1, tags={**tags, "reason": "no_cuda"}
        )
        return tensor, meta

    if mode == "NONE":
        return tensor, meta

    # Ensure tensor is on GPU
    if not tensor.is_cuda:
        log.warning_once("GPU kernel requested on CPU tensor. Moving to CUDA.")
        tensor = tensor.cuda()

    processed_tensor = tensor
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    try:
        start_time.record()

        if mode == "FP8":
            if TRANSFORMER_ENGINE_AVAILABLE:
                # Example: processed_tensor, scale = te.quantize_fp8(tensor)
                # meta['scale'] = scale
                log.error("FP8 mode: Simulated - Transformer Engine available but actual kernel not implemented yet. Returns original tensor.") # Changed from warning_once
                processed_tensor = tensor
                # monitor.log_metric("gpu_kernel_quantize_rate", 1, tags=tags) # Keep commented until real
            else:
                log.error("FP8 mode requested but Transformer Engine not available. Returns original tensor.")
                monitor.log_metric(
                    "gpu_kernel_error_count", 1, tags={**tags, "reason": "missing_lib_fp8_te"}
                )

        elif mode == "TOP_K":
            # Using hook_config directly. Future: pass specific params if needed.
            k_fraction = getattr(config, 'top_k_fraction', 0.1) # Default to 10%
            if not (0 < k_fraction < 1):
                log.warning(f"Invalid k_fraction {k_fraction} for TOP_K on layer {layer_idx}. Must be between 0 and 1. Using 0.1.")
                k_fraction = 0.1

            # Calculate k based on the last dimension of the tensor
            k = int(tensor.shape[-1] * k_fraction)

            # Ensure k is at least 1 if the dimension is not empty, to avoid issues with topk(0)
            if k == 0 and tensor.shape[-1] > 0:
                k = 1

            if k > 0:
                # Get the top-k values and their indices
                top_k_values, _ = torch.topk(tensor, k, dim=-1)

                # Create a new tensor filled with zeros (or another fill value if desired)
                processed_tensor = torch.zeros_like(tensor)

                # Determine the threshold: the smallest value among the top k
                # This handles cases where k might be equal to tensor.shape[-1]
                if k < tensor.shape[-1]:
                    threshold = top_k_values[..., -1].unsqueeze(-1) # Smallest of the top k
                    # Create a mask where tensor values are >= threshold
                    # This approach keeps all values if there are ties at the k-th value.
                    mask = tensor >= threshold
                    processed_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
                else: # k is equal or greater than the last dim, so keep all original values
                    processed_tensor = tensor.clone() # Use clone to ensure it's a new tensor if no ops applied

                meta['k_fraction'] = k_fraction
                meta['k_value'] = k
                monitor.log_metric("gpu_kernel_top_k_applied_rate", 1, tags=tags) # Renamed metric
                log.debug(f"Applied TOP_K (k={k}, fraction={k_fraction:.2f}) to tensor on layer {layer_idx} with shape {tensor.shape}")
            else:
                log.warning(f"TOP_K resulted in k=0 for layer {layer_idx} (shape {tensor.shape}, k_fraction {k_fraction:.2f}). Returning original tensor.")
                processed_tensor = tensor # Return original if k is 0

        elif mode == "QUANTIZE_FP16":
            if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
                processed_tensor = tensor.to(torch.float16)
                meta['quantized_to'] = 'float16'
                monitor.log_metric("gpu_kernel_quantize_fp16_rate", 1, tags=tags) # Specific metric
                log.debug(f"Applied QUANTIZE_FP16 to tensor on layer {layer_idx} from {tensor.dtype}")
            else:
                log.warning(f"QUANTIZE_FP16 requested for layer {layer_idx} but tensor dtype is already {tensor.dtype} or not suitable. Returning original tensor.")
                processed_tensor = tensor # Return original if not suitable for fp16 conversion

        elif mode == "FP8_LZ4":
            if NVCOMP_AVAILABLE and TRANSFORMER_ENGINE_AVAILABLE:
                log.error("FP8_LZ4 mode: Simulated - Dependencies available but actual kernel not implemented. Returns original tensor.") # Changed from warning_once
                processed_tensor = tensor
                # monitor.log_metric("gpu_kernel_compress_rate", 1, tags=tags) # Keep commented until real
            else:
                log.error(
                    "FP8_LZ4 mode requested but Transformer Engine or NVCOMP not available. Returns original tensor."
                )
                monitor.log_metric(
                    "gpu_kernel_error_count", 1, tags={**tags, "reason": "missing_lib_fp8_lz4_nvcomp_te"}
                )

        else:
            log.warning(f"Unknown GPU kernel mode: {mode} for layer {layer_idx}. Returning original tensor.")
            monitor.log_metric(
                "gpu_kernel_error_count", 1, tags={**tags, "reason": "unknown_mode"}
            )

        end_time.record()
        # Must sync stream or wait on event to get correct timing outside stream context
        # torch.cuda.current_stream().synchronize()
        # latency_ms = start_time.elapsed_time(end_time)
        # monitor.log_metric("gpu_kernel_latency_ms", latency_ms , tags=tags)
        # Note: Actual latency logging should happen after stream sync in hooker

    except Exception as e:
        log.error(f"Error in GPU kernel mode {mode}: {e}")
        monitor.log_metric(
            "gpu_kernel_error_count", 1, tags={**tags, "reason": "exception"}
        )
        return tensor, {
            "error": str(e),
            "mode": mode,
        }  # Return original tensor on error

    meta["original_dtype"] = str(tensor.dtype)
    meta["original_shape"] = tuple(tensor.shape)
    return processed_tensor, meta
