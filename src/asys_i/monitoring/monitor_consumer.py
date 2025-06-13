# src/asys_i/monitoring/monitor_consumer.py
"""
Core Philosophy: Graceful Degradation, Observability-First.
CONSUMER mode monitor implementation using Logging, CSV, and TensorBoard.
"""
import csv
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import torch

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter  # fallback

        TENSORBOARD_AVAILABLE = True
    except ImportError:
        SummaryWriter = None
        TENSORBOARD_AVAILABLE = False
        logging.warning(
            "TensorBoard/TensorBoardX not found. TensorBoard logging disabled."
        )


from asys_i.common.types import ComponentID
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MonitorConfig, ProjectConfig

log = logging.getLogger(__name__)


class LoggingCSVTensorBoardMonitor(BaseMonitor):
    """
    CONSUMER Mode Monitor:
    - Logs metrics to standard logging.
    - Appends metrics to a CSV file.
    - Writes metrics to TensorBoard.
    """

    def __init__(
        self,
        monitor_config: MonitorConfig,
        project_config: ProjectConfig,
        shared_heartbeats_dict: dict,
    ):
        super().__init__(monitor_config, project_config, shared_heartbeats_dict)
        self.log_dir = os.path.join(project_config.log_dir, project_config.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self._lock = threading.Lock()  # Ensure thread safety for file/writer access

        # CSV
        csv_path = os.path.join(self.log_dir, "metrics.csv")
        self._csv_file = open(csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_header_written = os.path.getsize(csv_path) > 0
        log.info(f"Logging metrics to CSV: {csv_path}")

        # TensorBoard
        self._writer: Optional[SummaryWriter] = None
        if TENSORBOARD_AVAILABLE:
            tb_log_dir = os.path.join(self.log_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=tb_log_dir)
            log.info(f"Logging metrics to TensorBoard: {tb_log_dir}")
        else:
            log.warning("TensorBoard writer is not available.")

        self._last_flush = time.time()

    def __getstate__(self):
        """Prepare the object for pickling. Exclude unpickleable attributes."""
        state = self.__dict__.copy()
        del state["_lock"]
        del state["_csv_file"]
        del state["_csv_writer"]
        if "_writer" in state:  # Tensorboard writer
            del state["_writer"]
        # Store paths and other config needed to re-initialize
        state["_csv_path_for_pickle"] = (
            self._csv_file.name
            if hasattr(self, "_csv_file") and self._csv_file
            else None
        )
        state["_tb_log_dir_for_pickle"] = (
            self._writer.log_dir if hasattr(self, "_writer") and self._writer else None
        )
        return state

    def __setstate__(self, state):
        """Re-initialize unpickleable attributes after unpickling."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

        # Re-open CSV file
        csv_path = state.get("_csv_path_for_pickle")
        if csv_path:
            self._csv_file = open(csv_path, "a", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_header_written = os.path.getsize(csv_path) > 0  # Recheck header
        else:  # Should not happen if correctly pickled from an initialized object
            self._csv_file = None
            self._csv_writer = None
            self._csv_header_written = False

        # Re-initialize TensorBoard writer
        tb_log_dir = state.get("_tb_log_dir_for_pickle")
        if TENSORBOARD_AVAILABLE and tb_log_dir:
            self._writer = SummaryWriter(log_dir=tb_log_dir)
        else:
            self._writer = None

        # Remove helper pickle fields
        if "_csv_path_for_pickle" in self.__dict__:
            del self.__dict__["_csv_path_for_pickle"]
        if "_tb_log_dir_for_pickle" in self.__dict__:
            del self.__dict__["_tb_log_dir_for_pickle"]

    def _format_tag_key(self, name: str, tags: Optional[Dict[str, Any]]) -> str:
        """Creates a tag key like 'metric_name/layer=1,worker=0'"""
        if not tags:
            return name
        tag_str = ",".join([f"{k}={v}" for k, v in sorted(tags.items())])
        return f"{name}/{{{tag_str}}}"

    def _write_csv(
        self,
        timestamp: float,
        name: str,
        value: float,
        step: Optional[int],
        tags: Optional[Dict[str, Any]],
    ):
        row = [timestamp, name, value, step, str(tags)]
        header = ["timestamp", "metric_name", "value", "step", "tags"]
        with self._lock:
            if not self._csv_header_written:
                self._csv_writer.writerow(header)
                self._csv_header_written = True
            self._csv_writer.writerow(row)
            # Auto-flush check
            if (
                time.time() - self._last_flush
                > self.monitor_config.metrics_flush_interval_sec
            ):
                self._csv_file.flush()
                os.fsync(self._csv_file.fileno())
                if self._writer:
                    self._writer.flush()
                self._last_flush = time.time()

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        log.info(f"METRIC: {name}={value} (step={step}, tags={tags})")
        now = time.time()
        tag_key = self._format_tag_key(name, tags)

        self._write_csv(now, name, value, step, tags)

        if self._writer:
            try:
                with self._lock:
                    # Use global_step = 0 if None for consistency
                    self._writer.add_scalar(tag_key, value, global_step=step or 0)
            except Exception as e:
                log.error(f"Tensorboard write error for {tag_key}: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        scalars_dict = {}
        # Log each and prepare for add_scalars / csv
        for name, value in metrics.items():
            log.info(f"METRIC: {name}={value} (step={step}, tags={tags})")
            self._write_csv(now, name, value, step, tags)
            scalars_dict[name] = value

        if self._writer:
            try:
                with self._lock:
                    # Use tag_key for add_scalars main_tag
                    tag_str = ",".join(
                        [f"{k}={v}" for k, v in sorted((tags or {}).items())]
                    )
                    main_tag = f"batch_metrics/{{{tag_str}}}"
                    # add_scalars writes all metrics under main_tag/metric_name
                    self._writer.add_scalars(
                        main_tag, scalars_dict, global_step=step or 0
                    )
            except Exception as e:
                log.error(f"Tensorboard add_scalars error: {e}")

    def log_hyperparams(
        self, params: Dict[str, Any], metrics: Optional[Dict[str, float]] = None
    ) -> None:
        log.info(f"HYPERPARAMETERS: {params}")
        if self._writer:
            try:
                # Ensure all values are serializable for hparams
                valid_params = {
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (int, float, str, bool, torch.Tensor))
                }
                valid_metrics = metrics or {"dummy_metric": 0.0}
                valid_metrics = {
                    k: v
                    for k, v in valid_metrics.items()
                    if isinstance(v, (int, float, str, bool, torch.Tensor))
                }
                with self._lock:
                    self._writer.add_hparams(valid_params, valid_metrics)
            except Exception as e:
                log.error(f"Tensorboard add_hparams error: {e}")

    def heartbeat(self, component_id: ComponentID) -> None:
        super().heartbeat(component_id)  # Update internal cache
        # Also log as metric
        self.log_metric(
            "component_heartbeat", time.time(), tags={"component": component_id}
        )

    def shutdown(self) -> None:
        super().shutdown()
        with self._lock:
            if self._csv_file and not self._csv_file.closed:
                self._csv_file.flush()
                os.fsync(self._csv_file.fileno())
                self._csv_file.close()
                log.info("CSV file closed.")
            if self._writer:
                self._writer.flush()
                self._writer.close()
                log.info("TensorBoard writer closed.")
