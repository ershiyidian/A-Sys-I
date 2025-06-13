# src/asys_i/monitoring/monitor_hpc.py
"""
Core Philosophy: Predictability (Performance), Observability-First.
HPC mode monitor implementation using Prometheus.
Low-overhead, pull-based metrics system.
"""
import logging
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Union

# HPC specific imports
try:
    from prometheus_client import (
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        Summary,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Gauge, Counter, Histogram, Summary = (None,) * 4  # type: ignore
    start_http_server = None  # type: ignore
    REGISTRY = None  # type: ignore


from asys_i.common.types import ComponentID
from asys_i.hpc import PROMETHEUS_AVAILABLE as HPC_PROM_AVAIL
from asys_i.hpc import check_hpc_prerequisites
from asys_i.monitoring.monitor_interface import BaseMonitor
from asys_i.orchestration.config_loader import MonitorConfig, ProjectConfig

log = logging.getLogger(__name__)

# Define metric types mapping for auto-creation
METRIC_TYPE_MAP = {
    "count": Counter,
    "latency": Histogram,  # Use Histogram for latency distribution (P50, P99)
    "ms": Histogram,
    "sec": Histogram,
    "rate": Gauge,
    "ratio": Gauge,
    "loss": Gauge,
    "sparsity": Gauge,
    "depth": Gauge,
    "queue": Gauge,
    "size": Gauge,
    "heartbeat": Gauge,
    "default": Gauge,
}
# Buckets for latency histograms (ms) - 0.1ms to 10s
LATENCY_BUCKETS_MS = (
    0.1,
    0.5,
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
)
# Standard BUCKETS
DEFAULT_BUCKETS = Histogram.DEFAULT_BUCKETS if Histogram else ()


class PrometheusMonitor(BaseMonitor):
    """
    HPC Mode Monitor:
    - Exposes metrics via HTTP for Prometheus scraping.
    - Uses a metric registry cache to avoid re-creating metric objects.
    """

    def __init__(
        self,
        monitor_config: MonitorConfig,
        project_config: ProjectConfig,
        shared_heartbeats_dict: dict,
    ):

        if not HPC_PROM_AVAIL:
            # We must fail fast if the dependency isn't met
            raise ImportError(
                "prometheus-client is not installed. Cannot use PrometheusMonitor. "
                "Install with `pip install a-sys-i[hpc]`"
            )
        # check_hpc_prerequisites() # Call this if other HPC deps are mandatory

        super().__init__(monitor_config, project_config, shared_heartbeats_dict)
        self._metrics_cache: Dict[str, Union[Gauge, Counter, Histogram, Summary]] = {}
        self._lock = Lock()  # Protect metric cache creation
        self._server_started = False

        self.port = monitor_config.prometheus_port
        # Pre-define heartbeat gauge
        self._heartbeat_gauge = self._get_or_create_metric(
            "component_heartbeat_timestamp_seconds",
            "Timestamp of the last heartbeat received from a component",
            ["component"],
            Gauge,
        )

        try:
            # Check if a server is already running on the port (e.g., in multiprocessing)
            # This is tricky, for now, just try to start.
            # Only start server in the main process or designated one.
            # TODO: Handle server start in multi-process scenario better (e.g. only main process)
            start_http_server(self.port, registry=REGISTRY)
            self._server_started = True
            log.info(
                f"Prometheus metrics server started at http://0.0.0.0:{self.port}/metrics"
            )
        except OSError as e:
            log.warning(
                f"Could not start Prometheus server on port {self.port} (maybe already running?): {e}"
            )
            self._server_started = False

    def _infer_metric_type(self, name: str) -> Any:
        """Guess the Prometheus metric type based on the name suffix."""
        name_lower = name.lower()
        for suffix, metric_type in METRIC_TYPE_MAP.items():
            if name_lower.endswith(f"_{suffix}"):
                return metric_type
        # Special cases
        if "latency" in name_lower:
            return Histogram
        if "count" in name_lower:
            return Counter
        return METRIC_TYPE_MAP["default"]  # Default to Gauge

    def _get_or_create_metric(
        self,
        name: str,
        description: str,
        label_names: List[str],
        metric_type: Any = None,
    ) -> Any:
        """Thread-safe retrieval or creation of a Prometheus metric object."""
        # Standardize name
        metric_name = f"asys_i_{name.replace('.', '_').replace('-', '_')}"

        with self._lock:
            if metric_name in self._metrics_cache:
                # TODO: Validate label names match
                return self._metrics_cache[metric_name]

            # Create new metric
            inferred_type = metric_type or self._infer_metric_type(name)
            log.debug(
                f"Creating new Prometheus metric: {metric_name} (Type: {inferred_type.__name__}) Labels: {label_names}"
            )

            kwargs = {}
            if inferred_type == Histogram:
                if "latency" in name or "_ms" in name or "_sec" in name:
                    kwargs["buckets"] = LATENCY_BUCKETS_MS
                else:
                    kwargs["buckets"] = DEFAULT_BUCKETS

            try:
                metric = inferred_type(
                    metric_name,
                    description or f"A-Sys-I metric: {name}",
                    labelnames=label_names,
                    **kwargs,
                )
                self._metrics_cache[metric_name] = metric
                return metric
            except ValueError as e:
                # Handle cases where metric exists with different labels or from other process
                log.warning(
                    f"Could not create or retrieve metric {metric_name}: {e}. Checking REGISTRY."
                )
                if metric_name in REGISTRY._names_to_collectors:
                    collector = REGISTRY._names_to_collectors[metric_name]
                    self._metrics_cache[metric_name] = collector  # cache existing
                    return collector
                else:
                    log.error(
                        f"Failed to access metric {metric_name} even from REGISTRY."
                    )
                    raise

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,  # Prometheus is step-agnostic (scraping)
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:

        tags = tags or {}
        # Ensure tags values are strings for prometheus
        label_values = {k: str(v) for k, v in tags.items()}
        label_names = sorted(label_values.keys())
        # Description needed for first creation only
        metric = self._get_or_create_metric(name, f"Metric {name}", label_names)

        # Sort keys to ensure consistent label order
        sorted_label_values = [label_values[k] for k in label_names]

        try:
            target_metric = metric
            if label_names:
                target_metric = metric.labels(*sorted_label_values)

            if isinstance(metric, Counter):
                # Counter must increment, value should be increment amount
                target_metric.inc(value)
            elif isinstance(metric, Gauge):
                target_metric.set(value)
            elif isinstance(metric, (Histogram, Summary)):
                target_metric.observe(value)
            else:
                log.warning(f"Unknown metric type {type(metric)} for {name}")
        except Exception as e:
            log.error(
                f"Error logging Prometheus metric {name} with labels {label_values}: {e}"
            )
            # Design for failure: don't crash the app due to monitoring error

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Log each individually
        for name, value in metrics.items():
            self.log_metric(name, value, step, tags)

    def log_hyperparams(
        self, params: Dict[str, Any], metrics: Optional[Dict[str, float]] = None
    ) -> None:
        # Prometheus uses 'info' metrics for static labels
        info_metric = self._get_or_create_metric(
            "experiment_info", "Experiment Hyperparameters", list(params.keys()), Gauge
        )
        # Represent params as labels, set value to 1
        try:
            labels = {k: str(v) for k, v in params.items()}
            info_metric.labels(**labels).set(1)
        except Exception as e:
            log.error(f"Error logging hyperparams info metric: {e}")

    def heartbeat(self, component_id: ComponentID) -> None:
        super().heartbeat(component_id)  # Update internal cache
        now = time.time()
        try:
            self._heartbeat_gauge.labels(component=component_id).set(now)
        except Exception as e:
            log.error(f"Error setting heartbeat gauge for {component_id}: {e}")

    def shutdown(self) -> None:
        super().shutdown()
        # Prometheus client lib doesn't provide a clean server shutdown func,
        # and it often runs in a daemon thread. In k8s, the pod just dies.
        log.info("PrometheusMonitor shutdown complete (HTTP server may remain active).")
        # Clear cache
        with self._lock:
            self._metrics_cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickleable entries
        if "_lock" in state:
            del state["_lock"]
        if (
            "_metrics_cache" in state
        ):  # Prometheus metric objects are not meant to be pickled
            del state["_metrics_cache"]
        if "_heartbeat_gauge" in state:  # This will be recreated
            del state["_heartbeat_gauge"]
        # Server is not owned by child processes
        state["_server_started"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize unpickleable entries
        self._lock = Lock()
        self._metrics_cache = {}  # Child gets its own cache
        # Re-define heartbeat gauge for this instance if needed
        if PROMETHEUS_AVAILABLE:  # Ensure it's only done if prometheus is available
            self._heartbeat_gauge = self._get_or_create_metric(
                "component_heartbeat_timestamp_seconds",
                "Timestamp of the last heartbeat received from a component",
                ["component"],
                Gauge,
            )
        else:
            self._heartbeat_gauge = None
