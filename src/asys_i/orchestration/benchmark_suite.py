# src/asys_i/orchestration/benchmark_suite.py
"""
(FINAL IMPLEMENTATION)
Runs a suite of standardized tests to measure the performance of the A-Sys-I
framework and generates a report. Does not assert against a fixed SLA, but
provides the data needed for validation.
"""
import logging
import time
import torch
import numpy as np
import threading
from typing import List, Optional

from asys_i.orchestration.config_loader import MasterConfig
from asys_i.orchestration.pipeline import ExperimentPipeline
from asys_i.common.types import RunProfile

log = logging.getLogger(__name__)

class BenchmarkSuite:
    def __init__(self, config: MasterConfig):
        # Force benchmark profile for consistency and specific settings
        config.run_profile = RunProfile.BENCHMARK
        # For stress testing, ensure high sampling rate and relevant layers
        config.hook.sampling_rate = 1.0
        if not config.hook.layers_to_hook:
            log.warning("No layers defined in hook.layers_to_hook. Benchmarks may not capture meaningful data.")
            # Default to first few layers for sanity if not set
            config.hook.layers_to_hook = [0,1,2] 

        self.config = config
        self.pipeline: Optional[ExperimentPipeline] = None # Will be initialized in setup
        self.results: dict = {}
        log.info("BenchmarkSuite initialized with BENCHMARK profile.")

    def setup(self):
        """Sets up the pipeline for benchmarking."""
        log.info("--- Setting up benchmark environment ---")
        self.pipeline = ExperimentPipeline(self.config)
        self.pipeline.setup()
        # For some tests, we don't want trainers/archivers running immediately
        # We stop them to ensure clean state for individual tests.
        if self.pipeline.trainer_manager:
            self.pipeline.trainer_manager.stop_all()
        if self.pipeline.archiver_manager:
            self.pipeline.archiver_manager.stop_all()

    def _run_host_for_duration(self, duration_sec: int) -> int:
        """
        Runs the PPO host loop in a separate thread for a fixed duration,
        and returns the number of steps completed.
        """
        assert self.pipeline and self.pipeline.ppo_host and self.pipeline.hooker
        log.info(f"Running host model for {duration_sec} seconds to generate data...")
        
        # Reset global step count for precise measurement
        initial_global_step = self.pipeline.ppo_host._global_step 
        
        # Reset the stop event for a fresh run
        self.pipeline.stop_event.clear()

        host_thread = threading.Thread(
            target=self.pipeline.ppo_host.run_training_loop,
            args=(self.pipeline.hooker,),
            daemon=True
        )
        host_thread.start()
        
        start_time = time.time()
        # Wait for the duration, periodically checking if host thread is still alive
        while host_thread.is_alive() and (time.time() - start_time < duration_sec):
            time.sleep(1) # Sleep to avoid busy-waiting

        # Signal the host thread to stop if it hasn't already
        log.info("Benchmark duration elapsed or host finished. Signaling host to stop.")
        self.pipeline.stop_event.set()
        
        # Wait for the host thread to finish gracefully
        host_thread.join(timeout=10)
        if host_thread.is_alive():
             log.error("Host thread did not stop cleanly within timeout for benchmark.")
             # Force terminate if it's stuck, though this is a harsher cleanup
             # if host_thread.ident: os.kill(host_thread.ident, signal.SIGTERM) 
        
        # Calculate steps completed during this duration
        steps_completed = self.pipeline.ppo_host._global_step - initial_global_step
        log.info(f"Host completed {steps_completed} steps during benchmark duration.")
        return steps_completed
        
    def test_hook_latency(self):
        """Measures the end-to-end latency of the activation hook."""
        log.info("\n--- Running Test: Hook Latency (Pure Hook + DataBus Push) ---")
        assert self.pipeline and self.pipeline.trainer_manager and self.pipeline.archiver_manager and self.pipeline.monitor

        # Ensure no consumers are running to measure pure hook+push performance
        self.pipeline.trainer_manager.stop_all()
        self.pipeline.archiver_manager.stop_all()

        # Collect metrics from the monitor. PrometheusMonitor collects histograms.
        # For CSV/Tensorboard, we would need to parse logs or capture through a custom monitor.
        # For this test, we assume the chosen monitor type (e.g., Prometheus) will have relevant data.
        # We can also temporarily swap the monitor's log_metric for direct capture.
        
        latency_metrics: List[float] = []
        original_log_metric = self.pipeline.monitor.log_metric
        # Temporarily override log_metric to capture values directly for this test
        def temp_metric_collector(name, value, step=None, tags=None):
             if name == "hook_total_latency_ms":
                 latency_metrics.append(value)
             original_log_metric(name, value, step, tags) # Still log to original monitor
        
        self.pipeline.monitor.log_metric = temp_metric_collector
        
        # Run host for a short duration to generate enough samples
        self._run_host_for_duration(duration_sec=10) # 10 seconds should generate many packets
        
        # Restore original monitor method
        self.pipeline.monitor.log_metric = original_log_metric 
        
        if not latency_metrics:
             log.error("No hook_total_latency_ms metrics were collected.")
             self.results['hook_latency'] = {"error": "No data collected. Check hook config and host model activity."}
             return

        # Calculate percentiles
        p50 = np.percentile(latency_metrics, 50)
        p95 = np.percentile(latency_metrics, 95)
        p99 = np.percentile(latency_metrics, 99)
        avg = np.mean(latency_metrics)
        
        self.results['hook_latency'] = {
            "p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "avg_ms": avg, 
            "samples": len(latency_metrics)
        }
        log.info(f"Hook Latency Results (n={len(latency_metrics)}): P99={p99:.3f}ms, P50={p50:.3f}ms, Avg={avg:.3f}ms")

    def test_bus_throughput(self):
        """Measures the data throughput of the DataBus."""
        log.info("\n--- Running Test: DataBus Throughput (Hooker + Single Consumer) ---")
        assert self.pipeline and self.pipeline.data_bus and self.pipeline.monitor
        
        # Ensure trainers and archivers are stopped to not consume data
        if self.pipeline.trainer_manager: self.pipeline.trainer_manager.stop_all()
        if self.pipeline.archiver_manager: self.pipeline.archiver_manager.stop_all()

        consumer_stop_event = threading.Event()
        pulled_bytes = 0
        
        def dummy_consumer():
            nonlocal pulled_bytes
            log.info("Dummy throughput consumer started.")
            # Dummy consumer subscribes to all available layers for max load
            all_layers = self.config.hook.layers_to_hook if self.config.hook.layers_to_hook else list(range(10)) # Heuristic if empty
            self.pipeline.data_bus.register_consumer("benchmark_consumer", all_layers)
            
            while not consumer_stop_event.is_set():
                packets = self.pipeline.data_bus.pull_batch("benchmark_consumer", timeout=0.5)
                for p in packets:
                    if isinstance(p['data'], torch.Tensor):
                        pulled_bytes += p['data'].nelement() * p['data'].element_size()
                    else: # If using TensorRef in HPC, need to "reconstruct" to get bytes
                        # This would be a more complex measure, assuming `data` is raw bytes or a numpy array
                        log.debug("Dummy consumer received non-tensor data type, skipping byte count.")
                if not packets: # Avoid busy-wait if queue is empty
                    time.sleep(0.01)
            log.info("Dummy throughput consumer stopped.")

        consumer_thread = threading.Thread(target=dummy_consumer, daemon=True)
        consumer_thread.start()

        # Run host for a fixed duration to generate data and allow consumer to pull
        test_duration_sec = 15
        log.info(f"Generating data for {test_duration_sec} seconds...")
        self._run_host_for_duration(duration_sec=test_duration_sec)
        
        log.info("Signaling dummy consumer to stop and collecting final metrics.")
        consumer_stop_event.set()
        consumer_thread.join(timeout=5)
        
        throughput_gbps = (pulled_bytes / 1e9) / test_duration_sec
        
        self.results['bus_throughput'] = {"gbps": throughput_gbps, "total_gb_transferred": pulled_bytes / 1e9}
        log.info(f"DataBus Throughput Results: {throughput_gbps:.4f} GB/s (Total: {pulled_bytes / 1e9:.2f} GB)")

    def test_isolation(self):
        """Measures the performance impact on the host model when A-Sys-I components are active."""
        log.info("\n--- Running Test: Host Model Isolation ---")
        assert self.pipeline and self.pipeline.trainer_manager
        
        test_duration_sec = 20 # Each run

        # 1. Baseline: Run host with hooks active, but no consumers (trainers/archivers)
        self.pipeline.trainer_manager.stop_all()
        if self.pipeline.archiver_manager: self.pipeline.archiver_manager.stop_all()
        
        log.info("Measuring baseline host performance (hooks active, no consumers)...")
        baseline_steps_completed = self._run_host_for_duration(duration_sec=test_duration_sec)
        baseline_steps_per_sec = baseline_steps_completed / test_duration_sec
        log.info(f"Baseline host performance: {baseline_steps_per_sec:.2f} steps/sec")

        # 2. Under Load: Run host with SAE trainers active (full A-Sys-I load)
        log.info("Measuring host performance under load (SAE trainers active)...")
        self.pipeline.trainer_manager.start_all() # Start all trainers
        
        loaded_steps_completed = self._run_host_for_duration(duration_sec=test_duration_sec)
        loaded_steps_per_sec = loaded_steps_completed / test_duration_sec
        log.info(f"Loaded host performance: {loaded_steps_per_sec:.2f} steps/sec")

        # Ensure trainers are stopped after the test
        self.pipeline.trainer_manager.stop_all()

        performance_impact_percent = 0.0
        if baseline_steps_per_sec > 0:
            performance_impact_percent = ((baseline_steps_per_sec - loaded_steps_per_sec) / baseline_steps_per_sec) * 100
        
        self.results['isolation'] = {
            "baseline_steps_per_sec": baseline_steps_per_sec,
            "loaded_steps_per_sec": loaded_steps_per_sec,
            "performance_impact_percent": performance_impact_percent
        }
        log.info(f"Host Isolation Results: Performance impact = {performance_impact_percent:.2f}%")

    def run_all_tests(self):
        """Executes the entire suite of benchmark tests."""
        # Ensure initial state is clean
        if not self.pipeline:
             log.error("Pipeline not set up for benchmarks. Aborting tests.")
             return
        if self.pipeline.trainer_manager: self.pipeline.trainer_manager.stop_all()
        if self.pipeline.archiver_manager: self.pipeline.archiver_manager.stop_all()
        self.pipeline.stop_event.clear() # Ensure no residual stop signals

        log.info("\n========== STARTING A-Sys-I BENCHMARK SUITE ==========")
        self.test_hook_latency()
        self.pipeline.stop_event.clear() # Clear for next test
        self.test_bus_throughput()
        self.pipeline.stop_event.clear() # Clear for next test
        self.test_isolation()
        log.info("\n========== A-Sys-I BENCHMARK SUITE COMPLETED ==========")

    def generate_report(self):
        """Prints a summarized benchmark report."""
        print("\n" + "="*50)
        print("          A-Sys-I Benchmark Report")
        print("="*50)
        print(f"Profile: {self.config.run_profile.value}, DataBus: {self.config.data_bus.type.value}")
        print(f"Host Model d_in: {self.config.sae_model.d_in}")
        print(f"Trainer Workers: {self.config.sae_trainer.num_workers}")
        print(f"CPU Affinity: {self.config.resource_manager.allocation_strategy} ({self.config.resource_manager.cpu_affinity_map if self.config.resource_manager.apply_bindings else 'Disabled'})")

        latency = self.results.get('hook_latency', {})
        print("\n[Hook Latency (ms)]")
        if "error" in latency:
            print(f"  Error: {latency['error']}")
        else:
            print(f"  - P99 Latency: {latency.get('p99_ms', -1):.3f}")
            print(f"  - P95 Latency: {latency.get('p95_ms', -1):.3f}")
            print(f"  - Avg Latency: {latency.get('avg_ms', -1):.3f}")
            print(f"  - Samples: {latency.get('samples', 0)}")
        
        throughput = self.results.get('bus_throughput', {})
        print("\n[DataBus Throughput]")
        print(f"  - Throughput: {throughput.get('gbps', -1):.4f} GB/s")

        isolation = self.results.get('isolation', {})
        print("\n[Host Model Isolation (Impact on steps/sec)]")
        print(f"  - Baseline Performance: {isolation.get('baseline_steps_per_sec',-1):.2f} steps/sec")
        print(f"  - Performance w/ A-Sys-I: {isolation.get('loaded_steps_per_sec',-1):.2f} steps/sec")
        print(f"  - Performance Impact: {isolation.get('performance_impact_percent',-1):.2f} %")
        print("="*50)

    def shutdown(self):
        """Performs a graceful shutdown of the benchmark environment."""
        log.info("--- Shutting down benchmark environment ---")
        if self.pipeline:
            self.pipeline.shutdown()
        log.info("BenchmarkSuite shutdown complete.")

