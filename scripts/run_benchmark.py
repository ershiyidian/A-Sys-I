# scripts/run_benchmark.py
"""
(FINAL IMPLEMENTATION)
Main entry point for running the A-Sys-I benchmark suite.
"""
import typer
import logging
import sys
import multiprocessing
# Add src to path if running script directly without pip install -e
sys.path.append('.') 

from asys_i.orchestration.benchmark_suite import BenchmarkSuite
from asys_i.orchestration.config_loader import load_config
from asys_i.utils import setup_logging

log = logging.getLogger(__name__)
app = typer.Typer()

@app.command()
def main(
     config_path: str = typer.Option(
          "configs/profile_hpc.yaml", # Default to HPC config for benchmarks
           "--config", "-c",
          help="Path to the benchmark configuration YAML file. (Ensure it's an HPC profile for best results)"
     ),
 ):
      """ Runs the A-Sys-I benchmark suite to measure performance. """
      suite = None
      exit_code = 0
      try:
           print("--- A-Sys-I BENCHMARK SUITE ---")
           # Load config (base.yaml will be loaded by default unless config_path specified without it)
           config = load_config(config_path, base_config_path="configs/base.yaml")
           setup_logging(config)
           
           suite = BenchmarkSuite(config)
           suite.setup() # Initialize pipeline and components
           suite.run_all_tests() # Execute all benchmarks
           suite.generate_report() # Print results

           log.info("Benchmark finished successfully.")

      except FileNotFoundError as e:
           print(f"CRITICAL ERROR: Config file not found: {e}", file=sys.stderr)
           exit_code = 1
      except Exception as e:
           log.exception("A critical error occurred during the benchmark:")
           exit_code = 1
      finally:
           if suite:
                log.info("Ensuring benchmark suite shutdown...")
                suite.shutdown() # Always call shutdown
           else:
                log.warning("Benchmark suite was not initialized, no shutdown performed.")
           log.info(f"Benchmark script finished with exit code {exit_code}.")
           sys.exit(exit_code)

if __name__ == "__main__":
      # Crucial for multiprocessing and CUDA stability, especially on Windows/macOS
      multiprocessing.freeze_support() 
      multiprocessing.set_start_method("spawn", force=True) 
      app()
