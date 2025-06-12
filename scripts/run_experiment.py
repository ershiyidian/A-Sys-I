"""
Core Philosophy: Config-Driven.
Main entry point for running an A-Sys-I experiment.
"""
import logging
import signal
import sys
import typer
from typing import Optional
import multiprocessing
# Add src to path if running script directly without pip install -e
sys.path.append('.') 
 
from asys_i.orchestration.config_loader import load_config, MasterConfig
from asys_i.orchestration.pipeline import ExperimentPipeline
from asys_i.utils import setup_logging

log = logging.getLogger(__name__)
app = typer.Typer()

# Global reference for signal handler
PIPELINE_INSTANCE: Optional[ExperimentPipeline] = None
 
def signal_handler(sig, frame):
     global PIPELINE_INSTANCE
     log.warning(f"!!! Signal {signal.Signals(sig).name} received. Initiating graceful shutdown... !!!")
     if PIPELINE_INSTANCE:
         # Setting the event is often enough for loops to break
         PIPELINE_INSTANCE.stop_event.set() 
         # Calling shutdown directly ensures cleanup, but finally block also calls it.
         # PIPELINE_INSTANCE.shutdown() 
     else:
          log.error("Pipeline instance not available for shutdown.")
          sys.exit(1)
     # Allow time for loops to detect event before finally block's shutdown
     # time.sleep(1) 

@app.command()
def main(
     config_path: str = typer.Option(
          "configs/profile_consumer.yaml",
           "--config", "-c",
          help="Path to the experiment configuration YAML file."
     ),
      base_config: str = typer.Option(
           "configs/base.yaml",
            "--base-config", "-b",
            help="Path to the base configuration YAML file."
      )
 ):
     """ Runs the A-Sys-I experiment pipeline based on CONFIG_PATH. """
     global PIPELINE_INSTANCE
     pipeline = None
     exit_code = 0
     try:
         config = load_config(config_path, base_config_path=base_config)
         setup_logging(config)
         log.info(f"Starting A-Sys-I experiment: {config.project.name}")
         log.info(f"Run Profile: {config.run_profile}")

         pipeline = ExperimentPipeline(config)
         PIPELINE_INSTANCE = pipeline
         
         # Register signal handlers
         signal.signal(signal.SIGINT, signal_handler)
         signal.signal(signal.SIGTERM, signal_handler)

         pipeline.setup()
         pipeline.run() # This blocks

         log.info("Experiment run completed successfully.")

     except FileNotFoundError as e:
          # Config loading error, logging might not be set up
          print(f"CRITICAL ERROR: Config file not found: {e}", file=sys.stderr)
          exit_code = 1
     except Exception as e:
         log.exception("A critical error occurred during the experiment:")
         exit_code = 1
     finally:
         if pipeline:
             log.info("Ensuring pipeline shutdown...")
             pipeline.shutdown() # Always call shutdown
         else:
              log.warning("Pipeline was not initialized, no shutdown performed.")
         log.info(f"Script finished with exit code {exit_code}.")
         # Need explicit exit for signals sometimes
         sys.exit(exit_code)

if __name__ == "__main__":
     # Make multiprocessing work on different OS (especially Windows/macOS spawn)
     multiprocessing.freeze_support() 
     multiprocessing.set_start_method("spawn", force=True) # 'spawn' is safer with CUDA/torch
     app()

