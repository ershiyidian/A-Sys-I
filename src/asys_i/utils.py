 # src/asys_i/utils.py
import logging
import sys
import os
from typing import TYPE_CHECKING
try:
     from rich.logging import RichHandler
     RICH_AVAILABLE = True
except ImportError:
      RICH_AVAILABLE = False
      RichHandler = None # type:ignore

if TYPE_CHECKING:
      from asys_i.orchestration.config_loader import MasterConfig

def setup_logging(config: 'MasterConfig'):
    """ Configures logging based on project config. """
    log_level_str = config.project.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_dir = config.project.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{config.project.name}.log")

    handlers: List[logging.Handler] = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
         '%(asctime)s - %(name)s - %(levelname)s - PID:%(process)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)

    # Console handler
    if RICH_AVAILABLE:
         console_handler = RichHandler(rich_tracebacks=True, show_path=False)
         # RichHandler doesn't need formatter
         handlers.append(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    logging.basicConfig(
         level=log_level,
         handlers=handlers,
         force=True # Override any existing basicConfig
     )
    # Silence noisy libraries
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("prometheus_client").setLevel(logging.INFO)
     
    log = logging.getLogger(__name__)
    log.info(f"Logging initialized. Level: {log_level_str}, File: {log_file}")
    if not RICH_AVAILABLE:
         log.info("Install 'rich' for better console logging.")

